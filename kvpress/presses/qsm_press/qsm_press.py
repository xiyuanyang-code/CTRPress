from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .config import SemanticScoreConfig
from .merge_utils import compute_removed_indices, merge_removed_values
from .query_aware_press import QueryAwarePress
from .semantic_score import compute_semantic_weights


@dataclass
class QSMPress(QueryAwarePress):
    tokenizer: object | None = None
    use_semantic: bool = True
    lambda_sem: float = 0.3
    semantic_config: SemanticScoreConfig = field(default_factory=SemanticScoreConfig)
    use_merge: bool = True
    merge_in_pseudo_query: bool = False
    merge_alpha: float = 0.2
    merge_target: str = "nearest"
    merge_weighting: str = "score"
    merge_score_power: float = 4.0
    merge_min_score_ratio: float = 0.5
    merge_count_power: float = 0.75
    semantic_weights: torch.Tensor | None = field(default=None, init=False, repr=False)
    last_merge_scores_by_layer: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.lambda_sem < 0:
            raise ValueError("lambda_sem must be non-negative.")
        if self.merge_alpha < 0:
            raise ValueError("merge_alpha must be non-negative.")
        if self.merge_weighting not in {"uniform", "score"}:
            raise ValueError("merge_weighting must be 'uniform' or 'score'.")
        if self.merge_score_power <= 0:
            raise ValueError("merge_score_power must be positive.")
        if not 0.0 <= self.merge_min_score_ratio <= 1.0:
            raise ValueError("merge_min_score_ratio must be in [0, 1].")
        if not 0.0 <= self.merge_count_power <= 1.0:
            raise ValueError("merge_count_power must be in [0, 1].")

    def set_runtime_state(
        self,
        input_ids: torch.Tensor | None = None,
        query_length: int = 0,
        tokenizer=None,
    ) -> None:
        super().set_runtime_state(input_ids=input_ids, query_length=query_length)
        tokenizer = tokenizer or self.tokenizer
        if self.use_semantic and input_ids is not None and tokenizer is not None:
            self.semantic_weights = compute_semantic_weights(input_ids, tokenizer, self.semantic_config)
        else:
            self.semantic_weights = None

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        hidden_states = kwargs.get("hidden_states", input[0] if input else None)
        if hidden_states is not None:
            self.runtime_input_ids = hidden_states
        if self.use_semantic and self.tokenizer is not None and self.semantic_weights is None:
            self.semantic_weights = compute_semantic_weights(
                self.runtime_input_ids, self.tokenizer, self.semantic_config
            ) if self.runtime_input_ids is not None and self.runtime_input_ids.dim() == 2 else None
        return super().forward_hook(module, input, kwargs, output)

    def _removed_merge_weights(
        self,
        scores: torch.Tensor,
        forced_keep_mask: torch.Tensor,
        kept_indices: torch.Tensor,
        removed_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.merge_weighting == "uniform" or removed_indices.numel() == 0:
            return None

        aggregate_scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if aggregate_scores.dim() == 3:
            aggregate_scores = aggregate_scores.mean(dim=1)
        removed_scores = aggregate_scores.gather(1, removed_indices).clamp_min(0.0)

        kept_mask = torch.zeros_like(forced_keep_mask)
        kept_flat = kept_indices.reshape(kept_indices.shape[0], -1) if kept_indices.dim() == 3 else kept_indices
        kept_mask.scatter_(1, kept_flat, True)
        nonforced_kept_mask = kept_mask & ~forced_keep_mask
        thresholds = aggregate_scores.masked_fill(~nonforced_kept_mask, torch.inf).amin(dim=-1, keepdim=True)
        thresholds = torch.where(torch.isfinite(thresholds), thresholds.clamp_min(1e-6), torch.ones_like(thresholds))

        relative_scores = (removed_scores / thresholds).clamp(0.0, 1.0)
        if self.merge_min_score_ratio > 0:
            relative_scores = relative_scores.masked_fill(relative_scores < self.merge_min_score_ratio, 0.0)
        return relative_scores.pow(self.merge_score_power)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> torch.Tensor:
        scores = super().score(module, hidden_states, keys, values, attentions, kwargs)
        self.last_merge_scores_by_layer[int(module.layer_idx)] = scores.detach()
        if not self.use_semantic or self.semantic_weights is None:
            return scores

        semantic_weights = self.semantic_weights.to(device=scores.device, dtype=scores.dtype)
        semantic_weights = semantic_weights[:, : scores.shape[-1]]
        forced_keep_mask = self.last_forced_keep_mask_by_layer[int(module.layer_idx)]
        candidate_scores = scores.masked_fill(forced_keep_mask[:, None, :], 0.0)
        semantic_gate = torch.tanh(semantic_weights)[:, None, :]
        modified_scores = candidate_scores * (1.0 + self.lambda_sem * semantic_gate)
        modified_scores = torch.nan_to_num(modified_scores, nan=0.0, posinf=0.0, neginf=0.0)
        modified_scores = modified_scores.masked_fill(forced_keep_mask[:, None, :], torch.finfo(scores.dtype).max)
        self.last_scores_by_layer[int(module.layer_idx)] = modified_scores.detach()
        return modified_scores

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio == 0:
            return keys, values

        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        seq_len = keys.shape[2]
        layer_idx = int(module.layer_idx)
        forced_keep_mask = self.last_forced_keep_mask_by_layer[layer_idx]

        # Select top-k indices
        n_kept = int(seq_len * (1.0 - self.compression_ratio))
        safe_scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
        forced_count = forced_keep_mask.sum(dim=-1)
        n_kept = max(n_kept, int(forced_count[0].item()))
        n_kept = min(n_kept, seq_len)
        ranking_scores = safe_scores.masked_fill(forced_keep_mask[:, None, :], torch.finfo(safe_scores.dtype).max)
        kept_indices = ranking_scores.topk(n_kept, dim=-1).indices
        kept_indices = torch.sort(kept_indices, dim=-1).values
        self.retained_indices_by_layer[layer_idx] = kept_indices.detach()

        # Gather compressed keys
        head_dim = module.head_dim
        expand = kept_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
        compressed_keys = keys.gather(2, expand).contiguous()

        # Merge values if enabled
        if not self.use_merge or (self.runtime_query_length == 0 and not self.merge_in_pseudo_query):
            return compressed_keys, values.gather(2, expand).contiguous()

        removed_indices = compute_removed_indices(seq_len, kept_indices)
        merge_scores = self.last_merge_scores_by_layer[layer_idx]
        removed_weights = self._removed_merge_weights(merge_scores, forced_keep_mask, kept_indices, removed_indices)
        compressed_values = merge_removed_values(
            values=values,
            kept_indices=kept_indices,
            removed_indices=removed_indices,
            merge_alpha=self.merge_alpha,
            removed_weights=removed_weights,
            target_strategy=self.merge_target,
            count_power=self.merge_count_power,
        )
        return compressed_keys, compressed_values


@dataclass
class QASemanticPress(QSMPress):
    use_semantic: bool = True
    use_merge: bool = False


@dataclass
class QAMergePress(QSMPress):
    use_semantic: bool = False
    use_merge: bool = True
