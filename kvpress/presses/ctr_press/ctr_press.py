from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.qsm_press import QSMPress
from kvpress.presses.qsm_press.merge_utils import compute_removed_indices, merge_removed_values
from .rescue import adjust_kept_by_entropy, compute_score_entropy, rescue_score


@dataclass
class CTRPress(QSMPress):
    """
    Compress-then-Refine Press.

    Extends QSM-Press with two mechanisms:
    1. Layer-adaptive compression: adjust per-layer compression based on attention entropy.
    2. Rescue merge: after first-pass compression, identify and rescue high-value
       discarded tokens by merging their values back into the kept set.

    The flow:
        Pass 1: QSM-Press compress (query-aware + semantic + merge)
        Analyze: Compute attention entropy per layer
        Refine: Rescue important discarded tokens via secondary merge
    """

    # Rescue parameters
    rescue_ratio: float = 0.1
    rescue_scoring: str = "key_miss"

    # Layer-adaptive parameters
    use_layer_adaptive: bool = True
    entropy_beta: float = 0.2

    # Internal state
    _layer_entropy: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _max_layer_idx: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.rescue_ratio <= 1.0:
            raise ValueError("rescue_ratio must be in [0, 1].")
        if self.rescue_scoring not in {"key_miss"}:
            raise ValueError("rescue_scoring must be 'key_miss'.")
        if self.entropy_beta < 0:
            raise ValueError("entropy_beta must be non-negative.")

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

        layer_idx = int(module.layer_idx)
        self._max_layer_idx = max(self._max_layer_idx, layer_idx)

        # === Pass 1: Standard QSM-Press compress ===
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        seq_len = keys.shape[2]
        forced_keep_mask = self.last_forced_keep_mask_by_layer[layer_idx]

        # Compute base n_kept
        n_kept = int(seq_len * (1.0 - self.compression_ratio))
        safe_scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
        forced_count = forced_keep_mask.sum(dim=-1)
        n_kept = max(n_kept, int(forced_count[0].item()))
        n_kept = min(n_kept, seq_len)

        # Layer-adaptive: adjust n_kept based on score entropy
        if self.use_layer_adaptive:
            entropy = compute_score_entropy(scores)
            self._layer_entropy[layer_idx] = entropy.mean().item()
            n_kept = adjust_kept_by_entropy(n_kept, seq_len, entropy, self.entropy_beta)
        else:
            self._layer_entropy[layer_idx] = 0.0

        # Select top-k indices
        ranking_scores = safe_scores.masked_fill(
            forced_keep_mask[:, None, :], torch.finfo(safe_scores.dtype).max
        )
        kept_indices = ranking_scores.topk(n_kept, dim=-1).indices
        kept_indices = torch.sort(kept_indices, dim=-1).values
        self.retained_indices_by_layer[layer_idx] = kept_indices.detach()

        # Gather compressed keys
        head_dim = module.head_dim
        expand = kept_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
        compressed_keys = keys.gather(2, expand).contiguous()

        # Compute removed indices
        removed_indices = compute_removed_indices(seq_len, kept_indices)

        # First-pass merge (standard QSM merge)
        use_merge = self.use_merge and (
            self.runtime_query_length > 0 or self.merge_in_pseudo_query
        )
        if not use_merge:
            compressed_values = values.gather(2, expand).contiguous()
        else:
            merge_scores = self.last_merge_scores_by_layer[layer_idx]
            removed_weights = self._removed_merge_weights(
                merge_scores, forced_keep_mask, kept_indices, removed_indices
            )
            compressed_values = merge_removed_values(
                values=values,
                kept_indices=kept_indices,
                removed_indices=removed_indices,
                merge_alpha=self.merge_alpha,
                removed_weights=removed_weights,
                target_strategy=self.merge_target,
                count_power=self.merge_count_power,
            )

        # === Pass 2: Rescue refinement ===
        if self.rescue_ratio > 0 and removed_indices.numel() > 0:
            compressed_values = self._rescue_refine(
                module=module,
                hidden_states=hidden_states,
                keys=keys,
                values=values,
                compressed_keys=compressed_keys,
                compressed_values=compressed_values,
                kept_indices=kept_indices,
                removed_indices=removed_indices,
                kwargs=kwargs,
            )

        return compressed_keys, compressed_values

    def _rescue_refine(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        compressed_keys: torch.Tensor,
        compressed_values: torch.Tensor,
        kept_indices: torch.Tensor,
        removed_indices: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        """
        Second pass: identify high-value discarded tokens and merge them
        back into the compressed values.
        """
        n_removed = removed_indices.shape[1]
        n_rescue = max(1, int(n_removed * self.rescue_ratio))

        # Gather removed keys for rescue scoring
        removed_expand = removed_indices[:, None, :, None].expand(
            -1, keys.shape[1], -1, module.head_dim
        )
        removed_keys = keys.gather(2, removed_expand).contiguous()

        # Compute rescue scores
        position_embeddings = kwargs.get("position_embeddings")
        scores = rescue_score(
            module, hidden_states, compressed_keys, removed_keys, position_embeddings
        )

        # Select top rescue candidates
        n_rescue = min(n_rescue, n_removed)
        _, rescue_local_indices = scores.topk(n_rescue, dim=-1)
        rescue_local_indices = torch.sort(rescue_local_indices, dim=-1).values

        # Map local indices back to original sequence positions
        rescue_orig_indices = removed_indices.gather(1, rescue_local_indices)

        # Gather rescued values
        rescue_expand = rescue_orig_indices[:, None, :, None].expand(
            -1, values.shape[1], -1, module.head_dim
        )
        rescued_values = values.gather(2, rescue_expand).contiguous()

        # Normalize rescue weights
        rescued_weights = scores.gather(1, rescue_local_indices)
        rescued_weights = rescued_weights / rescued_weights.amax(dim=-1, keepdim=True).clamp_min(1e-6)

        # Find nearest kept token for each rescued token
        # Use first head's kept positions if 3D (valid for GQA/MQA)
        kept_2d = kept_indices[:, 0, :] if kept_indices.dim() == 3 else kept_indices
        distances = (rescue_orig_indices.unsqueeze(-1) - kept_2d.unsqueeze(1)).abs()
        nearest_slots = distances.argmin(dim=-1)  # [batch, n_rescue]

        # Inline merge: blend rescued values into nearest kept values
        batch_size, num_heads, n_kept, head_dim = compressed_values.shape
        slot_expand = nearest_slots[:, None, :, None].expand(batch_size, num_heads, n_rescue, head_dim)

        # Ensure dtype consistency (rescue_score returns float32, values may be float16)
        rescued_weights = rescued_weights.to(dtype=compressed_values.dtype)

        contributions = torch.zeros_like(compressed_values)
        weighted_rescued = rescued_values * rescued_weights[:, None, :, None]
        contributions.scatter_add_(2, slot_expand, weighted_rescued)

        weight_sums = torch.zeros(batch_size, 1, n_kept, 1, dtype=compressed_values.dtype, device=compressed_values.device)
        weight_sums.scatter_add_(2, nearest_slots[:, None, :, None], rescued_weights[:, None, :, None])

        merge_alpha = self.merge_alpha * 2.0
        effective_mass = weight_sums.pow(0.25)
        mean_contributions = contributions / weight_sums.clamp_min(torch.finfo(values.dtype).eps)
        blended = (compressed_values + merge_alpha * effective_mass * mean_contributions) / (
            1.0 + merge_alpha * effective_mass
        )
        return torch.where(weight_sums > 0, blended, compressed_values)


@dataclass
class CTRSemanticPress(CTRPress):
    """CTR with semantic scoring, no rescue."""
    use_semantic: bool = True
    rescue_ratio: float = 0.0


@dataclass
class CTRRefinePress(CTRPress):
    """CTR with rescue only, no layer-adaptive or semantic."""
    use_semantic: bool = False
    use_layer_adaptive: bool = False
