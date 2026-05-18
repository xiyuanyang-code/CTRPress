from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import get_prerope_query_states


def _group_queries(query_states: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    batch_size, num_heads, seq_len, head_dim = query_states.shape
    if num_heads == num_kv_heads:
        return query_states
    group_size = num_heads // num_kv_heads
    return query_states.view(batch_size, num_kv_heads, group_size, seq_len, head_dim).mean(dim=2)


def _normalize_scores(scores: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    min_score = scores.amin(dim=-1, keepdim=True)
    max_score = scores.amax(dim=-1, keepdim=True)
    return (scores - min_score) / (max_score - min_score).clamp_min(eps)


@dataclass
class QueryAwarePress(ScorerPress):
    pseudo_query_len: int = 128
    pseudo_query_max_fraction: float = 0.125
    qa_alpha: float = 0.5
    sink_tokens: int = 4
    recent_tokens: int = 32
    keep_pseudo_query: bool = True
    use_query_aware: bool = True
    score_normalization: bool = True

    runtime_input_ids: torch.Tensor | None = field(default=None, init=False, repr=False)
    runtime_query_length: int = field(default=0, init=False, repr=False)
    last_forced_keep_mask_by_layer: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    last_scores_by_layer: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    retained_indices_by_layer: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.qa_alpha <= 1:
            raise ValueError("qa_alpha must be in [0, 1].")
        if not 0 < self.pseudo_query_max_fraction <= 1:
            raise ValueError("pseudo_query_max_fraction must be in (0, 1].")
        if self.pseudo_query_len < 0 or self.sink_tokens < 0 or self.recent_tokens < 0:
            raise ValueError("Token counts must be non-negative.")

    def set_runtime_state(self, input_ids: torch.Tensor | None = None, query_length: int = 0) -> None:
        self.runtime_input_ids = input_ids
        self.runtime_query_length = int(query_length)

    def _query_span(self, seq_len: int) -> tuple[int, int]:
        if self.runtime_query_length > 0:
            query_len = min(self.runtime_query_length, seq_len)
            return seq_len - query_len, seq_len
        max_fraction_len = max(1, int(seq_len * self.pseudo_query_max_fraction))
        query_len = min(self.pseudo_query_len, max_fraction_len, seq_len)
        return seq_len - query_len, seq_len

    def _forced_keep_mask(self, seq_len: int, batch_size: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        query_start, query_end = self._query_span(seq_len)
        has_explicit_query = self.runtime_query_length > 0
        context_end = query_start if has_explicit_query else seq_len

        if self.sink_tokens:
            mask[:, : min(self.sink_tokens, context_end)] = True
        if self.recent_tokens:
            recent_start = max(0, context_end - self.recent_tokens)
            mask[:, recent_start:context_end] = True
        if has_explicit_query and query_end > query_start:
            mask[:, query_start:query_end] = True
        if not has_explicit_query and self.keep_pseudo_query and query_end > query_start:
            mask[:, query_start:query_end] = True
        return mask

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> torch.Tensor:
        batch_size, _, seq_len, head_dim = keys.shape
        num_kv_heads = keys.shape[1]
        query_start, query_end = self._query_span(seq_len)
        query_len = query_end - query_start

        if not self.use_query_aware or query_len == 0:
            scores = torch.zeros(batch_size, num_kv_heads, seq_len, dtype=torch.float32, device=keys.device)
        else:
            # Get query hidden states and project to query space
            query_hidden = hidden_states[:, query_start:query_end]
            query_states = get_prerope_query_states(module, query_hidden)

            # Apply RoPE to query states
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is not None:
                cos, sin = position_embeddings
                cos_q = cos[:, query_start:query_end]
                sin_q = sin[:, query_start:query_end]
                rotary_dim = cos_q.shape[-1]
                q_rot = query_states[..., :rotary_dim]
                q_pass = query_states[..., rotary_dim:]
                q_rot = (q_rot * cos_q.unsqueeze(1)) + (rotate_half(q_rot) * sin_q.unsqueeze(1))
                query_states = torch.cat([q_rot, q_pass], dim=-1)

            # Group queries to match KV heads (for GQA)
            query_states = _group_queries(query_states, num_kv_heads)

            # Compute attention logits
            attention_logits = torch.matmul(
                query_states.float(), keys.float().transpose(-1, -2)
            ) / math.sqrt(head_dim)

            # Apply causal mask: query can only attend to keys at or before its position
            key_positions = torch.arange(seq_len, device=keys.device)
            query_positions = torch.arange(query_start, query_end, device=keys.device)
            causal_mask = key_positions[None, :] > query_positions[:, None]
            attention_logits = attention_logits.masked_fill(causal_mask[None, None, :, :], float("-inf"))

            # Compute attention probabilities
            attention_probs = F.softmax(attention_logits, dim=-1, dtype=torch.float32)

            # Aggregate across query positions: blend of mean and max
            mean_score = attention_probs.mean(dim=-2)
            max_score = attention_probs.amax(dim=-2)
            scores = self.qa_alpha * mean_score + (1.0 - self.qa_alpha) * max_score

            if self.score_normalization:
                scores = _normalize_scores(scores)
            else:
                scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)

        # Zero out pseudo-query region so it doesn't self-rank
        if self.runtime_query_length == 0 and query_end > query_start:
            scores = scores.clone()
            scores[:, :, query_start:query_end] = 0.0

        # Apply forced keep mask
        forced_keep_mask = self._forced_keep_mask(seq_len, batch_size, keys.device)
        scores = scores.masked_fill(forced_keep_mask[:, None, :], torch.finfo(scores.dtype).max)

        layer_idx = int(module.layer_idx)
        self.last_scores_by_layer[layer_idx] = scores.detach()
        self.last_forced_keep_mask_by_layer[layer_idx] = forced_keep_mask.detach()
        return scores

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
        n_kept = int(seq_len * (1.0 - self.compression_ratio))
        layer_idx = int(module.layer_idx)
        forced_keep_mask = self.last_forced_keep_mask_by_layer[layer_idx]

        # Select top-k indices (per-head)
        safe_scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
        forced_count = forced_keep_mask.sum(dim=-1)
        n_kept = max(n_kept, int(forced_count[0].item()))
        n_kept = min(n_kept, seq_len)
        ranking_scores = safe_scores.masked_fill(forced_keep_mask[:, None, :], torch.finfo(safe_scores.dtype).max)
        kept_indices = ranking_scores.topk(n_kept, dim=-1).indices
        kept_indices = torch.sort(kept_indices, dim=-1).values

        self.retained_indices_by_layer[layer_idx] = kept_indices.detach()

        # Gather compressed keys and values
        head_dim = module.head_dim
        expand = kept_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
        return keys.gather(2, expand).contiguous(), values.gather(2, expand).contiguous()
