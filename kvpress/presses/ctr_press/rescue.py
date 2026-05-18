from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def rescue_score(
    module: nn.Module,
    hidden_states: torch.Tensor,
    kept_keys: torch.Tensor,
    removed_keys: torch.Tensor,
    position_embeddings: tuple | None = None,
) -> torch.Tensor:
    """
    Score removed tokens by how much the compressed cache "misses" them.

    For each removed token, compute max attention from any kept query to its key.
    High score = the compressed cache would attend to this token if it were available.

    Args:
        module: Attention module (for head_dim, config).
        hidden_states: Hidden states used to derive queries [batch, seq_len, hidden_dim].
        kept_keys: Keys of kept tokens [batch, num_kv_heads, n_kept, head_dim].
        removed_keys: Keys of removed tokens [batch, num_kv_heads, n_removed, head_dim].
        position_embeddings: RoPE (cos, sin) if applicable.

    Returns:
        Rescue scores [batch, n_removed], aggregated across heads.
    """
    if removed_keys.numel() == 0:
        return torch.zeros(hidden_states.shape[0], 0, device=hidden_states.device)

    from kvpress.utils import get_prerope_query_states
    from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

    bsz, num_kv_heads, n_removed, head_dim = removed_keys.shape
    num_heads = module.config.num_attention_heads
    num_key_value_groups = num_heads // num_kv_heads
    n_kept = kept_keys.shape[2]

    # Get pre-RoPE query states from hidden states
    try:
        query_states = get_prerope_query_states(module, hidden_states)
    except (NotImplementedError, AttributeError):
        # Fallback: use kept_keys transposed as a rough proxy for query direction
        mean_kept_key = kept_keys.mean(dim=2, keepdim=True)  # [bsz, kv_heads, 1, head_dim]
        attn_to_removed = torch.matmul(
            mean_kept_key.float(), removed_keys.float().transpose(-1, -2)
        ) / math.sqrt(head_dim)
        return attn_to_removed.squeeze(-2).max(dim=1).values  # [bsz, n_removed]

    # Apply RoPE
    if position_embeddings is not None:
        cos, sin = position_embeddings
        rotary_dim = cos.shape[-1]
        q_rot = query_states[..., :rotary_dim]
        q_pass = query_states[..., rotary_dim:]
        q_rot = (q_rot * cos.unsqueeze(1)) + (rotate_half(q_rot) * sin.unsqueeze(1))
        query_states = torch.cat([q_rot, q_pass], dim=-1)

    # Repeat KV heads for GQA
    key_states_expanded = repeat_kv(removed_keys, num_key_value_groups)

    # Compute attention: query @ removed_keys^T
    attn_weights = torch.matmul(
        query_states.float(), key_states_expanded.float().transpose(-1, -2)
    ) / math.sqrt(head_dim)

    # Causal mask: query at position i can only attend to keys at position <= i
    # Since removed tokens came from the original sequence, and we use the full
    # hidden_states as queries, apply causal mask based on original positions.
    # For simplicity, use max attention (aggressive rescue) — if ANY query
    # position strongly attends to a removed token, it's worth rescuing.
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

    # Aggregate: max attention weight to each removed token, across all query
    # positions and all heads
    max_attn = attn_weights.amax(dim=(-2, -3))  # [bsz, n_removed] via amax over heads and query positions
    if max_attn.dim() == 3:
        max_attn = max_attn.amax(dim=1)

    return max_attn


def compute_score_entropy(scores: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of the score distribution per batch item.

    High entropy = many tokens with similar scores (model is uncertain).
    Low entropy = a few tokens have much higher scores (model is focused).

    Args:
        scores: [batch, num_kv_heads, seq_len]

    Returns:
        Entropy [batch]
    """
    safe_scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    # Average across heads
    agg = safe_scores.mean(dim=1)  # [batch, seq_len]
    # Normalize to probability distribution
    probs = F.softmax(agg, dim=-1)
    # Compute entropy: -sum(p * log(p))
    log_probs = torch.log_softmax(agg, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # [batch]
    return entropy


def adjust_kept_by_entropy(
    n_kept: int,
    seq_len: int,
    entropy: torch.Tensor,
    beta: float = 0.2,
) -> int:
    """
    Adjust the number of kept tokens based on score entropy.

    - High entropy (uniform scores) → keep fewer tokens (compression is already
      information-preserving since no token stands out).
    - Low entropy (concentrated scores) → keep more tokens (a few tokens are
      critical, don't risk losing them).

    Args:
        n_kept: Base number of tokens to keep.
        seq_len: Total sequence length.
        entropy: Per-batch entropy [batch].
        beta: Adaptation strength (0 = no adaptation).

    Returns:
        Adjusted n_kept (scalar, using mean across batch).
    """
    max_entropy = math.log(seq_len)
    if max_entropy <= 0:
        return n_kept
    normalized_entropy = max(0.0, min(1.0, entropy.mean().item() / max_entropy))
    scale = 1.0 + beta * (1.0 - 2.0 * normalized_entropy)
    adjusted = int(round(n_kept * scale))
    return max(1, min(adjusted, seq_len))
