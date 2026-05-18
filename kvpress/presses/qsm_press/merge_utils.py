from __future__ import annotations

import torch


def compute_removed_indices(seq_len: int, kept_indices: torch.Tensor) -> torch.Tensor:
    """
    Build sorted removed indices from kept indices.

    Handles both 2D [batch, n_kept] and 3D [batch, num_heads, n_kept] indices.
    For 3D, flattens across heads to get the union of all kept positions.
    """
    batch_size = kept_indices.shape[0]
    # Flatten to 2D if needed (union of kept positions across all heads)
    if kept_indices.dim() == 3:
        kept_flat = kept_indices.reshape(batch_size, -1).unique(dim=-1)
    else:
        kept_flat = kept_indices
    keep_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=kept_flat.device)
    keep_mask.scatter_(1, kept_flat, True)
    all_indices = torch.arange(seq_len, device=kept_flat.device).expand(batch_size, -1)
    n_removed = seq_len - keep_mask.sum(dim=-1).min().item()
    return all_indices.masked_select(~keep_mask).view(batch_size, int(n_removed))


def nearest_kept_slots(kept_indices: torch.Tensor, removed_indices: torch.Tensor) -> torch.Tensor:
    if removed_indices.numel() == 0:
        return removed_indices
    distances = (removed_indices.unsqueeze(-1) - kept_indices.unsqueeze(1)).abs()
    return distances.argmin(dim=-1)


def merge_removed_values(
    values: torch.Tensor,
    kept_indices: torch.Tensor,
    removed_indices: torch.Tensor,
    merge_alpha: float,
    removed_weights: torch.Tensor | None = None,
    target_strategy: str = "nearest",
    count_power: float = 0.75,
) -> torch.Tensor:
    batch_size, num_heads, _, head_dim = values.shape
    n_kept = kept_indices.shape[1]

    kept_expand = kept_indices[:, None, :, None].expand(batch_size, num_heads, n_kept, head_dim)
    kept_values = values.gather(2, kept_expand).contiguous()

    if removed_indices.numel() == 0 or merge_alpha == 0:
        return kept_values
    if target_strategy != "nearest":
        raise ValueError(f"Unsupported merge target strategy: {target_strategy}")

    n_removed = removed_indices.shape[1]
    removed_expand = removed_indices[:, None, :, None].expand(batch_size, num_heads, n_removed, head_dim)
    removed_values = values.gather(2, removed_expand)

    target_slots = nearest_kept_slots(kept_indices, removed_indices)
    slot_expand = target_slots[:, None, :, None].expand(batch_size, num_heads, n_removed, head_dim)

    if removed_weights is None:
        removed_weights = torch.ones(batch_size, n_removed, dtype=values.dtype, device=values.device)
    else:
        removed_weights = torch.nan_to_num(
            removed_weights.to(device=values.device, dtype=values.dtype),
            nan=0.0, posinf=0.0, neginf=0.0,
        ).clamp_min(0.0)
    weighted_removed_values = removed_values * removed_weights[:, None, :, None]

    contributions = torch.zeros_like(kept_values)
    contributions.scatter_add_(2, slot_expand, weighted_removed_values)

    weight_sums = torch.zeros(batch_size, 1, n_kept, 1, dtype=values.dtype, device=values.device)
    count_slots = target_slots[:, None, :, None]
    weight_sums.scatter_add_(2, count_slots, removed_weights[:, None, :, None])

    mean_contributions = contributions / weight_sums.clamp_min(torch.finfo(values.dtype).eps)
    effective_mass = weight_sums.pow(1.0 - count_power)
    blended = (kept_values + merge_alpha * effective_mass * mean_contributions) / (
        1.0 + merge_alpha * effective_mass
    )
    return torch.where(weight_sums > 0, blended, kept_values)
