# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.compactor_press import CompactorPress
from kvpress.presses.expected_attention_press import ExpectedAttentionPress
from kvpress.presses.keydiff_press import KeyDiffPress
from kvpress.presses.scorer_press import ScorerPress


def _rank_normalize(scores: torch.Tensor) -> torch.Tensor:
    """Head-wise rank normalization to [0, 1].

    Parameters
    ----------
    scores : torch.Tensor
        Shape (B, H, S).

    Returns
    -------
    torch.Tensor
        Rank-normalized scores in [0, 1], same shape.
    """
    # Handle NaN/Inf before ranking
    scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)
    order = scores.argsort(dim=-1)
    ranks = torch.empty_like(order, dtype=scores.dtype)
    rank_values = torch.linspace(0, 1, scores.shape[-1], device=scores.device, dtype=scores.dtype)
    ranks.scatter_(-1, order, rank_values.view(1, 1, -1).expand_as(scores))
    return ranks


_DEFAULT_PRESSES = None


def _get_default_presses():
    global _DEFAULT_PRESSES
    if _DEFAULT_PRESSES is None:
        _DEFAULT_PRESSES = [
            ExpectedAttentionPress(use_vnorm=False, epsilon=1e-2),
            CompactorPress(),
            KeyDiffPress(),
        ]
    return list(_DEFAULT_PRESSES)


@dataclass
class RiskAwareEnsemblePress(ScorerPress):
    """Risk-aware ensemble KV cache compression.

    Combines multiple scorer presses via consensus and disagreement signals.
    Tokens where scorers disagree are treated as higher eviction risk and
    receive a bonus score. Only tokens with low importance *and* low
    disagreement are evicted first.

    Parameters
    ----------
    compression_ratio : float
        Fraction of key-value pairs to remove.
    presses : list[ScorerPress] | None
        Sub-scorers to ensemble.  Defaults to
        [ExpectedAttentionPress, CompactorPress, KeyDiffPress].
    alpha : float
        Weight for max vs mean consensus (0 = pure mean, 1 = pure max).
    base_disagreement_weight : float
        Constant term in the disagreement weighting.
    ratio_disagreement_weight : float
        Slope that scales with compression_ratio.
    floor_weight : float
        Weight for the 2nd-largest score floor protection term.
    n_sink : int
        Number of initial tokens to unconditionally protect.
    n_recent : int
        Number of final tokens to unconditionally protect.
    normalization : str
        Score normalization strategy: ``"rank"`` (default) or ``"zscore"``.
    """

    compression_ratio: float = 0.0
    presses: list = field(default_factory=_get_default_presses)
    alpha: float = 0.7
    base_disagreement_weight: float = 0.05
    ratio_disagreement_weight: float = 0.30
    floor_weight: float = 0.0
    n_sink: int = 4
    n_recent: int = 0
    normalization: str = "rank"

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"
        assert len(self.presses) >= 1, "At least one sub-press is required"
        assert self.normalization in ("rank", "zscore"), f"Unknown normalization: {self.normalization!r}"
        # Sync compression_ratio to children (they are created with default 0.0)
        for p in self.presses:
            p.compression_ratio = self.compression_ratio

    def post_init_from_model(self, model):
        for press in self.presses:
            press.post_init_from_model(model)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == "compression_ratio":
            if "presses" in self.__dict__:
                for p in self.__dict__["presses"]:
                    p.compression_ratio = value

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zscore_normalize(scores: torch.Tensor) -> torch.Tensor:
        mean = scores.mean(dim=-1, keepdim=True)
        std = scores.std(dim=-1, keepdim=True, unbiased=False)
        z = (scores - mean) / (std + 1e-8)
        return torch.sigmoid(z)

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        # Sync compression_ratio to child presses
        for press in self.presses:
            press.compression_ratio = self.compression_ratio

        # Collect raw scores from each sub-scorer
        raw_scores = []
        for press in self.presses:
            s = press.score(module, hidden_states, keys, values, attentions, kwargs)
            assert s.shape == keys.shape[:3], (
                f"Expected score shape {keys.shape[:3]} from {type(press).__name__}, got {s.shape}"
            )
            raw_scores.append(s)

        # Normalize
        if self.normalization == "rank":
            normalized = [_rank_normalize(s) for s in raw_scores]
        else:
            normalized = [self._zscore_normalize(s) for s in raw_scores]

        stack = torch.stack(normalized, dim=0)  # (M, B, H, S)

        # Consensus: alpha * max + (1 - alpha) * mean
        max_score = stack.max(dim=0).values
        mean_score = stack.mean(dim=0)
        consensus = self.alpha * max_score + (1.0 - self.alpha) * mean_score

        # Disagreement (std) weighted by compression ratio
        disagreement = stack.std(dim=0, unbiased=False)
        disagreement_weight = self.base_disagreement_weight + self.ratio_disagreement_weight * self.compression_ratio

        final = consensus + disagreement_weight * disagreement

        # Floor protection (2nd-largest score)
        if self.floor_weight > 0 and stack.shape[0] > 1:
            top2 = stack.topk(2, dim=0).values
            final = final + self.floor_weight * top2[1]

        # Structural bias: protect sink and recent tokens
        if self.n_sink > 0 or self.n_recent > 0:
            protected_value = final.detach().amax()
            if self.n_sink > 0:
                final[..., : self.n_sink] = protected_value
            if self.n_recent > 0:
                final[..., -self.n_recent :] = protected_value

        return final
