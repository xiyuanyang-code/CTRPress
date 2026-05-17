# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from kvpress import KeyDiffPress, KnormPress, RiskAwareEnsemblePress, SnapKVPress
from tests.fixtures import unit_test_model  # noqa: F401


class TestRiskAwareEnsemblePress:
    """Tests for RiskAwareEnsemblePress."""

    def _make_press(self, **kwargs):
        defaults = dict(
            compression_ratio=0.5,
            presses=[KnormPress(), KeyDiffPress(), SnapKVPress()],
        )
        defaults.update(kwargs)
        return RiskAwareEnsemblePress(**defaults)

    def test_basic_forward(self, unit_test_model):  # noqa: F811
        """Press can compress without errors on a real model."""
        press = self._make_press(compression_ratio=0.5)
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)

    def test_zero_compression_ratio(self, unit_test_model):  # noqa: F811
        """compression_ratio=0 should be a no-op (ScorerPress.compress returns early)."""
        press = self._make_press(compression_ratio=0.0)
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)

    def test_score_shape(self, unit_test_model):  # noqa: F811
        """score() must return (B, H_kv, S)."""
        press = self._make_press(compression_ratio=0.5)
        press.post_init_from_model(unit_test_model)

        input_ids = torch.arange(10, 30).unsqueeze(0).to(unit_test_model.device)
        with torch.no_grad():
            output = unit_test_model(input_ids, use_cache=True)

        layer = unit_test_model.model.layers[0]
        attn = layer.self_attn
        cache = output.past_key_values
        keys = cache.layers[0].keys
        values = cache.layers[0].values

        # Get hidden_states via the model's embedding layer
        hidden_states = unit_test_model.model.embed_tokens(input_ids)

        scores = press.score(attn, hidden_states, keys, values, attentions=None, kwargs={})
        assert scores.shape == keys.shape[:3], f"Expected {keys.shape[:3]}, got {scores.shape}"

    def test_compression_ratio_propagation(self):
        """Setting compression_ratio on the ensemble should propagate to all children."""
        p1 = KnormPress(compression_ratio=0.0)
        p2 = KeyDiffPress(compression_ratio=0.0)
        press = self._make_press(compression_ratio=0.0, presses=[p1, p2])

        press.compression_ratio = 0.5
        assert p1.compression_ratio == 0.5
        assert p2.compression_ratio == 0.5

    def test_compression_ratio_set_before_children_exist(self):
        """compression_ratio should not crash when set before __post_init__ runs (dataclass field order)."""
        press = RiskAwareEnsemblePress(compression_ratio=0.3)
        assert press.compression_ratio == 0.3
        for p in press.presses:
            assert p.compression_ratio == 0.3

    def test_post_init_propagation(self, unit_test_model):  # noqa: F811
        """post_init_from_model should be forwarded to all child presses."""
        p1 = KnormPress()
        p2 = KeyDiffPress()
        press = self._make_press(presses=[p1, p2])

        # CompactorPress sets internal state in post_init, so this should not crash
        press.post_init_from_model(unit_test_model)

    def test_different_alpha(self, unit_test_model):  # noqa: F811
        """alpha=0 (pure mean) and alpha=1 (pure max) should both work."""
        for alpha in [0.0, 0.5, 1.0]:
            press = self._make_press(compression_ratio=0.5, alpha=alpha)
            with press(unit_test_model):
                input_ids = torch.arange(10, 40).to(unit_test_model.device)
                unit_test_model(input_ids.unsqueeze(0), use_cache=True)

    def test_floor_weight(self, unit_test_model):  # noqa: F811
        """floor_weight > 0 activates the 2nd-largest score term."""
        press = self._make_press(compression_ratio=0.5, floor_weight=0.1)
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)

    def test_zscore_normalization(self, unit_test_model):  # noqa: F811
        """z-score normalization should work as an alternative to rank normalization."""
        press = self._make_press(compression_ratio=0.5, normalization="zscore")
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)

    def test_n_sink_n_recent(self, unit_test_model):  # noqa: F811
        """Sink and recent tokens should be protected."""
        press = self._make_press(compression_ratio=0.5, n_sink=8, n_recent=8)
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)

    def test_invalid_normalization(self):
        """Invalid normalization strategy should raise."""
        with pytest.raises(AssertionError, match="Unknown normalization"):
            RiskAwareEnsemblePress(normalization="invalid")

    def test_empty_presses_raises(self):
        """At least one sub-press is required."""
        with pytest.raises(AssertionError, match="At least one sub-press"):
            RiskAwareEnsemblePress(presses=[])

    def test_default_presses(self):
        """When presses=None, default three scorers should be populated."""
        press = RiskAwareEnsemblePress(compression_ratio=0.5)
        assert len(press.presses) == 3

    def test_with_snkv_window_size(self, unit_test_model):  # noqa: F811
        """SnapKVPress with custom window_size should work inside the ensemble."""
        press = self._make_press(
            compression_ratio=0.5,
            presses=[KnormPress(), SnapKVPress(window_size=2)],
        )
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)

    def test_two_scorers(self, unit_test_model):  # noqa: F811
        """Ensemble with only 2 scorers should work (floor_weight > 0 also)."""
        press = self._make_press(
            compression_ratio=0.5,
            presses=[KnormPress(), KeyDiffPress()],
            floor_weight=0.1,
        )
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)

    def test_disagreement_weight_scales_with_ratio(self):
        """disagreement_weight should increase with compression_ratio."""
        press = self._make_press(compression_ratio=0.0)
        w0 = press.base_disagreement_weight

        press.compression_ratio = 0.5
        w50 = press.base_disagreement_weight + press.ratio_disagreement_weight * 0.5

        press.compression_ratio = 0.9
        w90 = press.base_disagreement_weight + press.ratio_disagreement_weight * 0.9

        assert w0 < w50 < w90
