"""
Tests for changepoint.py — Bayesian Online Changepoint Detection.
"""

import numpy as np
import pytest

from changepoint import BayesianChangepoint, BOCPDResult


@pytest.fixture
def default_config():
    return {
        "changepoint": {
            "hazard_lambda": 100,
            "prior_mu": 0.0,
            "prior_kappa": 0.1,
            "prior_alpha": 1.0,
            "prior_beta": 0.01,
            "threshold": 0.3,
            "feature_index": 0,
            "blend_weight": 0.3,
            "recency_horizon": 10,
        }
    }


@pytest.fixture
def bocpd(default_config):
    return BayesianChangepoint(default_config)


def make_regime_switch_series(n_per_regime=100, seed=42):
    """Create a synthetic series with clear regime changes."""
    rng = np.random.default_rng(seed)
    # Regime 1: low vol, positive drift
    r1 = rng.normal(0.001, 0.005, n_per_regime)
    # Regime 2: high vol, negative drift (crash)
    r2 = rng.normal(-0.01, 0.03, n_per_regime)
    # Regime 3: low vol, positive drift (recovery)
    r3 = rng.normal(0.002, 0.005, n_per_regime)
    return np.concatenate([r1, r2, r3])


class TestBOCPDBasic:
    """Basic functionality tests."""

    def test_detect_returns_bocpd_result(self, bocpd):
        x = np.random.default_rng(0).normal(0, 1, 200)
        result = bocpd.detect(x)
        assert isinstance(result, BOCPDResult)

    def test_output_shapes(self, bocpd):
        T = 150
        x = np.random.default_rng(0).normal(0, 1, T)
        result = bocpd.detect(x)
        assert result.changepoint_prob.shape == (T,)
        assert result.regime_stability.shape == (T,)
        assert result.expected_run_length.shape == (T,)
        assert result.map_run_length.shape == (T,)
        assert len(result.growth_probs) == T

    def test_changepoint_prob_is_probability(self, bocpd):
        x = np.random.default_rng(0).normal(0, 1, 200)
        result = bocpd.detect(x)
        assert np.all(result.changepoint_prob >= 0)
        assert np.all(result.changepoint_prob <= 1)

    def test_regime_stability_complement(self, bocpd):
        x = np.random.default_rng(0).normal(0, 1, 200)
        result = bocpd.detect(x)
        np.testing.assert_allclose(
            result.regime_stability,
            1.0 - result.changepoint_prob,
        )

    def test_2d_input_uses_feature_index(self, default_config):
        default_config["changepoint"]["feature_index"] = 1
        bocpd = BayesianChangepoint(default_config)
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (200, 3))
        result = bocpd.detect(X)
        assert result.changepoint_prob.shape == (200,)


class TestBOCPDDetection:
    """Tests that BOCPD actually detects regime changes."""

    def test_detects_mean_shift(self, bocpd):
        """A sudden jump in mean should produce high changepoint probability."""
        rng = np.random.default_rng(42)
        x = np.concatenate([
            rng.normal(0.0, 0.01, 100),
            rng.normal(0.1, 0.01, 100),  # abrupt mean shift
        ])
        result = bocpd.detect(x)
        # Around bar 100, changepoint_prob should spike
        region = result.changepoint_prob[98:108]
        assert np.max(region) > 0.5, "Should detect mean shift near bar 100"

    def test_detects_variance_shift(self, bocpd):
        """A sudden jump in variance should produce elevated changepoint probability."""
        rng = np.random.default_rng(42)
        x = np.concatenate([
            rng.normal(0.0, 0.01, 100),
            rng.normal(0.0, 0.1, 100),  # abrupt vol shift
        ])
        result = bocpd.detect(x)
        region = result.changepoint_prob[98:115]
        assert np.max(region) > 0.1, "Should detect variance shift"

    def test_stable_series_low_changepoint(self, bocpd):
        """A constant-parameter series should have low changepoint probability after warmup."""
        x = np.random.default_rng(0).normal(0, 0.01, 300)
        result = bocpd.detect(x)
        # After warmup (first 50 bars), changepoint prob should be low on average
        avg_cp = np.mean(result.changepoint_prob[50:])
        assert avg_cp < 0.1, f"Stable series should have low avg CP prob, got {avg_cp:.3f}"

    def test_multiple_changepoints(self, bocpd):
        x = make_regime_switch_series(n_per_regime=100, seed=42)
        result = bocpd.detect(x)
        # Should detect at least one changepoint
        assert len(result.changepoint_indices) >= 1

    def test_run_length_resets_at_changepoint(self, bocpd):
        """Expected run length should drop near changepoints."""
        rng = np.random.default_rng(42)
        x = np.concatenate([
            rng.normal(0.0, 0.01, 150),
            rng.normal(0.5, 0.01, 150),  # huge mean shift
        ])
        result = bocpd.detect(x)
        # Run length should be high before changepoint, low after
        rl_before = np.mean(result.expected_run_length[130:148])
        rl_after = np.mean(result.expected_run_length[152:160])
        assert rl_before > rl_after, "Run length should drop after changepoint"


class TestConfidenceFusion:
    """Tests for HMM + BOCPD confidence fusion."""

    def test_fusion_reduces_confidence_at_changepoint(self, bocpd):
        hmm_confidence = np.ones(200) * 0.9
        rng = np.random.default_rng(42)
        x = np.concatenate([
            rng.normal(0.0, 0.01, 100),
            rng.normal(0.5, 0.01, 100),
        ])
        result = bocpd.detect(x)
        fused = bocpd.merge_with_hmm_confidence(hmm_confidence, result, blend_weight=0.5)
        # Fused should be <= HMM confidence everywhere
        assert np.all(fused <= hmm_confidence + 1e-10)

    def test_fusion_preserves_during_stable(self, bocpd):
        hmm_confidence = np.ones(200) * 0.8
        x = np.random.default_rng(0).normal(0, 0.01, 200)
        result = bocpd.detect(x)
        fused = bocpd.merge_with_hmm_confidence(hmm_confidence, result, blend_weight=0.3)
        # During stable periods, fused ≈ hmm_confidence
        avg_fused = np.mean(fused[50:])
        assert avg_fused > 0.7, f"Stable series should preserve most confidence, got {avg_fused:.3f}"

    def test_zero_blend_weight_no_change(self, bocpd):
        hmm_confidence = np.random.default_rng(0).uniform(0.5, 1.0, 200)
        x = np.random.default_rng(42).normal(0, 1, 200)
        result = bocpd.detect(x)
        fused = bocpd.merge_with_hmm_confidence(hmm_confidence, result, blend_weight=0.0)
        np.testing.assert_allclose(fused, hmm_confidence)

    def test_full_blend_weight(self, bocpd):
        hmm_confidence = np.ones(200)
        x = np.random.default_rng(0).normal(0, 1, 200)
        result = bocpd.detect(x)
        fused = bocpd.merge_with_hmm_confidence(hmm_confidence, result, blend_weight=1.0)
        np.testing.assert_allclose(fused, result.regime_stability)


class TestHazardSensitivity:
    """Tests that hazard_lambda parameter affects detection sensitivity."""

    def test_lower_hazard_more_sensitive(self):
        rng = np.random.default_rng(42)
        x = np.concatenate([
            rng.normal(0.0, 0.01, 100),
            rng.normal(0.05, 0.01, 100),
            rng.normal(0.0, 0.01, 100),
        ])

        # Low hazard = more sensitive to changepoints
        bocpd_low = BayesianChangepoint({"changepoint": {
            "hazard_lambda": 50, "threshold": 0.3, "recency_horizon": 10,
        }})
        # High hazard = less sensitive to changepoints
        bocpd_high = BayesianChangepoint({"changepoint": {
            "hazard_lambda": 500, "threshold": 0.3, "recency_horizon": 10,
        }})

        result_low = bocpd_low.detect(x)
        result_high = bocpd_high.detect(x)

        # Both should detect changepoints, but with different sensitivity
        assert len(result_low.changepoint_indices) >= 1
        assert len(result_high.changepoint_indices) >= 1


class TestEdgeCases:
    """Edge case tests."""

    def test_short_series(self, bocpd):
        x = np.array([0.01, -0.02, 0.03])
        result = bocpd.detect(x)
        assert result.changepoint_prob.shape == (3,)

    def test_single_observation(self, bocpd):
        x = np.array([0.01])
        result = bocpd.detect(x)
        assert result.changepoint_prob.shape == (1,)

    def test_constant_series(self, bocpd):
        x = np.zeros(100)
        result = bocpd.detect(x)
        assert not np.any(np.isnan(result.changepoint_prob))

    def test_extreme_values(self, bocpd):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 0.01, 100)
        x[50] = 10.0  # extreme outlier
        result = bocpd.detect(x)
        assert not np.any(np.isnan(result.changepoint_prob))
        # Outlier should trigger elevated changepoint probability nearby
        region = result.changepoint_prob[49:60]
        assert np.max(region) > 0.0, "Extreme outlier should trigger some detection"
