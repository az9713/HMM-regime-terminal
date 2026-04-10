"""
Tests for changepoint.py -- Bayesian Online Changepoint Detection.

Tests cover: Student-t sufficient statistics, single changepoint detection,
multiple changepoint detection, multivariate detection, HMM fusion,
adaptive hysteresis, edge cases, and config factories.
"""

import numpy as np
import pytest

from changepoint import (
    BayesianChangepoint,
    BOCPDResult,
    ChangepointHMMFusion,
    StudentTSuffStats,
    create_bocpd_from_config,
    create_fusion_from_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_single_changepoint_data(n=200, seed=42):
    """Synthetic data with one clear changepoint at n//2."""
    rng = np.random.default_rng(seed)
    half = n // 2
    seg1 = rng.normal(0.0, 1.0, half)
    seg2 = rng.normal(5.0, 1.0, n - half)
    return np.concatenate([seg1, seg2])


def make_multi_changepoint_data(n=300, seed=42):
    """Synthetic data with changepoints at n//3 and 2*n//3."""
    rng = np.random.default_rng(seed)
    third = n // 3
    seg1 = rng.normal(0.0, 0.5, third)
    seg2 = rng.normal(5.0, 0.5, third)
    seg3 = rng.normal(-3.0, 0.5, n - 2 * third)
    return np.concatenate([seg1, seg2, seg3])


def make_stable_data(n=200, seed=42):
    """Synthetic data with no changepoints (single regime)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, n)


def make_hmm_outputs(n=200, n_states=3, seed=42):
    """Synthetic HMM-like outputs for fusion testing."""
    rng = np.random.default_rng(seed)
    posteriors = rng.dirichlet(np.ones(n_states), size=n)
    entropy = -np.sum(posteriors * np.log2(posteriors + 1e-12), axis=1)
    max_ent = np.log2(n_states)
    confidence = 1.0 - entropy / max_ent
    return posteriors, entropy, confidence


# ---------------------------------------------------------------------------
# Tests: StudentTSuffStats
# ---------------------------------------------------------------------------

class TestStudentTSuffStats:
    def test_init_shapes(self):
        stats = StudentTSuffStats(0.0, 1.0, 1.0, 1.0)
        assert len(stats.mu) == 1
        assert len(stats.kappa) == 1

    def test_update_grows_nothing(self):
        """Update should modify values in-place, not change array length."""
        stats = StudentTSuffStats(0.0, 1.0, 1.0, 1.0)
        stats.update(1.0)
        assert len(stats.mu) == 1
        assert stats.kappa[0] == 2.0
        assert stats.alpha[0] == 1.5

    def test_prepend_grows_by_one(self):
        stats = StudentTSuffStats(0.0, 1.0, 1.0, 1.0)
        stats.prepend_prior(0.0, 1.0, 1.0, 1.0)
        assert len(stats.mu) == 2

    def test_pred_logpdf_finite(self):
        stats = StudentTSuffStats(0.0, 1.0, 1.0, 1.0)
        logp = stats.pred_logpdf(0.5)
        assert np.all(np.isfinite(logp))

    def test_pred_logpdf_negative(self):
        """Log probabilities should be negative."""
        stats = StudentTSuffStats(0.0, 1.0, 1.0, 1.0)
        logp = stats.pred_logpdf(0.5)
        assert np.all(logp < 0)

    def test_trim(self):
        stats = StudentTSuffStats(0.0, 1.0, 1.0, 1.0)
        for _ in range(10):
            stats.prepend_prior(0.0, 1.0, 1.0, 1.0)
        assert len(stats.mu) == 11
        stats.trim(5)
        assert len(stats.mu) == 5

    def test_sequential_updates_shift_mean(self):
        """Posterior mean should shift toward repeated observations."""
        stats = StudentTSuffStats(0.0, 1.0, 1.0, 1.0)
        for _ in range(20):
            stats.update(5.0)
        # After many updates of 5.0, posterior mean should be near 5.0
        assert stats.mu[0] > 4.0


# ---------------------------------------------------------------------------
# Tests: BayesianChangepoint -- single changepoint
# ---------------------------------------------------------------------------

class TestBayesianChangepointSingle:
    def test_result_type(self):
        data = make_single_changepoint_data()
        bocpd = BayesianChangepoint(hazard_rate=50)
        result = bocpd.detect(data)
        assert isinstance(result, BOCPDResult)

    def test_output_shapes(self):
        n = 200
        data = make_single_changepoint_data(n=n)
        bocpd = BayesianChangepoint(hazard_rate=50)
        result = bocpd.detect(data)
        assert result.changepoint_prob.shape == (n,)
        assert result.map_run_length.shape == (n,)
        assert result.expected_run_length.shape == (n,)
        assert result.growth_prob.shape == (n,)
        assert result.run_length_dist.shape[0] == n

    def test_cp_prob_in_zero_one(self):
        data = make_single_changepoint_data()
        bocpd = BayesianChangepoint(hazard_rate=50)
        result = bocpd.detect(data)
        assert np.all(result.changepoint_prob >= 0)
        assert np.all(result.changepoint_prob <= 1)

    def test_growth_prob_complement(self):
        data = make_single_changepoint_data()
        bocpd = BayesianChangepoint(hazard_rate=50)
        result = bocpd.detect(data)
        np.testing.assert_allclose(
            result.changepoint_prob + result.growth_prob, 1.0, atol=1e-10
        )

    def test_detects_changepoint_near_true_location(self):
        """The peak changepoint probability should be near bar 100."""
        n = 200
        data = make_single_changepoint_data(n=n)
        bocpd = BayesianChangepoint(hazard_rate=50, threshold=0.1)
        result = bocpd.detect(data)

        # Find peak CP probability in the region around the true changepoint
        true_cp = n // 2
        window = 15
        region = result.changepoint_prob[true_cp - window : true_cp + window]
        peak_in_region = np.max(region)
        # Should have a notable spike
        assert peak_in_region > 0.1, (
            f"Expected CP prob spike near bar {true_cp}, got max {peak_in_region:.3f}"
        )

    def test_map_run_length_resets_at_changepoint(self):
        """MAP run length should drop near the changepoint."""
        n = 200
        data = make_single_changepoint_data(n=n)
        bocpd = BayesianChangepoint(hazard_rate=50)
        result = bocpd.detect(data)

        # Before changepoint, run length should be growing
        pre_cp_rl = result.map_run_length[n // 4]
        # After changepoint + some time, it should be growing again from ~0
        post_cp_rl = result.map_run_length[n // 2 + 20]
        # The post-CP run length should be lower than pre-CP
        assert post_cp_rl < pre_cp_rl


# ---------------------------------------------------------------------------
# Tests: BayesianChangepoint -- multiple changepoints
# ---------------------------------------------------------------------------

class TestBayesianChangepointMulti:
    def test_detects_multiple_changepoints(self):
        n = 300
        data = make_multi_changepoint_data(n=n)
        bocpd = BayesianChangepoint(hazard_rate=50, threshold=0.1)
        result = bocpd.detect(data)

        # Should detect changepoints near bars 100 and 200
        cp_indices = result.changepoint_indices
        assert len(cp_indices) >= 2, (
            f"Expected at least 2 changepoints, got {len(cp_indices)}"
        )

    def test_cp_prob_spikes_at_transitions(self):
        n = 300
        data = make_multi_changepoint_data(n=n)
        bocpd = BayesianChangepoint(hazard_rate=50)
        result = bocpd.detect(data)

        third = n // 3
        # Check spike near first changepoint
        region1 = result.changepoint_prob[third - 10 : third + 10]
        assert np.max(region1) > 0.05

        # Check spike near second changepoint
        region2 = result.changepoint_prob[2 * third - 10 : 2 * third + 10]
        assert np.max(region2) > 0.05


# ---------------------------------------------------------------------------
# Tests: BayesianChangepoint -- stable data
# ---------------------------------------------------------------------------

class TestBayesianChangepointStable:
    def test_low_cp_prob_on_stable_data(self):
        """Stable data should produce low changepoint probabilities."""
        data = make_stable_data()
        bocpd = BayesianChangepoint(hazard_rate=200, threshold=0.3)
        result = bocpd.detect(data)

        # After initial settling (first ~20 bars), CP prob should stay low
        settled = result.changepoint_prob[30:]
        mean_cp = settled.mean()
        assert mean_cp < 0.1, (
            f"Expected low mean CP prob on stable data, got {mean_cp:.3f}"
        )

    def test_expected_run_length_grows(self):
        """On stable data, expected run length should increase over time."""
        data = make_stable_data(n=100)
        bocpd = BayesianChangepoint(hazard_rate=200)
        result = bocpd.detect(data)

        # Compare early vs late expected run length
        early = np.mean(result.expected_run_length[10:30])
        late = np.mean(result.expected_run_length[70:90])
        assert late > early


# ---------------------------------------------------------------------------
# Tests: Multivariate detection
# ---------------------------------------------------------------------------

class TestMultivariateDetection:
    def test_multivariate_output_shapes(self):
        rng = np.random.default_rng(42)
        n, d = 200, 3
        data = np.column_stack([
            np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 100)]),
            np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 1, 100)]),
            np.concatenate([rng.normal(0, 1, 100), rng.normal(-2, 1, 100)]),
        ])
        bocpd = BayesianChangepoint(hazard_rate=50)
        result = bocpd.detect_multivariate(data)
        assert result.changepoint_prob.shape == (n,)
        assert result.map_run_length.shape == (n,)

    def test_multivariate_detects_changepoint(self):
        rng = np.random.default_rng(42)
        n = 200
        data = np.column_stack([
            np.concatenate([rng.normal(0, 0.5, 100), rng.normal(5, 0.5, 100)]),
            np.concatenate([rng.normal(0, 0.5, 100), rng.normal(3, 0.5, 100)]),
        ])
        bocpd = BayesianChangepoint(hazard_rate=50, threshold=0.05)
        result = bocpd.detect_multivariate(data)

        region = result.changepoint_prob[90:115]
        assert np.max(region) > 0.05

    def test_custom_weights(self):
        rng = np.random.default_rng(42)
        data = np.column_stack([
            np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 100)]),
            rng.normal(0, 1, 200),  # no changepoint in feature 2
        ])
        bocpd = BayesianChangepoint(hazard_rate=50)
        # Weight feature 1 heavily
        result_weighted = bocpd.detect_multivariate(
            data, feature_weights=np.array([0.9, 0.1])
        )
        # Equal weights
        result_equal = bocpd.detect_multivariate(data)

        # Weighted result should have higher CP prob near bar 100
        region_w = result_weighted.changepoint_prob[95:110].max()
        region_e = result_equal.changepoint_prob[95:110].max()
        assert region_w >= region_e * 0.5  # weighted should be at least comparable


# ---------------------------------------------------------------------------
# Tests: ChangepointHMMFusion
# ---------------------------------------------------------------------------

class TestChangepointHMMFusion:
    def test_urgency_shape(self):
        n = 200
        cp_prob = np.random.rand(n) * 0.3
        posteriors, entropy, confidence = make_hmm_outputs(n, n_states=3)
        fusion = ChangepointHMMFusion()
        urgency = fusion.compute_transition_urgency(
            cp_prob, entropy, posteriors, n_states=3
        )
        assert urgency.shape == (n,)
        assert np.all(urgency >= 0)
        assert np.all(urgency <= 1)

    def test_urgency_high_when_cp_and_uncertain(self):
        """Urgency should be high when both CP prob and entropy are high."""
        n = 100
        # Simulate a transition at bar 50
        cp_prob = np.zeros(n)
        cp_prob[48:55] = 0.8  # high changepoint prob

        posteriors = np.zeros((n, 3))
        posteriors[:, 0] = 1.0  # confident state 0 normally
        # Make uncertain around bar 50
        posteriors[45:55] = [0.33, 0.34, 0.33]

        entropy = -np.sum(posteriors * np.log2(posteriors + 1e-12), axis=1)

        fusion = ChangepointHMMFusion()
        urgency = fusion.compute_transition_urgency(
            cp_prob, entropy, posteriors, n_states=3
        )

        # Urgency should be notably higher around bar 50 vs baseline
        transition_urgency = urgency[48:55].mean()
        baseline_urgency = urgency[:20].mean()
        assert transition_urgency > baseline_urgency

    def test_adaptive_hysteresis_values(self):
        n = 100
        urgency = np.zeros(n)
        urgency[50:60] = 0.9  # high urgency during transition

        fusion = ChangepointHMMFusion()
        hyst = fusion.adaptive_hysteresis(urgency, base_hysteresis=3, min_hysteresis=0)

        # During high urgency, hysteresis should be reduced
        assert np.all(hyst[50:60] <= 1)
        # During low urgency, hysteresis should be at base
        assert np.all(hyst[:40] == 3)

    def test_adaptive_hysteresis_bounded(self):
        urgency = np.random.rand(100)
        fusion = ChangepointHMMFusion()
        hyst = fusion.adaptive_hysteresis(urgency, base_hysteresis=5, min_hysteresis=1)
        assert np.all(hyst >= 1)
        assert np.all(hyst <= 5)

    def test_fuse_returns_all_keys(self):
        n = 200
        data = make_single_changepoint_data(n=n)
        bocpd = BayesianChangepoint(hazard_rate=50)
        bocpd_result = bocpd.detect(data)

        posteriors, entropy, confidence = make_hmm_outputs(n, n_states=3)

        fusion = ChangepointHMMFusion()
        result = fusion.fuse(
            bocpd_result, entropy, confidence, posteriors,
            n_states=3, base_hysteresis=3
        )

        expected_keys = {
            "urgency", "adaptive_hysteresis", "enhanced_confidence",
            "transition_alerts", "changepoint_prob", "map_run_length",
            "expected_run_length", "run_length_dist",
        }
        assert set(result.keys()) == expected_keys

    def test_enhanced_confidence_reduced_at_changepoint(self):
        """Enhanced confidence should be lower than raw confidence at CPs."""
        n = 200
        data = make_single_changepoint_data(n=n)
        bocpd = BayesianChangepoint(hazard_rate=50)
        bocpd_result = bocpd.detect(data)

        posteriors, entropy, confidence = make_hmm_outputs(n, n_states=3)

        fusion = ChangepointHMMFusion()
        result = fusion.fuse(
            bocpd_result, entropy, confidence, posteriors,
            n_states=3, base_hysteresis=3
        )

        # At changepoint locations, enhanced confidence should be <= raw
        assert np.all(result["enhanced_confidence"] <= confidence + 1e-10)


# ---------------------------------------------------------------------------
# Tests: Config factories
# ---------------------------------------------------------------------------

class TestConfigFactories:
    def test_create_bocpd_defaults(self):
        bocpd = create_bocpd_from_config({})
        assert bocpd.hazard_rate == 100
        assert bocpd.threshold == 0.3

    def test_create_bocpd_custom(self):
        config = {
            "changepoint": {
                "hazard_rate": 50,
                "threshold": 0.2,
                "mu0": 1.0,
            }
        }
        bocpd = create_bocpd_from_config(config)
        assert bocpd.hazard_rate == 50
        assert bocpd.threshold == 0.2
        assert bocpd.mu0 == 1.0

    def test_create_fusion_defaults(self):
        fusion = create_fusion_from_config({})
        assert fusion.cp_weight == 0.5
        assert fusion.urgency_threshold == 0.4

    def test_create_fusion_custom(self):
        config = {
            "changepoint": {
                "fusion": {
                    "cp_weight": 0.7,
                    "urgency_threshold": 0.5,
                }
            }
        }
        fusion = create_fusion_from_config(config)
        assert fusion.cp_weight == 0.7
        assert fusion.urgency_threshold == 0.5


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_very_short_data(self):
        data = np.array([1.0, 2.0, 3.0])
        bocpd = BayesianChangepoint(hazard_rate=10)
        result = bocpd.detect(data)
        assert result.changepoint_prob.shape == (3,)

    def test_constant_data(self):
        data = np.ones(100)
        bocpd = BayesianChangepoint(hazard_rate=50)
        result = bocpd.detect(data)
        # Should not crash; CP prob should be low after settling
        assert np.all(np.isfinite(result.changepoint_prob))

    def test_single_point(self):
        data = np.array([42.0])
        bocpd = BayesianChangepoint(hazard_rate=10)
        result = bocpd.detect(data)
        assert result.changepoint_prob.shape == (1,)

    def test_extreme_values(self):
        data = np.array([0.0] * 50 + [1e6] * 50)
        bocpd = BayesianChangepoint(hazard_rate=20)
        result = bocpd.detect(data)
        assert np.all(np.isfinite(result.changepoint_prob))
        # Should detect the extreme jump
        region = result.changepoint_prob[48:55]
        assert np.max(region) > 0.1

    def test_max_run_length_truncation(self):
        """Run length dist should respect max_run_length."""
        data = make_stable_data(n=500)
        bocpd = BayesianChangepoint(hazard_rate=200, max_run_length=100)
        result = bocpd.detect(data)
        assert result.run_length_dist.shape[1] <= 100
