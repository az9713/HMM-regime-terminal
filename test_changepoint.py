"""
Tests for changepoint.py — Bayesian Online Changepoint Detection.

Tests cover: mean-shift detection, run-length distribution properties,
changepoint probability bounds, multiple changepoints, HMM comparison,
combined instability score, and edge cases.
"""

import numpy as np
import pytest

from changepoint import BayesianChangepointDetector, ChangepointResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_mean_shift_data(n_per_segment=200, shift=0.05, noise=0.01, seed=42):
    """Synthetic data: N(0, noise) then N(shift, noise)."""
    rng = np.random.default_rng(seed)
    seg1 = rng.normal(0, noise, n_per_segment)
    seg2 = rng.normal(shift, noise, n_per_segment)
    return np.concatenate([seg1, seg2])


def make_multi_shift_data(seed=42):
    """Three segments with distinct means."""
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.normal(0.0, 0.01, 150),
        rng.normal(0.08, 0.01, 150),
        rng.normal(-0.04, 0.01, 150),
    ])


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

class TestBOCDBasic:
    """Core BOCD algorithm tests."""

    def test_output_shapes(self):
        """Output arrays should have correct shapes."""
        data = np.random.default_rng(1).normal(0, 0.01, 100)
        detector = BayesianChangepointDetector(max_run_length=100)
        result = detector.run(data)

        assert result.changepoint_prob.shape == (100,)
        assert result.max_run_length.shape == (100,)
        assert result.run_length_dist.shape[0] == 100

    def test_changepoint_prob_range(self):
        """All changepoint probabilities should be in [0, 1]."""
        data = np.random.default_rng(2).normal(0, 0.01, 200)
        detector = BayesianChangepointDetector()
        result = detector.run(data)

        assert np.all(result.changepoint_prob >= 0)
        assert np.all(result.changepoint_prob <= 1)

    def test_run_length_dist_sums_to_one(self):
        """Each row of the run-length distribution should sum to ~1."""
        data = np.random.default_rng(3).normal(0, 0.01, 100)
        detector = BayesianChangepointDetector(max_run_length=150)
        result = detector.run(data)

        for t in range(5, 100):
            row_sum = result.run_length_dist[t].sum()
            assert abs(row_sum - 1.0) < 0.02, (
                f"Row {t} sums to {row_sum}, expected ~1.0"
            )

    def test_map_run_length_non_negative(self):
        """MAP run length should always be >= 0."""
        data = np.random.default_rng(4).normal(0, 0.01, 200)
        detector = BayesianChangepointDetector()
        result = detector.run(data)

        assert np.all(result.max_run_length >= 0)


# ---------------------------------------------------------------------------
# Changepoint detection
# ---------------------------------------------------------------------------

class TestBOCDDetection:
    """Changepoint detection accuracy."""

    def test_detects_mean_shift(self):
        """Should detect a clear mean shift near the true changepoint."""
        data = make_mean_shift_data()
        detector = BayesianChangepointDetector(
            hazard_rate=1 / 100,
            detection_threshold=0.3,
            cp_window=15,
        )
        result = detector.run(data)

        assert len(result.detected_changepoints) > 0, (
            f"No changepoints detected; max cp_prob={result.changepoint_prob.max():.3f}"
        )
        closest = min(
            result.detected_changepoints, key=lambda x: abs(x - 200)
        )
        assert abs(closest - 200) < 30, (
            f"Closest changepoint at {closest}, expected near 200"
        )

    def test_detects_multiple_changepoints(self):
        """Should detect multiple changepoints in multi-segment data."""
        data = make_multi_shift_data()
        detector = BayesianChangepointDetector(
            hazard_rate=1 / 80,
            detection_threshold=0.3,
            cp_window=15,
        )
        result = detector.run(data)

        assert len(result.detected_changepoints) >= 2, (
            f"Detected {len(result.detected_changepoints)} changepoints, "
            f"expected >= 2; max cp_prob={result.changepoint_prob.max():.3f}"
        )

    def test_spike_at_changepoint(self):
        """Changepoint score should spike near the true break."""
        data = make_mean_shift_data()
        detector = BayesianChangepointDetector(
            hazard_rate=1 / 100, cp_window=15,
        )
        result = detector.run(data)

        # The max cp_score near the break (195-230) should be much higher
        # than in the stable region (50-180)
        region = result.changepoint_prob[195:235]
        stable = result.changepoint_prob[50:180]
        assert region.max() > stable.mean() * 2, (
            f"Max score near break ({region.max():.3f}) should be >> "
            f"stable mean ({stable.mean():.3f})"
        )

    def test_low_cp_prob_in_stable_region(self):
        """Changepoint score should be low in stable segments."""
        data = np.random.default_rng(10).normal(0, 0.01, 300)
        detector = BayesianChangepointDetector(
            hazard_rate=1 / 100, cp_window=10,
        )
        result = detector.run(data)

        # After initial transient (first 50 bars), cp score should be low
        # In stable data, P(r < 10) ≈ 1 - (1-H)^10 ≈ 0.10
        mean_cp = np.mean(result.changepoint_prob[50:])
        assert mean_cp < 0.25, (
            f"Mean cp_score in stable data is {mean_cp:.3f}, expected < 0.25"
        )

    def test_detection_cooldown(self):
        """Detected changepoints should respect minimum distance."""
        data = make_mean_shift_data()
        detector = BayesianChangepointDetector(detection_threshold=0.05)
        result = detector.run(data)

        for i in range(1, len(result.detected_changepoints)):
            gap = (
                result.detected_changepoints[i]
                - result.detected_changepoints[i - 1]
            )
            assert gap >= 5, f"Changepoints too close: gap={gap}"


# ---------------------------------------------------------------------------
# HMM comparison
# ---------------------------------------------------------------------------

class TestBOCDHMMComparison:
    """Tests for compare_with_hmm_transitions."""

    def test_early_detection_computed(self):
        """Should compute early detection bars vs HMM transitions."""
        data = make_mean_shift_data()
        detector = BayesianChangepointDetector(
            hazard_rate=1 / 100,
            detection_threshold=0.3,
            cp_window=15,
        )
        result = detector.run(data)

        # Pretend HMM detected the transition 15 bars late
        result = detector.compare_with_hmm_transitions(result, [215])

        assert len(result.early_detection_bars) > 0, (
            f"No early detection; BOCD cps={result.detected_changepoints}"
        )
        assert result.avg_early_detection > 0

    def test_no_match_returns_empty(self):
        """When BOCD detects nothing near HMM transitions, return empty."""
        data = np.random.default_rng(20).normal(0, 0.01, 100)
        detector = BayesianChangepointDetector(detection_threshold=0.99)
        result = detector.run(data)

        result = detector.compare_with_hmm_transitions(result, [50])

        assert len(result.early_detection_bars) == 0
        assert result.avg_early_detection == 0.0

    def test_exact_match_excluded(self):
        """A BOCD detection at the exact same bar as HMM should not
        count as 'early' (lead = 0)."""
        data = make_mean_shift_data()
        detector = BayesianChangepointDetector(
            hazard_rate=1 / 100,
            detection_threshold=0.2,
        )
        result = detector.run(data)

        # If BOCD detects at bar X and HMM also at bar X, lead = 0 → excluded
        if result.detected_changepoints:
            cp_bar = result.detected_changepoints[0]
            result2 = detector.compare_with_hmm_transitions(result, [cp_bar])
            # All entries in early_detection_bars should be > 0
            assert all(b > 0 for b in result2.early_detection_bars)


# ---------------------------------------------------------------------------
# Combined instability score
# ---------------------------------------------------------------------------

class TestCombinedInstability:
    """Tests for combined_instability_score."""

    def test_output_range(self):
        """Combined score should be in [0, 1]."""
        cp_prob = np.random.default_rng(30).uniform(0, 0.5, 100)
        entropy_grad = np.random.default_rng(31).uniform(-0.1, 0.3, 100)

        detector = BayesianChangepointDetector()
        score = detector.combined_instability_score(cp_prob, entropy_grad)

        assert np.all(score >= 0)
        assert np.all(score <= 1)

    def test_zero_inputs(self):
        """Zero cp_prob and entropy_grad → zero instability."""
        cp_prob = np.zeros(50)
        entropy_grad = np.zeros(50)

        detector = BayesianChangepointDetector()
        score = detector.combined_instability_score(cp_prob, entropy_grad)

        np.testing.assert_array_almost_equal(score, 0.0)

    def test_length_mismatch_handled(self):
        """Should handle arrays of different lengths (use min)."""
        cp_prob = np.ones(100) * 0.5
        entropy_grad = np.ones(80) * 0.3

        detector = BayesianChangepointDetector()
        score = detector.combined_instability_score(cp_prob, entropy_grad)

        assert len(score) == 80


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestBOCDEdgeCases:
    """Edge case handling."""

    def test_short_data(self):
        """Should handle very short time series without error."""
        data = np.array([0.01, -0.02, 0.03])
        detector = BayesianChangepointDetector()
        result = detector.run(data)

        assert len(result.changepoint_prob) == 3
        assert result.run_length_dist.shape[0] == 3

    def test_single_observation(self):
        """Should handle a single observation."""
        data = np.array([0.01])
        detector = BayesianChangepointDetector()
        result = detector.run(data)

        assert len(result.changepoint_prob) == 1

    def test_constant_data(self):
        """Constant data should have low changepoint score."""
        data = np.full(200, 0.001)
        detector = BayesianChangepointDetector(
            hazard_rate=1 / 100, cp_window=10,
        )
        result = detector.run(data)

        # In stable data, P(r < 10) should be modest
        mean_cp = np.mean(result.changepoint_prob[30:])
        assert mean_cp < 0.25

    def test_large_jump(self):
        """A very large sudden jump should be detected."""
        rng = np.random.default_rng(50)
        data = np.concatenate([
            rng.normal(0, 0.005, 100),
            rng.normal(0.2, 0.005, 100),  # massive shift
        ])
        detector = BayesianChangepointDetector(
            hazard_rate=1 / 50,
            detection_threshold=0.3,
            cp_window=15,
        )
        result = detector.run(data)

        assert len(result.detected_changepoints) >= 1, (
            f"No changepoints; max score={result.changepoint_prob.max():.3f}"
        )
        closest = min(
            result.detected_changepoints, key=lambda x: abs(x - 100)
        )
        assert abs(closest - 100) < 15

    def test_hazard_rate_effect(self):
        """Higher hazard rate → more changepoints detected."""
        data = np.random.default_rng(60).normal(0, 0.01, 300)

        det_low = BayesianChangepointDetector(
            hazard_rate=1 / 200, detection_threshold=0.1
        )
        det_high = BayesianChangepointDetector(
            hazard_rate=1 / 20, detection_threshold=0.1
        )

        res_low = det_low.run(data)
        res_high = det_high.run(data)

        # Higher hazard → more detected changepoints (or equal)
        assert len(res_high.detected_changepoints) >= len(
            res_low.detected_changepoints
        )
