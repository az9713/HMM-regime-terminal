"""
test_changepoint.py — Tests for Bayesian Online Changepoint Detection.

Tests the BOCPD algorithm, ensemble metrics, and detection lead computation.
"""

import numpy as np
import pytest
from changepoint import (
    BayesianChangepointDetector,
    BOCPDResult,
    ensemble_hmm_bocpd,
    compute_detection_lead,
)


@pytest.fixture
def default_config():
    return {
        "changepoint": {
            "hazard_rate": 1 / 50,
            "mu_prior": 0.0,
            "kappa_prior": 1.0,
            "alpha_prior": 1.0,
            "beta_prior": 1.0,
            "threshold": 0.2,
        }
    }


@pytest.fixture
def regime_shift_data():
    """Synthetic data with a clear regime shift at bar 50."""
    np.random.seed(42)
    segment1 = np.random.normal(0.0, 0.5, 50)   # low-return regime
    segment2 = np.random.normal(2.0, 0.5, 50)    # high-return regime
    return np.concatenate([segment1, segment2])


@pytest.fixture
def multi_regime_data():
    """Synthetic data with 3 regimes."""
    np.random.seed(123)
    seg1 = np.random.normal(-1.0, 0.3, 40)   # bear
    seg2 = np.random.normal(0.0, 0.3, 40)    # neutral
    seg3 = np.random.normal(1.5, 0.3, 40)    # bull
    return np.concatenate([seg1, seg2, seg3])


class TestBayesianChangepointDetector:
    def test_init_defaults(self):
        detector = BayesianChangepointDetector({})
        assert detector.hazard_rate == pytest.approx(1 / 150)
        assert detector.threshold == 0.3

    def test_init_from_config(self, default_config):
        detector = BayesianChangepointDetector(default_config)
        assert detector.hazard_rate == pytest.approx(1 / 50)
        assert detector.threshold == 0.2

    def test_detect_returns_bocpd_result(self, default_config, regime_shift_data):
        detector = BayesianChangepointDetector(default_config)
        result = detector.detect(regime_shift_data)
        assert isinstance(result, BOCPDResult)
        assert len(result.changepoint_prob) == len(regime_shift_data)
        assert len(result.expected_run_length) == len(regime_shift_data)
        assert len(result.map_run_length) == len(regime_shift_data)

    def test_changepoint_prob_range(self, default_config, regime_shift_data):
        detector = BayesianChangepointDetector(default_config)
        result = detector.detect(regime_shift_data)
        assert np.all(result.changepoint_prob >= 0.0)
        assert np.all(result.changepoint_prob <= 1.0)

    def test_detects_regime_shift(self, default_config, regime_shift_data):
        """BOCPD should detect a changepoint near bar 50."""
        # Use a lower threshold for small synthetic data
        cfg = {**default_config, "changepoint": {**default_config["changepoint"], "threshold": 0.05}}
        detector = BayesianChangepointDetector(cfg)
        result = detector.detect(regime_shift_data)
        # At least one detected changepoint should be within 10 bars of 50
        nearby = [cp for cp in result.detected_changepoints if abs(cp - 50) <= 10]
        assert len(nearby) > 0, (
            f"No changepoint detected near bar 50. "
            f"Detected: {result.detected_changepoints}, "
            f"Max cp_prob: {result.changepoint_prob.max():.4f} at bar {result.changepoint_prob.argmax()}"
        )

    def test_expected_run_length_resets(self, default_config, regime_shift_data):
        """Expected run length should drop near changepoints."""
        detector = BayesianChangepointDetector(default_config)
        result = detector.detect(regime_shift_data)
        erl = result.expected_run_length
        # Run length should be growing in each segment, then drop
        # Check that max run length in first 45 bars > run length right after change
        max_first = np.max(erl[20:45])
        min_near_change = np.min(erl[48:55])
        assert max_first > min_near_change

    def test_segments_cover_data(self, default_config, regime_shift_data):
        detector = BayesianChangepointDetector(default_config)
        result = detector.detect(regime_shift_data)
        total_bars = sum(seg["length"] for seg in result.segments)
        assert total_bars == len(regime_shift_data)

    def test_segments_have_stats(self, default_config, regime_shift_data):
        detector = BayesianChangepointDetector(default_config)
        result = detector.detect(regime_shift_data)
        for seg in result.segments:
            assert "mean" in seg
            assert "std" in seg
            assert "cumulative" in seg
            assert seg["length"] > 0

    def test_multi_regime_detection(self, default_config, multi_regime_data):
        """Should detect at least one of the two boundaries."""
        cfg = {**default_config, "changepoint": {**default_config["changepoint"], "threshold": 0.05}}
        detector = BayesianChangepointDetector(cfg)
        result = detector.detect(multi_regime_data)
        # At least 1 changepoint in the ballpark of bar 40 or 80
        near_40 = [cp for cp in result.detected_changepoints if abs(cp - 40) <= 10]
        near_80 = [cp for cp in result.detected_changepoints if abs(cp - 80) <= 10]
        assert len(near_40) + len(near_80) >= 1

    def test_constant_data_no_changepoints(self, default_config):
        """Constant data should produce few/no changepoints."""
        data = np.zeros(100)
        detector = BayesianChangepointDetector(default_config)
        result = detector.detect(data)
        # Should detect very few changepoints after initial settling
        late_cps = [cp for cp in result.detected_changepoints if cp > 10]
        assert len(late_cps) <= 2

    def test_run_length_map_shape(self, default_config, regime_shift_data):
        detector = BayesianChangepointDetector(default_config)
        result = detector.detect(regime_shift_data)
        T = len(regime_shift_data)
        assert result.run_length_map.shape == (T, T)


class TestEnsembleHmmBocpd:
    @pytest.fixture
    def mock_hmm_data(self):
        T = 100
        np.random.seed(42)
        states = np.array([0] * 50 + [1] * 50)
        posteriors = np.zeros((T, 2))
        posteriors[:50, 0] = 0.9
        posteriors[:50, 1] = 0.1
        posteriors[50:, 0] = 0.1
        posteriors[50:, 1] = 0.9
        entropy = np.full(T, 0.3)
        confidence = np.full(T, 0.7)
        labels = {0: "bear", 1: "bull"}
        return states, posteriors, entropy, confidence, labels

    @pytest.fixture
    def mock_bocpd_result(self):
        T = 100
        cp_prob = np.full(T, 0.01)
        cp_prob[48] = 0.5  # BOCPD detects 2 bars early
        return BOCPDResult(
            changepoint_prob=cp_prob,
            run_length_map=np.zeros((T, T)),
            expected_run_length=np.arange(T, dtype=float),
            map_run_length=np.arange(T),
            detected_changepoints=[48],
            segments=[
                {"start": 0, "end": 48, "length": 48, "mean": 0.0, "std": 0.5, "cumulative": 0.0},
                {"start": 48, "end": 100, "length": 52, "mean": 2.0, "std": 0.5, "cumulative": 104.0},
            ],
        )

    def test_ensemble_output_shape(self, mock_hmm_data, mock_bocpd_result):
        states, posteriors, entropy, confidence, labels = mock_hmm_data
        df = ensemble_hmm_bocpd(
            states, posteriors, entropy, confidence, labels, mock_bocpd_result
        )
        assert len(df) == 100
        assert "regime_change_score" in df.columns
        assert "early_warning" in df.columns
        assert "consensus_confidence" in df.columns
        assert "regime_stability" in df.columns

    def test_early_warning_detected(self, mock_hmm_data, mock_bocpd_result):
        """BOCPD fires at 48, HMM at 50 — early warning should be nonzero near 48."""
        states, posteriors, entropy, confidence, labels = mock_hmm_data
        df = ensemble_hmm_bocpd(
            states, posteriors, entropy, confidence, labels, mock_bocpd_result
        )
        # Early warning should be elevated near bar 48
        early_vals = df.loc[45:52, "early_warning"].values
        assert np.max(early_vals) > 0

    def test_consensus_confidence_range(self, mock_hmm_data, mock_bocpd_result):
        states, posteriors, entropy, confidence, labels = mock_hmm_data
        df = ensemble_hmm_bocpd(
            states, posteriors, entropy, confidence, labels, mock_bocpd_result
        )
        assert df["consensus_confidence"].min() >= 0.0
        assert df["consensus_confidence"].max() <= 1.5  # can boost up to 1.3x

    def test_regime_stability_range(self, mock_hmm_data, mock_bocpd_result):
        states, posteriors, entropy, confidence, labels = mock_hmm_data
        df = ensemble_hmm_bocpd(
            states, posteriors, entropy, confidence, labels, mock_bocpd_result
        )
        assert df["regime_stability"].min() >= 0.0
        assert df["regime_stability"].max() <= 1.0


class TestDetectionLead:
    def test_bocpd_earlier(self):
        hmm = [50, 100]
        bocpd = [47, 98]
        result = compute_detection_lead(hmm, bocpd)
        assert result["n_matched"] == 2
        assert result["mean_lead"] > 0  # BOCPD is earlier
        assert result["bocpd_early_pct"] == 1.0

    def test_hmm_earlier(self):
        hmm = [50, 100]
        bocpd = [53, 104]
        result = compute_detection_lead(hmm, bocpd)
        assert result["n_matched"] == 2
        assert result["mean_lead"] < 0  # HMM is earlier

    def test_no_match_too_far(self):
        hmm = [50]
        bocpd = [100]
        result = compute_detection_lead(hmm, bocpd, max_distance=10)
        assert result["n_matched"] == 0
        assert len(result["unmatched_hmm"]) == 1
        assert len(result["unmatched_bocpd"]) == 1

    def test_empty_inputs(self):
        result = compute_detection_lead([], [])
        assert result["n_matched"] == 0
        assert result["mean_lead"] == 0.0

    def test_simultaneous_detection(self):
        hmm = [50]
        bocpd = [50]
        result = compute_detection_lead(hmm, bocpd)
        assert result["n_matched"] == 1
        assert result["mean_lead"] == 0.0
