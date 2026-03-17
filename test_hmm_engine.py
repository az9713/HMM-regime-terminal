"""
TDD tests for hmm_engine.py — RegimeDetector.

Tests cover: parameter counting, BIC model selection, Viterbi decoding,
regime labeling, regime statistics, Shannon entropy, transition matrix,
and rolling log-likelihood.
"""

import numpy as np
import pandas as pd
import pytest

from hmm_engine import RegimeDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_config(min_states=2, max_states=4, n_restarts=3, n_iter=50,
                covariance_type="full"):
    return {
        "hmm": {
            "min_states": min_states,
            "max_states": max_states,
            "n_restarts": n_restarts,
            "n_iter": n_iter,
            "tol": 1e-4,
            "covariance_type": covariance_type,
        }
    }


def make_two_regime_data(n=500, seed=42):
    """Synthetic 2-regime data: low-vol vs high-vol returns."""
    rng = np.random.default_rng(seed)
    regime = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    returns = np.where(regime == 0,
                       rng.normal(0.001, 0.01, n),
                       rng.normal(-0.001, 0.04, n))
    vol = np.abs(returns)
    X = np.column_stack([returns, vol])
    return X, regime.astype(int)


# ---------------------------------------------------------------------------
# Tests: initialization
# ---------------------------------------------------------------------------

class TestRegimeDetectorInit:
    def test_default_config(self):
        rd = RegimeDetector({})
        assert rd.min_states == 2
        assert rd.max_states == 8
        assert rd.n_restarts == 20
        assert rd.model is None

    def test_custom_config(self):
        cfg = make_config(min_states=3, max_states=5, n_restarts=10)
        rd = RegimeDetector(cfg)
        assert rd.min_states == 3
        assert rd.max_states == 5
        assert rd.n_restarts == 10


# ---------------------------------------------------------------------------
# Tests: _count_params
# ---------------------------------------------------------------------------

class TestCountParams:
    def test_full_covariance(self):
        rd = RegimeDetector(make_config(covariance_type="full"))
        # n=2, d=3: (2-1) + 2*(2-1) + 2*3 + 2*3*4//2 = 1+2+6+12 = 21
        assert rd._count_params(2, 3) == 21

    def test_diag_covariance(self):
        rd = RegimeDetector(make_config(covariance_type="diag"))
        # n=2, d=3: (2-1) + 2*(2-1) + 2*3 + 2*3 = 1+2+6+6 = 15
        assert rd._count_params(2, 3) == 15

    def test_spherical_covariance(self):
        rd = RegimeDetector(make_config(covariance_type="spherical"))
        # n=2, d=3: (2-1) + 2*(2-1) + 2*3 + 2 = 1+2+6+2 = 11
        assert rd._count_params(2, 3) == 11

    def test_tied_covariance(self):
        rd = RegimeDetector(make_config(covariance_type="tied"))
        # n=2, d=3: (2-1) + 2*(2-1) + 2*3 + 3*4//2 = 1+2+6+6 = 15
        assert rd._count_params(2, 3) == 15

    def test_single_feature(self):
        rd = RegimeDetector(make_config(covariance_type="full"))
        # n=3, d=1: (3-1) + 3*(3-1) + 3*1 + 3*1*2//2 = 2+6+3+3 = 14
        assert rd._count_params(3, 1) == 14


# ---------------------------------------------------------------------------
# Tests: fit_and_select
# ---------------------------------------------------------------------------

class TestFitAndSelect:
    def test_returns_bic_dict(self):
        X, _ = make_two_regime_data()
        rd = RegimeDetector(make_config(min_states=2, max_states=3, n_restarts=2))
        bic = rd.fit_and_select(X)
        assert isinstance(bic, dict)
        assert all(isinstance(v, float) for v in bic.values())

    def test_model_is_fitted(self):
        X, _ = make_two_regime_data()
        rd = RegimeDetector(make_config(min_states=2, max_states=3, n_restarts=2))
        rd.fit_and_select(X)
        assert rd.model is not None
        assert rd.n_states in (2, 3)

    def test_selects_best_bic(self):
        X, _ = make_two_regime_data()
        rd = RegimeDetector(make_config(min_states=2, max_states=3, n_restarts=2))
        bic = rd.fit_and_select(X)
        # The selected model should have BIC equal to min
        selected_bic = bic[rd.n_states]
        assert selected_bic == min(bic.values())

    def test_all_fits_fail_raises(self):
        """Empty data should cause all fits to fail."""
        X = np.array([]).reshape(0, 2)
        rd = RegimeDetector(make_config(min_states=2, max_states=2, n_restarts=1))
        with pytest.raises(RuntimeError, match="All HMM fits failed"):
            rd.fit_and_select(X)


# ---------------------------------------------------------------------------
# Tests: decode
# ---------------------------------------------------------------------------

class TestDecode:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.X, _ = make_two_regime_data()
        self.rd = RegimeDetector(make_config(min_states=2, max_states=3, n_restarts=2))
        self.rd.fit_and_select(self.X)

    def test_states_shape(self):
        states, posteriors = self.rd.decode(self.X)
        assert states.shape == (len(self.X),)

    def test_posteriors_shape(self):
        states, posteriors = self.rd.decode(self.X)
        assert posteriors.shape == (len(self.X), self.rd.n_states)

    def test_posteriors_sum_to_one(self):
        _, posteriors = self.rd.decode(self.X)
        np.testing.assert_allclose(posteriors.sum(axis=1), 1.0, atol=1e-6)

    def test_states_in_valid_range(self):
        states, _ = self.rd.decode(self.X)
        assert all(0 <= s < self.rd.n_states for s in states)


# ---------------------------------------------------------------------------
# Tests: label_regimes
# ---------------------------------------------------------------------------

class TestLabelRegimes:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.X, _ = make_two_regime_data()
        self.rd = RegimeDetector(make_config(min_states=2, max_states=2, n_restarts=2))
        self.rd.fit_and_select(self.X)

    def test_two_state_labels(self):
        labels = self.rd.label_regimes(self.X)
        assert set(labels.values()) == {"bear", "bull"}

    def test_labels_cover_all_states(self):
        labels = self.rd.label_regimes(self.X)
        assert len(labels) == self.rd.n_states

    def test_three_state_labels(self):
        X, _ = make_two_regime_data(n=800)
        rd = RegimeDetector(make_config(min_states=3, max_states=3, n_restarts=3))
        rd.fit_and_select(X)
        labels = rd.label_regimes(X)
        assert set(labels.values()) == {"bear", "neutral", "bull"}

    def test_four_state_labels(self):
        X, _ = make_two_regime_data(n=800)
        rd = RegimeDetector(make_config(min_states=4, max_states=4, n_restarts=3))
        rd.fit_and_select(X)
        labels = rd.label_regimes(X)
        assert set(labels.values()) == {"crash", "bear", "bull", "bull_run"}

    def test_five_state_labels(self):
        X, _ = make_two_regime_data(n=1000)
        rd = RegimeDetector(make_config(min_states=5, max_states=5, n_restarts=3))
        rd.fit_and_select(X)
        labels = rd.label_regimes(X)
        assert set(labels.values()) == {"crash", "bear", "neutral", "bull", "bull_run"}


# ---------------------------------------------------------------------------
# Tests: regime_statistics
# ---------------------------------------------------------------------------

class TestRegimeStatistics:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.X, _ = make_two_regime_data()
        self.rd = RegimeDetector(make_config(min_states=2, max_states=2, n_restarts=2))
        self.rd.fit_and_select(self.X)
        self.rd.label_regimes(self.X)

    def test_returns_dataframe(self):
        stats = self.rd.regime_statistics()
        assert isinstance(stats, pd.DataFrame)

    def test_has_expected_columns(self):
        stats = self.rd.regime_statistics()
        expected = {"state", "label", "mean_return", "volatility",
                    "expected_duration", "stationary_weight"}
        assert expected.issubset(set(stats.columns))

    def test_row_count_matches_states(self):
        stats = self.rd.regime_statistics()
        assert len(stats) == self.rd.n_states

    def test_stationary_weights_sum_to_one(self):
        stats = self.rd.regime_statistics()
        np.testing.assert_allclose(stats["stationary_weight"].sum(), 1.0, atol=1e-6)

    def test_expected_duration_positive(self):
        stats = self.rd.regime_statistics()
        assert (stats["expected_duration"] > 0).all()

    def test_volatility_positive(self):
        stats = self.rd.regime_statistics()
        assert (stats["volatility"] > 0).all()


# ---------------------------------------------------------------------------
# Tests: shannon_entropy
# ---------------------------------------------------------------------------

class TestShannonEntropy:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.X, _ = make_two_regime_data()
        self.rd = RegimeDetector(make_config(min_states=2, max_states=2, n_restarts=2))
        self.rd.fit_and_select(self.X)

    def test_entropy_shape(self):
        _, posteriors = self.rd.decode(self.X)
        entropy, confidence = self.rd.shannon_entropy(posteriors)
        assert entropy.shape == (len(self.X),)
        assert confidence.shape == (len(self.X),)

    def test_entropy_non_negative(self):
        _, posteriors = self.rd.decode(self.X)
        entropy, _ = self.rd.shannon_entropy(posteriors)
        assert (entropy >= 0).all()

    def test_confidence_in_zero_one(self):
        _, posteriors = self.rd.decode(self.X)
        _, confidence = self.rd.shannon_entropy(posteriors)
        assert (confidence >= 0 - 1e-10).all()
        assert (confidence <= 1 + 1e-10).all()

    def test_certain_posteriors_give_max_confidence(self):
        """If posteriors are [1,0] or [0,1], confidence should be 1."""
        certain = np.array([[1.0, 0.0], [0.0, 1.0]])
        _, confidence = self.rd.shannon_entropy(certain)
        np.testing.assert_allclose(confidence, 1.0, atol=1e-6)

    def test_uniform_posteriors_give_zero_confidence(self):
        """If posteriors are uniform, confidence should be ~0."""
        uniform = np.array([[0.5, 0.5], [0.5, 0.5]])
        _, confidence = self.rd.shannon_entropy(uniform)
        np.testing.assert_allclose(confidence, 0.0, atol=1e-2)


# ---------------------------------------------------------------------------
# Tests: transition_matrix
# ---------------------------------------------------------------------------

class TestTransitionMatrix:
    def test_shape(self):
        X, _ = make_two_regime_data()
        rd = RegimeDetector(make_config(min_states=2, max_states=2, n_restarts=2))
        rd.fit_and_select(X)
        tm = rd.transition_matrix()
        assert tm.shape == (rd.n_states, rd.n_states)

    def test_rows_sum_to_one(self):
        X, _ = make_two_regime_data()
        rd = RegimeDetector(make_config(min_states=2, max_states=2, n_restarts=2))
        rd.fit_and_select(X)
        tm = rd.transition_matrix()
        np.testing.assert_allclose(tm.sum(axis=1), 1.0, atol=1e-6)

    def test_is_copy(self):
        X, _ = make_two_regime_data()
        rd = RegimeDetector(make_config(min_states=2, max_states=2, n_restarts=2))
        rd.fit_and_select(X)
        tm = rd.transition_matrix()
        tm[0, 0] = -999
        assert rd.model.transmat_[0, 0] != -999


# ---------------------------------------------------------------------------
# Tests: log_likelihood_series
# ---------------------------------------------------------------------------

class TestLogLikelihoodSeries:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.X, _ = make_two_regime_data(n=200)
        self.rd = RegimeDetector(make_config(min_states=2, max_states=2, n_restarts=2))
        self.rd.fit_and_select(self.X)

    def test_shape(self):
        ll = self.rd.log_likelihood_series(self.X, window=20)
        assert len(ll) == len(self.X)

    def test_nans_at_start(self):
        window = 20
        ll = self.rd.log_likelihood_series(self.X, window=window)
        assert np.all(np.isnan(ll[:window - 1]))

    def test_values_after_window(self):
        window = 20
        ll = self.rd.log_likelihood_series(self.X, window=window)
        valid = ll[window - 1:]
        assert np.all(np.isfinite(valid))
