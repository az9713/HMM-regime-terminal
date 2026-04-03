"""
TDD tests for regime_forecast.py — RegimeForecaster.

Tests cover: Chapman-Kolmogorov projection, regime half-life computation,
expected return/volatility forecasts, return cones, transition countdown,
posterior momentum, anticipatory signals, survival curves, and mean
first-passage times.
"""

import numpy as np
import pandas as pd
import pytest

from regime_forecast import RegimeForecaster, RegimeForecast, ForecastSeries


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_config(**overrides):
    cfg = {
        "strategy": {
            "hysteresis_bars": 3,
            "confirmations": {"min_confidence": 0.6},
        },
    }
    cfg.update(overrides)
    return cfg


def make_simple_transmat():
    """2-state transition matrix with clear persistence."""
    return np.array([
        [0.9, 0.1],
        [0.2, 0.8],
    ])


def make_simple_hmm_params():
    """2-state HMM with distinct emission parameters."""
    transmat = make_simple_transmat()
    means = np.array([
        [-0.01, 0.03],   # state 0: bear (negative return, high vol)
        [0.01, 0.01],    # state 1: bull (positive return, low vol)
    ])
    covars = np.array([
        [[0.0016, 0.0], [0.0, 0.0004]],  # state 0
        [[0.0004, 0.0], [0.0, 0.0001]],  # state 1
    ])
    labels = {0: "bear", 1: "bull"}
    return transmat, means, covars, labels


# ---------------------------------------------------------------------------
# Tests: Chapman-Kolmogorov projection
# ---------------------------------------------------------------------------

class TestChapmanKolmogorov:
    def test_forecast_probs_sum_to_one(self):
        """Each row of forecast_probs must be a valid probability distribution."""
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=10)
        transmat, means, covars, labels = make_simple_hmm_params()

        posterior = np.array([0.0, 1.0])  # certain bull
        forecast = fc.forecast_at_bar(
            posterior, transmat, means, covars, labels,
            current_state=1,
        )

        for h in range(10):
            row_sum = forecast.forecast_probs[h].sum()
            assert abs(row_sum - 1.0) < 1e-10, f"Horizon {h}: sum = {row_sum}"

    def test_forecast_converges_to_stationary(self):
        """At large horizons, forecast should approach stationary distribution."""
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=200)
        transmat, means, covars, labels = make_simple_hmm_params()

        # Compute stationary distribution
        eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary /= stationary.sum()

        posterior = np.array([1.0, 0.0])  # start certain bear
        forecast = fc.forecast_at_bar(
            posterior, transmat, means, covars, labels,
            current_state=0,
        )

        # At horizon 200, should be close to stationary
        np.testing.assert_allclose(
            forecast.forecast_probs[-1], stationary, atol=1e-4,
        )

    def test_one_step_matches_transmat(self):
        """One-step-ahead forecast should be posterior @ A."""
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=5)
        transmat, means, covars, labels = make_simple_hmm_params()

        posterior = np.array([0.3, 0.7])
        forecast = fc.forecast_at_bar(
            posterior, transmat, means, covars, labels,
            current_state=1,
        )

        expected = posterior @ transmat
        np.testing.assert_allclose(forecast.forecast_probs[0], expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: Regime half-life
# ---------------------------------------------------------------------------

class TestHalfLife:
    def test_half_life_certain_persistent_state(self):
        """A state with a_ii = 0.9 should have a half-life around 6-7 bars."""
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=50)
        transmat, means, covars, labels = make_simple_hmm_params()

        posterior = np.array([0.0, 1.0])  # certain bull
        forecast = fc.forecast_at_bar(
            posterior, transmat, means, covars, labels,
            current_state=1,
        )

        # Geometric decay: 0.8^h = 0.5 => h = log(0.5)/log(0.8) ~ 3.1
        # But with 2 states, there's inflow from state 0 too
        assert 1.0 < forecast.half_life < 20.0

    def test_half_life_absorbing_state(self):
        """A nearly absorbing state (a_ii ~ 1) should have max half-life."""
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=20)
        transmat = np.array([[0.99, 0.01], [0.01, 0.99]])
        means = np.array([[-0.01, 0.03], [0.01, 0.01]])
        covars = np.array([
            [[0.0016, 0.0], [0.0, 0.0004]],
            [[0.0004, 0.0], [0.0, 0.0001]],
        ])
        labels = {0: "bear", 1: "bull"}

        posterior = np.array([0.0, 1.0])
        forecast = fc.forecast_at_bar(
            posterior, transmat, means, covars, labels,
            current_state=1,
        )

        assert forecast.half_life == 20.0  # never drops below 0.5 in horizon


# ---------------------------------------------------------------------------
# Tests: Expected returns and volatility
# ---------------------------------------------------------------------------

class TestExpectedReturns:
    def test_certain_state_returns(self):
        """When certain of a regime, expected return should match regime mean."""
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=5)
        transmat = np.eye(2)  # no transitions
        means = np.array([[-0.01, 0.03], [0.02, 0.01]])
        covars = np.array([
            [[0.0001, 0.0], [0.0, 0.0001]],
            [[0.0001, 0.0], [0.0, 0.0001]],
        ])
        labels = {0: "bear", 1: "bull"}

        posterior = np.array([0.0, 1.0])
        forecast = fc.forecast_at_bar(
            posterior, transmat, means, covars, labels,
            current_state=1,
        )

        # With identity transmat, always in bull: expected return = 0.02
        np.testing.assert_allclose(forecast.expected_returns, 0.02, atol=1e-10)

    def test_expected_returns_mixed(self):
        """With mixed posteriors, expected return is weighted average."""
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=1)
        transmat = np.eye(2)
        means = np.array([[-0.01, 0.0], [0.03, 0.0]])
        covars = np.array([
            [[0.0001, 0.0], [0.0, 0.0001]],
            [[0.0001, 0.0], [0.0, 0.0001]],
        ])
        labels = {0: "bear", 1: "bull"}

        posterior = np.array([0.4, 0.6])
        forecast = fc.forecast_at_bar(
            posterior, transmat, means, covars, labels,
            current_state=1,
        )

        expected = 0.4 * (-0.01) + 0.6 * 0.03
        assert abs(forecast.expected_returns[0] - expected) < 1e-10


# ---------------------------------------------------------------------------
# Tests: Return cone
# ---------------------------------------------------------------------------

class TestReturnCone:
    def test_cone_shape(self):
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=10)
        transmat, means, covars, labels = make_simple_hmm_params()

        posterior = np.array([0.5, 0.5])
        forecast = fc.forecast_at_bar(
            posterior, transmat, means, covars, labels,
            current_state=0,
        )

        assert forecast.return_cone.shape == (10, 5)

    def test_cone_ordering(self):
        """Percentiles should be ordered: 5th <= 25th <= 50th <= 75th <= 95th."""
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=10)
        transmat, means, covars, labels = make_simple_hmm_params()

        posterior = np.array([0.3, 0.7])
        forecast = fc.forecast_at_bar(
            posterior, transmat, means, covars, labels,
            current_state=1,
        )

        for h in range(10):
            for i in range(4):
                assert forecast.return_cone[h, i] <= forecast.return_cone[h, i + 1], \
                    f"Horizon {h}: percentile order violated"


# ---------------------------------------------------------------------------
# Tests: Transition countdown
# ---------------------------------------------------------------------------

class TestTransitionCountdown:
    def test_starts_low_for_persistent_state(self):
        """P(change within 1 bar) should be low for persistent states."""
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=10)
        transmat, means, covars, labels = make_simple_hmm_params()

        posterior = np.array([0.0, 1.0])
        forecast = fc.forecast_at_bar(
            posterior, transmat, means, covars, labels,
            current_state=1,
        )

        assert forecast.transition_prob_curve[0] < 0.3

    def test_monotonically_increasing(self):
        """Transition probability should generally increase with horizon."""
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=50)
        transmat, means, covars, labels = make_simple_hmm_params()

        posterior = np.array([0.0, 1.0])
        forecast = fc.forecast_at_bar(
            posterior, transmat, means, covars, labels,
            current_state=1,
        )

        # Should be non-decreasing (with small tolerance for numerical noise)
        for h in range(1, 50):
            assert forecast.transition_prob_curve[h] >= forecast.transition_prob_curve[h - 1] - 1e-10


# ---------------------------------------------------------------------------
# Tests: Posterior momentum
# ---------------------------------------------------------------------------

class TestPosteriorMomentum:
    def test_zero_without_history(self):
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=5)
        transmat, means, covars, labels = make_simple_hmm_params()

        posterior = np.array([0.5, 0.5])
        forecast = fc.forecast_at_bar(
            posterior, transmat, means, covars, labels,
            current_state=0,
            posterior_history=None,
        )

        np.testing.assert_array_equal(forecast.posterior_velocity, [0, 0])

    def test_velocity_with_history(self):
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=5)
        transmat, means, covars, labels = make_simple_hmm_params()

        history = np.array([
            [0.8, 0.2],
            [0.6, 0.4],
            [0.4, 0.6],
        ])

        forecast = fc.forecast_at_bar(
            history[-1], transmat, means, covars, labels,
            current_state=1,
            posterior_history=history,
        )

        expected_vel = history[-1] - history[-2]  # [-.2, .2]
        np.testing.assert_allclose(forecast.posterior_velocity, expected_vel)


# ---------------------------------------------------------------------------
# Tests: Forecast series (bulk computation)
# ---------------------------------------------------------------------------

class TestForecastSeries:
    def _make_synthetic_data(self, T=100, n_states=2, seed=42):
        rng = np.random.default_rng(seed)
        states = np.zeros(T, dtype=int)
        states[T // 2:] = 1

        posteriors = np.zeros((T, n_states))
        for t in range(T):
            posteriors[t, states[t]] = 0.85
            posteriors[t, 1 - states[t]] = 0.15

        return states, posteriors

    def test_output_lengths(self):
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=10)
        transmat, means, covars, labels = make_simple_hmm_params()
        states, posteriors = self._make_synthetic_data(T=100)

        result = fc.forecast_series(
            posteriors, states, transmat, means, covars, labels,
        )

        assert len(result.half_lives) == 100
        assert len(result.transition_urgency) == 100
        assert len(result.posterior_velocity_norm) == 100
        assert len(result.anticipated_regime) == 100
        assert len(result.anticipated_confidence) == 100
        assert len(result.expected_return_5) == 100
        assert len(result.expected_return_10) == 100
        assert len(result.anticipatory_signals) == 100

    def test_anticipatory_signals_valid(self):
        """Anticipatory signals should be in {-1, 0, 1}."""
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=10)
        transmat, means, covars, labels = make_simple_hmm_params()
        states, posteriors = self._make_synthetic_data(T=100)

        result = fc.forecast_series(
            posteriors, states, transmat, means, covars, labels,
        )

        assert set(np.unique(result.anticipatory_signals)).issubset({-1, 0, 1})

    def test_transition_urgency_bounded(self):
        """Transition urgency should be in [0, 1]."""
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=10)
        transmat, means, covars, labels = make_simple_hmm_params()
        states, posteriors = self._make_synthetic_data(T=100)

        result = fc.forecast_series(
            posteriors, states, transmat, means, covars, labels,
        )

        assert np.all(result.transition_urgency >= 0)
        assert np.all(result.transition_urgency <= 1)


# ---------------------------------------------------------------------------
# Tests: Survival curve
# ---------------------------------------------------------------------------

class TestSurvivalCurve:
    def test_starts_near_one(self):
        """Survival at step 1 should equal a_ii."""
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=10)
        transmat = make_simple_transmat()

        survival = fc.regime_survival_curve(transmat, state=1, max_steps=10)
        assert abs(survival[0] - 0.8) < 0.1  # a_11 = 0.8 but inflow matters

    def test_decays_over_time(self):
        """Survival probability should generally decay."""
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=50)
        transmat = make_simple_transmat()

        survival = fc.regime_survival_curve(transmat, state=0, max_steps=50)

        # First value should be highest, last should be lowest
        assert survival[0] > survival[-1]


# ---------------------------------------------------------------------------
# Tests: Mean first-passage times
# ---------------------------------------------------------------------------

class TestAbsorptionTimes:
    def test_diagonal_is_zero(self):
        """MFPT from state to itself should be 0."""
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=10)
        transmat, _, _, labels = make_simple_hmm_params()

        mfpt = fc.regime_absorption_times(transmat, labels)

        for label in mfpt.index:
            assert mfpt.loc[label, label] == 0.0

    def test_symmetric_for_symmetric_chain(self):
        """For a symmetric transition matrix, MFPT should be symmetric."""
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=10)
        transmat = np.array([[0.8, 0.2], [0.2, 0.8]])
        labels = {0: "A", 1: "B"}

        mfpt = fc.regime_absorption_times(transmat, labels)
        assert abs(mfpt.loc["A", "B"] - mfpt.loc["B", "A"]) < 1e-10

    def test_mfpt_positive(self):
        """All off-diagonal MFPT should be positive."""
        config = make_config()
        fc = RegimeForecaster(config, max_horizon=10)
        transmat, _, _, labels = make_simple_hmm_params()

        mfpt = fc.regime_absorption_times(transmat, labels)

        for i in mfpt.index:
            for j in mfpt.columns:
                if i != j:
                    assert mfpt.loc[i, j] > 0, f"MFPT[{i},{j}] should be positive"
