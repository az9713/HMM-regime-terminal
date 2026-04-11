"""Unit tests for regime_forecast."""
import numpy as np
import pytest

from regime_forecast import (
    ForecastResult,
    FragilityReport,
    expected_regime_durations,
    first_passage_distribution,
    forecast_from_detector,
    fragility_adjusted_position,
    hitting_probability,
    horizon_forecast,
    kl_divergence,
    n_step_distribution,
    regime_change_probability,
    regime_fragility_index,
    stationary_distribution,
    survival_curve,
)


# ──────────────────────────────────────────────────────────────────────────
# Test fixtures
# ──────────────────────────────────────────────────────────────────────────

def _two_state():
    # Persistent bull / bear.
    return np.array([
        [0.9, 0.1],
        [0.2, 0.8],
    ])


def _three_state():
    return np.array([
        [0.85, 0.10, 0.05],
        [0.15, 0.70, 0.15],
        [0.05, 0.10, 0.85],
    ])


def _uniform(n: int):
    return np.full((n, n), 1.0 / n)


# ──────────────────────────────────────────────────────────────────────────
# Stationary distribution
# ──────────────────────────────────────────────────────────────────────────

class TestStationary:
    def test_sums_to_one(self):
        pi = stationary_distribution(_two_state())
        assert abs(pi.sum() - 1.0) < 1e-10
        assert np.all(pi >= 0.0)

    def test_is_fixed_point(self):
        P = _three_state()
        pi = stationary_distribution(P)
        np.testing.assert_allclose(pi @ P, pi, atol=1e-10)

    def test_uniform_chain(self):
        pi = stationary_distribution(_uniform(4))
        np.testing.assert_allclose(pi, np.full(4, 0.25), atol=1e-10)


# ──────────────────────────────────────────────────────────────────────────
# N-step distribution
# ──────────────────────────────────────────────────────────────────────────

class TestNStepDistribution:
    def test_zero_step_returns_initial(self):
        init = np.array([0.3, 0.7])
        out = n_step_distribution(_two_state(), init, horizon=0)
        np.testing.assert_allclose(out[0], init)

    def test_shape(self):
        out = n_step_distribution(
            _three_state(), np.array([1.0, 0.0, 0.0]), horizon=10
        )
        assert out.shape == (11, 3)

    def test_rows_sum_to_one(self):
        out = n_step_distribution(
            _three_state(), np.array([0.2, 0.5, 0.3]), horizon=20
        )
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)

    def test_one_step_matches_matmul(self):
        P = _two_state()
        init = np.array([0.3, 0.7])
        out = n_step_distribution(P, init, horizon=1)
        np.testing.assert_allclose(out[1], init @ P)

    def test_converges_to_stationary(self):
        P = _three_state()
        pi = stationary_distribution(P)
        out = n_step_distribution(P, np.array([1.0, 0.0, 0.0]), horizon=500)
        np.testing.assert_allclose(out[-1], pi, atol=1e-6)

    def test_handles_unnormalised_initial(self):
        out = n_step_distribution(_two_state(), np.array([3.0, 7.0]), horizon=2)
        np.testing.assert_allclose(out[0].sum(), 1.0, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────
# Expected durations
# ──────────────────────────────────────────────────────────────────────────

class TestDurations:
    def test_two_state_durations(self):
        P = np.array([[0.95, 0.05], [0.10, 0.90]])
        dur = expected_regime_durations(P)
        np.testing.assert_allclose(dur, [20.0, 10.0])

    def test_degenerate_diagonal(self):
        # a_ii = 1.0 — should return a finite large value, not inf.
        P = np.eye(3)
        dur = expected_regime_durations(P)
        assert np.all(np.isfinite(dur))
        assert np.all(dur > 1e6)


# ──────────────────────────────────────────────────────────────────────────
# Survival curves
# ──────────────────────────────────────────────────────────────────────────

class TestSurvival:
    def test_monotone_non_increasing(self):
        s = survival_curve(_three_state(), state_idx=0, horizon=10)
        assert s[0] == 1.0
        assert np.all(np.diff(s) <= 1e-12)

    def test_exact_value(self):
        # a_00 = 0.9
        s = survival_curve(_two_state(), state_idx=0, horizon=3)
        np.testing.assert_allclose(s, [1.0, 0.9, 0.81, 0.729])


# ──────────────────────────────────────────────────────────────────────────
# First passage / hitting probability
# ──────────────────────────────────────────────────────────────────────────

class TestFirstPassage:
    def test_nonnegative_and_bounded(self):
        fpd = first_passage_distribution(
            _three_state(), from_state=0, to_state=2, horizon=100
        )
        assert np.all(fpd >= 0.0)
        assert fpd.sum() <= 1.0 + 1e-9

    def test_index_zero_is_zero_for_distinct_states(self):
        fpd = first_passage_distribution(
            _three_state(), from_state=0, to_state=2, horizon=5
        )
        assert fpd[0] == 0.0

    def test_self_passage_is_delta(self):
        fpd = first_passage_distribution(
            _three_state(), from_state=1, to_state=1, horizon=5
        )
        assert fpd[0] == 1.0
        assert fpd[1:].sum() == 0.0

    def test_hitting_monotone_in_horizon(self):
        P = _three_state()
        p5 = hitting_probability(P, 0, 2, horizon=5)
        p50 = hitting_probability(P, 0, 2, horizon=50)
        p500 = hitting_probability(P, 0, 2, horizon=500)
        assert p5 <= p50 <= p500
        assert p500 <= 1.0 + 1e-9
        # Irreducible aperiodic chain — will almost surely hit.
        assert p500 > 0.9

    def test_unreachable_state(self):
        # Two absorbing singletons — state 1 can never reach state 0.
        P = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert hitting_probability(P, 1, 0, horizon=100) == 0.0


# ──────────────────────────────────────────────────────────────────────────
# Regime change probability
# ──────────────────────────────────────────────────────────────────────────

class TestRegimeChange:
    def test_zero_horizon_is_zero(self):
        assert regime_change_probability(
            _two_state(), np.array([1.0, 0.0]), horizon=0
        ) == 0.0

    def test_monotone_in_horizon(self):
        P = _two_state()
        belief = np.array([0.8, 0.2])
        probs = [regime_change_probability(P, belief, h) for h in range(20)]
        diffs = np.diff(probs)
        assert np.all(diffs >= -1e-12)

    def test_identity_chain_never_changes(self):
        prob = regime_change_probability(
            np.eye(3), np.array([0.5, 0.5, 0.0]), horizon=100
        )
        assert prob == 0.0

    def test_more_persistent_state_lower_flip_prob(self):
        P = np.array([[0.99, 0.01], [0.20, 0.80]])
        p_persistent = regime_change_probability(P, np.array([1.0, 0.0]), horizon=10)
        p_volatile = regime_change_probability(P, np.array([0.0, 1.0]), horizon=10)
        assert p_persistent < p_volatile


# ──────────────────────────────────────────────────────────────────────────
# Horizon forecast
# ──────────────────────────────────────────────────────────────────────────

class TestHorizonForecast:
    def test_shapes(self):
        fc = horizon_forecast(
            _three_state(),
            np.array([0.1, 0.8, 0.1]),
            regime_means=np.array([-0.02, 0.0, 0.03]),
            regime_vars=np.array([0.01, 0.005, 0.008]),
            horizon=20,
        )
        assert isinstance(fc, ForecastResult)
        assert fc.distribution.shape == (21, 3)
        assert fc.step_expected_return.shape == (21,)
        assert fc.cum_expected_return.shape == (21,)
        assert fc.cum_variance.shape == (21,)
        assert fc.regime_change_prob.shape == (21,)

    def test_cumulative_variance_non_decreasing(self):
        fc = horizon_forecast(
            _three_state(),
            np.array([0.2, 0.5, 0.3]),
            regime_means=np.array([-0.01, 0.0, 0.02]),
            regime_vars=np.array([0.005, 0.003, 0.007]),
            horizon=30,
        )
        assert np.all(np.diff(fc.cum_variance) >= -1e-12)

    def test_optimal_horizon_consistent(self):
        fc = horizon_forecast(
            _three_state(),
            np.array([0.0, 0.0, 1.0]),
            regime_means=np.array([-0.01, 0.0, 0.02]),
            regime_vars=np.array([0.005, 0.003, 0.002]),
            horizon=40,
        )
        assert 1 <= fc.optimal_horizon <= 40
        assert fc.optimal_sharpe == fc.cum_sharpe[fc.optimal_horizon]
        # Must be max over h ≥ 1.
        assert fc.optimal_sharpe >= fc.cum_sharpe[1:].max() - 1e-12

    def test_to_dataframe_columns(self):
        fc = horizon_forecast(
            _two_state(),
            np.array([1.0, 0.0]),
            regime_means=np.array([-0.01, 0.01]),
            regime_vars=np.array([0.002, 0.002]),
            labels={0: "bear", 1: "bull"},
            horizon=5,
        )
        df = fc.to_dataframe()
        assert len(df) == 6
        for col in ("step", "cum_expected_return", "cum_sharpe", "regime_change_prob",
                    "P(bear)", "P(bull)"):
            assert col in df.columns

    def test_labels_fallback(self):
        fc = horizon_forecast(
            _two_state(),
            np.array([1.0, 0.0]),
            regime_means=np.array([-0.01, 0.01]),
            regime_vars=np.array([0.002, 0.002]),
            horizon=3,
        )
        assert fc.regime_labels == ["state_0", "state_1"]

    def test_zero_variance_regimes_give_zero_sharpe(self):
        fc = horizon_forecast(
            _two_state(),
            np.array([1.0, 0.0]),
            regime_means=np.array([0.0, 0.0]),
            regime_vars=np.array([0.0, 0.0]),
            horizon=5,
        )
        assert np.allclose(fc.cum_sharpe, 0.0)


# ──────────────────────────────────────────────────────────────────────────
# Fragility index
# ──────────────────────────────────────────────────────────────────────────

class TestFragility:
    def test_index_in_unit_interval(self):
        rep = regime_fragility_index(
            posteriors=np.tile(np.array([0.1, 0.8, 0.1]), (30, 1)),
            transmat=_three_state(),
            entropy_series=np.zeros(30),
            horizon=5,
        )
        assert isinstance(rep, FragilityReport)
        assert 0.0 <= rep.index <= 1.0

    def test_conviction_beats_balanced(self):
        P = _three_state()
        strong = np.tile(np.array([0.02, 0.96, 0.02]), (20, 1))
        balanced = np.tile(np.array([0.33, 0.34, 0.33]), (20, 1))
        s = regime_fragility_index(strong, P, np.zeros(20))
        b = regime_fragility_index(balanced, P, np.full(20, np.log2(3)))
        assert s.index < b.index

    def test_components_keys(self):
        rep = regime_fragility_index(
            np.tile(np.array([0.5, 0.5]), (10, 1)),
            _two_state(),
            np.zeros(10),
        )
        assert set(rep.components) == {
            "posterior_closeness",
            "entropy_gradient",
            "stationary_convergence",
            "horizon_change_probability",
        }

    def test_empty_posteriors_raises(self):
        with pytest.raises(ValueError):
            regime_fragility_index(
                np.zeros((0, 2)), _two_state(), np.zeros(0)
            )

    def test_rising_entropy_more_fragile(self):
        P = _two_state()
        posteriors = np.tile(np.array([0.7, 0.3]), (10, 1))
        flat_entropy = np.zeros(10)
        rising_entropy = np.linspace(0.0, 0.9, 10)
        flat_rep = regime_fragility_index(posteriors, P, flat_entropy)
        rising_rep = regime_fragility_index(posteriors, P, rising_entropy)
        assert (
            rising_rep.components["entropy_gradient"]
            > flat_rep.components["entropy_gradient"]
        )


# ──────────────────────────────────────────────────────────────────────────
# KL divergence
# ──────────────────────────────────────────────────────────────────────────

class TestKL:
    def test_zero_for_identical(self):
        p = np.array([0.2, 0.3, 0.5])
        assert abs(kl_divergence(p, p)) < 1e-10

    def test_positive_for_distinct(self):
        assert kl_divergence(np.array([0.9, 0.1]), np.array([0.1, 0.9])) > 0.0

    def test_asymmetric(self):
        p = np.array([0.1, 0.2, 0.7])
        q = np.array([0.4, 0.4, 0.2])
        assert not np.isclose(kl_divergence(p, q), kl_divergence(q, p))


# ──────────────────────────────────────────────────────────────────────────
# Fragility-adjusted position sizing
# ──────────────────────────────────────────────────────────────────────────

class TestPositionSizing:
    def test_zero_fragility_high_sharpe_saturates(self):
        # tanh(10) ≈ 1, so result ≈ base.
        s = fragility_adjusted_position(1.0, fragility=0.0, expected_sharpe=10.0)
        assert abs(s - 1.0) < 1e-6

    def test_monotone_in_fragility(self):
        s_low = fragility_adjusted_position(1.0, 0.1, 1.0)
        s_high = fragility_adjusted_position(1.0, 0.9, 1.0)
        assert s_high < s_low

    def test_negative_sharpe_forces_zero(self):
        assert fragility_adjusted_position(1.0, 0.2, -2.0) == 0.0

    def test_clamps_to_max(self):
        s = fragility_adjusted_position(2.0, 0.0, 5.0, max_size=0.5)
        assert s == 0.5


# ──────────────────────────────────────────────────────────────────────────
# forecast_from_detector (integration-lite)
# ──────────────────────────────────────────────────────────────────────────

class TestForecastFromDetector:
    def test_end_to_end_with_stub(self):
        import pandas as pd

        class StubDetector:
            labels = {0: "bear", 1: "bull"}

            def transition_matrix(self):
                return _two_state()

        posteriors = np.tile(np.array([0.3, 0.7]), (15, 1))
        regime_stats = pd.DataFrame({
            "state": [0, 1],
            "mean_return": [-0.01, 0.01],
            "volatility": [0.02, 0.015],
        })

        fc, frag = forecast_from_detector(
            StubDetector(), posteriors, regime_stats,
            horizon=10, fragility_horizon=5,
        )
        assert fc.horizon == 10
        assert fc.distribution.shape == (11, 2)
        assert 0.0 <= frag.index <= 1.0
        assert fc.regime_labels == ["bear", "bull"]
