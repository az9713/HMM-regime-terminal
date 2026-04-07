"""
Tests for regime_forecast.py — RegimeForecastEngine.

Cover analytic Markov-chain projection: posterior propagation,
stationary distribution, spectral gap / mixing time, per-state half-life,
expected first-passage times, and hit-by-horizon probabilities.
"""

import numpy as np
import pandas as pd
import pytest

from regime_forecast import RegimeForecastEngine, ForecastTrajectory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def three_state_chain():
    """A persistent 3-state chain: bear, neutral, bull."""
    T = np.array([
        [0.85, 0.10, 0.05],
        [0.10, 0.80, 0.10],
        [0.05, 0.15, 0.80],
    ])
    means = np.array([-0.003, 0.0, 0.004])
    labels = {0: "bear", 1: "neutral", 2: "bull"}
    return T, means, labels


@pytest.fixture
def crash_chain():
    """4-state chain with a crash absorbing-ish state."""
    T = np.array([
        [0.95, 0.04, 0.01, 0.00],   # crash (very persistent)
        [0.05, 0.85, 0.08, 0.02],   # bear
        [0.01, 0.10, 0.80, 0.09],   # neutral
        [0.00, 0.02, 0.18, 0.80],   # bull
    ])
    means = np.array([-0.01, -0.003, 0.0, 0.004])
    labels = {0: "crash", 1: "bear", 2: "neutral", 3: "bull"}
    return T, means, labels


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_constructor_validates_square(three_state_chain):
    T, means, labels = three_state_chain
    eng = RegimeForecastEngine(T, means, labels)
    assert eng.n_states == 3


def test_constructor_rejects_non_stochastic():
    T = np.array([[0.5, 0.4], [0.3, 0.6]])  # rows don't sum to 1
    with pytest.raises(ValueError):
        RegimeForecastEngine(T, np.zeros(2), {0: "a", 1: "b"})


def test_constructor_rejects_non_square():
    T = np.array([[0.5, 0.5, 0.0], [0.2, 0.5, 0.3]])
    with pytest.raises(ValueError):
        RegimeForecastEngine(T, np.zeros(2), {0: "a", 1: "b"})


def test_constructor_rejects_mean_mismatch(three_state_chain):
    T, _, labels = three_state_chain
    with pytest.raises(ValueError):
        RegimeForecastEngine(T, np.zeros(2), labels)


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


def test_propagate_horizon_zero_returns_pi0(three_state_chain):
    T, means, labels = three_state_chain
    eng = RegimeForecastEngine(T, means, labels)
    pi0 = np.array([0.0, 1.0, 0.0])
    traj = eng.propagate(pi0, 0)
    assert traj.distributions.shape == (1, 3)
    np.testing.assert_allclose(traj.distributions[0], pi0)


def test_propagate_one_step_matches_matrix_multiply(three_state_chain):
    T, means, labels = three_state_chain
    eng = RegimeForecastEngine(T, means, labels)
    pi0 = np.array([0.2, 0.3, 0.5])
    traj = eng.propagate(pi0, 1)
    np.testing.assert_allclose(traj.distributions[1], pi0 @ T)


def test_propagate_rows_sum_to_one(crash_chain):
    T, means, labels = crash_chain
    eng = RegimeForecastEngine(T, means, labels)
    pi0 = np.array([0.0, 0.0, 0.0, 1.0])
    traj = eng.propagate(pi0, 50)
    sums = traj.distributions.sum(axis=1)
    np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-10)


def test_propagate_converges_to_stationary(three_state_chain):
    T, means, labels = three_state_chain
    eng = RegimeForecastEngine(T, means, labels)
    pi0 = np.array([1.0, 0.0, 0.0])
    traj = eng.propagate(pi0, 500)
    stationary = eng.stationary_distribution()
    np.testing.assert_allclose(traj.distributions[-1], stationary, atol=1e-6)


def test_propagate_entropy_increases_then_plateaus(crash_chain):
    T, means, labels = crash_chain
    eng = RegimeForecastEngine(T, means, labels)
    pi0 = np.array([0.0, 0.0, 0.0, 1.0])  # certain start
    traj = eng.propagate(pi0, 100)
    # Entropy at k=0 is 0 (certain), should rise.
    assert traj.entropies[0] == pytest.approx(0.0, abs=1e-9)
    assert traj.entropies[10] > traj.entropies[0]
    # And eventually stabilize: |H[100] - H[80]| small
    assert abs(traj.entropies[100] - traj.entropies[80]) < 0.05


def test_propagate_expected_returns_match_dist_dot_means(three_state_chain):
    T, means, labels = three_state_chain
    eng = RegimeForecastEngine(T, means, labels)
    pi0 = np.array([0.1, 0.2, 0.7])
    traj = eng.propagate(pi0, 20)
    for k in range(21):
        expected = float(traj.distributions[k] @ means)
        assert traj.expected_returns[k] == pytest.approx(expected, rel=1e-9)


def test_propagate_normalizes_pi0(three_state_chain):
    T, means, labels = three_state_chain
    eng = RegimeForecastEngine(T, means, labels)
    pi0 = np.array([2.0, 4.0, 4.0])  # not normalized, sums to 10
    traj = eng.propagate(pi0, 1)
    np.testing.assert_allclose(traj.distributions[0].sum(), 1.0)


def test_propagate_rejects_negative_horizon(three_state_chain):
    T, means, labels = three_state_chain
    eng = RegimeForecastEngine(T, means, labels)
    with pytest.raises(ValueError):
        eng.propagate(np.array([1.0, 0.0, 0.0]), -1)


# ---------------------------------------------------------------------------
# Stationary / spectral
# ---------------------------------------------------------------------------


def test_stationary_distribution_is_left_eigvec(three_state_chain):
    T, means, labels = three_state_chain
    eng = RegimeForecastEngine(T, means, labels)
    pi = eng.stationary_distribution()
    np.testing.assert_allclose(pi @ T, pi, atol=1e-10)
    assert pi.sum() == pytest.approx(1.0)
    assert (pi >= 0).all()


def test_spectral_gap_in_unit_interval(three_state_chain):
    T, means, labels = three_state_chain
    eng = RegimeForecastEngine(T, means, labels)
    gap = eng.spectral_gap()
    assert 0.0 < gap <= 1.0


def test_mixing_time_finite_for_ergodic(three_state_chain):
    T, means, labels = three_state_chain
    eng = RegimeForecastEngine(T, means, labels)
    t_mix = eng.mixing_time(epsilon=0.01)
    assert np.isfinite(t_mix)
    assert t_mix > 0


def test_mixing_time_more_persistent_chain_is_slower(three_state_chain, crash_chain):
    T1, m1, l1 = three_state_chain
    T2, m2, l2 = crash_chain
    eng1 = RegimeForecastEngine(T1, m1, l1)
    eng2 = RegimeForecastEngine(T2, m2, l2)
    # Crash chain has 0.95 self-loop -> stickier -> slower mix
    assert eng2.mixing_time() > eng1.mixing_time()


# ---------------------------------------------------------------------------
# Half-life
# ---------------------------------------------------------------------------


def test_regime_half_life_formula(three_state_chain):
    T, means, labels = three_state_chain
    eng = RegimeForecastEngine(T, means, labels)
    df = eng.regime_half_life()
    assert set(df.columns) >= {
        "state", "label", "self_loop_prob", "expected_duration", "half_life"
    }
    # bear: 0.85 self-loop -> half_life = ln(0.5)/ln(0.85)
    bear_row = df[df["label"] == "bear"].iloc[0]
    expected_hl = np.log(0.5) / np.log(0.85)
    assert bear_row["half_life"] == pytest.approx(expected_hl)
    # Expected duration 1/(1-0.85) = 6.666...
    assert bear_row["expected_duration"] == pytest.approx(1.0 / 0.15)


def test_regime_half_life_handles_absorbing_state():
    T = np.array([
        [1.0, 0.0],   # absorbing
        [0.1, 0.9],
    ])
    eng = RegimeForecastEngine(T, np.array([0.0, 0.0]), {0: "absorb", 1: "live"})
    df = eng.regime_half_life()
    absorb = df[df["label"] == "absorb"].iloc[0]
    assert np.isinf(absorb["half_life"])
    assert np.isinf(absorb["expected_duration"])


# ---------------------------------------------------------------------------
# Hitting times
# ---------------------------------------------------------------------------


def test_expected_hitting_times_target_is_zero(crash_chain):
    T, means, labels = crash_chain
    eng = RegimeForecastEngine(T, means, labels)
    h = eng.expected_hitting_times([0])  # target = crash
    assert h[0] == 0.0


def test_expected_hitting_times_monotone_distance(crash_chain):
    """States farther from crash should have larger expected hitting time."""
    T, means, labels = crash_chain
    eng = RegimeForecastEngine(T, means, labels)
    h = eng.expected_hitting_times([0])
    # bull (3) is farther from crash than bear (1)
    assert h[3] > h[1] > h[0]


def test_expected_hitting_times_finite_when_reachable(crash_chain):
    T, means, labels = crash_chain
    eng = RegimeForecastEngine(T, means, labels)
    h = eng.expected_hitting_times([0])
    for v in h.values():
        assert np.isfinite(v)


def test_expected_hitting_times_unreachable_is_inf():
    # Two disconnected components: {0} and {1,2}
    T = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.0, 0.5, 0.5],
    ])
    eng = RegimeForecastEngine(T, np.zeros(3), {0: "iso", 1: "a", 2: "b"})
    h = eng.expected_hitting_times([0])
    assert np.isinf(h[1])
    assert np.isinf(h[2])


def test_expected_hitting_times_rejects_bad_target(three_state_chain):
    T, means, labels = three_state_chain
    eng = RegimeForecastEngine(T, means, labels)
    with pytest.raises(ValueError):
        eng.expected_hitting_times([5])


# ---------------------------------------------------------------------------
# Hit probability curves
# ---------------------------------------------------------------------------


def test_hit_probability_curve_monotone_in_horizon(crash_chain):
    T, means, labels = crash_chain
    eng = RegimeForecastEngine(T, means, labels)
    df = eng.hit_probability_by_horizon([0], horizon=50)
    # Each non-target column must be non-decreasing in horizon
    for col in df.columns:
        if col == "horizon":
            continue
        vals = df[col].values
        diffs = np.diff(vals)
        assert (diffs >= -1e-12).all(), f"{col} not monotone"


def test_hit_probability_curve_in_unit_interval(crash_chain):
    T, means, labels = crash_chain
    eng = RegimeForecastEngine(T, means, labels)
    df = eng.hit_probability_by_horizon([0], horizon=20)
    for col in df.columns:
        if col == "horizon":
            continue
        vals = df[col].values
        assert (vals >= -1e-12).all() and (vals <= 1 + 1e-12).all()


def test_hit_probability_curve_excludes_target(crash_chain):
    T, means, labels = crash_chain
    eng = RegimeForecastEngine(T, means, labels)
    df = eng.hit_probability_by_horizon([0], horizon=5)
    assert "crash" not in df.columns
    assert "bear" in df.columns


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def test_regime_probabilities_at(three_state_chain):
    T, means, labels = three_state_chain
    eng = RegimeForecastEngine(T, means, labels)
    pi0 = np.array([1.0, 0.0, 0.0])
    df = eng.regime_probabilities_at(pi0, [0, 1, 5, 20])
    assert list(df["horizon"]) == [0, 1, 5, 20]
    # k=0 row: bear=1, others 0
    row0 = df.iloc[0]
    assert row0["bear"] == pytest.approx(1.0)
    assert row0["bull"] == pytest.approx(0.0)
    assert "expected_log_return" in df.columns


def test_forecast_summary_complete(crash_chain):
    T, means, labels = crash_chain
    eng = RegimeForecastEngine(T, means, labels)
    pi0 = np.array([0.0, 0.0, 0.0, 1.0])  # start in bull
    summary = eng.forecast_summary(pi0, target_label="crash", horizon=72)
    assert isinstance(summary["trajectory"], ForecastTrajectory)
    assert summary["target_states"] == [0]
    assert np.isfinite(summary["expected_bars_to_target"])
    assert summary["expected_bars_to_target"] > 0
    assert isinstance(summary["half_lives"], pd.DataFrame)
    assert isinstance(summary["hit_probability_curve"], pd.DataFrame)
    assert summary["spectral_gap"] > 0
    assert summary["mixing_time"] > 0


def test_forecast_summary_no_matching_target(three_state_chain):
    T, means, labels = three_state_chain
    eng = RegimeForecastEngine(T, means, labels)
    pi0 = np.array([0.5, 0.3, 0.2])
    summary = eng.forecast_summary(pi0, target_label="crash", horizon=10)
    assert summary["target_states"] == []
    assert np.isnan(summary["expected_bars_to_target"])
    assert summary["hit_probability_curve"].empty
