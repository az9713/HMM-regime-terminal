"""
Tests for forecast.py — Analytical Forecast Engine.
"""

import numpy as np
import pytest

from forecast import ForecastEngine, ForecastResult


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def two_state():
    """Sticky two-state chain with clear bull/bear separation."""
    transmat = np.array([[0.9, 0.1], [0.1, 0.9]])
    means = np.array([[-0.01], [0.015]])                # bear, bull (log return only)
    # diag covariance layout: (n_states, n_features)
    covars = np.array([[0.0004], [0.0002]])
    labels = {0: "bear", 1: "bull"}
    return transmat, means, covars, labels


@pytest.fixture
def three_state():
    """Three-state chain: crash / neutral / bull."""
    transmat = np.array([
        [0.85, 0.12, 0.03],
        [0.05, 0.90, 0.05],
        [0.02, 0.10, 0.88],
    ])
    means = np.array([[-0.02], [0.0005], [0.012]])
    covars = np.array([[0.001], [0.0002], [0.0003]])
    labels = {0: "crash", 1: "neutral", 2: "bull"}
    return transmat, means, covars, labels


@pytest.fixture
def multi_feature_two_state():
    """Two-state chain with multiple emission features and full covariance."""
    transmat = np.array([[0.8, 0.2], [0.2, 0.8]])
    means = np.array([
        [-0.005, 0.02, 0.0],
        [0.008, 0.01, 0.0],
    ])
    covars = np.array([
        [[0.0005, 0.0001, 0.0],
         [0.0001, 0.0004, 0.0],
         [0.0,    0.0,    0.001]],
        [[0.0003, 0.00005, 0.0],
         [0.00005, 0.0002, 0.0],
         [0.0,     0.0,    0.001]],
    ])
    labels = {0: "bear", 1: "bull"}
    return transmat, means, covars, labels


@pytest.fixture
def engine():
    return ForecastEngine({"forecast": {"horizon": 20, "risk_aversion": 1.0, "max_size": 1.0}})


# ─── project_posterior ─────────────────────────────────────────────────────────


class TestProjectPosterior:
    def test_shape(self, engine, two_state):
        transmat, *_ = two_state
        traj = engine.project_posterior(np.array([0.6, 0.4]), transmat, horizon=10)
        assert traj.shape == (11, 2)

    def test_rows_are_probabilities(self, engine, three_state):
        transmat, *_ = three_state
        traj = engine.project_posterior(np.array([0.1, 0.2, 0.7]), transmat, horizon=30)
        np.testing.assert_allclose(traj.sum(axis=1), 1.0, atol=1e-10)
        assert (traj >= 0).all()

    def test_first_row_is_current(self, engine, two_state):
        transmat, *_ = two_state
        p0 = np.array([0.3, 0.7])
        traj = engine.project_posterior(p0, transmat, horizon=5)
        np.testing.assert_allclose(traj[0], p0)

    def test_converges_to_stationary(self, engine, three_state):
        """Over a long horizon, any initial belief should converge to π."""
        transmat, *_ = three_state
        # Ground-truth stationary distribution via left eigenvector
        eigvals, eigvecs = np.linalg.eig(transmat.T)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        pi = np.real(eigvecs[:, idx])
        pi = np.abs(pi) / np.abs(pi).sum()

        traj = engine.project_posterior(np.array([1.0, 0.0, 0.0]), transmat, horizon=500)
        np.testing.assert_allclose(traj[-1], pi, atol=1e-6)

    def test_chapman_kolmogorov_equivalence(self, engine, three_state):
        """p_T @ A^k should match p_T @ A @ A @ ... (k times)."""
        transmat, *_ = three_state
        p0 = np.array([0.2, 0.5, 0.3])
        traj = engine.project_posterior(p0, transmat, horizon=7)
        A_pow = np.linalg.matrix_power(transmat, 7)
        np.testing.assert_allclose(traj[7], p0 @ A_pow, atol=1e-10)

    def test_rejects_wrong_shape(self, engine, two_state):
        transmat, *_ = two_state
        with pytest.raises(ValueError):
            engine.project_posterior(np.array([0.5, 0.3, 0.2]), transmat)

    def test_rejects_zero_posterior(self, engine, two_state):
        transmat, *_ = two_state
        with pytest.raises(ValueError):
            engine.project_posterior(np.array([0.0, 0.0]), transmat)

    def test_renormalises_unnormalised_input(self, engine, two_state):
        transmat, *_ = two_state
        traj = engine.project_posterior(np.array([2.0, 3.0]), transmat, horizon=3)
        # Row 0 should be [0.4, 0.6], not the raw input
        np.testing.assert_allclose(traj[0], [0.4, 0.6])


# ─── expected_return_path / variance_path ─────────────────────────────────────


class TestReturnMoments:
    def test_expected_return_single_regime(self, engine, two_state):
        """If the chain is pinned to the bull state, E[r] == μ_bull per step."""
        transmat, means, covars, _ = two_state
        traj = engine.project_posterior(np.array([0.0, 1.0]), transmat, horizon=1)
        e = engine.expected_return_path(traj, means)
        # One step ahead, posterior is A[1] = [0.1, 0.9], so E[r] = 0.1*-0.01 + 0.9*0.015
        expected = 0.1 * -0.01 + 0.9 * 0.015
        np.testing.assert_allclose(e[0], expected)

    def test_expected_return_length_matches_horizon(self, engine, two_state):
        transmat, means, *_ = two_state
        traj = engine.project_posterior(np.array([0.5, 0.5]), transmat, horizon=15)
        e = engine.expected_return_path(traj, means)
        assert e.shape == (15,)

    def test_variance_non_negative(self, engine, three_state):
        transmat, means, covars, _ = three_state
        traj = engine.project_posterior(np.array([0.4, 0.4, 0.2]), transmat, horizon=20)
        v = engine.variance_path(traj, means, covars, covariance_type="diag")
        assert (v >= 0).all()
        assert v.shape == (20,)

    def test_law_of_total_variance_manual(self, engine, two_state):
        """Cross-check law of total variance on a hand-computed case."""
        transmat, means, covars, _ = two_state
        traj = engine.project_posterior(np.array([1.0, 0.0]), transmat, horizon=1)
        v = engine.variance_path(traj, means, covars, covariance_type="diag")

        p = traj[1]                     # [0.9, 0.1]
        mu = means[:, 0]                # [-0.01, 0.015]
        s2 = covars[:, 0]               # [0.0004, 0.0002]
        e_r = (p * mu).sum()
        e_r2 = (p * (s2 + mu * mu)).sum()
        expected_var = e_r2 - e_r * e_r
        np.testing.assert_allclose(v[0], expected_var, rtol=1e-10)

    def test_full_covariance_uses_diagonal_element(self, engine, multi_feature_two_state):
        transmat, means, covars, _ = multi_feature_two_state
        traj = engine.project_posterior(np.array([0.5, 0.5]), transmat, horizon=5)
        v = engine.variance_path(traj, means, covars, covariance_type="full", feature_idx=0)
        assert (v > 0).all()

    def test_spherical_covariance(self, engine):
        transmat = np.array([[0.7, 0.3], [0.3, 0.7]])
        means = np.array([[-0.01, 0.0], [0.01, 0.0]])
        covars = np.array([0.001, 0.0008])
        traj = ForecastEngine({}).project_posterior(np.array([0.5, 0.5]), transmat, horizon=3)
        v = ForecastEngine({}).variance_path(traj, means, covars, covariance_type="spherical")
        assert (v >= 0).all() and v.shape == (3,)


# ─── cumulative cone ───────────────────────────────────────────────────────────


class TestCumulativeCone:
    def test_cumulative_mean_is_sum(self, engine):
        e_path = np.array([0.01, -0.005, 0.003, 0.004])
        v_path = np.array([0.0001, 0.0002, 0.00015, 0.00012])
        cm, cv = engine.cumulative_return_stats(e_path, v_path)
        np.testing.assert_allclose(cm, np.cumsum(e_path))
        np.testing.assert_allclose(cv, np.cumsum(v_path))

    def test_cone_monotonic_width(self, engine, two_state):
        transmat, means, covars, _ = two_state
        traj = engine.project_posterior(np.array([0.3, 0.7]), transmat, horizon=30)
        e = engine.expected_return_path(traj, means)
        v = engine.variance_path(traj, means, covars, covariance_type="diag")
        _, cum_var = engine.cumulative_return_stats(e, v)
        # Cumulative variance should be non-decreasing
        assert np.all(np.diff(cum_var) >= -1e-12)


# ─── expected_time_in_regime ───────────────────────────────────────────────────


class TestExpectedTimeInRegime:
    def test_sums_to_horizon(self, engine, three_state):
        transmat, *_ = three_state
        traj = engine.project_posterior(np.array([0.2, 0.5, 0.3]), transmat, horizon=25)
        time_in = engine.expected_time_in_regime(traj)
        np.testing.assert_allclose(time_in.sum(), 25.0, atol=1e-10)

    def test_absorbing_state_dominates(self, engine):
        """A sticky state with high self-transition should absorb most of the time."""
        transmat = np.array([[0.99, 0.01], [0.01, 0.99]])
        traj = ForecastEngine({}).project_posterior(np.array([1.0, 0.0]), transmat, horizon=50)
        time_in = ForecastEngine({}).expected_time_in_regime(traj)
        assert time_in[0] > time_in[1]


# ─── first passage time ───────────────────────────────────────────────────────


class TestFirstPassage:
    def test_shape(self, engine, three_state):
        transmat, *_ = three_state
        fpt = engine.first_passage_time_matrix(transmat)
        assert fpt.shape == (3, 3)

    def test_return_time_matches_inverse_stationary(self, engine, three_state):
        """E[T_{i->i}] should equal 1/pi_i for an ergodic chain."""
        transmat, *_ = three_state
        fpt = engine.first_passage_time_matrix(transmat)

        eigvals, eigvecs = np.linalg.eig(transmat.T)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        pi = np.real(eigvecs[:, idx])
        pi = np.abs(pi) / np.abs(pi).sum()

        for j in range(3):
            np.testing.assert_allclose(fpt[j, j], 1.0 / pi[j], rtol=1e-6)

    def test_adjacent_passage_small(self, engine):
        """In a highly sticky 2-state chain the off-diagonal is ≈ 1/a_{ij}."""
        transmat = np.array([[0.9, 0.1], [0.2, 0.8]])
        fpt = ForecastEngine({}).first_passage_time_matrix(transmat)
        # From state 0 to 1: E[T] = 1 / 0.1 = 10
        np.testing.assert_allclose(fpt[0, 1], 10.0, rtol=1e-6)
        # From state 1 to 0: E[T] = 1 / 0.2 = 5
        np.testing.assert_allclose(fpt[1, 0], 5.0, rtol=1e-6)

    def test_expected_first_passage_from_posterior(self, engine):
        transmat = np.array([[0.9, 0.1], [0.2, 0.8]])
        fpt = ForecastEngine({}).first_passage_time_matrix(transmat)
        avg = ForecastEngine({}).expected_first_passage_from_posterior(
            np.array([0.5, 0.5]), fpt
        )
        assert avg.shape == (2,)
        # mean of diagonal return times
        assert np.all(np.isfinite(avg))


# ─── most likely path ─────────────────────────────────────────────────────────


class TestMostLikelyPath:
    def test_length(self, engine, two_state):
        transmat, *_ = two_state
        path = engine.most_likely_path(np.array([0.3, 0.7]), transmat, horizon=12)
        assert path.shape == (12,)

    def test_sticky_chain_holds(self, engine):
        """On a very sticky chain, the most likely path should stay in the starting state."""
        transmat = np.array([[0.99, 0.01], [0.01, 0.99]])
        path = ForecastEngine({}).most_likely_path(np.array([1.0, 0.0]), transmat, horizon=20)
        assert (path == 0).all()

    def test_path_values_are_valid_states(self, engine, three_state):
        transmat, *_ = three_state
        path = engine.most_likely_path(np.array([0.33, 0.34, 0.33]), transmat, horizon=40)
        assert set(np.unique(path)).issubset({0, 1, 2})


# ─── expected-utility signal ──────────────────────────────────────────────────


class TestExpectedUtilitySignal:
    def test_positive_drift_suggests_long(self, engine):
        cum_mean = np.array([0.01, 0.02, 0.03])
        cum_var = np.array([0.0001, 0.0002, 0.0003])
        action, eu, size = engine.expected_utility_signal(cum_mean, cum_var)
        assert action == 1
        assert size > 0
        assert eu[1] > eu[-1]

    def test_negative_drift_suggests_short(self, engine):
        cum_mean = np.array([-0.01, -0.02, -0.03])
        cum_var = np.array([0.0001, 0.0002, 0.0003])
        action, eu, size = engine.expected_utility_signal(cum_mean, cum_var)
        assert action == -1
        assert size > 0

    def test_high_variance_dominates_flat(self, engine):
        """When risk dwarfs reward, the best action should be flat."""
        cum_mean = np.array([0.0001])
        cum_var = np.array([10.0])
        action, eu, size = engine.expected_utility_signal(cum_mean, cum_var)
        assert action == 0
        assert size == 0.0

    def test_size_capped(self):
        eng = ForecastEngine({"forecast": {"max_size": 0.25, "risk_aversion": 0.01}})
        cum_mean = np.array([1.0])
        cum_var = np.array([0.01])
        _, _, size = eng.expected_utility_signal(cum_mean, cum_var)
        assert size <= 0.25

    def test_empty_horizon_returns_flat(self, engine):
        action, eu, size = engine.expected_utility_signal(np.array([]), np.array([]))
        assert action == 0
        assert size == 0.0


# ─── end-to-end run ────────────────────────────────────────────────────────────


class TestRun:
    def test_result_shapes(self, engine, three_state):
        transmat, means, covars, labels = three_state
        res = engine.run(
            current_posterior=np.array([0.2, 0.5, 0.3]),
            transmat=transmat,
            means=means,
            covars=covars,
            labels=labels,
            covariance_type="diag",
            horizon=15,
        )
        assert isinstance(res, ForecastResult)
        assert res.horizon == 15
        assert res.n_states == 3
        assert res.posterior_trajectory.shape == (16, 3)
        assert res.expected_return_path.shape == (15,)
        assert res.variance_path.shape == (15,)
        assert res.cumulative_mean.shape == (15,)
        assert res.cumulative_variance.shape == (15,)
        assert res.cone_lower_1.shape == (15,)
        assert res.cone_upper_2.shape == (15,)
        assert res.expected_time_in_regime.shape == (3,)
        assert res.first_passage_time.shape == (3, 3)
        assert res.most_likely_path.shape == (15,)
        assert res.eu_best_action in (-1, 0, 1)
        assert set(res.eu_values.keys()) == {-1, 0, 1}

    def test_run_respects_config_horizon(self, three_state):
        transmat, means, covars, labels = three_state
        eng = ForecastEngine({"forecast": {"horizon": 7}})
        res = eng.run(
            current_posterior=np.array([0.5, 0.3, 0.2]),
            transmat=transmat,
            means=means,
            covars=covars,
            labels=labels,
            covariance_type="diag",
        )
        assert res.horizon == 7

    def test_cone_widens_monotonically(self, engine, two_state):
        transmat, means, covars, labels = two_state
        res = engine.run(
            current_posterior=np.array([0.5, 0.5]),
            transmat=transmat,
            means=means,
            covars=covars,
            labels=labels,
            covariance_type="diag",
            horizon=25,
        )
        widths = res.cone_upper_2 - res.cone_lower_2
        assert np.all(np.diff(widths) >= -1e-12)

    def test_multi_feature_full_cov(self, engine, multi_feature_two_state):
        transmat, means, covars, labels = multi_feature_two_state
        res = engine.run(
            current_posterior=np.array([0.4, 0.6]),
            transmat=transmat,
            means=means,
            covars=covars,
            labels=labels,
            covariance_type="full",
            feature_idx=0,
            horizon=10,
        )
        assert (res.variance_path >= 0).all()
        assert np.all(np.isfinite(res.expected_return_path))
