"""
Tests for regime_forecast.py — Bayesian Regime Forecast Engine.

Validates:
  - Forward posterior propagation (π · T^k)
  - Soft-assignment fitting of per-regime return distributions
  - Closed-form marginal forecast moments and quantiles
  - Monte Carlo cumulative forecast consistency with closed-form mean
  - Calibration validation on synthetic data drawn from a known HMM
  - Shape contracts, edge cases, and dataclass integrity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from regime_forecast import (
    CalibrationResult,
    ForecastResult,
    RegimeForecaster,
    RegimeReturnParams,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_config():
    return {
        "forecast": {
            "horizons": [1, 3, 5, 10, 21],
            "n_paths": 2000,
            "seed": 7,
            "calibration_horizons": [1, 5],
            "calibration_min_history": 50,
        }
    }


@pytest.fixture
def two_state_transmat():
    """Persistent 2-state chain."""
    return np.array([[0.9, 0.1], [0.1, 0.9]])


@pytest.fixture
def two_state_params():
    """Known per-regime return parameters (log returns)."""
    mus = np.array([-0.002, 0.003])         # bear, bull
    sigmas = np.array([0.02, 0.012])
    return mus, sigmas


def _synthetic_hmm_data(
    transmat: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    T: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate (posteriors, returns, states) from a known HMM.
    Posteriors are constructed as one-hot on the true state (perfect inference)
    to isolate the forecaster math from HMM estimation noise.
    """
    rng = np.random.default_rng(seed)
    n_states = transmat.shape[0]
    stationary = np.ones(n_states) / n_states  # uniform initial

    states = np.empty(T, dtype=int)
    states[0] = rng.choice(n_states, p=stationary)
    for t in range(1, T):
        states[t] = rng.choice(n_states, p=transmat[states[t - 1]])

    returns = rng.normal(loc=mus[states], scale=sigmas[states])

    posteriors = np.zeros((T, n_states))
    posteriors[np.arange(T), states] = 1.0
    return posteriors, returns, states


@pytest.fixture
def synthetic_hmm(two_state_transmat, two_state_params):
    mus, sigmas = two_state_params
    posteriors, returns, states = _synthetic_hmm_data(
        two_state_transmat, mus, sigmas, T=5000, seed=123
    )
    return {
        "transmat": two_state_transmat,
        "posteriors": posteriors,
        "returns": returns,
        "states": states,
        "true_mus": mus,
        "true_sigmas": sigmas,
    }


@pytest.fixture
def fitted_forecaster(simple_config, synthetic_hmm):
    fc = RegimeForecaster(simple_config)
    fc.fit(
        transmat=synthetic_hmm["transmat"],
        posteriors=synthetic_hmm["posteriors"],
        returns=synthetic_hmm["returns"],
        labels={0: "bear", 1: "bull"},
    )
    return fc


# ─────────────────────────────────────────────────────────────────────────────
# Fitting
# ─────────────────────────────────────────────────────────────────────────────


class TestFit:
    def test_fit_recovers_true_params(self, fitted_forecaster, synthetic_hmm):
        fc = fitted_forecaster
        params = fc.regime_params
        assert params is not None
        # Hard-posterior assignment + large sample → accurate recovery
        np.testing.assert_allclose(
            params.means, synthetic_hmm["true_mus"], atol=5e-4
        )
        np.testing.assert_allclose(
            params.stds, synthetic_hmm["true_sigmas"], rtol=0.05
        )

    def test_fit_sets_labels(self, fitted_forecaster):
        assert fitted_forecaster.labels == {0: "bear", 1: "bull"}

    def test_fit_default_labels(self, simple_config, synthetic_hmm):
        fc = RegimeForecaster(simple_config)
        fc.fit(
            synthetic_hmm["transmat"],
            synthetic_hmm["posteriors"],
            synthetic_hmm["returns"],
        )
        assert fc.labels == {0: "state_0", 1: "state_1"}

    def test_fit_row_normalizes_transmat(self, simple_config, synthetic_hmm):
        bad_T = np.array([[9.0, 1.0], [1.0, 9.0]])  # rows sum to 10
        fc = RegimeForecaster(simple_config)
        fc.fit(bad_T, synthetic_hmm["posteriors"], synthetic_hmm["returns"])
        assert np.allclose(fc.transmat.sum(axis=1), 1.0)

    def test_fit_validates_shapes(self, simple_config, synthetic_hmm):
        fc = RegimeForecaster(simple_config)
        # mismatched posteriors/transmat
        with pytest.raises(ValueError):
            fc.fit(
                np.eye(3),
                synthetic_hmm["posteriors"],  # 2 states
                synthetic_hmm["returns"],
            )
        # non-square transmat
        with pytest.raises(ValueError):
            fc.fit(
                np.zeros((2, 3)),
                synthetic_hmm["posteriors"],
                synthetic_hmm["returns"],
            )
        # mismatched returns length
        with pytest.raises(ValueError):
            fc.fit(
                synthetic_hmm["transmat"],
                synthetic_hmm["posteriors"],
                synthetic_hmm["returns"][:-5],
            )

    def test_fit_handles_nan_returns(self, simple_config, synthetic_hmm):
        returns = synthetic_hmm["returns"].copy()
        returns[0:10] = np.nan
        fc = RegimeForecaster(simple_config)
        fc.fit(synthetic_hmm["transmat"], synthetic_hmm["posteriors"], returns)
        assert np.all(np.isfinite(fc.regime_params.means))
        assert np.all(np.isfinite(fc.regime_params.stds))


# ─────────────────────────────────────────────────────────────────────────────
# Forward propagation
# ─────────────────────────────────────────────────────────────────────────────


class TestPropagation:
    def test_zero_horizon_identity(self, fitted_forecaster):
        pi = np.array([0.3, 0.7])
        fwd = fitted_forecaster.propagate_posterior(pi, [0])
        np.testing.assert_allclose(fwd[0], pi)

    def test_one_step_matches_T(self, fitted_forecaster, two_state_transmat):
        pi = np.array([0.3, 0.7])
        fwd = fitted_forecaster.propagate_posterior(pi, [1])
        np.testing.assert_allclose(fwd[0], pi @ two_state_transmat)

    def test_k_step_matches_Tk(self, fitted_forecaster, two_state_transmat):
        pi = np.array([0.8, 0.2])
        k = 5
        expected = pi @ np.linalg.matrix_power(two_state_transmat, k)
        fwd = fitted_forecaster.propagate_posterior(pi, [k])
        np.testing.assert_allclose(fwd[0], expected, atol=1e-12)

    def test_converges_to_stationary(self, fitted_forecaster, two_state_transmat):
        pi = np.array([1.0, 0.0])
        fwd = fitted_forecaster.propagate_posterior(pi, [200])
        # Symmetric chain → uniform stationary
        np.testing.assert_allclose(fwd[0], [0.5, 0.5], atol=1e-4)

    def test_multiple_horizons_shape(self, fitted_forecaster):
        pi = np.array([0.5, 0.5])
        fwd = fitted_forecaster.propagate_posterior(pi, [1, 3, 10])
        assert fwd.shape == (3, 2)

    def test_normalization(self, fitted_forecaster):
        pi = np.array([0.4, 0.6])
        fwd = fitted_forecaster.propagate_posterior(pi, [1, 5, 20])
        np.testing.assert_allclose(fwd.sum(axis=1), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Marginal forecast
# ─────────────────────────────────────────────────────────────────────────────


class TestMarginalForecast:
    def test_one_hot_posterior_matches_regime(self, fitted_forecaster):
        # If current posterior is entirely in bull, one-step ahead mean should
        # be close to bull mean since bull is persistent (0.9 stay).
        pi_bull = np.array([0.0, 1.0])
        mu_bull = fitted_forecaster.regime_params.means[1]
        mu_bear = fitted_forecaster.regime_params.means[0]
        res = fitted_forecaster.forecast_marginal(pi_bull, horizons=[1])
        # One-step forward posterior = [0.1, 0.9]
        expected_mean = 0.1 * mu_bear + 0.9 * mu_bull
        assert abs(res["mean"][0] - expected_mean) < 1e-10

    def test_mean_matches_mixture_formula(self, fitted_forecaster):
        pi = np.array([0.4, 0.6])
        res = fitted_forecaster.forecast_marginal(pi, horizons=[1, 5, 21])
        mus = fitted_forecaster.regime_params.means
        # Manually propagate and compute weighted means
        T_mat = fitted_forecaster.transmat
        for i, h in enumerate([1, 5, 21]):
            fwd = pi @ np.linalg.matrix_power(T_mat, h)
            expected = float(np.sum(fwd * mus))
            assert abs(res["mean"][i] - expected) < 1e-12

    def test_variance_via_law_of_total_variance(self, fitted_forecaster):
        pi = np.array([0.3, 0.7])
        res = fitted_forecaster.forecast_marginal(pi, horizons=[1])
        mus = fitted_forecaster.regime_params.means
        sigmas = fitted_forecaster.regime_params.stds
        T_mat = fitted_forecaster.transmat
        fwd = pi @ T_mat
        expected_mean = float(np.sum(fwd * mus))
        expected_var = float(np.sum(fwd * (sigmas ** 2 + mus ** 2))) - expected_mean ** 2
        assert abs(res["std"][0] - np.sqrt(expected_var)) < 1e-12

    def test_p_up_in_unit_interval(self, fitted_forecaster):
        pi = np.array([0.5, 0.5])
        res = fitted_forecaster.forecast_marginal(pi, horizons=[1, 5, 10, 21])
        assert np.all(res["p_up"] >= 0.0)
        assert np.all(res["p_up"] <= 1.0)

    def test_quantiles_monotone(self, fitted_forecaster):
        pi = np.array([0.5, 0.5])
        res = fitted_forecaster.forecast_marginal(pi, horizons=[1, 5])
        for i in range(len(res["horizons"])):
            assert res["q05"][i] < res["q16"][i] < res["q84"][i] < res["q95"][i]

    def test_68_band_matches_std_for_single_regime(self, simple_config):
        """When forward posterior is concentrated on one regime, the 16/84
        band should equal mu ± sigma."""
        fc = RegimeForecaster(simple_config)
        # Absorbing bull state: once in bull, stays in bull.
        T = np.array([[0.5, 0.5], [0.0, 1.0]])
        # Construct fake posteriors/returns that give clean params
        post = np.zeros((200, 2))
        post[:100, 0] = 1.0
        post[100:, 1] = 1.0
        rng = np.random.default_rng(0)
        r = np.concatenate([
            rng.normal(-0.01, 0.02, 100),
            rng.normal(0.01, 0.015, 100),
        ])
        fc.fit(T, post, r)
        # Start in bull → next step stays in bull deterministically
        pi = np.array([0.0, 1.0])
        res = fc.forecast_marginal(pi, horizons=[1])
        mu = fc.regime_params.means[1]
        sd = fc.regime_params.stds[1]
        assert abs(res["q16"][0] - (mu - sd)) < 1e-3
        assert abs(res["q84"][0] - (mu + sd)) < 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Cumulative Monte Carlo forecast
# ─────────────────────────────────────────────────────────────────────────────


class TestCumulativeForecast:
    def test_shapes(self, fitted_forecaster):
        pi = np.array([0.5, 0.5])
        res = fitted_forecaster.forecast_cumulative(pi, horizons=[1, 5, 10])
        assert res["mean_log"].shape == (3,)
        assert res["mean_simple"].shape == (3,)
        assert res["sim_cum_log"].shape[1] == 10
        assert res["sim_cum_simple"].shape[1] == 10
        assert res["sim_regime_paths"].shape[1] == 10

    def test_mc_mean_matches_closed_form(self, fitted_forecaster):
        """MC cumulative mean at horizon k should approximate the sum of
        closed-form marginal means over 1..k."""
        pi = np.array([0.4, 0.6])
        horizons = [1, 3, 5, 10]
        marg = fitted_forecaster.forecast_marginal(
            pi, horizons=list(range(1, max(horizons) + 1))
        )
        cum = fitted_forecaster.forecast_cumulative(
            pi, horizons=horizons, n_paths=10000, seed=11
        )
        for i, h in enumerate(horizons):
            cf_mean = float(marg["mean"][:h].sum())
            assert abs(cum["mean_log"][i] - cf_mean) < 0.003

    def test_std_grows_with_horizon(self, fitted_forecaster):
        pi = np.array([0.5, 0.5])
        res = fitted_forecaster.forecast_cumulative(
            pi, horizons=[1, 5, 10, 21], n_paths=5000
        )
        stds = res["std_log"]
        for i in range(1, len(stds)):
            assert stds[i] > stds[i - 1]

    def test_p_up_in_unit_interval(self, fitted_forecaster):
        pi = np.array([0.3, 0.7])
        res = fitted_forecaster.forecast_cumulative(pi, horizons=[1, 5, 21])
        assert np.all(res["p_up"] >= 0.0)
        assert np.all(res["p_up"] <= 1.0)

    def test_quantile_ordering(self, fitted_forecaster):
        pi = np.array([0.5, 0.5])
        res = fitted_forecaster.forecast_cumulative(pi, horizons=[5])
        assert res["q05"][0] < res["q16"][0] < res["q50"][0]
        assert res["q50"][0] < res["q84"][0] < res["q95"][0]

    def test_regime_paths_valid_states(self, fitted_forecaster):
        pi = np.array([0.5, 0.5])
        res = fitted_forecaster.forecast_cumulative(
            pi, horizons=[5], n_paths=500
        )
        assert set(np.unique(res["sim_regime_paths"])).issubset({0, 1})

    def test_reproducibility_with_seed(self, fitted_forecaster):
        pi = np.array([0.5, 0.5])
        a = fitted_forecaster.forecast_cumulative(
            pi, horizons=[5], n_paths=500, seed=42
        )
        b = fitted_forecaster.forecast_cumulative(
            pi, horizons=[5], n_paths=500, seed=42
        )
        np.testing.assert_allclose(a["mean_log"], b["mean_log"])
        np.testing.assert_allclose(a["q95"], b["q95"])


# ─────────────────────────────────────────────────────────────────────────────
# Full forecast convenience
# ─────────────────────────────────────────────────────────────────────────────


class TestFullForecast:
    def test_returns_forecast_result(self, fitted_forecaster):
        res = fitted_forecaster.forecast(np.array([0.5, 0.5]), horizons=[1, 5])
        assert isinstance(res, ForecastResult)
        assert len(res.horizons) == 2
        assert res.forward_posteriors.shape == (2, 2)
        assert res.marginal_mean.shape == (2,)
        assert res.cumulative_mean_simple.shape == (2,)

    def test_include_paths_false(self, fitted_forecaster):
        res = fitted_forecaster.forecast(
            np.array([0.5, 0.5]), horizons=[1], include_paths=False
        )
        assert res.sim_cum_simple is None
        assert res.sim_regime_paths is None

    def test_summary_table_columns(self, fitted_forecaster):
        res = fitted_forecaster.forecast(np.array([0.5, 0.5]), horizons=[1, 5])
        df = fitted_forecaster.summary_table(res)
        assert isinstance(df, pd.DataFrame)
        for col in ["horizon_bars", "cum_return_mean", "cum_return_p_up",
                    "cum_return_q05", "cum_return_q95", "bar_return_mean",
                    "bar_return_std", "bar_return_p_up"]:
            assert col in df.columns
        assert len(df) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Posterior trajectory
# ─────────────────────────────────────────────────────────────────────────────


class TestPosteriorTrajectory:
    def test_shape_and_row_zero(self, fitted_forecaster):
        pi = np.array([0.2, 0.8])
        traj = fitted_forecaster.posterior_trajectory(pi, max_horizon=10)
        assert traj.shape == (11, 2)
        np.testing.assert_allclose(traj[0], pi)

    def test_convergence_in_trajectory(self, fitted_forecaster):
        pi = np.array([1.0, 0.0])
        traj = fitted_forecaster.posterior_trajectory(pi, max_horizon=300)
        np.testing.assert_allclose(traj[-1], [0.5, 0.5], atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────


class TestCalibration:
    def test_well_specified_coverage(self, fitted_forecaster, synthetic_hmm):
        """On data drawn from the same HMM used to fit the forecaster,
        empirical coverage should be close to nominal quantile levels."""
        results = fitted_forecaster.calibrate(
            posteriors=synthetic_hmm["posteriors"],
            returns=synthetic_hmm["returns"],
            horizons=[1, 5],
            nominal_levels=[0.05, 0.5, 0.95],
            min_history=500,
            stride=1,
        )
        assert len(results) == 2
        for r in results:
            # Median coverage close to 50%
            median_idx = 1
            assert abs(r.empirical_coverage[median_idx] - 0.5) < 0.1
            # 95% quantile covers ~95%
            assert abs(r.empirical_coverage[2] - 0.95) < 0.05
            assert abs(r.empirical_coverage[0] - 0.05) < 0.05
            assert r.n_observations > 0
            assert r.crps >= 0.0

    def test_calibration_result_fields(self, fitted_forecaster, synthetic_hmm):
        results = fitted_forecaster.calibrate(
            posteriors=synthetic_hmm["posteriors"],
            returns=synthetic_hmm["returns"],
            horizons=[1],
            min_history=500,
            stride=5,  # stride to save time
        )
        r = results[0]
        assert isinstance(r, CalibrationResult)
        assert r.horizon == 1
        assert r.empirical_coverage.shape == r.nominal_levels.shape
        assert isinstance(r.crps, float)
        assert isinstance(r.mae, float)
        assert isinstance(r.mean_bias, float)

    def test_calibration_requires_fit(self, simple_config):
        fc = RegimeForecaster(simple_config)
        with pytest.raises(RuntimeError):
            fc.calibrate(np.zeros((100, 2)), np.zeros(100))

    def test_insufficient_history_returns_empty(
        self, fitted_forecaster, synthetic_hmm
    ):
        # Set min_history larger than available data
        results = fitted_forecaster.calibrate(
            posteriors=synthetic_hmm["posteriors"][:10],
            returns=synthetic_hmm["returns"][:10],
            horizons=[1],
            min_history=100,
        )
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# Error paths
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorPaths:
    def test_forecast_before_fit(self, simple_config):
        fc = RegimeForecaster(simple_config)
        with pytest.raises(RuntimeError):
            fc.forecast_marginal(np.array([0.5, 0.5]))
        with pytest.raises(RuntimeError):
            fc.forecast_cumulative(np.array([0.5, 0.5]))
        with pytest.raises(RuntimeError):
            fc.propagate_posterior(np.array([0.5, 0.5]), [1])

    def test_negative_horizons_rejected(self, fitted_forecaster):
        with pytest.raises(ValueError):
            fitted_forecaster.propagate_posterior(np.array([0.5, 0.5]), [-1])
