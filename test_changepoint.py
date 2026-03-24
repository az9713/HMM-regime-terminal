"""Tests for changepoint.py — Bayesian Online Changepoint Detection."""

import numpy as np
import pandas as pd
import pytest

from changepoint import (
    BOCDEngine,
    BOCDResult,
    StudentTPredictor,
    MultivariatePredictor,
)


@pytest.fixture
def default_config():
    return {
        "changepoint": {
            "hazard_lambda": 100,
            "threshold": 0.3,
            "prior_mu": 0.0,
            "prior_kappa": 1.0,
            "prior_alpha": 1.0,
            "prior_beta": 1.0,
            "min_run_length": 5,
        }
    }


@pytest.fixture
def regime_switch_series():
    """Generate a time series with a clear regime switch at bar 100."""
    rng = np.random.default_rng(42)
    n = 200
    # Regime 1: N(0, 0.01) for bars 0-99
    segment1 = rng.normal(0.0, 0.01, 100)
    # Regime 2: N(0.05, 0.03) for bars 100-199 — big mean/vol shift
    segment2 = rng.normal(0.05, 0.03, 100)
    return np.concatenate([segment1, segment2])


@pytest.fixture
def multi_regime_series():
    """Generate a series with 3 distinct regimes."""
    rng = np.random.default_rng(123)
    seg1 = rng.normal(0.0, 0.01, 80)    # calm
    seg2 = rng.normal(-0.03, 0.05, 60)  # crash
    seg3 = rng.normal(0.02, 0.015, 80)  # recovery
    return np.concatenate([seg1, seg2, seg3])


# ── StudentTPredictor tests ─────────────────────────────────────────────────


class TestStudentTPredictor:
    def test_predictive_logprob_returns_finite(self):
        pred = StudentTPredictor()
        mu = np.array([0.0, 0.0])
        kappa = np.array([1.0, 2.0])
        alpha = np.array([1.0, 1.5])
        beta = np.array([1.0, 1.0])
        logp = pred.predictive_logprob(0.01, mu, kappa, alpha, beta)
        assert logp.shape == (2,)
        assert np.all(np.isfinite(logp))
        assert np.all(logp <= 0)  # log probabilities are non-positive

    def test_logprob_higher_at_mean(self):
        pred = StudentTPredictor(mu0=0.0)
        mu = np.array([0.0])
        kappa = np.array([10.0])
        alpha = np.array([5.0])
        beta = np.array([1.0])
        logp_at_mean = pred.predictive_logprob(0.0, mu, kappa, alpha, beta)
        logp_far = pred.predictive_logprob(5.0, mu, kappa, alpha, beta)
        assert logp_at_mean[0] > logp_far[0]

    def test_update_sufficient_stats(self):
        pred = StudentTPredictor()
        mu = np.array([0.0])
        kappa = np.array([1.0])
        alpha = np.array([1.0])
        beta = np.array([1.0])
        new_mu, new_kappa, new_alpha, new_beta = pred.update_sufficient_stats(
            0.5, mu, kappa, alpha, beta
        )
        assert new_kappa[0] == 2.0
        assert new_alpha[0] == 1.5
        assert np.isclose(new_mu[0], 0.25)  # (1*0 + 0.5) / 2
        assert new_beta[0] > beta[0]  # beta always grows


# ── MultivariatePredictor tests ─────────────────────────────────────────────


class TestMultivariatePredictor:
    def test_predictive_logprob_multivariate(self):
        pred = MultivariatePredictor(n_features=3)
        stats = [
            (np.array([0.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]))
            for _ in range(3)
        ]
        x = np.array([0.01, -0.02, 0.005])
        logp = pred.predictive_logprob(x, stats)
        assert logp.shape == (1,)
        assert np.isfinite(logp[0])

    def test_update_stats_multivariate(self):
        pred = MultivariatePredictor(n_features=2)
        stats = [
            (np.array([0.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]))
            for _ in range(2)
        ]
        x = np.array([0.5, -0.3])
        new_stats = pred.update_sufficient_stats(x, stats)
        assert len(new_stats) == 2
        for dim_stats in new_stats:
            assert len(dim_stats) == 4
            assert dim_stats[1][0] == 2.0  # kappa updated


# ── BOCDEngine tests ────────────────────────────────────────────────────────


class TestBOCDEngine:
    def test_detect_returns_correct_shape(self, default_config):
        engine = BOCDEngine(default_config)
        X = np.random.default_rng(0).normal(0, 1, 50)
        result = engine.detect(X)

        assert isinstance(result, BOCDResult)
        assert result.changepoint_prob.shape == (50,)
        assert result.map_run_length.shape == (50,)
        assert result.growth_prob.shape == (50,)
        assert np.all(result.changepoint_prob >= 0)
        assert np.all(result.changepoint_prob <= 1)

    def test_detect_finds_obvious_changepoint(self, default_config, regime_switch_series):
        engine = BOCDEngine(default_config)
        result = engine.detect(regime_switch_series)

        # Should detect a changepoint near bar 100
        if len(result.changepoints) > 0:
            closest = result.changepoints[
                np.argmin(np.abs(result.changepoints - 100))
            ]
            assert abs(closest - 100) < 20, (
                f"Expected changepoint near bar 100, closest was {closest}"
            )

        # Changepoint probability should spike around bar 100
        window = result.changepoint_prob[90:120]
        assert np.max(window) > 0.1, "Expected elevated cp probability near regime switch"

    def test_detect_multivariate(self, default_config):
        rng = np.random.default_rng(42)
        X = np.column_stack([
            rng.normal(0, 0.01, 100),
            rng.normal(0, 0.5, 100),
        ])
        engine = BOCDEngine(default_config)
        result = engine.detect(X)
        assert result.changepoint_prob.shape == (100,)

    def test_detect_multi_regime(self, default_config, multi_regime_series):
        engine = BOCDEngine(default_config)
        result = engine.detect(multi_regime_series)

        # Should detect changepoints near bars 80 and 140
        assert len(result.changepoints) >= 1, "Should detect at least one changepoint"

    def test_constant_series_no_changepoints(self, default_config):
        """A perfectly constant series should have low changepoint probability."""
        engine = BOCDEngine(default_config)
        X = np.ones(100) * 0.01
        result = engine.detect(X)
        # After initial transient, cp probability should be very low
        assert np.mean(result.changepoint_prob[20:]) < 0.1

    def test_hazard_function(self, default_config):
        engine = BOCDEngine(default_config)
        r = np.array([0, 10, 50, 100])
        h = engine.hazard(r)
        assert np.allclose(h, 1.0 / 100)

    def test_store_posterior(self, default_config):
        default_config["changepoint"]["store_posterior"] = True
        engine = BOCDEngine(default_config)
        X = np.random.default_rng(0).normal(0, 1, 30)
        result = engine.detect(X)
        assert result.run_length_posterior is not None
        assert len(result.run_length_posterior) == 30
        # Each entry should be a valid probability distribution
        for rl_dist in result.run_length_posterior:
            assert np.isclose(rl_dist.sum(), 1.0, atol=1e-6)

    def test_detect_on_features(self, default_config):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "log_return": rng.normal(0, 0.01, 100),
            "rolling_vol": rng.exponential(0.01, 100),
        })
        engine = BOCDEngine(default_config)
        result = engine.detect_on_features(df, feature_cols=["log_return"])
        assert result.changepoint_prob.shape == (100,)

    def test_detect_on_features_missing_col(self, default_config):
        df = pd.DataFrame({"close": np.ones(10)})
        engine = BOCDEngine(default_config)
        with pytest.raises(ValueError, match="Missing feature columns"):
            engine.detect_on_features(df, feature_cols=["log_return"])

    def test_changepoint_confirmation(self, default_config, regime_switch_series):
        engine = BOCDEngine(default_config)
        result = engine.detect(regime_switch_series)
        confirmed = engine.changepoint_confirmation(result, window=10)
        assert confirmed.shape == (200,)
        assert np.all(confirmed >= 0)
        assert np.all(confirmed <= 1)
        # Confirmed should be >= raw cp prob (it's a rolling max)
        assert np.all(confirmed >= result.changepoint_prob - 1e-10)

    def test_regime_stability_score(self, default_config, regime_switch_series):
        engine = BOCDEngine(default_config)
        result = engine.detect(regime_switch_series)
        stability = engine.regime_stability_score(result)
        assert stability.shape == (200,)
        assert np.all(stability >= 0)
        assert np.all(stability <= 1)

    def test_min_run_length_filtering(self):
        config = {
            "changepoint": {
                "hazard_lambda": 50,
                "threshold": 0.01,  # very low threshold to trigger many
                "min_run_length": 20,
            }
        }
        engine = BOCDEngine(config)
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, 200)
        result = engine.detect(X)
        # Check that detected changepoints are at least min_run_length apart
        if len(result.changepoints) > 1:
            gaps = np.diff(result.changepoints)
            assert np.all(gaps >= 20), f"Gaps too small: {gaps}"

    def test_probabilities_sum_correctly(self, default_config):
        """Changepoint prob + growth prob should be close to 1."""
        engine = BOCDEngine(default_config)
        X = np.random.default_rng(0).normal(0, 0.01, 50)
        result = engine.detect(X)
        total = result.changepoint_prob + result.growth_prob
        assert np.allclose(total, 1.0, atol=0.05)

    def test_empty_config_uses_defaults(self):
        engine = BOCDEngine({})
        assert engine.hazard_lambda == 200
        assert engine.threshold == 0.5
        X = np.random.default_rng(0).normal(0, 1, 30)
        result = engine.detect(X)
        assert result.changepoint_prob.shape == (30,)
