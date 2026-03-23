"""
Tests for monte_carlo.py — Monte Carlo Regime Simulation Engine.
"""

import numpy as np
import pytest
from monte_carlo import MonteCarloEngine, SimulationResult, ScenarioResult


@pytest.fixture
def simple_config():
    return {
        "monte_carlo": {
            "n_paths": 100,
            "n_steps": 50,
            "ruin_threshold": 0.5,
            "seed": 42,
        },
        "backtest": {"initial_capital": 100000, "commission_pct": 0.001},
        "strategy": {"hysteresis_bars": 2},
        "risk": {"kelly_fraction": 0.5, "max_position_pct": 1.0},
    }


@pytest.fixture
def two_state_hmm():
    """Simple 2-state HMM: bear and bull."""
    transmat = np.array([[0.95, 0.05], [0.05, 0.95]])
    means = np.array([[-0.002, 0.02, 0.0, 0.01, 40],
                       [0.003, 0.01, 0.0, 0.008, 60]])
    covars = np.array([
        np.diag([0.0004, 0.0001, 0.001, 0.0001, 25.0]),
        np.diag([0.0002, 0.00005, 0.001, 0.00005, 20.0]),
    ])
    labels = {0: "bear", 1: "bull"}
    return transmat, means, covars, labels


@pytest.fixture
def three_state_hmm():
    """3-state HMM: crash, neutral, bull."""
    transmat = np.array([
        [0.90, 0.08, 0.02],
        [0.05, 0.85, 0.10],
        [0.02, 0.08, 0.90],
    ])
    means = np.array([
        [-0.005, 0.03, 0.0, 0.015, 30],
        [0.000, 0.015, 0.0, 0.010, 50],
        [0.004, 0.01, 0.0, 0.008, 65],
    ])
    covars = np.array([
        np.diag([0.001, 0.0002, 0.001, 0.0002, 30.0]),
        np.diag([0.0003, 0.0001, 0.001, 0.0001, 20.0]),
        np.diag([0.0002, 0.00005, 0.001, 0.00005, 15.0]),
    ])
    labels = {0: "crash", 1: "neutral", 2: "bull"}
    return transmat, means, covars, labels


class TestRegimePathSimulation:
    def test_shape(self, simple_config, two_state_hmm):
        engine = MonteCarloEngine(simple_config)
        transmat, _, _, _ = two_state_hmm
        stationary = np.array([0.5, 0.5])
        paths = engine.simulate_regime_paths(transmat, stationary)
        assert paths.shape == (100, 50)

    def test_valid_states(self, simple_config, two_state_hmm):
        engine = MonteCarloEngine(simple_config)
        transmat, _, _, _ = two_state_hmm
        stationary = np.array([0.5, 0.5])
        paths = engine.simulate_regime_paths(transmat, stationary)
        assert set(np.unique(paths)).issubset({0, 1})

    def test_persistence(self, simple_config, two_state_hmm):
        """High self-transition probability should produce long runs."""
        engine = MonteCarloEngine(simple_config)
        transmat, _, _, _ = two_state_hmm  # 0.95 self-transition
        stationary = np.array([0.5, 0.5])
        paths = engine.simulate_regime_paths(transmat, stationary, n_paths=1000)
        # Count transitions per path
        transitions = np.sum(np.diff(paths, axis=1) != 0, axis=1)
        mean_transitions = transitions.mean()
        # With p_switch = 0.05 over 49 steps, expect ~2.45 transitions
        assert mean_transitions < 5  # generous upper bound

    def test_deterministic_seed(self, simple_config, two_state_hmm):
        engine = MonteCarloEngine(simple_config)
        transmat, _, _, _ = two_state_hmm
        stationary = np.array([0.5, 0.5])
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = engine.simulate_regime_paths(transmat, stationary, rng=rng1)
        p2 = engine.simulate_regime_paths(transmat, stationary, rng=rng2)
        np.testing.assert_array_equal(p1, p2)


class TestReturnSimulation:
    def test_shape(self, simple_config, two_state_hmm):
        engine = MonteCarloEngine(simple_config)
        transmat, means, covars, _ = two_state_hmm
        stationary = np.array([0.5, 0.5])
        regime_paths = engine.simulate_regime_paths(transmat, stationary)
        returns = engine.simulate_returns(regime_paths, means, covars)
        assert returns.shape == (100, 50)

    def test_regime_conditional_means(self, simple_config, two_state_hmm):
        """Returns should differ by regime (bear < bull on average)."""
        engine = MonteCarloEngine(simple_config)
        transmat, means, covars, labels = two_state_hmm
        stationary = np.array([0.5, 0.5])

        # Large sample for statistical power
        regime_paths = engine.simulate_regime_paths(
            transmat, stationary, n_paths=5000, n_steps=100
        )
        returns = engine.simulate_returns(regime_paths, means, covars)

        bear_returns = returns[regime_paths == 0].mean()
        bull_returns = returns[regime_paths == 1].mean()
        assert bear_returns < bull_returns

    def test_no_nans(self, simple_config, two_state_hmm):
        engine = MonteCarloEngine(simple_config)
        transmat, means, covars, _ = two_state_hmm
        stationary = np.array([0.5, 0.5])
        regime_paths = engine.simulate_regime_paths(transmat, stationary)
        returns = engine.simulate_returns(regime_paths, means, covars)
        assert not np.any(np.isnan(returns))


class TestStrategySimulation:
    def test_equity_shape(self, simple_config, two_state_hmm):
        engine = MonteCarloEngine(simple_config)
        transmat, means, covars, labels = two_state_hmm
        stationary = np.array([0.5, 0.5])
        regime_paths = engine.simulate_regime_paths(transmat, stationary)
        returns = engine.simulate_returns(regime_paths, means, covars)
        equity, signals = engine.simulate_strategy(returns, regime_paths, labels)
        assert equity.shape == (100, 50)
        assert signals.shape == (100, 50)

    def test_equity_starts_at_capital(self, simple_config, two_state_hmm):
        engine = MonteCarloEngine(simple_config)
        transmat, means, covars, labels = two_state_hmm
        stationary = np.array([0.5, 0.5])
        regime_paths = engine.simulate_regime_paths(transmat, stationary)
        returns = engine.simulate_returns(regime_paths, means, covars)
        equity, _ = engine.simulate_strategy(returns, regime_paths, labels)
        np.testing.assert_allclose(equity[:, 0], 100000)

    def test_signals_valid(self, simple_config, two_state_hmm):
        engine = MonteCarloEngine(simple_config)
        transmat, means, covars, labels = two_state_hmm
        stationary = np.array([0.5, 0.5])
        regime_paths = engine.simulate_regime_paths(transmat, stationary)
        returns = engine.simulate_returns(regime_paths, means, covars)
        _, signals = engine.simulate_strategy(returns, regime_paths, labels)
        assert set(np.unique(signals)).issubset({-1, 0, 1})


class TestRiskMetrics:
    def test_var_cvar_ordering(self, simple_config, two_state_hmm):
        """CVaR should be more extreme than VaR."""
        engine = MonteCarloEngine(simple_config)
        transmat, means, covars, labels = two_state_hmm
        result = engine.run(transmat, means, covars, labels, n_paths=500)
        assert result.cvar_95 <= result.var_95
        assert result.cvar_99 <= result.var_99

    def test_ruin_probability_bounded(self, simple_config, two_state_hmm):
        engine = MonteCarloEngine(simple_config)
        transmat, means, covars, labels = two_state_hmm
        result = engine.run(transmat, means, covars, labels)
        assert 0.0 <= result.ruin_probability <= 1.0

    def test_percentile_bands_ordered(self, simple_config, two_state_hmm):
        engine = MonteCarloEngine(simple_config)
        transmat, means, covars, labels = two_state_hmm
        result = engine.run(transmat, means, covars, labels)
        bands = result.percentile_bands
        # At each timestep, 5th < 25th < 50th < 75th < 95th
        for t in range(result.n_steps):
            assert bands[5][t] <= bands[25][t] <= bands[50][t]
            assert bands[50][t] <= bands[75][t] <= bands[95][t]

    def test_max_drawdowns_negative(self, simple_config, two_state_hmm):
        engine = MonteCarloEngine(simple_config)
        transmat, means, covars, labels = two_state_hmm
        result = engine.run(transmat, means, covars, labels)
        assert np.all(result.max_drawdowns <= 0)


class TestFullPipeline:
    def test_run_returns_result(self, simple_config, two_state_hmm):
        engine = MonteCarloEngine(simple_config)
        transmat, means, covars, labels = two_state_hmm
        result = engine.run(transmat, means, covars, labels)
        assert isinstance(result, SimulationResult)
        assert result.n_paths == 100
        assert result.n_steps == 50

    def test_regime_time_fractions_sum_to_one(self, simple_config, two_state_hmm):
        engine = MonteCarloEngine(simple_config)
        transmat, means, covars, labels = two_state_hmm
        result = engine.run(transmat, means, covars, labels)
        total = sum(result.regime_time_fractions.values())
        assert abs(total - 1.0) < 1e-6

    def test_three_state(self, simple_config, three_state_hmm):
        engine = MonteCarloEngine(simple_config)
        transmat, means, covars, labels = three_state_hmm
        result = engine.run(transmat, means, covars, labels)
        assert isinstance(result, SimulationResult)
        assert len(result.regime_time_fractions) == 3


class TestStressScenarios:
    def test_build_scenarios(self, simple_config, three_state_hmm):
        engine = MonteCarloEngine(simple_config)
        _, _, _, labels = three_state_hmm
        scenarios = engine.build_stress_scenarios(labels)
        assert len(scenarios) >= 3
        for name, desc, seq in scenarios:
            assert len(seq) == 50
            assert isinstance(name, str)

    def test_run_scenario(self, simple_config, three_state_hmm):
        engine = MonteCarloEngine(simple_config)
        transmat, means, covars, labels = three_state_hmm
        seq = np.zeros(50, dtype=int)  # all crash
        result = engine.run_scenario(
            "Test Crash", "All crash", seq, means, covars, labels, n_paths=50
        )
        assert isinstance(result, ScenarioResult)
        assert result.name == "Test Crash"
        assert len(result.terminal_wealth) == 50

    def test_crash_scenario_strategy_shorts(self, simple_config, three_state_hmm):
        """In a prolonged crash, the strategy should go short and profit."""
        engine = MonteCarloEngine(simple_config)
        transmat, means, covars, labels = three_state_hmm
        # crash state = 0, with mean return -0.005
        seq = np.zeros(200, dtype=int)
        result = engine.run_scenario(
            "Long Crash", "200 bars of crash", seq, means, covars, labels,
            n_paths=500,
        )
        # Strategy shorts crash regimes, so median should be positive
        # (the strategy is working correctly by profiting from the crash)
        assert result.median_return > 0

    def test_run_all_stress_tests(self, simple_config, three_state_hmm):
        engine = MonteCarloEngine(simple_config)
        transmat, means, covars, labels = three_state_hmm
        results = engine.run_all_stress_tests(
            means, covars, labels, n_paths=50, n_steps=50
        )
        assert len(results) >= 3
        assert all(isinstance(r, ScenarioResult) for r in results)


class TestCovarianceTypes:
    def test_diag_covariance(self, simple_config):
        engine = MonteCarloEngine(simple_config)
        transmat = np.array([[0.9, 0.1], [0.1, 0.9]])
        means = np.array([[-0.002, 0.02, 0.0, 0.01, 40],
                           [0.003, 0.01, 0.0, 0.008, 60]])
        # Diag: (n_states, n_features)
        covars = np.array([[0.0004, 0.0001, 0.001, 0.0001, 25.0],
                            [0.0002, 0.00005, 0.001, 0.00005, 20.0]])
        labels = {0: "bear", 1: "bull"}
        result = engine.run(transmat, means, covars, labels, covariance_type="diag")
        assert isinstance(result, SimulationResult)

    def test_spherical_covariance(self, simple_config):
        engine = MonteCarloEngine(simple_config)
        transmat = np.array([[0.9, 0.1], [0.1, 0.9]])
        means = np.array([[-0.002, 0.02, 0.0, 0.01, 40],
                           [0.003, 0.01, 0.0, 0.008, 60]])
        # Spherical: scalar per state
        covars = np.array([0.001, 0.0005])
        labels = {0: "bear", 1: "bull"}
        result = engine.run(transmat, means, covars, labels, covariance_type="spherical")
        assert isinstance(result, SimulationResult)
