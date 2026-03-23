"""
monte_carlo.py — Monte Carlo Regime Simulation Engine.

Leverages the HMM's generative nature to sample thousands of synthetic
market paths from fitted transition dynamics and emission distributions.
Runs the strategy on each path to produce forward-looking risk metrics:
VaR, CVaR, ruin probability, drawdown distributions, and scenario stress tests.

Turns "what happened?" into "what COULD happen?"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.linalg import cholesky


@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation outputs."""
    n_paths: int
    n_steps: int
    # (n_paths, n_steps) arrays
    regime_paths: np.ndarray        # integer state IDs per path per step
    return_paths: np.ndarray        # simulated log returns per path per step
    equity_paths: np.ndarray        # cumulative equity per path per step
    signal_paths: np.ndarray        # strategy signals per path per step

    # Aggregate risk metrics
    terminal_wealth: np.ndarray     # final equity per path
    max_drawdowns: np.ndarray       # worst drawdown per path
    path_sharpes: np.ndarray        # Sharpe ratio per path

    # Forward-looking risk
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    ruin_probability: float = 0.0   # P(equity < ruin_threshold)
    median_return: float = 0.0
    mean_return: float = 0.0

    # Percentile bands for fan chart (5th, 25th, 50th, 75th, 95th)
    percentile_bands: dict = field(default_factory=dict)

    # Regime time-in-state statistics
    regime_time_fractions: dict = field(default_factory=dict)

    # Scenario results (if any)
    scenario_results: list = field(default_factory=list)


@dataclass
class ScenarioResult:
    """Result from a single stress scenario."""
    name: str
    description: str
    regime_sequence: np.ndarray
    equity_paths: np.ndarray
    terminal_wealth: np.ndarray
    max_drawdowns: np.ndarray
    var_95: float
    cvar_95: float
    median_return: float


class MonteCarloEngine:
    """
    Monte Carlo simulation engine using fitted HMM parameters.

    Samples regime paths from the transition matrix, generates synthetic
    returns from regime-conditional multivariate Gaussian emissions,
    and evaluates strategy performance across the distribution of outcomes.
    """

    def __init__(self, config: dict):
        mc_cfg = config.get("monte_carlo", {})
        self.n_paths = mc_cfg.get("n_paths", 1000)
        self.n_steps = mc_cfg.get("n_steps", 252)
        self.ruin_threshold = mc_cfg.get("ruin_threshold", 0.5)
        self.seed = mc_cfg.get("seed", 42)
        self.initial_capital = config.get("backtest", {}).get("initial_capital", 100000)
        self.config = config

    def simulate_regime_paths(
        self,
        transmat: np.ndarray,
        stationary: np.ndarray,
        n_paths: int | None = None,
        n_steps: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Sample regime paths from the Markov chain.

        Args:
            transmat: (n_states, n_states) transition matrix
            stationary: (n_states,) stationary distribution for initial state
            n_paths: number of paths to simulate
            n_steps: number of steps per path

        Returns:
            (n_paths, n_steps) integer array of state IDs
        """
        n_paths = n_paths or self.n_paths
        n_steps = n_steps or self.n_steps
        if rng is None:
            rng = np.random.default_rng(self.seed)

        n_states = transmat.shape[0]
        paths = np.empty((n_paths, n_steps), dtype=int)

        # Sample initial states from stationary distribution
        paths[:, 0] = rng.choice(n_states, size=n_paths, p=stationary)

        # Forward-sample from transition matrix
        for t in range(1, n_steps):
            for s in range(n_states):
                mask = paths[:, t - 1] == s
                count = mask.sum()
                if count > 0:
                    paths[mask, t] = rng.choice(n_states, size=count, p=transmat[s])

        return paths

    def simulate_returns(
        self,
        regime_paths: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
        covariance_type: str = "full",
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Generate synthetic returns from regime-conditional emission distributions.

        For each (path, step), sample from N(mu_s, Sigma_s) where s is the regime.

        Args:
            regime_paths: (n_paths, n_steps) state IDs
            means: (n_states, n_features) emission means
            covars: emission covariances (shape depends on covariance_type)
            covariance_type: 'full', 'diag', 'spherical', or 'tied'

        Returns:
            (n_paths, n_steps) log returns (first feature dimension only)
        """
        if rng is None:
            rng = np.random.default_rng(self.seed + 1)

        n_paths, n_steps = regime_paths.shape
        n_states, n_features = means.shape

        # Pre-compute Cholesky factors for efficient sampling
        chol_factors = []
        for s in range(n_states):
            cov = self._get_covariance_matrix(covars, s, n_features, covariance_type)
            # Add small jitter for numerical stability
            cov = cov + np.eye(n_features) * 1e-8
            try:
                L = cholesky(cov, lower=True)
            except np.linalg.LinAlgError:
                # Fallback: use diagonal
                L = np.diag(np.sqrt(np.maximum(np.diag(cov), 1e-8)))
            chol_factors.append(L)

        # Sample all returns at once per state for efficiency
        returns = np.empty((n_paths, n_steps))

        for s in range(n_states):
            mask = regime_paths == s
            count = mask.sum()
            if count == 0:
                continue

            # Sample multivariate normal, take first feature (log_return)
            z = rng.standard_normal((count, n_features))
            samples = z @ chol_factors[s].T + means[s]
            returns[mask] = samples[:, 0]  # log_return is feature 0

        return returns

    def _get_covariance_matrix(
        self, covars, state: int, n_features: int, covariance_type: str
    ) -> np.ndarray:
        """Extract a full covariance matrix for a given state."""
        if covariance_type == "full":
            return covars[state]
        elif covariance_type == "diag":
            return np.diag(covars[state])
        elif covariance_type == "spherical":
            return np.eye(n_features) * covars[state]
        elif covariance_type == "tied":
            return covars  # same matrix for all states
        else:
            return np.eye(n_features) * 0.01

    def simulate_strategy(
        self,
        return_paths: np.ndarray,
        regime_paths: np.ndarray,
        labels: dict[int, str],
        confidence_paths: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run a simplified strategy on each simulated path.

        Uses regime-based entry logic:
          - Long in bull/bull_run regimes
          - Short in bear/crash regimes
          - Flat otherwise

        Applies hysteresis and position sizing from config.

        Args:
            return_paths: (n_paths, n_steps) log returns
            regime_paths: (n_paths, n_steps) state IDs
            labels: {state_id: regime_label}
            confidence_paths: optional (n_paths, n_steps) confidence scores

        Returns:
            (equity_paths, signal_paths) each (n_paths, n_steps)
        """
        strat = self.config.get("strategy", {})
        risk = self.config.get("risk", {})
        hysteresis = strat.get("hysteresis_bars", 3)
        kelly_frac = risk.get("kelly_fraction", 0.5)
        max_position = risk.get("max_position_pct", 1.0)
        commission = self.config.get("backtest", {}).get("commission_pct", 0.001)

        n_paths, n_steps = return_paths.shape

        bull_states = {s for s, l in labels.items() if l in ("bull", "bull_run")}
        bear_states = {s for s, l in labels.items() if l in ("bear", "crash")}

        equity_paths = np.ones((n_paths, n_steps)) * self.initial_capital
        signal_paths = np.zeros((n_paths, n_steps), dtype=int)

        for p in range(n_paths):
            position = 0
            persist = 0
            size = kelly_frac * max_position

            for t in range(1, n_steps):
                state = regime_paths[p, t]

                # Track regime persistence
                if t > 0 and regime_paths[p, t] == regime_paths[p, t - 1]:
                    persist += 1
                else:
                    persist = 0

                # Determine target signal
                if persist >= hysteresis:
                    if state in bull_states:
                        target = 1
                    elif state in bear_states:
                        target = -1
                    else:
                        target = 0
                else:
                    target = position  # hold current

                # Apply confidence scaling if available
                if confidence_paths is not None:
                    size = kelly_frac * max_position * confidence_paths[p, t]

                # Position change cost
                if target != position:
                    equity_paths[p, t] = equity_paths[p, t - 1] * (1 - commission)
                    position = target
                else:
                    equity_paths[p, t] = equity_paths[p, t - 1]

                # Apply return
                ret = np.expm1(return_paths[p, t])  # convert log return to simple
                equity_paths[p, t] *= (1 + ret * position * size)
                signal_paths[p, t] = position

        return equity_paths, signal_paths

    def compute_risk_metrics(
        self,
        equity_paths: np.ndarray,
    ) -> dict:
        """
        Compute forward-looking risk metrics from simulated equity paths.

        Returns dict with VaR, CVaR, ruin probability, drawdown stats, etc.
        """
        n_paths, n_steps = equity_paths.shape

        # Terminal wealth and returns
        terminal = equity_paths[:, -1]
        terminal_returns = terminal / self.initial_capital - 1

        # Per-path max drawdown
        max_drawdowns = np.empty(n_paths)
        for p in range(n_paths):
            cummax = np.maximum.accumulate(equity_paths[p])
            dd = (equity_paths[p] - cummax) / cummax
            max_drawdowns[p] = dd.min()

        # Per-path Sharpe (annualized, assume daily steps)
        path_sharpes = np.empty(n_paths)
        for p in range(n_paths):
            rets = np.diff(equity_paths[p]) / equity_paths[p, :-1]
            if rets.std() > 0:
                path_sharpes[p] = rets.mean() / rets.std() * np.sqrt(252)
            else:
                path_sharpes[p] = 0.0

        # VaR and CVaR
        var_95 = float(np.percentile(terminal_returns, 5))
        var_99 = float(np.percentile(terminal_returns, 1))
        cvar_95 = float(terminal_returns[terminal_returns <= var_95].mean()) if (terminal_returns <= var_95).any() else var_95
        cvar_99 = float(terminal_returns[terminal_returns <= var_99].mean()) if (terminal_returns <= var_99).any() else var_99

        # Ruin probability
        ruin_prob = float((terminal < self.initial_capital * self.ruin_threshold).mean())

        # Percentile bands for fan chart
        percentiles = [5, 25, 50, 75, 95]
        bands = {}
        for pct in percentiles:
            bands[pct] = np.percentile(equity_paths, pct, axis=0)

        # Regime time fractions
        return {
            "terminal_wealth": terminal,
            "max_drawdowns": max_drawdowns,
            "path_sharpes": path_sharpes,
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "ruin_probability": ruin_prob,
            "median_return": float(np.median(terminal_returns)),
            "mean_return": float(np.mean(terminal_returns)),
            "percentile_bands": bands,
        }

    def compute_regime_time_fractions(
        self,
        regime_paths: np.ndarray,
        labels: dict[int, str],
    ) -> dict[str, float]:
        """Compute average fraction of time spent in each regime."""
        fractions = {}
        total = regime_paths.size
        for state_id, label in labels.items():
            fractions[label] = float((regime_paths == state_id).sum() / total)
        return fractions

    def run(
        self,
        transmat: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
        labels: dict[int, str],
        covariance_type: str = "full",
        n_paths: int | None = None,
        n_steps: int | None = None,
    ) -> SimulationResult:
        """
        Full Monte Carlo simulation pipeline.

        1. Sample regime paths from transition matrix
        2. Generate returns from emission distributions
        3. Run strategy on each path
        4. Compute risk metrics
        """
        n_paths = n_paths or self.n_paths
        n_steps = n_steps or self.n_steps
        rng = np.random.default_rng(self.seed)

        # Compute stationary distribution
        eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = np.abs(stationary)
        stationary /= stationary.sum()

        # 1. Sample regime paths
        regime_paths = self.simulate_regime_paths(
            transmat, stationary, n_paths, n_steps, rng
        )

        # 2. Generate synthetic returns
        return_paths = self.simulate_returns(
            regime_paths, means, covars, covariance_type,
            np.random.default_rng(self.seed + 1)
        )

        # 3. Run strategy on each path
        equity_paths, signal_paths = self.simulate_strategy(
            return_paths, regime_paths, labels
        )

        # 4. Compute risk metrics
        metrics = self.compute_risk_metrics(equity_paths)

        # 5. Regime time fractions
        regime_fracs = self.compute_regime_time_fractions(regime_paths, labels)

        return SimulationResult(
            n_paths=n_paths,
            n_steps=n_steps,
            regime_paths=regime_paths,
            return_paths=return_paths,
            equity_paths=equity_paths,
            signal_paths=signal_paths,
            terminal_wealth=metrics["terminal_wealth"],
            max_drawdowns=metrics["max_drawdowns"],
            path_sharpes=metrics["path_sharpes"],
            var_95=metrics["var_95"],
            var_99=metrics["var_99"],
            cvar_95=metrics["cvar_95"],
            cvar_99=metrics["cvar_99"],
            ruin_probability=metrics["ruin_probability"],
            median_return=metrics["median_return"],
            mean_return=metrics["mean_return"],
            percentile_bands=metrics["percentile_bands"],
            regime_time_fractions=regime_fracs,
        )

    def run_scenario(
        self,
        name: str,
        description: str,
        regime_sequence: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
        labels: dict[int, str],
        covariance_type: str = "full",
        n_paths: int = 500,
    ) -> ScenarioResult:
        """
        Run a stress scenario with a fixed regime sequence.

        Instead of sampling regimes from the transition matrix, uses a
        predetermined sequence (e.g., prolonged crash, whipsaw, etc.).
        Returns are still sampled stochastically from the regime emissions.

        Args:
            name: scenario name
            description: human-readable description
            regime_sequence: (n_steps,) array of state IDs
            means: emission means
            covars: emission covariances
            labels: regime labels
            covariance_type: covariance parameterization
            n_paths: number of return samples per scenario
        """
        rng = np.random.default_rng(self.seed + 100)
        n_steps = len(regime_sequence)

        # Broadcast regime sequence to all paths (same regime path, different returns)
        regime_paths = np.tile(regime_sequence, (n_paths, 1))

        # Generate stochastic returns
        return_paths = self.simulate_returns(
            regime_paths, means, covars, covariance_type, rng
        )

        # Run strategy
        equity_paths, _ = self.simulate_strategy(
            return_paths, regime_paths, labels
        )

        # Metrics
        terminal = equity_paths[:, -1]
        terminal_returns = terminal / self.initial_capital - 1
        max_dds = np.empty(n_paths)
        for p in range(n_paths):
            cummax = np.maximum.accumulate(equity_paths[p])
            dd = (equity_paths[p] - cummax) / cummax
            max_dds[p] = dd.min()

        var_95 = float(np.percentile(terminal_returns, 5))
        cvar_95_vals = terminal_returns[terminal_returns <= var_95]
        cvar_95 = float(cvar_95_vals.mean()) if len(cvar_95_vals) > 0 else var_95

        return ScenarioResult(
            name=name,
            description=description,
            regime_sequence=regime_sequence,
            equity_paths=equity_paths,
            terminal_wealth=terminal,
            max_drawdowns=max_dds,
            var_95=var_95,
            cvar_95=cvar_95,
            median_return=float(np.median(terminal_returns)),
        )

    def build_stress_scenarios(
        self,
        labels: dict[int, str],
        n_steps: int | None = None,
    ) -> list[tuple[str, str, np.ndarray]]:
        """
        Build a set of predefined stress scenarios.

        Returns list of (name, description, regime_sequence) tuples.
        """
        n_steps = n_steps or self.n_steps
        state_by_label = {l: s for s, l in labels.items()}

        scenarios = []

        # 1. Prolonged crash
        crash_state = state_by_label.get("crash", state_by_label.get("bear"))
        if crash_state is not None:
            seq = np.full(n_steps, crash_state, dtype=int)
            scenarios.append((
                "Prolonged Crash",
                f"Market stays in {labels[crash_state]} regime for {n_steps} bars",
                seq,
            ))

        # 2. Prolonged bull
        bull_state = state_by_label.get("bull_run", state_by_label.get("bull"))
        if bull_state is not None:
            seq = np.full(n_steps, bull_state, dtype=int)
            scenarios.append((
                "Prolonged Bull",
                f"Market stays in {labels[bull_state]} regime for {n_steps} bars",
                seq,
            ))

        # 3. Whipsaw: rapid regime alternation
        if crash_state is not None and bull_state is not None:
            seq = np.empty(n_steps, dtype=int)
            cycle_len = 10
            for i in range(n_steps):
                seq[i] = crash_state if (i // cycle_len) % 2 == 0 else bull_state
            scenarios.append((
                "Whipsaw",
                f"Rapid alternation between {labels[crash_state]} and "
                f"{labels[bull_state]} every {cycle_len} bars",
                seq,
            ))

        # 4. Recovery: crash then bull
        if crash_state is not None and bull_state is not None:
            mid = n_steps // 3
            seq = np.empty(n_steps, dtype=int)
            seq[:mid] = crash_state
            seq[mid:] = bull_state
            scenarios.append((
                "V-Recovery",
                f"{mid} bars of {labels[crash_state]} followed by "
                f"{n_steps - mid} bars of {labels[bull_state]}",
                seq,
            ))

        # 5. Slow bleed: neutral with periodic crashes
        neutral_state = state_by_label.get("neutral")
        if neutral_state is not None and crash_state is not None:
            seq = np.full(n_steps, neutral_state, dtype=int)
            # Insert crash episodes every 50 bars lasting 10 bars
            for start in range(0, n_steps, 50):
                end = min(start + 10, n_steps)
                seq[start:end] = crash_state
            scenarios.append((
                "Slow Bleed",
                f"Mostly {labels[neutral_state]} with periodic "
                f"{labels[crash_state]} episodes",
                seq,
            ))

        return scenarios

    def run_all_stress_tests(
        self,
        means: np.ndarray,
        covars: np.ndarray,
        labels: dict[int, str],
        covariance_type: str = "full",
        n_paths: int = 500,
        n_steps: int | None = None,
    ) -> list[ScenarioResult]:
        """Run all predefined stress scenarios."""
        scenarios = self.build_stress_scenarios(labels, n_steps)
        results = []
        for name, desc, seq in scenarios:
            result = self.run_scenario(
                name, desc, seq, means, covars, labels, covariance_type, n_paths
            )
            results.append(result)
        return results
