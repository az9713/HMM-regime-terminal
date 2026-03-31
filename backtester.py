"""
backtester.py — Walk-forward engine, performance metrics, bootstrap CIs.

Implements anchored walk-forward validation: train on expanding/rolling
window, decode test period, generate signals, simulate trades, advance.
No future data leakage by construction.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from data_loader import standardize, get_feature_matrix
from hmm_engine import RegimeDetector
from strategy import SignalGenerator
from regime_analyzer import RegimeTransitionAnalyzer
from changepoint import BayesianChangepoint


@dataclass
class TradeRecord:
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    direction: int  # 1 = long, -1 = short
    pnl: float
    pnl_pct: float
    regime: str
    n_confirmations: int
    position_size: float


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    trades: list[TradeRecord] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    regime_series: pd.Series = None
    confidence_series: pd.Series = None
    changepoint_prob: pd.Series = None
    fused_confidence: pd.Series = None
    ci_lower: dict = field(default_factory=dict)
    ci_upper: dict = field(default_factory=dict)
    regime_attribution: pd.DataFrame = None


class WalkForwardBacktester:
    """Walk-forward backtesting engine with bootstrap confidence intervals."""

    def __init__(self, config: dict):
        bt = config.get("backtest", {})
        self.train_window = bt.get("train_window_bars", 500)
        self.test_window = bt.get("test_window_bars", 100)
        self.step_bars = bt.get("step_bars", 50)
        self.initial_capital = bt.get("initial_capital", 100000)
        self.commission_pct = bt.get("commission_pct", 0.001)
        self.slippage_pct = bt.get("slippage_pct", 0.0005)
        self.bootstrap_samples = bt.get("bootstrap_samples", 1000)
        self.bootstrap_ci = bt.get("bootstrap_ci", 0.90)

        self.config = config
        self.feature_cols = config.get("data", {}).get(
            "features",
            ["log_return", "rolling_vol", "volume_change", "intraday_range", "rsi"],
        )

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Walk-forward loop:
          1. Train HMM on [start : start+train_window]
          2. Decode [start+train_window : start+train_window+test_window]
          3. Generate signals on test period
          4. Simulate trades
          5. Advance by step_bars and repeat
        """
        T = len(df)
        min_required = self.train_window + self.test_window
        if T < min_required:
            raise ValueError(
                f"Need at least {min_required} bars, got {T}. "
                f"Increase lookback or reduce train/test windows."
            )

        all_signals = pd.Series(0, index=df.index, dtype=int)
        all_sizes = pd.Series(0.0, index=df.index)
        all_regimes = pd.Series("unknown", index=df.index)
        all_confidence = pd.Series(0.0, index=df.index)
        all_changepoint_prob = pd.Series(0.0, index=df.index)
        all_fused_confidence = pd.Series(0.0, index=df.index)

        # BOCPD setup
        cp_cfg = self.config.get("changepoint", {})
        use_changepoint = cp_cfg.get("enabled", False)
        blend_weight = cp_cfg.get("blend_weight", 0.3)
        bocpd = BayesianChangepoint(self.config) if use_changepoint else None

        start = 0
        while start + self.train_window + self.test_window <= T:
            train_end = start + self.train_window
            test_end = min(train_end + self.test_window, T)

            train_df = df.iloc[start:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()

            # Standardize using train stats only
            train_z, test_z, _ = standardize(train_df, test_df, self.feature_cols)

            X_train = get_feature_matrix(train_z, self.feature_cols)
            X_test = get_feature_matrix(test_z, self.feature_cols)

            # Fit HMM
            detector = RegimeDetector(self.config)
            try:
                detector.fit_and_select(X_train)
            except RuntimeError:
                start += self.step_bars
                continue

            # Decode test period
            states, posteriors = detector.decode(X_test)
            labels = detector.label_regimes(X_test)
            entropy, confidence = detector.shannon_entropy(posteriors)

            # Run BOCPD on test period features
            fused_confidence = confidence.copy()
            cp_prob = np.zeros(len(X_test))
            if bocpd is not None:
                bocpd_result = bocpd.detect(X_test)
                cp_prob = bocpd_result.changepoint_prob
                fused_confidence = bocpd.merge_with_hmm_confidence(
                    confidence, bocpd_result, blend_weight
                )
                # Inject regime_stability into test_df for confirmation gate
                test_df = test_df.copy()
                test_df["regime_stability"] = bocpd_result.regime_stability

            # Generate signals (with changepoint-aware confirmations)
            sig_gen = SignalGenerator(self.config)
            test_with_conf = sig_gen.compute_confirmations(test_df)
            signals = sig_gen.generate_signals(
                test_with_conf, states, posteriors, labels, fused_confidence
            )

            # Position sizing (uses fused confidence)
            sizes = pd.Series(0.0, index=test_df.index)
            for i in range(len(test_df)):
                sizes.iloc[i] = sig_gen.compute_position_size(fused_confidence[i])

            # Store results for this fold
            idx = test_df.index
            all_signals.loc[idx] = signals.values
            all_sizes.loc[idx] = sizes.values
            for i, s in enumerate(states):
                all_regimes.iloc[train_end + i] = labels.get(s, "unknown")
            all_confidence.loc[idx] = confidence
            all_changepoint_prob.loc[idx] = cp_prob
            all_fused_confidence.loc[idx] = fused_confidence

            start += self.step_bars

        # Simulate trades over the full series
        equity, trades = self.simulate_trades(df, all_signals, all_sizes)

        # Benchmark: buy-and-hold
        benchmark = self.initial_capital * (df["Close"] / df["Close"].iloc[0])

        result = BacktestResult(
            equity_curve=equity,
            benchmark_curve=benchmark,
            trades=trades,
            regime_series=all_regimes,
            confidence_series=all_confidence,
            changepoint_prob=all_changepoint_prob if use_changepoint else None,
            fused_confidence=all_fused_confidence if use_changepoint else None,
        )

        # Compute metrics
        result.metrics = self.compute_metrics(equity, trades, benchmark)

        # Bootstrap CIs
        ci_lo, ci_hi = self.bootstrap_confidence_intervals(equity)
        result.ci_lower = ci_lo
        result.ci_upper = ci_hi

        # Regime attribution
        returns = df["Close"].pct_change().fillna(0).values
        state_ids = np.array([
            next((s for s, l in {} .items() if l == r), 0)
            for r in all_regimes.values
        ])
        regime_labels_map = {i: lab for i, lab in enumerate(sorted(set(all_regimes.values)))}
        # Build a simple integer state array from regime labels
        label_to_int = {lab: i for i, lab in enumerate(sorted(set(all_regimes.values)))}
        int_states = np.array([label_to_int.get(r, 0) for r in all_regimes.values])

        analyzer = RegimeTransitionAnalyzer()
        result.regime_attribution = analyzer.regime_attribution(
            returns, int_states, regime_labels_map, all_signals.values,
        )

        return result

    def simulate_trades(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        sizes: pd.Series,
    ) -> tuple[pd.Series, list[TradeRecord]]:
        """
        Simulate trades from signal series.
        Returns equity curve and list of TradeRecords.
        """
        capital = self.initial_capital
        equity = pd.Series(capital, index=df.index, dtype=float)
        trades = []
        position = 0  # current direction
        entry_price = 0.0
        entry_bar = 0
        position_size = 0.0

        for t in range(1, len(df)):
            price = df["Close"].iloc[t]
            prev_price = df["Close"].iloc[t - 1]
            signal = signals.iloc[t]

            # Mark-to-market
            if position != 0:
                ret = (price / prev_price - 1) * position * position_size
                capital *= (1 + ret)

            # Check for position change
            if signal != position:
                # Close existing position
                if position != 0:
                    exit_price = price * (1 - self.slippage_pct * abs(position))
                    commission = capital * self.commission_pct
                    capital -= commission

                    pnl = (exit_price / entry_price - 1) * position
                    trades.append(TradeRecord(
                        entry_bar=entry_bar,
                        exit_bar=t,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction=position,
                        pnl=pnl * position_size * self.initial_capital,
                        pnl_pct=pnl,
                        regime="",
                        n_confirmations=0,
                        position_size=position_size,
                    ))

                # Open new position
                if signal != 0:
                    entry_price = price * (1 + self.slippage_pct * abs(signal))
                    entry_bar = t
                    position_size = sizes.iloc[t]
                    commission = capital * self.commission_pct
                    capital -= commission

                position = signal

            equity.iloc[t] = capital

        return equity, trades

    def compute_metrics(
        self,
        equity: pd.Series,
        trades: list[TradeRecord],
        benchmark: pd.Series,
    ) -> dict:
        """
        Compute performance metrics:
          Sharpe, Sortino, Calmar, CVaR(5%), max drawdown,
          max DD duration, win rate, profit factor, alpha.
        """
        returns = equity.pct_change().dropna()
        bench_returns = benchmark.pct_change().dropna()

        # Annualization factor (assume hourly data ~ 24*365 bars/year)
        ann_factor = np.sqrt(8760)

        # Sharpe
        sharpe = returns.mean() / returns.std() * ann_factor if returns.std() > 0 else 0

        # Sortino
        downside = returns[returns < 0]
        sortino = returns.mean() / downside.std() * ann_factor if len(downside) > 0 and downside.std() > 0 else 0

        # Max drawdown
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd = drawdown.min()

        # Max DD duration (bars)
        dd_duration = 0
        max_dd_duration = 0
        for dd in drawdown:
            if dd < 0:
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
            else:
                dd_duration = 0

        # Calmar
        total_return_ann = (equity.iloc[-1] / equity.iloc[0]) ** (8760 / len(equity)) - 1
        calmar = total_return_ann / abs(max_dd) if max_dd != 0 else 0

        # CVaR (5%)
        var_5 = returns.quantile(0.05)
        cvar_5 = returns[returns <= var_5].mean() if len(returns[returns <= var_5]) > 0 else var_5

        # Trade stats
        if trades:
            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl <= 0]
            win_rate = len(wins) / len(trades)
            gross_profit = sum(t.pnl for t in wins) if wins else 0
            gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        else:
            win_rate = 0
            profit_factor = 0

        # Alpha (excess return vs benchmark)
        strat_return = equity.iloc[-1] / equity.iloc[0] - 1
        bench_return = benchmark.iloc[-1] / benchmark.iloc[0] - 1
        alpha = strat_return - bench_return

        return {
            "total_return": strat_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd,
            "max_dd_duration": max_dd_duration,
            "cvar_5pct": cvar_5,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "alpha": alpha,
            "n_trades": len(trades),
        }

    def bootstrap_confidence_intervals(
        self, equity: pd.Series
    ) -> tuple[dict, dict]:
        """
        Bootstrap 90% CIs on key metrics by resampling returns.
        Returns (ci_lower, ci_upper) dicts.
        """
        returns = equity.pct_change().dropna().values
        n = len(returns)
        if n < 10:
            empty = {k: 0 for k in ["sharpe_ratio", "total_return", "max_drawdown"]}
            return empty, empty

        rng = np.random.default_rng(42)
        ann_factor = np.sqrt(8760)

        boot_sharpe = []
        boot_return = []
        boot_dd = []

        for _ in range(self.bootstrap_samples):
            sample = rng.choice(returns, size=n, replace=True)
            cumulative = np.cumprod(1 + sample)

            # Sharpe
            s = sample.mean() / sample.std() * ann_factor if sample.std() > 0 else 0
            boot_sharpe.append(s)

            # Total return
            boot_return.append(cumulative[-1] - 1)

            # Max drawdown
            peak = np.maximum.accumulate(cumulative)
            dd = (cumulative - peak) / peak
            boot_dd.append(dd.min())

        alpha_lo = (1 - self.bootstrap_ci) / 2
        alpha_hi = 1 - alpha_lo

        ci_lower = {
            "sharpe_ratio": float(np.quantile(boot_sharpe, alpha_lo)),
            "total_return": float(np.quantile(boot_return, alpha_lo)),
            "max_drawdown": float(np.quantile(boot_dd, alpha_lo)),
        }
        ci_upper = {
            "sharpe_ratio": float(np.quantile(boot_sharpe, alpha_hi)),
            "total_return": float(np.quantile(boot_return, alpha_hi)),
            "max_drawdown": float(np.quantile(boot_dd, alpha_hi)),
        }

        return ci_lower, ci_upper
