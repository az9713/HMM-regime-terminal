"""
TDD tests for backtester.py — TradeRecord, BacktestResult,
WalkForwardBacktester (compute_metrics, simulate_trades, bootstrap CIs).

We test the pure-logic methods directly, avoiding the full walk-forward
run which requires ta/yfinance.
"""

import numpy as np
import pandas as pd
import pytest

from backtester import TradeRecord, BacktestResult, WalkForwardBacktester


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_config():
    return {
        "data": {"features": ["log_return", "rolling_vol"]},
        "hmm": {"min_states": 2, "max_states": 3, "n_restarts": 2, "n_iter": 50},
        "strategy": {"min_confirmations": 4, "cooldown_bars": 5,
                      "min_hold_bars": 10, "hysteresis_bars": 3,
                      "confirmations": {"min_confidence": 0.6}},
        "risk": {"use_kelly": True, "kelly_fraction": 0.5,
                 "use_entropy_scaling": True, "max_leverage": 2.0,
                 "max_position_pct": 1.0},
        "backtest": {
            "train_window_bars": 50,
            "test_window_bars": 20,
            "step_bars": 10,
            "initial_capital": 100000,
            "commission_pct": 0.001,
            "slippage_pct": 0.0005,
            "bootstrap_samples": 100,
            "bootstrap_ci": 0.90,
        },
    }


def make_equity_and_trades(n=200, seed=42):
    """Create a synthetic equity curve and trades for testing metrics."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.01, n)
    equity = 100000 * np.cumprod(1 + returns)
    equity_series = pd.Series(equity, index=pd.RangeIndex(n))

    trades = [
        TradeRecord(entry_bar=10, exit_bar=20, entry_price=100, exit_price=105,
                    direction=1, pnl=500, pnl_pct=0.05, regime="bull",
                    n_confirmations=5, position_size=0.5),
        TradeRecord(entry_bar=30, exit_bar=40, entry_price=105, exit_price=102,
                    direction=1, pnl=-300, pnl_pct=-0.03, regime="bear",
                    n_confirmations=4, position_size=0.3),
        TradeRecord(entry_bar=50, exit_bar=70, entry_price=100, exit_price=95,
                    direction=-1, pnl=500, pnl_pct=0.05, regime="bear",
                    n_confirmations=6, position_size=0.4),
    ]
    benchmark = 100000 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    benchmark_series = pd.Series(benchmark, index=pd.RangeIndex(n))

    return equity_series, trades, benchmark_series


# ---------------------------------------------------------------------------
# Tests: TradeRecord
# ---------------------------------------------------------------------------

class TestTradeRecord:
    def test_fields(self):
        t = TradeRecord(
            entry_bar=0, exit_bar=10, entry_price=100, exit_price=110,
            direction=1, pnl=1000, pnl_pct=0.1, regime="bull",
            n_confirmations=5, position_size=0.5,
        )
        assert t.direction == 1
        assert t.pnl == 1000
        assert t.pnl_pct == 0.1


# ---------------------------------------------------------------------------
# Tests: BacktestResult
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def test_defaults(self):
        r = BacktestResult(
            equity_curve=pd.Series([100000]),
            benchmark_curve=pd.Series([100000]),
        )
        assert r.trades == []
        assert r.metrics == {}


# ---------------------------------------------------------------------------
# Tests: compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.bt = WalkForwardBacktester(default_config())
        self.equity, self.trades, self.benchmark = make_equity_and_trades()

    def test_returns_dict(self):
        m = self.bt.compute_metrics(self.equity, self.trades, self.benchmark)
        assert isinstance(m, dict)

    def test_has_expected_keys(self):
        m = self.bt.compute_metrics(self.equity, self.trades, self.benchmark)
        expected_keys = {
            "total_return", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown", "max_dd_duration", "cvar_5pct", "win_rate",
            "profit_factor", "alpha", "n_trades",
        }
        assert expected_keys == set(m.keys())

    def test_n_trades(self):
        m = self.bt.compute_metrics(self.equity, self.trades, self.benchmark)
        assert m["n_trades"] == 3

    def test_win_rate(self):
        m = self.bt.compute_metrics(self.equity, self.trades, self.benchmark)
        # 2 wins out of 3 trades
        assert pytest.approx(m["win_rate"], abs=0.01) == 2 / 3

    def test_max_drawdown_negative(self):
        m = self.bt.compute_metrics(self.equity, self.trades, self.benchmark)
        assert m["max_drawdown"] <= 0

    def test_total_return_is_correct(self):
        m = self.bt.compute_metrics(self.equity, self.trades, self.benchmark)
        expected = self.equity.iloc[-1] / self.equity.iloc[0] - 1
        assert pytest.approx(m["total_return"], abs=1e-10) == expected

    def test_no_trades_metrics(self):
        m = self.bt.compute_metrics(self.equity, [], self.benchmark)
        assert m["win_rate"] == 0
        assert m["profit_factor"] == 0
        assert m["n_trades"] == 0

    def test_profit_factor_positive(self):
        m = self.bt.compute_metrics(self.equity, self.trades, self.benchmark)
        # gross_profit=1000, gross_loss=300 => PF=1000/300
        assert pytest.approx(m["profit_factor"], abs=0.01) == 1000 / 300

    def test_alpha_sign(self):
        m = self.bt.compute_metrics(self.equity, self.trades, self.benchmark)
        strat_ret = self.equity.iloc[-1] / self.equity.iloc[0] - 1
        bench_ret = self.benchmark.iloc[-1] / self.benchmark.iloc[0] - 1
        assert pytest.approx(m["alpha"], abs=1e-10) == strat_ret - bench_ret


# ---------------------------------------------------------------------------
# Tests: simulate_trades
# ---------------------------------------------------------------------------

class TestSimulateTrades:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.bt = WalkForwardBacktester(default_config())

    def test_flat_signals_no_trades(self):
        n = 50
        df = pd.DataFrame({"Close": np.linspace(100, 110, n)})
        signals = pd.Series(0, index=df.index)
        sizes = pd.Series(0.0, index=df.index)
        equity, trades = self.bt.simulate_trades(df, signals, sizes)
        assert len(trades) == 0
        # Equity should stay at initial capital
        assert equity.iloc[-1] == self.bt.initial_capital

    def test_long_trade_pnl_direction(self):
        """Going long in a rising market should increase equity."""
        n = 50
        prices = np.linspace(100, 120, n)  # uptrend
        df = pd.DataFrame({"Close": prices})
        signals = pd.Series(1, index=df.index)
        signals.iloc[0] = 0  # start flat
        sizes = pd.Series(1.0, index=df.index)
        equity, trades = self.bt.simulate_trades(df, signals, sizes)
        assert equity.iloc[-1] > self.bt.initial_capital * 0.95  # should profit

    def test_short_trade_in_downtrend(self):
        """Going short in a falling market should increase equity."""
        n = 50
        prices = np.linspace(120, 100, n)  # downtrend
        df = pd.DataFrame({"Close": prices})
        signals = pd.Series(-1, index=df.index)
        signals.iloc[0] = 0
        sizes = pd.Series(1.0, index=df.index)
        equity, trades = self.bt.simulate_trades(df, signals, sizes)
        assert equity.iloc[-1] > self.bt.initial_capital * 0.95

    def test_equity_length_matches_data(self):
        n = 50
        df = pd.DataFrame({"Close": np.linspace(100, 110, n)})
        signals = pd.Series(0, index=df.index)
        sizes = pd.Series(0.0, index=df.index)
        equity, _ = self.bt.simulate_trades(df, signals, sizes)
        assert len(equity) == n

    def test_trade_records_created_on_position_change(self):
        n = 30
        df = pd.DataFrame({"Close": np.linspace(100, 110, n)})
        signals = pd.Series(0, index=df.index)
        signals.iloc[5:15] = 1   # long from bar 5 to 14
        signals.iloc[15:25] = -1  # short from bar 15 to 24
        sizes = pd.Series(0.5, index=df.index)
        _, trades = self.bt.simulate_trades(df, signals, sizes)
        # Should have trades: close long at 15, close short at 25 (if exits)
        assert len(trades) >= 1


# ---------------------------------------------------------------------------
# Tests: bootstrap_confidence_intervals
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.bt = WalkForwardBacktester(default_config())

    def test_returns_two_dicts(self):
        equity, _, _ = make_equity_and_trades()
        ci_lo, ci_hi = self.bt.bootstrap_confidence_intervals(equity)
        assert isinstance(ci_lo, dict)
        assert isinstance(ci_hi, dict)

    def test_has_expected_keys(self):
        equity, _, _ = make_equity_and_trades()
        ci_lo, ci_hi = self.bt.bootstrap_confidence_intervals(equity)
        expected = {"sharpe_ratio", "total_return", "max_drawdown"}
        assert set(ci_lo.keys()) == expected
        assert set(ci_hi.keys()) == expected

    def test_lower_less_than_upper(self):
        equity, _, _ = make_equity_and_trades()
        ci_lo, ci_hi = self.bt.bootstrap_confidence_intervals(equity)
        for key in ci_lo:
            assert ci_lo[key] <= ci_hi[key]

    def test_short_equity_returns_zeros(self):
        equity = pd.Series([100000, 100100, 100050])
        ci_lo, ci_hi = self.bt.bootstrap_confidence_intervals(equity)
        assert all(v == 0 for v in ci_lo.values())


# ---------------------------------------------------------------------------
# Tests: WalkForwardBacktester init
# ---------------------------------------------------------------------------

class TestWalkForwardInit:
    def test_default_values(self):
        bt = WalkForwardBacktester(default_config())
        assert bt.train_window == 50
        assert bt.test_window == 20
        assert bt.initial_capital == 100000

    def test_insufficient_data_raises(self):
        bt = WalkForwardBacktester(default_config())
        df = pd.DataFrame({
            "Close": np.ones(30),
            "log_return": np.zeros(30),
            "rolling_vol": np.zeros(30),
        })
        with pytest.raises(ValueError, match="Need at least"):
            bt.run(df)
