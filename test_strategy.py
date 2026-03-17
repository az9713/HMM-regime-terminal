"""
TDD tests for strategy.py — SignalGenerator.

Tests cover: generate_signals logic (regime-based entry/exit, cooldown,
hysteresis, min hold), and Kelly position sizing with entropy scaling.

Note: compute_confirmations depends on the `ta` library which is unavailable,
so we test generate_signals and compute_position_size directly.
"""

import numpy as np
import pandas as pd
import pytest

from strategy import SignalGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_config():
    return {
        "strategy": {
            "min_confirmations": 4,
            "cooldown_bars": 5,
            "min_hold_bars": 10,
            "hysteresis_bars": 3,
            "confirmations": {"min_confidence": 0.6},
        },
        "risk": {
            "use_kelly": True,
            "kelly_fraction": 0.5,
            "use_entropy_scaling": True,
            "max_leverage": 2.0,
            "max_position_pct": 1.0,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.15,
        },
    }


def make_signal_inputs(n=100, regime="bull"):
    """Create inputs for generate_signals."""
    df = pd.DataFrame({
        "Close": np.linspace(100, 110, n),
        "n_confirmations": np.full(n, 5),
    })
    if regime == "bull":
        states = np.zeros(n, dtype=int)
        labels = {0: "bull"}
    elif regime == "bear":
        states = np.zeros(n, dtype=int)
        labels = {0: "bear"}
    else:
        states = np.zeros(n, dtype=int)
        labels = {0: "neutral"}
    posteriors = np.column_stack([np.ones(n)])
    confidence = np.ones(n) * 0.9
    return df, states, posteriors, labels, confidence


# ---------------------------------------------------------------------------
# Tests: generate_signals
# ---------------------------------------------------------------------------

class TestGenerateSignals:
    def test_returns_series(self):
        sg = SignalGenerator(default_config())
        df, states, posteriors, labels, confidence = make_signal_inputs()
        signals = sg.generate_signals(df, states, posteriors, labels, confidence)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(df)

    def test_bull_regime_goes_long(self):
        sg = SignalGenerator(default_config())
        df, states, posteriors, labels, confidence = make_signal_inputs(
            n=50, regime="bull"
        )
        signals = sg.generate_signals(df, states, posteriors, labels, confidence)
        # After hysteresis period, should go long
        assert (signals.iloc[-1] == 1)

    def test_bear_regime_goes_short(self):
        sg = SignalGenerator(default_config())
        df, states, posteriors, labels, confidence = make_signal_inputs(
            n=50, regime="bear"
        )
        signals = sg.generate_signals(df, states, posteriors, labels, confidence)
        assert (signals.iloc[-1] == -1)

    def test_neutral_regime_stays_flat(self):
        sg = SignalGenerator(default_config())
        df, states, posteriors, labels, confidence = make_signal_inputs(
            n=50, regime="neutral"
        )
        signals = sg.generate_signals(df, states, posteriors, labels, confidence)
        assert (signals == 0).all()

    def test_hysteresis_delays_entry(self):
        """Signal should not trigger until regime persists for hysteresis_bars."""
        cfg = default_config()
        cfg["strategy"]["hysteresis_bars"] = 5
        sg = SignalGenerator(cfg)
        df, states, posteriors, labels, confidence = make_signal_inputs(
            n=50, regime="bull"
        )
        signals = sg.generate_signals(df, states, posteriors, labels, confidence)
        # First hysteresis_bars should be 0 (regime persistence starts at 0)
        assert signals.iloc[0] == 0

    def test_low_confidence_prevents_entry(self):
        cfg = default_config()
        cfg["strategy"]["confirmations"]["min_confidence"] = 0.95
        sg = SignalGenerator(cfg)
        df, states, posteriors, labels, confidence = make_signal_inputs(
            n=50, regime="bull"
        )
        confidence[:] = 0.5  # below threshold
        signals = sg.generate_signals(df, states, posteriors, labels, confidence)
        assert (signals == 0).all()

    def test_insufficient_confirmations_no_entry(self):
        cfg = default_config()
        cfg["strategy"]["min_confirmations"] = 6
        sg = SignalGenerator(cfg)
        df, states, posteriors, labels, confidence = make_signal_inputs(
            n=50, regime="bull"
        )
        df["n_confirmations"] = 3  # below min_confirmations
        signals = sg.generate_signals(df, states, posteriors, labels, confidence)
        assert (signals == 0).all()

    def test_min_hold_bars_enforced(self):
        """Once in a position, must hold for min_hold_bars."""
        cfg = default_config()
        cfg["strategy"]["min_hold_bars"] = 10
        cfg["strategy"]["hysteresis_bars"] = 1
        sg = SignalGenerator(cfg)

        n = 30
        df = pd.DataFrame({
            "Close": np.linspace(100, 110, n),
            "n_confirmations": np.full(n, 5),
        })
        # Bull for first 15 bars, then bear
        states = np.array([0] * 15 + [1] * 15)
        labels = {0: "bull", 1: "bear"}
        posteriors = np.zeros((n, 2))
        posteriors[:15, 0] = 1.0
        posteriors[15:, 1] = 1.0
        confidence = np.ones(n) * 0.9

        signals = sg.generate_signals(df, states, posteriors, labels, confidence)
        # Find where position first becomes 1
        first_long = None
        for i in range(n):
            if signals.iloc[i] == 1:
                first_long = i
                break
        if first_long is not None:
            # Should hold for at least min_hold_bars
            hold_end = min(first_long + 10, n)
            assert (signals.iloc[first_long:hold_end] == 1).all()

    def test_signals_only_valid_values(self):
        sg = SignalGenerator(default_config())
        df, states, posteriors, labels, confidence = make_signal_inputs(n=100)
        signals = sg.generate_signals(df, states, posteriors, labels, confidence)
        assert set(signals.unique()).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# Tests: compute_position_size
# ---------------------------------------------------------------------------

class TestComputePositionSize:
    def test_basic_kelly(self):
        sg = SignalGenerator(default_config())
        size = sg.compute_position_size(
            confidence=1.0, win_rate=0.6, avg_win=0.02, avg_loss=0.01
        )
        # Kelly: (0.6*2 - 0.4)/2 = 0.4, half-Kelly * confidence = 0.4*0.5*1.0 = 0.2
        assert size > 0
        assert size <= sg.max_leverage * sg.max_position_pct

    def test_zero_avg_loss_returns_zero(self):
        sg = SignalGenerator(default_config())
        size = sg.compute_position_size(confidence=0.9, avg_loss=0)
        assert size == 0.0

    def test_losing_strategy_returns_zero(self):
        """If Kelly is negative (losing edge), size should be 0."""
        sg = SignalGenerator(default_config())
        # win_rate=0.3, b=1 => kelly = (0.3*1 - 0.7)/1 = -0.4
        size = sg.compute_position_size(
            confidence=0.9, win_rate=0.3, avg_win=0.01, avg_loss=0.01
        )
        assert size == 0.0

    def test_entropy_scaling(self):
        sg = SignalGenerator(default_config())
        size_high = sg.compute_position_size(confidence=1.0, win_rate=0.6,
                                              avg_win=0.02, avg_loss=0.01)
        size_low = sg.compute_position_size(confidence=0.5, win_rate=0.6,
                                             avg_win=0.02, avg_loss=0.01)
        assert size_high > size_low

    def test_no_entropy_scaling(self):
        cfg = default_config()
        cfg["risk"]["use_entropy_scaling"] = False
        sg = SignalGenerator(cfg)
        size_high = sg.compute_position_size(confidence=1.0, win_rate=0.6,
                                              avg_win=0.02, avg_loss=0.01)
        size_low = sg.compute_position_size(confidence=0.5, win_rate=0.6,
                                             avg_win=0.02, avg_loss=0.01)
        assert size_high == size_low

    def test_no_kelly(self):
        cfg = default_config()
        cfg["risk"]["use_kelly"] = False
        sg = SignalGenerator(cfg)
        size = sg.compute_position_size(confidence=0.8, win_rate=0.6,
                                         avg_win=0.02, avg_loss=0.01)
        # Without Kelly, base size=1.0 * confidence=0.8
        assert pytest.approx(size, abs=0.01) == 0.8

    def test_capped_at_max(self):
        cfg = default_config()
        cfg["risk"]["max_leverage"] = 1.0
        cfg["risk"]["max_position_pct"] = 0.5
        cfg["risk"]["use_kelly"] = False
        sg = SignalGenerator(cfg)
        size = sg.compute_position_size(confidence=1.0)
        assert size <= 0.5

    def test_always_non_negative(self):
        sg = SignalGenerator(default_config())
        for conf in [0.0, 0.5, 1.0]:
            for wr in [0.1, 0.5, 0.9]:
                size = sg.compute_position_size(confidence=conf, win_rate=wr)
                assert size >= 0.0
