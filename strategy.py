"""
strategy.py — Confirmation signals, entry/exit logic, Kelly sizing.

Multi-confirmation gating: 8 boolean conditions must pass a minimum
threshold before a trade signal is generated. Includes cooldown,
hysteresis, and entropy-scaled Kelly position sizing.
"""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator, MACD

from changepoint import BOCDEngine


class SignalGenerator:
    """Generate trading signals with multi-confirmation gating."""

    def __init__(self, config: dict):
        strat = config.get("strategy", {})
        conf = strat.get("confirmations", {})
        risk = config.get("risk", {})

        self.rsi_oversold = conf.get("rsi_oversold", 30)
        self.rsi_overbought = conf.get("rsi_overbought", 70)
        self.momentum_window = conf.get("momentum_window", 10)
        self.vol_low_pct = conf.get("vol_low_pct", 20)
        self.vol_high_pct = conf.get("vol_high_pct", 80)
        self.volume_threshold = conf.get("volume_threshold", 1.2)
        self.adx_threshold = conf.get("adx_threshold", 20)
        self.ema_period = conf.get("ema_period", 50)
        self.macd_fast = conf.get("macd_fast", 12)
        self.macd_slow = conf.get("macd_slow", 26)
        self.macd_signal = conf.get("macd_signal", 9)
        self.min_confidence = conf.get("min_confidence", 0.6)

        self.min_confirmations = strat.get("min_confirmations", 4)
        self.cooldown_bars = strat.get("cooldown_bars", 5)
        self.min_hold_bars = strat.get("min_hold_bars", 10)
        self.hysteresis_bars = strat.get("hysteresis_bars", 3)

        # BOCD confirmation: penalize signals during detected changepoints
        self.use_changepoint = conf.get("use_changepoint", True)
        self.changepoint_window = conf.get("changepoint_window", 10)

        self.use_kelly = risk.get("use_kelly", True)
        self.kelly_fraction = risk.get("kelly_fraction", 0.5)
        self.use_entropy_scaling = risk.get("use_entropy_scaling", True)
        self.max_leverage = risk.get("max_leverage", 2.0)
        self.max_position_pct = risk.get("max_position_pct", 1.0)
        self.stop_loss_pct = risk.get("stop_loss_pct", 0.05)
        self.take_profit_pct = risk.get("take_profit_pct", 0.15)

    def compute_confirmations(self, df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
        """
        Compute 9 boolean confirmation conditions:
          1. RSI not overbought (RSI < overbought threshold for longs)
          2. RSI not oversold (RSI > oversold threshold — avoid catching knife)
          3. Positive momentum (close > close[momentum_window] bars ago)
          4. Volatility in acceptable range (between low_pct and high_pct percentiles)
          5. Volume above threshold (volume / SMA20 > threshold)
          6. ADX above threshold (trending market)
          7. Price above EMA (trend confirmation)
          8. MACD bullish (MACD line > signal line)
          9. Regime stable (no BOCD changepoint detected recently)
        """
        out = df.copy()

        # RSI
        rsi = RSIIndicator(out["Close"], window=14).rsi()
        out["conf_rsi_not_overbought"] = rsi < self.rsi_overbought
        out["conf_rsi_not_oversold"] = rsi > self.rsi_oversold

        # Momentum
        out["conf_momentum"] = out["Close"] > out["Close"].shift(self.momentum_window)

        # Volatility range
        if "rolling_vol" in out.columns:
            vol_low = out["rolling_vol"].quantile(self.vol_low_pct / 100)
            vol_high = out["rolling_vol"].quantile(self.vol_high_pct / 100)
            out["conf_vol_range"] = (out["rolling_vol"] >= vol_low) & (out["rolling_vol"] <= vol_high)
        else:
            out["conf_vol_range"] = True

        # Volume
        vol_sma = out["Volume"].rolling(20).mean()
        out["conf_volume"] = (out["Volume"] / vol_sma) > self.volume_threshold

        # ADX
        adx = ADXIndicator(out["High"], out["Low"], out["Close"], window=14)
        out["conf_adx"] = adx.adx() > self.adx_threshold

        # Price > EMA
        ema = EMAIndicator(out["Close"], window=self.ema_period).ema_indicator()
        out["conf_above_ema"] = out["Close"] > ema

        # MACD bullish
        macd = MACD(out["Close"], window_slow=self.macd_slow,
                     window_fast=self.macd_fast, window_sign=self.macd_signal)
        out["conf_macd"] = macd.macd() > macd.macd_signal()

        # 9. BOCD regime stability: no changepoint detected in recent window
        if self.use_changepoint and "log_return" in out.columns:
            try:
                bocd = BOCDEngine(config or {})
                bocd_result = bocd.detect_on_features(out, feature_cols=["log_return"])
                confirmed_cp = bocd.changepoint_confirmation(
                    bocd_result, window=self.changepoint_window
                )
                # Regime is "stable" when no recent changepoint detected
                out["conf_regime_stable"] = confirmed_cp < bocd.threshold
                out["changepoint_prob"] = bocd_result.changepoint_prob
            except Exception:
                out["conf_regime_stable"] = True
                out["changepoint_prob"] = 0.0
        else:
            out["conf_regime_stable"] = True
            out["changepoint_prob"] = 0.0

        conf_cols = [c for c in out.columns if c.startswith("conf_")]
        out["n_confirmations"] = out[conf_cols].sum(axis=1)

        return out

    def generate_signals(
        self,
        df: pd.DataFrame,
        states: np.ndarray,
        posteriors: np.ndarray,
        labels: dict[int, str],
        confidence: np.ndarray,
    ) -> pd.Series:
        """
        Generate trading signals: 1 (long), -1 (short), 0 (flat).

        Rules:
          - Enter long: bullish regime + min confirmations met + confidence above threshold
          - Enter short: bearish regime + min confirmations met + confidence above threshold
          - Hysteresis: regime must persist for hysteresis_bars before acting
          - Cooldown: no new signal within cooldown_bars of last signal
          - Min hold: once in a position, hold for at least min_hold_bars
        """
        n = len(df)
        signals = np.zeros(n, dtype=int)

        # Determine bullish/bearish states
        bull_states = {s for s, l in labels.items() if l in ("bull", "bull_run")}
        bear_states = {s for s, l in labels.items() if l in ("bear", "crash")}

        # Regime persistence counter
        regime_persist = np.zeros(n, dtype=int)
        for t in range(1, n):
            if states[t] == states[t - 1]:
                regime_persist[t] = regime_persist[t - 1] + 1
            else:
                regime_persist[t] = 0

        last_signal_bar = -self.cooldown_bars - 1
        position_start = -self.min_hold_bars - 1
        current_pos = 0

        confs = df["n_confirmations"].values if "n_confirmations" in df.columns else np.full(n, self.min_confirmations)

        for t in range(n):
            # Check if we're in min hold period
            if current_pos != 0 and (t - position_start) < self.min_hold_bars:
                signals[t] = current_pos
                continue

            # Check cooldown
            if (t - last_signal_bar) < self.cooldown_bars and current_pos == 0:
                signals[t] = 0
                continue

            # Check hysteresis
            if regime_persist[t] < self.hysteresis_bars:
                signals[t] = current_pos  # maintain current position
                continue

            # Check confidence
            if confidence[t] < self.min_confidence:
                signals[t] = current_pos
                continue

            # Generate signal based on regime + confirmations
            state = states[t]
            n_conf = confs[t]

            if state in bull_states and n_conf >= self.min_confirmations:
                new_signal = 1
            elif state in bear_states and n_conf >= self.min_confirmations:
                new_signal = -1
            else:
                new_signal = 0

            if new_signal != current_pos:
                current_pos = new_signal
                last_signal_bar = t
                if new_signal != 0:
                    position_start = t

            signals[t] = current_pos

        return pd.Series(signals, index=df.index, name="signal")

    def compute_position_size(
        self,
        confidence: float,
        win_rate: float = 0.5,
        avg_win: float = 0.02,
        avg_loss: float = 0.01,
    ) -> float:
        """
        Position size using Kelly criterion with entropy scaling.

        Kelly: f* = (p * b - q) / b
        where p = win_rate, q = 1-p, b = avg_win / avg_loss

        Then scale by:
          - kelly_fraction (half-Kelly by default)
          - confidence (entropy-based) if enabled
          - cap at max_leverage * max_position_pct
        """
        if avg_loss == 0:
            return 0.0

        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p

        if self.use_kelly:
            kelly = (p * b - q) / b
            kelly = max(kelly, 0.0)
            size = kelly * self.kelly_fraction
        else:
            size = 1.0

        if self.use_entropy_scaling:
            size *= confidence

        size = min(size, self.max_leverage * self.max_position_pct)
        return max(size, 0.0)
