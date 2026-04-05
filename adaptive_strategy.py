"""
adaptive_strategy.py — Regime-Adaptive Strategy Engine with Posterior Blending.

The core insight: regime detection is only valuable if strategy behavior
*adapts* to the detected regime. This module replaces hard regime-to-signal
mapping with posterior-probability-weighted blending of regime-specific
playbooks, smooth adaptive exits, and regime-conditional position sizing.

Key innovations:
  1. Regime Playbooks: each regime (bull, bear, crash, neutral) gets its own
     signal logic, confirmation thresholds, and risk parameters.
  2. Posterior Blending: instead of switching strategies at regime boundaries
     (which causes whipsaws), blend playbook outputs proportionally to the
     HMM posterior probabilities. 70% bull / 30% neutral → weighted mix.
  3. Adaptive Exits: trailing ATR stops that auto-adjust width by regime
     volatility, time-decay that force-reduces stale positions, and
     uncertainty-triggered de-risking when entropy spikes.
  4. Regime-Conditional Sizing: Kelly fraction scales with regime clarity
     and regime type (aggressive in trending, conservative in uncertain).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Regime Playbook definitions
# ---------------------------------------------------------------------------

@dataclass
class RegimePlaybook:
    """Strategy parameters for a single regime type."""
    name: str
    bias: int                    # +1 = long-biased, -1 = short-biased, 0 = flat
    min_confirmations: int       # out of 8 conditions
    min_confidence: float        # entropy-based threshold
    kelly_multiplier: float      # scale on base Kelly fraction (0 = no trade)
    trailing_atr_multiplier: float  # ATR units for trailing stop width
    max_hold_bars: int           # force-reduce after this many bars
    entry_aggression: float      # 0-1, how eagerly to enter (scales confirmation weight)


# Sensible defaults per regime type
BULL_PLAYBOOK = RegimePlaybook(
    name="bull",
    bias=1,
    min_confirmations=3,       # easier entry in confirmed uptrend
    min_confidence=0.55,
    kelly_multiplier=1.0,      # full Kelly allocation
    trailing_atr_multiplier=2.5,  # wide trailing stop — let winners run
    max_hold_bars=200,
    entry_aggression=0.8,
)

BEAR_PLAYBOOK = RegimePlaybook(
    name="bear",
    bias=-1,
    min_confirmations=5,       # harder to enter shorts (need more confirmation)
    min_confidence=0.65,
    kelly_multiplier=0.6,      # reduced size — shorting is riskier
    trailing_atr_multiplier=1.5,  # tighter stop on shorts
    max_hold_bars=80,
    entry_aggression=0.5,
)

CRASH_PLAYBOOK = RegimePlaybook(
    name="crash",
    bias=0,                    # flat — preserve capital
    min_confirmations=8,       # virtually impossible to enter
    min_confidence=0.9,
    kelly_multiplier=0.0,      # no new positions
    trailing_atr_multiplier=1.0,  # tightest stops to exit existing
    max_hold_bars=10,          # rapid forced exit
    entry_aggression=0.0,
)

NEUTRAL_PLAYBOOK = RegimePlaybook(
    name="neutral",
    bias=0,
    min_confirmations=5,       # moderate bar for mean-reversion
    min_confidence=0.6,
    kelly_multiplier=0.4,      # small positions
    trailing_atr_multiplier=1.8,
    max_hold_bars=40,          # don't hold long in choppy markets
    entry_aggression=0.3,
)

DEFAULT_PLAYBOOKS = {
    "bull": BULL_PLAYBOOK,
    "bull_run": BULL_PLAYBOOK,
    "bear": BEAR_PLAYBOOK,
    "crash": CRASH_PLAYBOOK,
    "neutral": NEUTRAL_PLAYBOOK,
}


# ---------------------------------------------------------------------------
# Blended signal result
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveSignal:
    """Output of the adaptive strategy for a single bar."""
    position: int              # +1, -1, 0
    raw_blend: float           # continuous blended signal before discretization
    size: float                # position size [0, max_leverage]
    dominant_regime: str
    blend_confidence: float    # how decisive the blend is (0 = ambiguous, 1 = pure regime)
    trailing_stop: float       # current trailing stop price
    exit_reason: str           # "" if holding, else reason for exit


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class AdaptiveStrategyEngine:
    """
    Regime-adaptive strategy with posterior-weighted playbook blending.

    Instead of picking one strategy per regime, it blends all regime playbooks
    weighted by HMM posterior probabilities. This produces smooth signal
    transitions that avoid the whipsaw problem of hard regime switching.
    """

    def __init__(self, config: dict):
        adaptive = config.get("adaptive_strategy", {})
        risk = config.get("risk", {})

        # Core parameters
        self.use_blending = adaptive.get("use_blending", True)
        self.blend_temperature = adaptive.get("blend_temperature", 1.0)
        self.discretization_threshold = adaptive.get("discretization_threshold", 0.25)

        # Exit parameters
        self.atr_period = adaptive.get("atr_period", 14)
        self.time_decay_start = adaptive.get("time_decay_start", 0.7)
        self.entropy_exit_threshold = adaptive.get("entropy_exit_threshold", 0.85)
        self.max_entropy_bars = adaptive.get("max_entropy_bars", 5)

        # Position sizing
        self.base_kelly_fraction = risk.get("kelly_fraction", 0.5)
        self.max_leverage = risk.get("max_leverage", 2.0)
        self.max_position_pct = risk.get("max_position_pct", 1.0)

        # Hysteresis and cooldown
        self.cooldown_bars = adaptive.get("cooldown_bars", 3)
        self.min_hold_bars = adaptive.get("min_hold_bars", 5)

        # Playbooks (can be overridden from config)
        self.playbooks = self._load_playbooks(adaptive.get("playbooks", {}))

    def _load_playbooks(self, playbook_cfg: dict) -> dict[str, RegimePlaybook]:
        """Load playbooks from config, falling back to defaults."""
        playbooks = dict(DEFAULT_PLAYBOOKS)
        for regime_name, params in playbook_cfg.items():
            if regime_name in playbooks:
                base = playbooks[regime_name]
                playbooks[regime_name] = RegimePlaybook(
                    name=regime_name,
                    bias=params.get("bias", base.bias),
                    min_confirmations=params.get("min_confirmations", base.min_confirmations),
                    min_confidence=params.get("min_confidence", base.min_confidence),
                    kelly_multiplier=params.get("kelly_multiplier", base.kelly_multiplier),
                    trailing_atr_multiplier=params.get("trailing_atr_multiplier", base.trailing_atr_multiplier),
                    max_hold_bars=params.get("max_hold_bars", base.max_hold_bars),
                    entry_aggression=params.get("entry_aggression", base.entry_aggression),
                )
        return playbooks

    def _get_playbook(self, regime_label: str) -> RegimePlaybook:
        """Get playbook for a regime label, with fallback to neutral."""
        return self.playbooks.get(regime_label, NEUTRAL_PLAYBOOK)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators needed by playbooks (pure numpy/pandas)."""
        out = df.copy()
        close = out["Close"]
        high = out["High"]
        low = out["Low"]

        # --- RSI (Wilder's smoothed) ---
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out["rsi"] = 100 - (100 / (1 + rs))

        # --- ADX ---
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        # Zero out when the other is larger
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < plus_dm] = 0
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_14 = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr_14
        minus_di = 100 * minus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr_14
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        out["adx"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

        # --- EMA 50 ---
        out["ema_50"] = close.ewm(span=50, min_periods=50, adjust=False).mean()

        # --- MACD ---
        ema_fast = close.ewm(span=12, min_periods=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, min_periods=26, adjust=False).mean()
        out["macd_line"] = ema_fast - ema_slow
        out["macd_signal"] = out["macd_line"].ewm(span=9, min_periods=9, adjust=False).mean()

        # --- ATR ---
        out["atr"] = tr.ewm(alpha=1 / self.atr_period, min_periods=self.atr_period, adjust=False).mean()

        # --- Volume ratio ---
        vol_sma = out["Volume"].rolling(20).mean()
        out["volume_ratio"] = out["Volume"] / vol_sma

        # --- Momentum ---
        out["momentum_10"] = close > close.shift(10)

        # --- Volatility percentile rank ---
        if "rolling_vol" in out.columns:
            out["vol_pctrank"] = out["rolling_vol"].rank(pct=True)
        else:
            log_ret = np.log(close / close.shift(1))
            roll_vol = log_ret.rolling(21).std()
            out["vol_pctrank"] = roll_vol.rank(pct=True)

        return out

    def _score_confirmations(self, row: pd.Series) -> float:
        """
        Score 8 confirmation conditions as a continuous value [0, 8].
        Each condition contributes 1.0 if fully met, partial credit otherwise.
        """
        score = 0.0

        rsi = row.get("rsi", 50)
        if not np.isnan(rsi):
            # RSI not overbought: full credit < 65, linear decay 65-80
            if rsi < 65:
                score += 1.0
            elif rsi < 80:
                score += 1.0 - (rsi - 65) / 15
            # RSI not oversold: full credit > 35, linear decay 20-35
            if rsi > 35:
                score += 1.0
            elif rsi > 20:
                score += (rsi - 20) / 15

        # Momentum
        if row.get("momentum_10", False):
            score += 1.0

        # Volatility in range
        vol_pct = row.get("vol_pctrank", 0.5)
        if not np.isnan(vol_pct):
            if 0.2 <= vol_pct <= 0.8:
                score += 1.0
            else:
                # Partial credit near boundaries
                dist = min(abs(vol_pct - 0.2), abs(vol_pct - 0.8))
                score += max(0, 1.0 - dist / 0.2)

        # Volume above average
        vol_ratio = row.get("volume_ratio", 1.0)
        if not np.isnan(vol_ratio):
            score += min(1.0, max(0, (vol_ratio - 0.8) / 0.6))

        # ADX trending
        adx = row.get("adx", 20)
        if not np.isnan(adx):
            score += min(1.0, max(0, (adx - 15) / 15))

        # Price above EMA
        close = row.get("Close", 0)
        ema = row.get("ema_50", close)
        if not np.isnan(ema) and ema > 0:
            ratio = close / ema
            if ratio > 1.0:
                score += min(1.0, (ratio - 1.0) / 0.02)
            else:
                score += max(0, 1.0 - (1.0 - ratio) / 0.02)

        # MACD bullish
        macd_l = row.get("macd_line", 0)
        macd_s = row.get("macd_signal", 0)
        if not np.isnan(macd_l) and not np.isnan(macd_s):
            diff = macd_l - macd_s
            if diff > 0:
                score += min(1.0, diff / (abs(macd_s) + 1e-8))
            else:
                score += max(0, 1.0 + diff / (abs(macd_s) + 1e-8))

        return score

    def _playbook_signal(
        self,
        playbook: RegimePlaybook,
        confirmation_score: float,
        confidence: float,
    ) -> float:
        """
        Compute a single playbook's raw signal strength [-1, +1].
        Combines regime bias, confirmation score, and confidence.
        """
        if playbook.kelly_multiplier == 0:
            return 0.0

        if confidence < playbook.min_confidence:
            return 0.0

        # Effective confirmation score scaled by entry aggression
        # Higher aggression means fewer confirmations needed
        effective_threshold = playbook.min_confirmations * (1 - playbook.entry_aggression * 0.3)
        if confirmation_score < effective_threshold:
            return 0.0

        # Signal strength: how far above threshold we are, normalized
        excess = (confirmation_score - effective_threshold) / max(8 - effective_threshold, 1)
        strength = min(1.0, excess + 0.5)  # base 0.5 strength when just at threshold

        return playbook.bias * strength * confidence

    def blend_signals(
        self,
        posteriors: np.ndarray,
        labels: dict[int, str],
        confirmation_score: float,
        confidence: float,
    ) -> tuple[float, str, float]:
        """
        Blend playbook signals weighted by posterior probabilities.

        Returns (blended_signal, dominant_regime, blend_confidence).
        blended_signal is in [-1, +1].
        blend_confidence measures how decisive the blend is.
        """
        n_states = len(posteriors)

        # Apply temperature scaling for sharper/softer blending
        if self.blend_temperature != 1.0:
            scaled = posteriors ** (1 / self.blend_temperature)
            scaled = scaled / (scaled.sum() + 1e-12)
        else:
            scaled = posteriors

        # Compute each playbook's signal and weight
        blended = 0.0
        total_weight = 0.0
        dominant_idx = np.argmax(scaled)
        dominant_regime = labels.get(int(dominant_idx), "neutral")

        for state_id in range(n_states):
            label = labels.get(int(state_id), "neutral")
            playbook = self._get_playbook(label)
            signal = self._playbook_signal(playbook, confirmation_score, confidence)
            weight = scaled[state_id]
            blended += signal * weight
            total_weight += weight

        if total_weight > 0:
            blended /= total_weight

        # Blend confidence: how much the dominant regime dominates
        blend_confidence = scaled[dominant_idx] if n_states > 0 else 0.0

        return blended, dominant_regime, float(blend_confidence)

    def generate_adaptive_signals(
        self,
        df: pd.DataFrame,
        states: np.ndarray,
        posteriors: np.ndarray,
        labels: dict[int, str],
        confidence: np.ndarray,
        entropy: np.ndarray,
    ) -> list[AdaptiveSignal]:
        """
        Generate adaptive signals for the full DataFrame.

        This is the main entry point. For each bar:
          1. Compute confirmation score
          2. Blend playbook signals by posterior weights
          3. Apply adaptive exits (trailing stop, time decay, entropy de-risk)
          4. Compute regime-conditional position size
          5. Discretize the blended signal into {-1, 0, +1}

        Returns list of AdaptiveSignal, one per bar.
        """
        df_ind = self.compute_indicators(df)
        n = len(df)
        signals = []

        # State tracking
        current_pos = 0
        entry_bar = -1
        entry_price = 0.0
        trailing_stop = 0.0
        peak_price = 0.0
        last_signal_bar = -self.cooldown_bars - 1
        high_entropy_count = 0

        for t in range(n):
            row = df_ind.iloc[t]
            price = row["Close"]
            atr = row.get("atr", 0.0)
            if np.isnan(atr):
                atr = 0.0

            # --- 1. Confirmation score ---
            conf_score = self._score_confirmations(row)

            # --- 2. Blend playbook signals ---
            if self.use_blending and posteriors is not None:
                raw_blend, dominant_regime, blend_conf = self.blend_signals(
                    posteriors[t], labels, conf_score, confidence[t]
                )
            else:
                # Fallback: use dominant regime only
                dominant_regime = labels.get(int(states[t]), "neutral")
                playbook = self._get_playbook(dominant_regime)
                raw_blend = self._playbook_signal(playbook, conf_score, confidence[t])
                blend_conf = confidence[t]

            # --- 3. Adaptive exits ---
            exit_reason = ""

            if current_pos != 0:
                bars_held = t - entry_bar
                dominant_playbook = self._get_playbook(dominant_regime)

                # 3a. Trailing stop
                if current_pos == 1:
                    peak_price = max(peak_price, price)
                    stop_width = atr * dominant_playbook.trailing_atr_multiplier
                    new_stop = peak_price - stop_width
                    trailing_stop = max(trailing_stop, new_stop)
                    if price <= trailing_stop and stop_width > 0:
                        exit_reason = "trailing_stop"
                else:  # short
                    peak_price = min(peak_price, price)
                    stop_width = atr * dominant_playbook.trailing_atr_multiplier
                    new_stop = peak_price + stop_width
                    trailing_stop = min(trailing_stop, new_stop) if trailing_stop > 0 else new_stop
                    if price >= trailing_stop and stop_width > 0:
                        exit_reason = "trailing_stop"

                # 3b. Max hold time (force-reduce)
                if bars_held >= dominant_playbook.max_hold_bars:
                    exit_reason = "max_hold_time"

                # 3c. Time decay — after time_decay_start fraction of max_hold,
                # linearly reduce conviction
                time_frac = bars_held / max(dominant_playbook.max_hold_bars, 1)
                if time_frac > self.time_decay_start:
                    decay = 1.0 - (time_frac - self.time_decay_start) / (1.0 - self.time_decay_start)
                    raw_blend *= max(decay, 0)

                # 3d. Entropy-based de-risking
                if entropy[t] > self.entropy_exit_threshold:
                    high_entropy_count += 1
                else:
                    high_entropy_count = 0

                if high_entropy_count >= self.max_entropy_bars:
                    exit_reason = "entropy_spike"

            # --- 4. Discretize ---
            if exit_reason:
                new_pos = 0
            elif current_pos == 0 and (t - last_signal_bar) < self.cooldown_bars:
                new_pos = 0
            elif abs(raw_blend) >= self.discretization_threshold:
                new_pos = 1 if raw_blend > 0 else -1
            else:
                new_pos = 0

            # Min hold enforcement
            if current_pos != 0 and not exit_reason:
                bars_held = t - entry_bar
                if bars_held < self.min_hold_bars:
                    new_pos = current_pos

            # --- 5. Position sizing ---
            if new_pos != 0:
                dominant_playbook = self._get_playbook(dominant_regime)
                size = self._compute_adaptive_size(
                    confidence[t],
                    blend_conf,
                    dominant_playbook.kelly_multiplier,
                )
            else:
                size = 0.0

            # --- Update tracking ---
            if new_pos != current_pos:
                if new_pos != 0:
                    entry_bar = t
                    entry_price = price
                    peak_price = price
                    # Initialize trailing stop
                    if new_pos == 1:
                        trailing_stop = price - atr * self._get_playbook(dominant_regime).trailing_atr_multiplier
                    else:
                        trailing_stop = price + atr * self._get_playbook(dominant_regime).trailing_atr_multiplier
                last_signal_bar = t
                high_entropy_count = 0

            current_pos = new_pos

            signals.append(AdaptiveSignal(
                position=new_pos,
                raw_blend=raw_blend,
                size=size,
                dominant_regime=dominant_regime,
                blend_confidence=blend_conf,
                trailing_stop=trailing_stop if current_pos != 0 else 0.0,
                exit_reason=exit_reason,
            ))

        return signals

    def _compute_adaptive_size(
        self,
        confidence: float,
        blend_confidence: float,
        kelly_multiplier: float,
    ) -> float:
        """
        Regime-conditional position sizing.

        size = base_kelly * kelly_multiplier * confidence * blend_clarity
        where blend_clarity rewards decisive regime assignments.
        """
        base = self.base_kelly_fraction
        blend_clarity = 0.5 + 0.5 * blend_confidence  # [0.5, 1.0] — always at least half

        size = base * kelly_multiplier * confidence * blend_clarity
        size = min(size, self.max_leverage * self.max_position_pct)
        return max(size, 0.0)

    def signals_to_series(
        self,
        signals: list[AdaptiveSignal],
        index: pd.Index,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Convert AdaptiveSignal list to pandas Series for backtester integration.
        Returns (positions, sizes, regimes).
        """
        positions = pd.Series(
            [s.position for s in signals], index=index, name="signal", dtype=int
        )
        sizes = pd.Series(
            [s.size for s in signals], index=index, name="position_size", dtype=float
        )
        regimes = pd.Series(
            [s.dominant_regime for s in signals], index=index, name="regime"
        )
        return positions, sizes, regimes

    def get_exit_reasons(self, signals: list[AdaptiveSignal]) -> dict[str, int]:
        """Summarize exit reasons for diagnostics."""
        reasons: dict[str, int] = {}
        for s in signals:
            if s.exit_reason:
                reasons[s.exit_reason] = reasons.get(s.exit_reason, 0) + 1
        return reasons

    def get_blend_diagnostics(self, signals: list[AdaptiveSignal]) -> pd.DataFrame:
        """Return per-bar diagnostic DataFrame for the blended signals."""
        return pd.DataFrame([
            {
                "position": s.position,
                "raw_blend": s.raw_blend,
                "size": s.size,
                "dominant_regime": s.dominant_regime,
                "blend_confidence": s.blend_confidence,
                "trailing_stop": s.trailing_stop,
                "exit_reason": s.exit_reason,
            }
            for s in signals
        ])

