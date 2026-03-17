# Regime Terminal: Analysis of the YouTube HMM Trading App

**Source**: YouTube video by AI Pathways (https://www.youtube.com/watch?v=EUSXhJNwRqI)

---

## 1. Overview

The video demonstrates building a "Regime Terminal" — a Python-based trading dashboard that uses Hidden Markov Models for market regime detection, layered with confirmation-based entry strategies. The app was built using Claude Code in VS Code and uses a Streamlit-style dashboard for visualization.

---

## 2. Architecture (4 Main Files)

| File | Purpose |
|------|---------|
| **Data Loader** | Fetches live OHLCV market data via Yahoo Finance (yfinance) |
| **Back Tester** | Core engine: HMM training, regime auto-labeling, strategy logic, risk management, backtesting with leverage/cooldown |
| **Dashboard** | Web UI showing signals, charts, metrics, trade logs |
| **Strategy/Config** | Entry/exit rules, confirmation conditions, leverage settings |

---

## 3. Core HMM Engine

### Configuration
- **Model**: Gaussian HMM (Gaussian distributions)
- **Number of states**: 7 regimes (state 0 through state 6)
- **Training features** (3 features):
  1. **Returns** — price returns
  2. **Range** — (High - Low) / Close (intra-day range)
  3. **Volume change** — fractional change in volume
- **Data**: Hourly candles, 730 days (~17,000 data points for Bitcoin)
- **Data source**: Yahoo Finance via yfinance

### Regime Auto-Labeling
After fitting the 7-state Gaussian HMM, regimes are automatically labeled by sorting states based on their mean return:
- **Highest positive mean return** → labeled as **Bull Run**
- **Lowest mean return** → labeled as **Bear/Crash**
- **Middle states** → various grades of chop, noise, mixed, mild trends

### Regime Types Referenced
- Bull Run (best case — trade aggressively)
- Crash / Bear (worst case — exit immediately)
- Noise / Chop (avoid trading)
- Mixed / Transitional states

---

## 4. Strategy Layer (On Top of HMM)

The key insight from the video: **HMM detects the regime; strategies are layered on top.** This is described as "two-factor authentication" — the regime must be favorable before any strategy signals are even considered.

### Entry Rules
- **Regime prerequisite**: Only enter trades when regime is detected as **bullish**
- **Confirmation system**: 7 out of 8 conditions must be met simultaneously
- **8 confirmation indicators**:
  1. RSI < 90 (not extremely overbought)
  2. Momentum (positive momentum)
  3. Volatility (within acceptable range)
  4. Volume (sufficient volume)
  5. ADX (trend strength)
  6. Price (price-based condition)
  7. MACD (MACD signal alignment)
  8. (8th condition not explicitly named — likely a composite or regime confidence threshold)

### Exit Rules
- **Immediate exit** when regime flips from bullish to bear/crash
- Exit is regime-driven, not indicator-driven

### Risk Management
- **48-hour cooldown** after any exit (bot cannot re-enter for 48 hours)
- **Leverage**: 2.5x (configurable)
- **Regime confidence**: Probability score indicating confidence in detected regime

### Signal Types
The dashboard outputs one of these signals:
- **Long Holding** — maintain existing long positions (still in bull regime)
- **Cash / Neutral** — do nothing (regime unfavorable or insufficient confirmations)
- **Enter Long** — open new long position (bull regime + 7/8 confirmations met)
- **Exit** — close positions (regime flipped to bear/crash)

---

## 5. Minimum Hold / Signal Hysteresis

The host emphasizes **"minimum hold"** (also called "signal hysteresis" or "regime confirmation lag"):
- Don't buy immediately when bull regime is detected
- Don't sell immediately when choppy regime appears
- Wait for regime stability before acting
- Prevents whipsawing on rapid regime changes
- This is described as "where a lot of the magic and profitability is truly shown"

---

## 6. Backtesting Results (as shown in video)

| Metric | Value |
|--------|-------|
| Asset | Bitcoin (BTC) |
| Timeframe | Hourly candles |
| Period | ~2 years (2024-2026) |
| Data points | ~17,000 hourly samples (11,000 used for state characterization) |
| Total Return | ~65% (one run), ~3x portfolio (another mention) |
| Alpha vs Buy & Hold | ~63% |
| Max Drawdown | ~41% |
| Win Rate | Not explicitly stated |
| Leverage | 2.5x |

---

## 7. Iterative Refinement Process

The host describes a key workflow for continuous improvement:

### Phase 1: Validate Core HMM Logic
1. Use ChatGPT to generate initial HMM Python script
2. Run in Google Colab to validate regime detection visually
3. Verify the scatter plot of regimes overlaid on price chart makes sense

### Phase 2: Build Full App with Claude Code
1. Open VS Code with Claude Code extension
2. Provide detailed prompt with:
   - Core HMM engine specs (7 states, 3 features, Gaussian)
   - Strategy logic (7/8 confirmations, specific indicators)
   - Risk management rules (cooldown, exit triggers, leverage)
   - Dashboard requirements (signals, charts, metrics, trade logs)
3. Claude Code creates the 4-file architecture autonomously

### Phase 3: Continuous AI-Assisted Tuning
Examples given:
- "Strategy is working great, but I want to test a high-risk variant"
  - Increase leverage from 2.5x to 4x
  - Reduce confirmations from 7/8 to 5/8
  - Add trailing stop
  - Add checkbox to toggle aggressive mode
- "Drawdown is too high — tighten the entry signal"
- Adjust number of HMM components
- Change confirmation thresholds

### Key Philosophy
> "Even though the HMMs stay stable, the strategies always adapt."
> "A 2020 bull run regime would have required a breakout strategy, whereas in 2024 the bull run regime would have required a mean reversion strategy instead."

The regime detection (HMM core) remains stable. The strategies on top are what get tuned as markets evolve.

---

## 8. Comparison: This App vs Trading View

| Aspect | Trading View | Regime Terminal |
|--------|-------------|-----------------|
| Logic | Linear if/then rules | Probabilistic regime detection |
| Adaptability | Static scripts | Retrain HMM on new data |
| Complexity | Simple indicators (RSI > 70 → sell) | Gaussian distributions, matrix operations, regime probabilities |
| Market evolution | Strategy dies when market changes | Retrain model, adjust strategies |
| Power | Calculator-level | Full Python ML stack |

---

## 9. Key Takeaways for Our Implementation

### What to Replicate
1. **7-state Gaussian HMM** as core engine (can test 2-7 states with BIC)
2. **3 features**: returns, range, volume change (matches academic literature)
3. **Regime auto-labeling** by sorting state means
4. **Confirmation-based entry** — require multiple indicators to agree before entering
5. **Regime-driven exits** — exit on regime flip, not indicator signals
6. **Cooldown period** — prevent whipsawing after exits
7. **Confidence scoring** — regime probability as a filter
8. **Signal hysteresis / minimum hold** — don't react to instant regime flips
9. **Dashboard** with: current signal, regime status, backtest results, equity curve, trade log

### What to Improve (Based on Academic Research)
1. **Walk-forward validation** — the video doesn't explicitly mention avoiding lookahead bias; our implementation must use strict expanding/rolling window training
2. **Multiple random restarts** — handle local optima in Baum-Welch
3. **BIC-based state selection** — rather than hardcoding 7 states, test 2-7 and select optimal
4. **Multi-asset support** — the video mentions multiple tickers
5. **Regime transition probabilities** — display the transition matrix for understanding regime persistence
6. **Overfitting awareness** — the host briefly mentions this but doesn't build it into the tool

### Proposed Enhancements Beyond the Video
1. Add **walk-forward backtest** mode (train on window, test on next period, advance)
2. Add **regime duration analysis** (how long does each regime typically last?)
3. Add **transition probability heatmap** visualization
4. Add **model comparison panel** (2-state vs 3-state vs 5-state vs 7-state)
5. Add **out-of-sample vs in-sample performance split** in metrics
6. Add **Monte Carlo simulation** for robustness testing
7. Support **multiple assets simultaneously** with per-asset regime detection

---

## 10. Prompt Template Extracted from Video

The host's prompt to Claude Code (reconstructed from transcript):

```
Build a professional regime-based trading app.

CORE ENGINE:
- Use Gaussian distributions with 7 components to determine market regimes
- Train on 3 features: returns, range, and volume volatility
- Automatically identify the bull run state (highest positive returns)
- Automatically identify bear/crash state (lowest returns)

STRATEGY LOGIC:
- Only enter trade if regime is bullish AND 7/8 conditions met:
  1. RSI < 90
  2. Positive momentum
  3. Volatility within range
  4. Sufficient volume
  5. ADX confirms trend
  6. Price condition met
  7. MACD alignment
  8. [8th condition]

RISK MANAGEMENT:
- 48-hour cooldown after any exit
- Exit immediately if regime flips to bear/crash
- Leverage: 2.5x (configurable)

ARCHITECTURE:
- Data loader: fetch data via Yahoo Finance
- Backtester: HMM training + strategy simulation
- Dashboard: signals, charts, metrics
- Charts: price with regime overlay, equity curve
- Metrics: total return, alpha vs buy & hold, win rate, max drawdown
```

---

## 11. Summary

The Regime Terminal is a practical implementation that aligns well with the academic literature reviewed in our main research report. Its core innovation is the separation of concerns: **HMM handles regime detection** (the "when" to trade), while **configurable strategies handle execution** (the "how" to trade). This two-layer architecture allows the regime engine to remain stable while strategies adapt to changing market conditions.

The main gaps versus academic best practices are the lack of explicit walk-forward validation, BIC-based model selection, and overfitting safeguards — all of which we should incorporate in our implementation.
