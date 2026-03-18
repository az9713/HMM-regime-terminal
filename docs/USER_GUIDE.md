# HMM Regime Terminal -- User Guide

A complete guide from installation to productive use. No prior knowledge
of HMMs or quantitative finance is assumed.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
   - [Windows (Git Bash)](#windows-git-bash)
   - [Windows (cmd / PowerShell)](#windows-cmd--powershell)
   - [Linux](#linux)
   - [macOS](#macos)
3. [First Run](#first-run)
4. [Dashboard Overview](#dashboard-overview)
5. [Sidebar Controls](#sidebar-controls)
6. [Tab 1: Current Signal](#tab-1-current-signal)
7. [Tab 2: Regime Analysis](#tab-2-regime-analysis)
8. [Tab 3: Backtest Results](#tab-3-backtest-results)
9. [Tab 4: Trade Log](#tab-4-trade-log)
10. [Tab 5: Model Diagnostics](#tab-5-model-diagnostics)
11. [Parameter Tuning Guide](#parameter-tuning-guide)
12. [Common Workflows](#common-workflows)
13. [Interpreting Results](#interpreting-results)
14. [Troubleshooting](#troubleshooting)
15. [FAQ](#faq)
16. [Glossary](#glossary)

---

## Prerequisites

### Required Software

| Software   | Minimum Version | How to Check           | Install From                       |
|------------|----------------|------------------------|-------------------------------------|
| Python     | 3.10           | `python --version`     | https://python.org/downloads        |
| pip        | 21.0           | `pip --version`        | Bundled with Python                 |
| Git        | 2.30           | `git --version`        | https://git-scm.com/downloads       |

### Required Python Packages

| Package      | Minimum Version | Purpose                              |
|--------------|----------------|--------------------------------------|
| yfinance     | 0.2.31         | OHLCV market data from Yahoo Finance |
| hmmlearn     | 0.3.0          | Hidden Markov Model fitting          |
| numpy        | 1.24           | Numerical computation                |
| pandas       | 2.0            | Data manipulation and time series    |
| scipy        | 1.11           | Statistical functions                |
| ta           | 0.11           | Technical analysis indicators        |
| streamlit    | 1.30           | Web dashboard framework              |
| plotly       | 5.18           | Interactive charts                   |
| pyyaml       | 6.0            | Configuration file parsing           |

All packages are installed automatically from requirements.txt.

### System Requirements

- **RAM:** 2 GB minimum, 4 GB recommended
- **CPU:** Any modern processor. Faster CPU = faster HMM fitting.
- **Internet:** Required for fetching market data from Yahoo Finance.
- **Browser:** Any modern browser (Chrome, Firefox, Edge, Safari).

---

## Installation

### Windows (Git Bash)

```bash
# Clone the repository
git clone https://github.com/az9713/HMM-trading-platform.git
cd HMM-trading-platform

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run app.py
```

### Windows (cmd / PowerShell)

```cmd
:: Clone the repository
git clone https://github.com/az9713/HMM-trading-platform.git
cd HMM-trading-platform

:: Create a virtual environment
python -m venv .venv

:: Activate the virtual environment
.venv\Scripts\activate

:: Install dependencies
pip install -r requirements.txt

:: Launch the application
streamlit run app.py
```

### Linux

```bash
# Clone the repository
git clone https://github.com/az9713/HMM-trading-platform.git
cd HMM-trading-platform

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run app.py
```

### macOS

```bash
# Clone the repository
git clone https://github.com/az9713/HMM-trading-platform.git
cd HMM-trading-platform

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run app.py
```

### Verifying Installation

After running `streamlit run app.py`, you should see:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open `http://localhost:8501` in your browser. You should see the
"HMM Regime Terminal" title and a sidebar with configuration controls.

If the page does not load, check the Troubleshooting section.

---

## First Run

### What to Expect

1. **Open the dashboard.** After `streamlit run app.py`, your browser
   opens to the HMM Regime Terminal. You see a sidebar on the left and
   a message in the main area: "Configure parameters in the sidebar and
   click Run Analysis to start."

2. **Leave defaults.** For your first run, do not change any settings.
   The defaults are configured for BTC-USD on 1-hour bars over 90 days.

3. **Click "Run Analysis".** This button is at the bottom of the sidebar.

4. **Wait for data fetch.** A spinner appears: "Fetching data..." This
   takes 1-3 seconds as yfinance downloads the OHLCV data. You will see
   a success message like "Loaded 2136 bars for BTC-USD (1h)".

5. **Wait for HMM fitting.** A second spinner: "Fitting HMM (BIC
   selection with random restarts)..." This is the slowest step. It takes
   20-120 seconds depending on your CPU. The system is testing models with
   2 through 8 hidden states, running 20 random restarts for each. You
   will see a success message like "Selected 3 states (BIC)".

6. **Wait for backtesting.** A third spinner: "Running walk-forward
   backtest..." This runs the HMM fitting again, multiple times, in a
   walk-forward fashion. This can take 1-5 minutes.

7. **Explore the tabs.** Once complete, all 5 tabs are populated with
   results. Click through them to explore.

### How Long Does It Take?

| Lookback | Interval | Approx Time | Notes                          |
|----------|----------|-------------|--------------------------------|
| 30 days  | 1h       | 30-60s      | ~700 bars, quick run           |
| 90 days  | 1h       | 1-3 min     | ~2100 bars, default            |
| 365 days | 1d       | 1-3 min     | ~250 bars, fewer bars but same |
|          |          |             | number of HMM fits             |
| 365 days | 1h       | 3-10 min    | ~8700 bars, slow fitting       |
| 60 days  | 15m      | 2-5 min     | ~5700 bars                     |

The dominant cost is HMM fitting (number of states * restarts * data size),
so reducing restarts from 20 to 10, or narrowing the state range from
2-8 to 2-4, gives roughly proportional speedups.

---

## Dashboard Overview

The dashboard has two main areas:

### Sidebar (Left Panel)

Contains all configuration controls organized in 5 sections:
- **Data Settings:** What asset, what timeframe, how much history
- **HMM Settings:** Model complexity and fitting parameters
- **Strategy Settings:** Signal generation filters
- **Risk Settings:** Position sizing controls
- **Backtest Settings:** Walk-forward window configuration
- **Run Analysis button:** Starts the analysis

### Main Area (5 Tabs)

| Tab               | Purpose                                              |
|-------------------|------------------------------------------------------|
| Current Signal    | What is the model saying right now?                  |
| Regime Analysis   | What regimes exist and how do they behave?           |
| Backtest Results  | How would this strategy have performed historically? |
| Trade Log         | Individual trade details                             |
| Model Diagnostics | Is the model healthy?                                |

---

## Sidebar Controls

### Data Settings

**Ticker** (default: `BTC-USD`)

Any symbol recognized by Yahoo Finance:
- US Stocks: `AAPL`, `TSLA`, `GOOGL`, `AMZN`, `MSFT`
- Crypto: `BTC-USD`, `ETH-USD`, `SOL-USD`, `DOGE-USD`
- Indices: `^SPX`, `^IXIC`, `^DJI`
- ETFs: `SPY`, `QQQ`, `GLD`, `TLT`
- Forex: `EURUSD=X`, `GBPUSD=X`

**Interval** (default: `1h`)

The bar size. Smaller intervals have shorter maximum lookback periods:

| Interval | Max Lookback | Typical Bars/Day |
|----------|-------------|------------------|
| 1m       | 7 days      | 390 (US stocks)  |
| 5m       | 60 days     | 78               |
| 15m      | 60 days     | 26               |
| 1h       | 730 days    | 6.5              |
| 1d       | unlimited   | 1                |

**Lookback (days)** (default: 90, range: 7-730)

How many calendar days of history to fetch. If the lookback exceeds the
interval limit, it is silently clamped to the maximum.

### HMM Settings

**Min states** (default: 2, range: 2-6)

The minimum number of hidden states to test. States correspond to
market regimes. Two states give a simple bull/bear model.

**Max states** (default: 8, range: 3-10)

The maximum number of hidden states to test. Higher values allow more
nuanced regime detection but are slower to fit and risk overfitting.

**Model type** (default: `gaussian`)

- `gaussian`: One Gaussian distribution per state. Simpler, faster.
- `gmm`: Gaussian Mixture Model per state. More flexible, slower.
  (Note: GMM support is configured but the current implementation uses
  GaussianHMM regardless.)

**Covariance** (default: `full`)

The structure of the emission covariance matrix:
- `full`: Each state has a full covariance matrix. Captures correlations
  between features. Most flexible but requires the most data.
- `diag`: Diagonal covariance only. Assumes features are independent
  within each state. Faster, more stable with limited data.
- `tied`: All states share one covariance matrix. Assumes volatility
  structure is the same across regimes (only means differ).
- `spherical`: One variance scalar per state. All features have equal
  variance. Most restrictive.

**Random restarts** (default: 20, range: 5-50)

Number of random initializations per state count. More restarts increase
the chance of finding the best model but take longer. 20 is a good
balance for most use cases.

### Strategy Settings

**Min confirmations** (default: 4, range: 1-8)

How many of the 8 technical conditions must be true to generate a signal.
Lower values produce more signals; higher values produce fewer but
higher-conviction signals.

**Cooldown bars** (default: 5, range: 0-20)

After closing a position, the system waits this many bars before opening
a new one. Prevents overtrading during regime oscillations.

**Min hold bars** (default: 10, range: 1-50)

Minimum number of bars to hold a position before allowing exit. Prevents
premature exits on noise.

**Min regime confidence** (default: 0.6, range: 0.0-1.0)

The entropy-based confidence threshold. If the model is less confident
than this about the current regime, no signal is generated. This is the
single most impactful filter.

### Risk Settings

**Kelly sizing** (default: On)

When enabled, position sizes are determined by the Kelly criterion based
on estimated win rate and payoff ratio. When disabled, position size is
fixed at 100%.

**Entropy scaling** (default: On)

When enabled, position size is multiplied by the regime confidence (0 to
1). This means the system bets less when the model is uncertain.

**Max leverage** (default: 2.0x, range: 0.5-5.0)

The maximum allowed position size as a multiple of equity. 1.0 = no
leverage, 2.0 = up to 2x leverage.

### Backtest Settings

**Train window (bars)** (default: 500, range: 100-2000)

Number of bars used to fit the HMM in each walk-forward fold. Larger
windows give the model more data to learn from, but may include outdated
market structure.

**Test window (bars)** (default: 100, range: 20-500)

Number of bars to generate signals on after each training period. This
is the out-of-sample evaluation period.

**Step size (bars)** (default: 50, range: 10-200)

How many bars to advance the window between walk-forward folds. Smaller
steps give more folds (more robust estimates) but take longer.

**Important:** You need at least `train_window + test_window` total bars
of data. If you see an error about insufficient bars, increase lookback
or decrease these window sizes.

---

## Tab 1: Current Signal

This tab shows the model's assessment of the current market state, as of
the most recent bar of data.

### Regime Banner

A large colored box showing the current regime classification:

| Regime    | Color  | Meaning                              |
|-----------|--------|--------------------------------------|
| CRASH     | Red    | Severe bearish conditions            |
| BEAR      | Orange | Negative returns, elevated volatility|
| NEUTRAL   | Gray   | No clear directional bias            |
| BULL      | Green  | Positive returns, moderate volatility|
| BULL_RUN  | Blue   | Strong positive returns              |

The number of possible regimes depends on the HMM's selected state count.
A 2-state model produces only BEAR and BULL. A 5-state model produces
all five labels.

### Confidence Metric

A percentage from 0% to 100% derived from Shannon entropy:
- **90-100%:** The model is very confident about the current regime.
  Signals based on this classification are more reliable.
- **60-80%:** Moderate confidence. Signals are generated if other
  conditions are met.
- **Below 60%:** Low confidence (below default threshold). The model
  cannot clearly distinguish between regimes. Signals are suppressed.

### Signal Indicator

The current trading recommendation:
- **LONG:** The model recommends being long (buying/holding).
- **SHORT:** The model recommends being short.
- **FLAT:** No position recommended. Either the model is uncertain,
  confirmations are insufficient, or a cooldown is active.

### Position Size

The recommended allocation as a percentage of equity. This is computed
from the Kelly criterion, scaled by the half-Kelly fraction and regime
confidence.

Example: If Kelly says bet 40% and confidence is 80%, the recommended
size is 40% * 0.5 (half-Kelly) * 80% (confidence) = 16%.

### Confirmation Breakdown

A table showing which of the 8 technical conditions are currently met:

| Condition          | What It Checks                            |
|--------------------|-------------------------------------------|
| Rsi Not Overbought | RSI is below 70 (not overheated)          |
| Rsi Not Oversold   | RSI is above 30 (not capitulating)        |
| Momentum           | Price is higher than N bars ago           |
| Vol Range          | Volatility is in the 20th-80th percentile |
| Volume             | Volume exceeds 1.2x the 20-bar average   |
| Adx                | ADX above 20 (market is trending)         |
| Above Ema          | Price is above the 50-period EMA          |
| Macd               | MACD line is above the signal line        |

A green checkmark (True) means the condition is met. The total count is
shown below the table (e.g., "5 / 8").

### Reading the Current Signal Tab

A high-conviction long signal looks like:
- Regime: BULL or BULL_RUN
- Confidence: 80%+
- Signal: LONG
- Position Size: > 10%
- Confirmations: 6+ out of 8

If the signal is FLAT despite a bullish regime, look at the confirmation
breakdown to understand why. Common reasons:
- Confidence below threshold (model uncertain)
- Not enough confirmations met (technical indicators disagree)
- Regime has not persisted long enough (hysteresis filter)
- System is in cooldown after a recent position close

---

## Tab 2: Regime Analysis

This tab provides deep analysis of the detected market regimes.

### Price with Regime Overlay

An interactive price chart where each bar is color-coded by its regime.
The continuous black line shows price, while colored dots show regime
assignments.

**How to use it:**
- Hover over any point to see price and date
- Zoom in on specific periods to see regime transitions
- Look for regime clusters -- long stretches of one color indicate
  persistent regimes
- Rapid color changes indicate regime uncertainty

### Transition Matrix Heatmap

A square heatmap showing the probability of transitioning between
regimes in one bar. Read it as "probability of going FROM the row state
TO the column state."

**What to look for:**
- **Diagonal values:** These are self-transition probabilities. Values
  of 0.95+ mean regimes are sticky (they persist for 20+ bars on average).
- **Off-diagonal values:** These show transition probabilities. A high
  value (e.g., bull -> bear = 0.08) means that transition is relatively
  common.
- **Asymmetry:** If bull -> bear = 0.05 but bear -> bull = 0.10, the
  market transitions out of bear regimes faster than out of bull regimes.

### Regime Statistics Table

A table with per-regime metrics computed from the fitted HMM:

| Column             | Meaning                                       |
|--------------------|-----------------------------------------------|
| state              | Internal state index (integer)                |
| label              | Semantic label (bear, bull, etc.)              |
| mean_return        | Average log return per bar in this regime      |
| volatility         | Standard deviation of returns from the HMM's  |
|                    | emission covariance                           |
| expected_duration  | Average number of bars the regime persists:   |
|                    | 1 / (1 - self_transition_probability)         |
| stationary_weight  | Long-run proportion of time in this regime    |

### Return Distributions

Overlaid histograms showing the distribution of log returns for each
regime. Each color corresponds to a regime.

**What to look for:**
- **Separation:** Well-separated distributions (different means, minimal
  overlap) indicate the HMM found genuinely distinct regimes.
- **Overlap:** Heavily overlapping distributions suggest the model may be
  over-splitting regimes that are not meaningfully different.
- **Tail behavior:** Crash/bear regimes should have fatter left tails
  (more extreme negative returns).

### BIC / AIC Model Selection

A line chart showing BIC and AIC scores for each tested number of states.
The selected model has the lowest BIC.

**What to look for:**
- **Clear minimum:** A V-shaped or U-shaped BIC curve with a clear
  minimum indicates the data strongly prefers that number of states.
- **Flat curve:** A flat BIC curve means the data does not strongly
  differentiate between state counts. The simplest model is usually best.
- **BIC vs AIC disagreement:** If BIC selects 3 states but AIC selects
  5, the extra 2 states improve fit slightly but are not justified by
  the data. Trust BIC in this case.

---

## Tab 3: Backtest Results

This tab shows how the strategy would have performed historically using
walk-forward backtesting (no lookahead bias).

### Performance Metrics

Eleven metrics are displayed in cards. Some have a tooltip showing the
90% bootstrap confidence interval.

| Metric         | What It Means                      | Good Values                  |
|----------------|------------------------------------|------------------------------|
| Total Return   | Cumulative return over the period  | Positive, > benchmark        |
| Sharpe Ratio   | Risk-adjusted return (annualized)  | > 1.0 good, > 2.0 excellent |
| Sortino Ratio  | Like Sharpe, penalizes only        | > 1.5 good                   |
|                | downside volatility                |                              |
| Calmar Ratio   | Annualized return / max drawdown   | > 1.0                        |
| Max Drawdown   | Largest peak-to-trough decline     | > -20%                       |
| Max DD Duration| Longest drawdown in bars           | Shorter is better            |
| CVaR (5%)      | Expected loss in worst 5% of bars  | Close to 0                   |
| Win Rate       | % of profitable trades             | > 50%                        |
| Profit Factor  | Gross profit / gross loss          | > 1.5 good, > 2.0 strong    |
| Alpha          | Return above buy-and-hold          | Positive                     |
| Total Trades   | Number of round-trip trades        | > 10 for statistical power   |

### Equity Curve vs Benchmark

Two lines on one chart:
- **Blue solid line:** Strategy equity over time
- **Gray dashed line:** Buy-and-hold equity for the same asset

The strategy should ideally:
- Stay above the benchmark during drawdowns (risk management working)
- Track or exceed the benchmark during bull runs
- Show smoother growth (lower volatility)

### Drawdown Chart

A red filled area chart showing the current drawdown from the strategy's
equity peak. Deeper red = larger drawdown.

**What to look for:**
- Shallow drawdowns (e.g., -5%) indicate good risk management
- Deep drawdowns (e.g., -30%) indicate the model was wrong during a
  significant period
- Long drawdowns indicate the model struggled to recover

### Interpreting Bootstrap Confidence Intervals

Hover over metrics with an (i) icon to see the 90% CI. The CI tells you
the range where the true metric value likely falls.

**Example:** Sharpe = 1.50, CI = [0.30, 2.80]

This means:
- The point estimate is 1.50 (looks good)
- But the true value could be as low as 0.30 (barely positive)
- Wide CIs indicate high uncertainty

**Red flags:**
- Sharpe CI crosses 0: The strategy may not have a genuine edge
- Total return CI includes negative values: You might have lost money
- CI width is larger than the point estimate: High uncertainty

---

## Tab 4: Trade Log

A table showing every round-trip trade from the backtest.

| Column         | Meaning                                    |
|----------------|--------------------------------------------|
| Entry Bar      | Bar index when the position was opened     |
| Exit Bar       | Bar index when the position was closed     |
| Direction      | Long or Short                              |
| Entry Price    | Execution price including slippage         |
| Exit Price     | Execution price including slippage         |
| PnL ($)        | Dollar profit or loss                      |
| PnL (%)        | Percentage return on the trade             |
| Position Size  | Kelly/entropy-scaled allocation            |
| Bars Held      | How long the position was held             |

**How to analyze the trade log:**

Click column headers to sort. Useful sorts:
- **Sort by PnL ($):** Find your best and worst trades
- **Sort by Bars Held:** See if longer trades perform better
- **Sort by Direction:** Compare long vs short performance
- **Sort by Position Size:** Check if larger positions performed
  differently

Look for patterns:
- Are losses clustered in time? (model may have been wrong during that
  period)
- Do short trades consistently lose? (may need to disable shorts or
  tune bear regime detection)
- Are there too many tiny trades? (increase min_hold_bars or cooldown)

---

## Tab 5: Model Diagnostics

This tab helps assess whether the HMM model is healthy and reliable.

### Rolling Log-Likelihood

A time series showing the model's per-bar log-likelihood computed over
a rolling 50-bar window.

**Interpretation:**
- **Stable, flat line:** The model consistently explains the data well.
  Good sign.
- **Downward trend:** Model degradation. The market structure has changed
  since the model was fitted. In the walk-forward backtest, this is
  handled by re-fitting, but for the full-sample model (shown here), it
  indicates the model is less reliable for recent data.
- **Sudden drops:** Regime breaks or black-swan events that the model
  cannot explain. These often coincide with major market events.
- **Upward trend:** The model fits recent data better than older data.
  This could mean the model is biased toward recent market structure.

### Shannon Entropy / Confidence

A dual-axis chart showing:
- **Orange line (left axis): Entropy.** Higher values = more uncertainty
  about the current regime. Maximum = log2(n_states) bits.
- **Teal line (right axis): Confidence.** Higher values = more certainty.
  This is 1 minus normalized entropy.

**Interpretation:**
- **Entropy spikes** often coincide with regime transitions. The model
  is briefly uncertain before settling into the new regime.
- **Persistently high entropy** means the model cannot distinguish between
  regimes. Try: fewer states, more data, or different features.
- **Low confidence periods** should correlate with FLAT signals. If the
  system is generating signals during low-confidence periods, the
  min_confidence threshold may be too low.

### Feature Correlation Matrix

A heatmap showing pairwise Pearson correlations between the 5 input
features. Values range from -1 (perfect negative correlation) to +1
(perfect positive correlation).

**Interpretation:**
- **Low correlations (|r| < 0.3):** Features provide independent
  information. Good for the HMM.
- **High correlations (|r| > 0.7):** Features are partially redundant.
  The HMM is not getting as much independent information as the number
  of features suggests.
- **Expected patterns:** rolling_vol and intraday_range may be moderately
  correlated (both measure volatility). log_return and rsi may be
  moderately correlated (both reflect direction/momentum).

---

## Parameter Tuning Guide

### If You See X, Try Adjusting Y

| Symptom                                 | Likely Cause                   | Adjustment                                |
|-----------------------------------------|--------------------------------|-------------------------------------------|
| All signals are FLAT                    | Filters too strict             | Lower min_confirmations to 3,             |
|                                         |                                | lower min_confidence to 0.4               |
| Too many trades (overtrading)           | Filters too loose              | Raise min_confirmations to 6,             |
|                                         |                                | raise cooldown to 10-15                   |
| BIC curve is flat                       | Insufficient data or too many  | Increase lookback, or narrow              |
|                                         | states tested                  | state range to 2-4                        |
| BIC curve is noisy                      | Not enough restarts            | Increase restarts to 30-50                |
| Very few trades in backtest             | Windows too large for data     | Reduce train_window to 200-300,           |
|                                         |                                | reduce test_window to 50                  |
| Backtest error: "Need at least N bars"  | Not enough data                | Increase lookback or reduce               |
|                                         |                                | train + test window sizes                 |
| HMM fitting is too slow                 | Too many states/restarts       | Narrow states to 2-4, reduce              |
|                                         |                                | restarts to 10, use diag covariance       |
| All HMM fits failed                     | Insufficient data for model    | Reduce max_states, increase lookback,     |
|                                         | complexity                     | switch covariance to diag                 |
| Regime labels keep changing between runs| Data sensitivity               | Increase restarts to 40+, or fix          |
|                                         |                                | state count (set min = max)               |
| Backtest performs much worse than        | Overfitting                    | Reduce max_states to 3-4, use diag        |
| full-sample analysis                    |                                | covariance, increase train_window         |
| High entropy everywhere                 | Poor regime separation         | Try fewer states (2-3), increase data,    |
|                                         |                                | verify the asset has distinct regimes     |
| Strategy loses during trending markets  | Bear regime detection too      | Lower confidence threshold,               |
|                                         | aggressive                     | increase min_confirmations for shorts     |
| Excessive drawdowns                     | Position sizing too aggressive | Enable Kelly sizing + entropy scaling,    |
|                                         |                                | reduce max_leverage to 1.0                |

### Recommended Starting Configurations

**Conservative (fewer trades, higher conviction):**
```
Min confirmations: 6
Cooldown: 10
Min hold: 20
Min confidence: 0.8
Max leverage: 1.0
States: 2-3
```

**Moderate (default):**
```
Min confirmations: 4
Cooldown: 5
Min hold: 10
Min confidence: 0.6
Max leverage: 2.0
States: 2-8
```

**Aggressive (more trades, lower conviction):**
```
Min confirmations: 2
Cooldown: 2
Min hold: 5
Min confidence: 0.4
Max leverage: 3.0
States: 2-8
```

---

## Common Workflows

### Workflow 1: Quick Regime Check

**Goal:** See what regime the market is in right now for a specific asset.

1. Enter the ticker (e.g., `AAPL`)
2. Set interval to `1d`, lookback to `365`
3. Leave HMM settings at defaults
4. Click "Run Analysis"
5. Go to **Tab 1 (Current Signal)** to see the current regime and
   confidence
6. Go to **Tab 2 (Regime Analysis)** to see the price chart with regime
   overlay and the regime statistics table
7. Focus on the transition matrix -- are we likely to stay in the
   current regime?

**Time: ~1 minute**

### Workflow 2: Crypto Analysis

**Goal:** Analyze crypto market regimes with higher-frequency data.

1. Enter `BTC-USD` (or `ETH-USD`, `SOL-USD`)
2. Set interval to `1h`, lookback to `90`
3. HMM: min_states=2, max_states=5 (crypto often has clear bull/bear)
4. Strategy: min_confirmations=3 (crypto is noisy, be less strict)
5. Click "Run Analysis"
6. Check all 5 tabs
7. If too many FLAT signals, lower min_confidence to 0.4
8. If too many whipsaw trades, increase cooldown to 8-10

**Time: 2-5 minutes**

### Workflow 3: Strategy Validation

**Goal:** Rigorously test whether the HMM-based strategy has a real edge.

1. Set a long lookback: 365 days, interval `1d` for maximum data
2. Backtest: train_window=250 (1 year of daily bars), test_window=60,
   step=30
3. Click "Run Analysis"
4. Go to **Tab 3 (Backtest Results)** and check:
   - Is the Sharpe ratio positive with CI not crossing zero?
   - Is total return positive?
   - Is profit factor > 1.5?
   - Are there enough trades (> 15) for statistical significance?
5. Go to **Tab 5 (Model Diagnostics)** and check:
   - Is rolling log-likelihood stable?
   - Are entropy spikes aligned with regime transitions?
6. Compare in-sample (Tab 2) vs walk-forward (Tab 3): if Tab 2 looks
   amazing but Tab 3 is mediocre, the model is overfitting.

**Time: 5-10 minutes**

### Workflow 4: Comparing Models

**Goal:** Determine the best number of states for your asset.

1. Set up for your asset (e.g., `SPY`, `1d`, 365 days)
2. **Run 1:** Set min_states=2, max_states=2 (force 2-state model)
3. Note the backtest metrics (Sharpe, return, win rate)
4. **Run 2:** Set min_states=3, max_states=3 (force 3-state model)
5. Note the metrics
6. **Run 3:** Set min_states=4, max_states=4 (force 4-state model)
7. Compare:
   - Which has the best walk-forward Sharpe?
   - Which has the tightest bootstrap CIs?
   - Which has the most stable rolling log-likelihood?
8. Finally, **Run 4:** Set min_states=2, max_states=8 (let BIC decide)
9. Verify BIC selects the model that also performed best in the
   comparison

**Time: 15-20 minutes**

### Workflow 5: Tuning for a Specific Asset

**Goal:** Find the best parameter configuration for a specific asset.

1. Start with defaults and run analysis
2. Check Tab 3 for baseline performance
3. Adjust one parameter at a time and re-run:
   - First: try different state counts (fix min=max to test each)
   - Then: try different covariance types (full vs diag)
   - Then: adjust min_confirmations (try 3, 4, 5, 6)
   - Then: adjust min_confidence (try 0.4, 0.6, 0.8)
4. Keep the configuration that gives the best walk-forward Sharpe
   with CIs not crossing zero
5. Save your preferred settings by editing config.yaml

**Time: 30-60 minutes**

---

## Interpreting Results

### What "Good" Looks Like

A well-performing HMM analysis shows:

1. **Clear BIC minimum:** The BIC curve has a distinct V or U shape,
   indicating the data genuinely has that many distinct regimes.

2. **Separated return distributions:** In Tab 2, the regime-specific
   return histograms have different means and minimal overlap.

3. **Walk-forward Sharpe > 1.0:** The strategy generates positive
   risk-adjusted returns out of sample. The 90% CI should not cross zero.

4. **Strategy equity above benchmark:** In Tab 3, the blue strategy line
   stays above or near the gray benchmark line, especially during drawdowns.

5. **Stable rolling log-likelihood:** In Tab 5, the log-likelihood is
   roughly flat over time, indicating the model's regime structure
   persists.

6. **Entropy aligned with transitions:** Entropy spikes correspond to
   regime changes, and is low during stable regime periods.

7. **Sufficient trades:** At least 10-15 round-trip trades in the backtest
   to give statistical credibility to the metrics.

### Warning Signs

1. **Flat BIC curve:** The data does not have distinct regimes, or there
   is not enough data for model selection to work. Try more data or fewer
   states.

2. **Bootstrap CI for Sharpe crosses zero:** The strategy may not have a
   genuine edge. The positive Sharpe could be due to chance.

3. **Very few trades (< 5):** Not enough data to evaluate the strategy.
   Increase lookback, decrease min_confirmations, or lower confidence
   threshold.

4. **High entropy everywhere:** The model cannot distinguish between
   regimes. This asset may not have distinct regime structure, or the
   features are not informative.

5. **Rolling log-likelihood declining:** The market has structurally
   changed since the model was fit. The walk-forward backtest handles
   this by re-fitting, but the full-sample model (Tabs 1, 2, 5) may be
   unreliable.

6. **Massive gap between in-sample and walk-forward:** If Tab 2 shows
   beautifully separated regimes but Tab 3 shows poor performance, the
   model is overfitting. Reduce complexity (fewer states, diag covariance).

7. **All signals FLAT:** The filters are too strict. Gradually relax
   min_confidence and min_confirmations.

### In-Sample vs Walk-Forward

The Regime Analysis tab (Tab 2) shows **in-sample** results: the HMM
was fit on ALL data and decoded on the same data. This always looks
better than reality because the model has seen the answers.

The Backtest tab (Tab 3) shows **walk-forward** results: at each step,
the HMM was fit only on past data and tested on future data. This is a
realistic estimate of how the strategy would have performed in practice.

Walk-forward metrics should be:
- Lower than in-sample (always)
- But still positive (if there is a genuine edge)
- Significantly lower = overfitting
- Negative = no edge, or the edge is too small to survive transaction costs

---

## Troubleshooting

### Error: "No data returned for TICKER"

**Cause:** yfinance could not find data for the specified ticker.

**Solutions:**
- Verify the ticker is correct on https://finance.yahoo.com
- Use the exact Yahoo Finance symbol (e.g., `AAPL` not `Apple`)
- Add suffix for non-US markets (e.g., `AAPL.L` for London)
- Crypto needs `-USD` suffix (e.g., `BTC-USD` not `BTC`)
- Very recent IPOs may not have enough data
- Check your internet connection

### Error: "Need at least N bars, got M"

**Cause:** The dataset has fewer bars than train_window + test_window.

**Solutions:**
- Increase lookback days
- Decrease train_window_bars in the sidebar (try 200-300)
- Decrease test_window_bars in the sidebar (try 50)
- Use a larger interval (1d instead of 1h gives more days of coverage)

### Error: "All HMM fits failed"

**Cause:** The model could not converge for any state count with any seed.

**Solutions:**
- Reduce max_states to 4 (fewer parameters to estimate)
- Increase lookback to get more data
- Switch covariance from `full` to `diag` (fewer parameters)
- Increase n_restarts to 30-50 (more chances to converge)
- The data may have extreme outliers -- try a different time period

### Dashboard is very slow

**Cause:** HMM fitting is computationally expensive. The total number of
model fits is (max_states - min_states + 1) * n_restarts.

**Solutions:**
- Reduce restarts from 20 to 10 (2x speedup)
- Narrow state range: try 2-4 instead of 2-8 (2.3x speedup)
- Use `diag` covariance instead of `full` (~3x speedup per fit)
- Reduce lookback (fewer bars = faster fitting)
- The walk-forward backtest multiplies fitting time by the number of folds

### Signals are always FLAT

**Cause:** The filters are too strict for this dataset.

**Solutions (try in order):**
- Lower min_confirmations from 4 to 3 (then 2 if still flat)
- Lower min_confidence from 0.6 to 0.4 (then 0.3)
- Lower hysteresis_bars to 1 (in config.yaml)
- Check Tab 5 entropy -- if it is persistently high, the model is
  uncertain and signals will be suppressed
- Try fewer states (2-3) for clearer regime separation

### Charts are not displaying

**Cause:** Browser or Streamlit rendering issue.

**Solutions:**
- Refresh the browser page
- Clear Streamlit cache: add `?clear_cache=true` to the URL
- Try a different browser
- Check that plotly is installed: `pip install plotly>=5.18`
- Restart the Streamlit server: Ctrl+C and re-run `streamlit run app.py`

### Import errors on startup

**Cause:** Missing or incompatible packages.

**Solutions:**
- Ensure you activated the virtual environment before `pip install`
- Re-run `pip install -r requirements.txt`
- Check Python version: `python --version` (must be 3.10+)
- If hmmlearn fails to install, ensure you have a C compiler:
  - Windows: Install Visual Studio Build Tools
  - Linux: `sudo apt install build-essential`
  - macOS: `xcode-select --install`

### "RuntimeError: dictionary changed size during iteration"

**Cause:** Streamlit version incompatibility.

**Solutions:**
- Upgrade Streamlit: `pip install --upgrade streamlit`
- This error is rare and usually occurs with Streamlit < 1.30

---

## FAQ

**Q: Is this a trading bot?**

A: No. This is a research and analysis tool. It detects market regimes
and backtests a signal-generation strategy. It does not execute trades.
Any trading decisions based on this tool are at your own risk.

**Q: Can I use this for stocks, or only crypto?**

A: It works with any asset available on Yahoo Finance: stocks, crypto,
ETFs, indices, forex. Enter the appropriate ticker symbol.

**Q: How often should I re-run the analysis?**

A: The model uses the most recent data available at the time of the run.
For hourly data, re-running every few hours captures new bars. For daily
data, once per day is sufficient. The walk-forward backtest results do not
change unless the data or parameters change.

**Q: Why does the model sometimes pick different numbers of states?**

A: BIC model selection is data-dependent. Different lookback periods,
intervals, or assets may produce different optimal state counts. This is
expected and correct -- different datasets have different regime structures.

**Q: Can I save my settings?**

A: Edit config.yaml with your preferred defaults. The sidebar widgets will
initialize from these values. You cannot save sidebar adjustments to
config.yaml from within the app.

**Q: Why is the walk-forward backtest so much slower than the full-sample
analysis?**

A: The walk-forward backtest fits a fresh HMM for each fold. With default
settings, that is ~20 folds, each performing BIC selection with 140 model
fits. Total: ~2800 model fits vs 140 for full-sample.

**Q: What does a negative Sharpe ratio mean?**

A: The strategy lost money on a risk-adjusted basis. This means the HMM-
based signals did not provide an edge after accounting for risk and
transaction costs. Try different parameters or a different asset.

**Q: Why are position sizes so small?**

A: Half-Kelly sizing with entropy scaling produces conservative position
sizes by design. If the win rate is 55% and confidence is 70%, the Kelly
bet is small. This is intentional -- it prevents catastrophic losses from
model errors.

**Q: Can I disable short selling?**

A: Not directly from the sidebar. To disable shorts, you would need to
modify strategy.py to skip the bear_states logic in generate_signals().
A simpler workaround: set min_confirmations to 8 -- bear signals almost
never have all 8 bull-oriented confirmations met, so shorts will
effectively never trigger.

**Q: Does the backtest account for dividends?**

A: No. yfinance's Close prices are unadjusted by default. For assets
that pay dividends, use the `Adj Close` price or account for dividends
separately.

---

## Glossary

**ADX (Average Directional Index):**
A technical indicator measuring trend strength, regardless of direction.
ADX > 20 suggests a trending market; ADX < 20 suggests a ranging market.

**AIC (Akaike Information Criterion):**
A model selection criterion: AIC = -2*LL + 2*k. Less conservative than
BIC (penalizes complexity less), so it tends to select more complex models.

**Alpha:**
The excess return of a strategy compared to a benchmark (usually
buy-and-hold). Positive alpha means the strategy outperformed.

**Backtest:**
A simulation of how a trading strategy would have performed on historical
data.

**Baum-Welch Algorithm:**
The Expectation-Maximization algorithm specialized for Hidden Markov
Models. It iteratively estimates the model parameters (initial state
probabilities, transition matrix, emission distributions) by computing
expected state occupancies and updating parameters to maximize likelihood.

**BIC (Bayesian Information Criterion):**
A model selection criterion: BIC = -2*LL + k*ln(T). Penalizes model
complexity more heavily than AIC. Lower BIC = better model.

**Bootstrap:**
A statistical resampling method that estimates the sampling distribution
of a statistic by repeatedly resampling with replacement from the
observed data.

**Calmar Ratio:**
Annualized return divided by the absolute value of maximum drawdown.
Measures return per unit of worst-case loss.

**Confidence (Regime Confidence):**
A value from 0 to 1 derived from Shannon entropy of the posterior state
probabilities. High confidence = the model is certain about the current
regime.

**Confirmation:**
One of 8 technical conditions that must be met before a trading signal is
generated. Confirmations include RSI levels, momentum, volume, ADX, EMA
trend, and MACD.

**Cooldown:**
A waiting period after closing a position, during which no new positions
can be opened. Prevents overtrading.

**Covariance Type:**
The structure of the Gaussian emission covariance matrix in the HMM. Full
covariance captures feature correlations; diagonal assumes independence.

**CVaR (Conditional Value at Risk) / Expected Shortfall:**
The expected loss given that the loss exceeds VaR. Measures tail risk.
CVaR at 5% answers: "When things go badly (worst 5% of bars), how bad
do they get on average?"

**Drawdown:**
The decline from a peak in equity to a subsequent trough, expressed as
a percentage.

**EMA (Exponential Moving Average):**
A moving average that gives more weight to recent observations. EMA(50)
uses the last 50 bars with exponentially decaying weights.

**Emission Distribution:**
In an HMM, the probability distribution of observations given a hidden
state. This application uses multivariate Gaussian emissions.

**Entropy (Shannon Entropy):**
A measure of uncertainty or information content. High entropy = high
uncertainty. Measured in bits (log base 2).

**Half-Kelly:**
Betting half the amount recommended by the full Kelly criterion. Retains
~75% of optimal growth while reducing variance by ~50%.

**HMM (Hidden Markov Model):**
A statistical model where the system is assumed to be in one of several
hidden states that switch according to a Markov chain. Each state produces
observations from a probability distribution.

**Hysteresis:**
A filter that requires a regime to persist for a minimum number of bars
before it can trigger a signal change. Prevents whipsawing on regime
boundaries.

**Kelly Criterion:**
A formula for optimal bet sizing that maximizes the long-run geometric
growth rate of wealth: f* = (p*b - q) / b, where p = win probability,
b = win/loss ratio, q = 1-p.

**Log Return:**
The natural logarithm of the price ratio: ln(P_t / P_{t-1}). Preferred
over simple returns because they are additive, symmetric, and approximately
normally distributed.

**MACD (Moving Average Convergence Divergence):**
A momentum indicator showing the relationship between two exponential
moving averages. MACD = EMA(12) - EMA(26). The signal line is EMA(9)
of the MACD.

**Markov Property:**
The assumption that the future state depends only on the current state,
not the history of previous states.

**Regime:**
A persistent market state characterized by specific return, volatility,
and volume patterns. Examples: bull, bear, crash, neutral.

**RSI (Relative Strength Index):**
A momentum oscillator ranging from 0 to 100. RSI > 70 = overbought;
RSI < 30 = oversold.

**Sharpe Ratio:**
Risk-adjusted return: (mean return / standard deviation) * annualization
factor. Higher = better risk-adjusted performance.

**Sortino Ratio:**
Like the Sharpe ratio, but only penalizes downside volatility. Better
for strategies with asymmetric return distributions.

**Stationary Distribution:**
The long-run probability of being in each state. Represents the proportion
of time spent in each regime as the number of transitions approaches
infinity.

**Transition Matrix:**
A square matrix where entry [i, j] is the probability of transitioning
from state i to state j in one time step. Rows sum to 1.

**VaR (Value at Risk):**
The threshold return such that losses exceed it with a given probability
(e.g., 5%). "There is a 5% chance the return will be worse than VaR."

**Viterbi Algorithm:**
A dynamic programming algorithm that finds the most likely sequence of
hidden states given the observations and model parameters.

**Walk-Forward Validation:**
A backtesting methodology that trains the model only on past data and
tests on future data, advancing the window through time. Prevents
lookahead bias.

**Z-Score Standardization:**
Transforming data to have zero mean and unit standard deviation:
z = (x - mean) / std. Used to make features comparable regardless of
their natural scale.
