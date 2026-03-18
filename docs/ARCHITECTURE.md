# HMM Regime Terminal -- Architecture Document

## Table of Contents

1. [System Overview](#system-overview)
2. [Full Data Flow Pipeline](#full-data-flow-pipeline)
3. [Module Dependency Graph](#module-dependency-graph)
4. [Module Reference](#module-reference)
   - [config.yaml](#configyaml)
   - [data_loader.py](#data_loaderpy)
   - [hmm_engine.py](#hmm_enginepy)
   - [strategy.py](#strategypy)
   - [backtester.py](#backtesterpy)
   - [app.py](#apppy)
5. [Data Flow Diagrams](#data-flow-diagrams)
   - [Full Analysis Pipeline](#full-analysis-pipeline)
   - [Walk-Forward Loop](#walk-forward-loop)
   - [Signal Generation State Machine](#signal-generation-state-machine)
6. [Configuration Cascade](#configuration-cascade)
7. [State Management in Streamlit](#state-management-in-streamlit)
8. [Threading and Performance Model](#threading-and-performance-model)
9. [Error Handling Strategy](#error-handling-strategy)
10. [Data Integrity Guarantees](#data-integrity-guarantees)
11. [Extension Points](#extension-points)

---

## System Overview

The HMM Regime Terminal is a six-module Python application that detects market
regimes using Hidden Markov Models, generates multi-confirmation trading
signals, and evaluates them through walk-forward backtesting with bootstrap
confidence intervals. The system is designed around three core principles:

1. **No lookahead bias.** Every standardization, model fit, and signal
   generation step uses only data available at decision time. Walk-forward
   backtesting enforces this structurally.

2. **Data-driven model selection.** The number of hidden states is chosen by
   BIC, not guessed. Multiple random restarts mitigate local optima in EM.

3. **Multi-layered signal gating.** A regime classification alone is never
   sufficient to trigger a trade. Eight independent technical confirmations,
   a confidence threshold, a hysteresis filter, and a cooldown period must
   all agree before a position is opened.

### Technology Stack

```
Layer               Technology              Role
---------------------------------------------------------------------------
Data source         yfinance                OHLCV retrieval from Yahoo Finance
HMM fitting         hmmlearn                Baum-Welch EM, Viterbi, forward-backward
Technical analysis  ta                      RSI, ADX, EMA, MACD indicators
Numerical compute   numpy, scipy, pandas    Array math, eigendecomposition, DataFrames
Visualization       plotly                  Interactive charts and heatmaps
Dashboard           streamlit               Reactive web UI with sidebar controls
Configuration       pyyaml                  YAML parsing for config.yaml
```

### ASCII System Diagram

```
                        +------------------+
                        |   config.yaml    |
                        |  (all defaults)  |
                        +--------+---------+
                                 |
                     +-----------+-----------+
                     |                       |
                     v                       v
            +----------------+      +----------------+
            |   Streamlit    |      |   Streamlit    |
            |   Sidebar      |      |   @cache_data  |
            |   (overrides)  |      |   load_config  |
            +-------+--------+      +-------+--------+
                    |                        |
                    +----------+-------------+
                               |
                               v
                    +---------------------+
                    |   build_config()    |
                    |   (merged runtime   |
                    |    config dict)     |
                    +----------+----------+
                               |
              +----------------+------------------+
              |                                   |
              v                                   v
     +------------------+              +--------------------+
     | FULL-SAMPLE PATH |              | WALK-FORWARD PATH  |
     | (Tabs 1, 2, 5)   |              | (Tabs 3, 4)        |
     +--------+---------+              +---------+----------+
              |                                   |
              v                                   v
     +------------------+              +--------------------+
     |  data_loader     |              |  backtester        |
     |  fetch_ohlcv()   |              |  WalkForwardBack-  |
     |  compute_feats() |              |  tester.run()      |
     |  standardize()   |              |  (orchestrates     |
     +--------+---------+              |   data_loader,     |
              |                        |   hmm_engine,      |
              v                        |   strategy for     |
     +------------------+              |   each fold)       |
     |  hmm_engine      |              +---------+----------+
     |  RegimeDetector  |                        |
     |  fit_and_select()|              +---------+----------+
     |  decode()        |              |                    |
     |  label_regimes() |              v                    v
     |  entropy()       |      +--------------+   +----------------+
     |  rolling_ll()    |      | equity curve |   | trade records  |
     +--------+---------+      | metrics      |   | (TradeRecord)  |
              |                | bootstrap CI |   +----------------+
              v                +--------------+
     +------------------+
     |  strategy        |
     |  SignalGenerator  |
     |  confirmations() |
     |  signals()       |
     |  position_size() |
     +--------+---------+
              |
              v
     +------------------+
     |  Plotly charts   |
     |  across 5 tabs   |
     +------------------+
```

---

## Full Data Flow Pipeline

The system has two execution paths that share the same data acquisition and
feature engineering steps but diverge afterward:

```
User clicks "Run Analysis"
         |
         v
+----------------------------------------------------------+
| STEP 1: DATA ACQUISITION                                 |
|   fetch_ohlcv(ticker, interval, lookback_days)            |
|   Input:  ticker string, interval string, int days        |
|   Output: DataFrame [Open, High, Low, Close, Volume]      |
|   Source: Yahoo Finance via yfinance                      |
|   Notes:  Lookback silently clamped to yfinance limits    |
|           MultiIndex columns flattened if present         |
+----------------------------+-----------------------------+
                             |
                             v
+----------------------------------------------------------+
| STEP 2: FEATURE ENGINEERING                               |
|   compute_features(df, config["data"])                    |
|   Input:  OHLCV DataFrame, data config dict               |
|   Output: DataFrame with 5 added columns:                 |
|     log_return    = ln(Close_t / Close_{t-1})             |
|     rolling_vol   = std(log_return, window=21)            |
|     volume_change = Volume / SMA(Volume,20) - 1           |
|     intraday_range= (High - Low) / Close                  |
|     rsi           = RSI(14)                               |
|   Notes:  NaN rows from rolling windows are dropped       |
|           Index is reset (DatetimeIndex -> "Date" column)  |
+----------------------------+-----------------------------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
+---------------------------+   +---------------------------+
| PATH A: FULL-SAMPLE       |   | PATH B: WALK-FORWARD     |
| (Tabs 1, 2, 5)            |   | (Tabs 3, 4)              |
|                            |   |                          |
| standardize(all_data)      |   | for each fold:           |
|   -> Z-score all at once   |   |   split train/test       |
|                            |   |   standardize(train,test)|
| fit_and_select(X)          |   |   fit_and_select(X_train)|
| decode(X)                  |   |   decode(X_test)         |
| label_regimes(X)           |   |   label_regimes(X_test)  |
| shannon_entropy(post)      |   |   shannon_entropy(post)  |
| regime_statistics()        |   |   compute_confirmations()|
| log_likelihood_series(X)   |   |   generate_signals()     |
|                            |   |   compute_position_size()|
| compute_confirmations(df)  |   |   advance window         |
| generate_signals(...)      |   |                          |
|                            |   | simulate_trades(...)     |
| -> regime overlay          |   | compute_metrics(...)     |
| -> transition matrix       |   | bootstrap_ci(...)        |
| -> BIC/AIC curve           |   |                          |
| -> current signal          |   | -> equity curve          |
| -> diagnostics             |   | -> trade log             |
+---------------------------+   +---------------------------+
```

**Why two paths?** The full-sample path gives the richest visualization of
regime structure but is in-sample (the model has seen all data). The
walk-forward path provides unbiased out-of-sample performance estimates.
Comparing the two reveals overfitting: if full-sample looks great but
walk-forward is poor, the model is memorizing noise.

---

## Module Dependency Graph

```
                    config.yaml
                        |
                        | (read by app.py, passed as dict to all modules)
                        |
  +---------------------+---------------------+
  |                     |                      |
  v                     v                      v
data_loader.py     hmm_engine.py         strategy.py
  (no internal       (no internal          (no internal
   imports)           imports)              imports)
  |                     |                      |
  |   External:         |   External:          |   External:
  |   yfinance          |   hmmlearn           |   ta
  |   ta.momentum       |   numpy              |   numpy
  |   numpy             |   pandas             |   pandas
  |   pandas            |   scipy              |
  |                     |                      |
  +----------+----------+----------+-----------+
             |                     |
             v                     v
         backtester.py
           imports: data_loader, hmm_engine, strategy
           External: numpy, pandas
                        |
                        v
                     app.py
           imports: data_loader, hmm_engine, strategy, backtester
           External: streamlit, plotly, yaml, numpy, pandas
```

### Why Dependencies Flow One Direction

The dependency graph is a strict DAG (directed acyclic graph). Each module
has a well-defined responsibility and depends only on modules "below" it:

- **data_loader.py** is a pure data module. It fetches and transforms data.
  It has no knowledge of HMMs, signals, or backtesting. Any project that
  needs OHLCV data with technical features can import it independently.

- **hmm_engine.py** is a pure modeling module. It fits HMMs and extracts
  regime information. It has no knowledge of trading signals, position sizing,
  or backtesting. It does not import data_loader because it receives its
  input as numpy arrays -- the caller is responsible for data preparation.

- **strategy.py** is a pure signal module. It evaluates conditions and
  produces signals. It does not import hmm_engine because it receives regime
  states and posteriors as numpy arrays. This separation allows the signal
  generation logic to work with any regime classifier, not just HMMs.

- **backtester.py** is the orchestrator. It is the only module that imports
  the three modules above. It coordinates the train-standardize-fit-decode-
  signal-simulate pipeline within each walk-forward fold. This concentration
  of orchestration logic in one module means that the individual components
  remain testable in isolation.

- **app.py** is the presentation layer. It imports everything and adds UI
  chrome. It delegates all computation to the other modules and focuses on
  layout, widget binding, and chart construction.

This architecture means:
- You can test hmm_engine without data_loader (pass synthetic numpy arrays)
- You can test strategy without hmm_engine (pass synthetic state sequences)
- You can test backtester end-to-end with a small synthetic DataFrame
- Changes to visualization do not affect computation
- Changes to signal logic do not affect model fitting

---

## Module Reference

### config.yaml

**Role:** Single source of truth for all tunable parameters.

**Structure:**

```
config.yaml
  +-- data
  |     +-- default_ticker: str         "BTC-USD"
  |     +-- default_interval: str       "1h"
  |     +-- default_lookback_days: int   90
  |     +-- features: list[str]         [log_return, rolling_vol, ...]
  |     +-- rolling_vol_window: int      21
  |     +-- rsi_period: int              14
  |
  +-- hmm
  |     +-- min_states: int              2
  |     +-- max_states: int              8
  |     +-- n_restarts: int              20
  |     +-- n_iter: int                  200
  |     +-- tol: float                   1e-4
  |     +-- model_type: str              "gaussian"
  |     +-- covariance_type: str         "full"
  |     +-- gmm_n_mix: int               2
  |
  +-- strategy
  |     +-- confirmations
  |     |     +-- rsi_oversold: int      30
  |     |     +-- rsi_overbought: int    70
  |     |     +-- momentum_window: int   10
  |     |     +-- vol_low_pct: int       20
  |     |     +-- vol_high_pct: int      80
  |     |     +-- volume_threshold: float 1.2
  |     |     +-- adx_threshold: int     20
  |     |     +-- ema_period: int        50
  |     |     +-- macd_fast: int         12
  |     |     +-- macd_slow: int         26
  |     |     +-- macd_signal: int       9
  |     |     +-- min_confidence: float  0.6
  |     +-- min_confirmations: int       4
  |     +-- cooldown_bars: int           5
  |     +-- min_hold_bars: int           10
  |     +-- hysteresis_bars: int         3
  |
  +-- risk
  |     +-- use_kelly: bool              true
  |     +-- kelly_fraction: float        0.5
  |     +-- use_entropy_scaling: bool    true
  |     +-- max_leverage: float          2.0
  |     +-- max_position_pct: float      1.0
  |     +-- stop_loss_pct: float         0.05
  |     +-- take_profit_pct: float       0.15
  |
  +-- backtest
        +-- train_window_bars: int       500
        +-- test_window_bars: int        100
        +-- step_bars: int               50
        +-- initial_capital: int         100000
        +-- commission_pct: float        0.001
        +-- slippage_pct: float          0.0005
        +-- bootstrap_samples: int       1000
        +-- bootstrap_ci: float          0.90
        +-- benchmark_ticker: null
```

**Design decisions:**

- All parameters have sensible defaults. The app can run with zero
  configuration changes.
- The sidebar exposes the most commonly tuned parameters. Less common
  parameters (kelly_fraction, commission_pct, rsi_period) require editing
  the YAML file.
- The config dict is passed by value to each module constructor. Modules
  extract only the keys they need, making them resilient to config additions.

---

### data_loader.py

**Role:** Data acquisition and feature engineering. No model logic.

**Public API:**

```python
fetch_ohlcv(ticker: str, interval: str, lookback_days: int) -> pd.DataFrame
```

Downloads OHLCV data from Yahoo Finance. Handles:
- Automatic lookback clamping based on yfinance interval limits (1m: 7d,
  5m/15m/30m: 60d, 1h: 730d, 1d+: effectively unlimited)
- MultiIndex column flattening (yfinance version inconsistency)
- Empty data validation with descriptive error message
- Column subsetting to [Open, High, Low, Close, Volume]
- NaN row removal

```python
compute_features(df: pd.DataFrame, config: dict) -> pd.DataFrame
```

Adds five feature columns to an OHLCV DataFrame:

| Feature         | Formula                           | Why this feature                                    |
|-----------------|-----------------------------------|-----------------------------------------------------|
| log_return      | ln(Close_t / Close_{t-1})         | Stationary, additive, approx normal for Gaussian HMM|
| rolling_vol     | std(log_return, window=21)        | Captures volatility clustering, key regime separator |
| volume_change   | Volume / SMA(Volume,20) - 1       | Scale-free relative volume; regime-independent units |
| intraday_range  | (High - Low) / Close              | Realized intra-bar volatility                       |
| rsi             | RSI(14) via ta library            | Mean-reversion signal, bounded [0,100]              |

Drops NaN rows introduced by rolling window calculations and resets the index
so that the DatetimeIndex becomes a "Date" column (preserving it for charting).

```python
standardize(train: DataFrame, test: DataFrame | None, cols: list[str] | None)
    -> (train_z, test_z, stats)
```

Z-score normalization using training set statistics only. For each feature
column c:
- Computes mu = train[c].mean() and sigma = train[c].std()
- Transforms: x_z = (x - mu) / sigma
- Handles sigma=0 edge case by setting sigma=1 (prevents division by zero
  for constant features)
- Returns the stats dict {col: {mean, std}} for downstream inspection

This is the primary anti-lookahead mechanism in the data pipeline. The test
set is always standardized using training statistics, never its own.

```python
get_feature_matrix(df: DataFrame, cols: list[str] | None) -> np.ndarray
```

Trivial extraction: returns df[cols].values as a 2D numpy array shaped
(T, d) for hmmlearn's fit/predict/score API.

**Internal design:**

The module is entirely stateless -- all functions are pure. There is no class
because there is no state to encapsulate. The config dict is read-only and
only used for rolling_vol_window and rsi_period defaults.

---

### hmm_engine.py

**Role:** HMM fitting, BIC model selection, regime decoding and analysis.
The mathematical core of the system.

**Class: RegimeDetector**

Constructor accepts the full config dict and extracts the `hmm` section.
Stores model parameters and initializes empty result containers.

**Instance state after fitting:**
- `self.model`: the selected GaussianHMM instance (best BIC)
- `self.n_states`: number of components in the selected model
- `self.bic_scores`: dict {n_states: bic_value}
- `self.aic_scores`: dict {n_states: aic_value}
- `self.labels`: dict {state_id: label_string}
- `self._regime_stats`: DataFrame of per-regime statistics

**Public API:**

```python
fit_and_select(X_train: ndarray) -> dict[int, float]
```

The core model selection loop. For each candidate state count n from
min_states to max_states:
1. Run n_restarts independent fits with different random seeds (0..19)
2. Each fit runs Baum-Welch EM for up to n_iter iterations or until
   convergence (tol=1e-4)
3. Keep the model with highest log-likelihood for this n
4. Compute BIC = -2*LL + k*ln(T) where k = _count_params(n, d)
5. Keep the model with lowest BIC across all n

Returns the bic_scores dict for visualization.

Total fits per call: (max_states - min_states + 1) * n_restarts.
Default: 7 * 20 = 140 fits. This is the dominant computational cost.

```python
decode(X: ndarray) -> (states: ndarray, posteriors: ndarray)
```

Two-pass decoding:
1. `model.predict(X)` runs the Viterbi algorithm, returning the single
   most likely state sequence (shape: T,)
2. `model.predict_proba(X)` runs the forward-backward algorithm, returning
   posterior probabilities P(S_t=i | X_{1:T}) (shape: T x n_states)

These serve different purposes: Viterbi states are used for regime labeling
and signal generation (discrete decisions). Posteriors are used for entropy
computation and confidence scoring (continuous uncertainty).

```python
label_regimes(X: ndarray) -> dict[int, str]
```

Assigns semantic labels to states by sorting on mean log-return (column 0
of model.means_). The label mapping depends on the number of states:

| n_states | Labels (lowest return to highest)                    |
|----------|------------------------------------------------------|
| 2        | bear, bull                                           |
| 3        | bear, neutral, bull                                  |
| 4        | crash, bear, bull, bull_run                           |
| 5        | crash, bear, neutral, bull, bull_run                  |
| 6+       | crash, regime_1, ..., regime_{n-2}, bull_run          |

```python
regime_statistics() -> pd.DataFrame
```

Extracts per-regime statistics from the fitted model:
- mean_return: model.means_[i, 0]
- volatility: sqrt(covars_[i][0,0]) (adjusted for covariance_type)
- expected_duration: 1 / (1 - transmat_[i,i])
- stationary_weight: left eigenvector of transmat_ at eigenvalue 1

The stationary distribution is computed via eigendecomposition of
transmat_.T. The eigenvector corresponding to the eigenvalue closest
to 1.0 is extracted, normalized to sum to 1, and taken as the
stationary distribution.

```python
shannon_entropy(posteriors: ndarray) -> (entropy: ndarray, confidence: ndarray)
```

Per-bar entropy: H_t = -sum(p_i * log2(p_i)) with epsilon clipping
to prevent log(0). Normalized confidence: 1 - H/log2(n_states).

```python
transition_matrix() -> ndarray
```

Returns a copy of model.transmat_ (prevents mutation of the fitted model).

```python
log_likelihood_series(X: ndarray, window: int = 50) -> ndarray
```

Computes model.score(X[t-window:t]) / window for each bar from t=window
to T. Returns NaN for the first window-1 bars. This detects model
degradation: a declining trend means the market structure has shifted.

**Internal design:**

The _count_params method encodes the parameter counting rules for each
covariance type:
- full: n * d*(d+1)/2 (upper triangle of symmetric matrix)
- diag: n * d (diagonal entries only)
- spherical: n (one scalar per state)
- tied: d*(d+1)/2 (one matrix shared across all states)

Plus (n-1) initial state parameters, n*(n-1) transition parameters,
and n*d mean parameters.

All warnings from hmmlearn are suppressed during fitting (ConvergenceWarning
is common with limited data). Exceptions during individual fits are caught
silently -- if a particular seed/state-count combination fails, it is
skipped. Only if ALL fits for ALL state counts fail does the method raise
RuntimeError.

---

### strategy.py

**Role:** Signal generation with multi-confirmation gating, hysteresis
filtering, and Kelly criterion position sizing.

**Class: SignalGenerator**

Constructor reads strategy.confirmations, strategy.*, and risk.* sections
from the config dict.

**Public API:**

```python
compute_confirmations(df: DataFrame) -> DataFrame
```

Evaluates 8 independent boolean conditions on the OHLCV + feature DataFrame:

| # | Column                   | Condition                | Rationale                        |
|---|--------------------------|--------------------------|----------------------------------|
| 1 | conf_rsi_not_overbought  | RSI < 70                 | Avoid entering longs at tops     |
| 2 | conf_rsi_not_oversold    | RSI > 30                 | Avoid catching falling knives    |
| 3 | conf_momentum            | Close > Close[N ago]     | Price trending upward            |
| 4 | conf_vol_range           | Vol in [20th, 80th] pctl | Avoid extreme vol environments   |
| 5 | conf_volume              | Volume/SMA20 > 1.2       | Confirm with participation       |
| 6 | conf_adx                 | ADX > 20                 | Market is trending, not ranging  |
| 7 | conf_above_ema           | Close > EMA(50)          | Intermediate uptrend             |
| 8 | conf_macd                | MACD > Signal            | Bullish momentum crossover       |

Adds `n_confirmations` = sum of all 8 booleans. All columns prefixed with
`conf_` are automatically discovered and summed.

```python
generate_signals(df, states, posteriors, labels, confidence) -> Series
```

Bar-by-bar state machine producing signal values: 1 (long), -1 (short),
0 (flat). See the Signal Generation State Machine section below for the
detailed flowchart.

```python
compute_position_size(confidence, win_rate, avg_win, avg_loss) -> float
```

Three-stage sizing:
1. Kelly criterion: f* = (p*b - q) / b
2. Fractional Kelly: size = f* * kelly_fraction (default 0.5)
3. Entropy scaling: size *= confidence (when enabled)
4. Cap: min(size, max_leverage * max_position_pct)

---

### backtester.py

**Role:** Walk-forward orchestration, trade simulation, performance
metrics, bootstrap confidence intervals.

**Data structures:**

```python
@dataclass
class TradeRecord:
    entry_bar: int          # bar index of entry
    exit_bar: int           # bar index of exit
    entry_price: float      # after slippage
    exit_price: float       # after slippage
    direction: int          # 1 = long, -1 = short
    pnl: float              # dollar P&L
    pnl_pct: float          # percentage P&L
    regime: str             # regime at entry
    n_confirmations: int    # confirmations at entry
    position_size: float    # Kelly-scaled size

@dataclass
class BacktestResult:
    equity_curve: pd.Series       # capital over time
    benchmark_curve: pd.Series    # buy-and-hold over time
    trades: list[TradeRecord]     # all round-trip trades
    metrics: dict                 # performance metrics
    regime_series: pd.Series      # regime label per bar
    confidence_series: pd.Series  # entropy confidence per bar
    ci_lower: dict                # bootstrap CI lower bounds
    ci_upper: dict                # bootstrap CI upper bounds
```

**Class: WalkForwardBacktester**

```python
run(df: DataFrame) -> BacktestResult
```

The walk-forward loop (detailed in the Walk-Forward Loop section below).
Validates that the DataFrame has sufficient bars (train_window + test_window).
Aggregates signals, sizes, regimes, and confidence from all folds.

```python
simulate_trades(df, signals, sizes) -> (equity: Series, trades: list)
```

Bar-by-bar trade simulation. See the Implementation Guide for details on
mark-to-market calculation, slippage, and commission modeling.

```python
compute_metrics(equity, trades, benchmark) -> dict
```

Computes 11 performance metrics:
- total_return: cumulative return over backtest
- sharpe_ratio: annualized (assumes hourly bars, ann_factor = sqrt(8760))
- sortino_ratio: annualized, penalizes only downside volatility
- calmar_ratio: annualized return / |max drawdown|
- max_drawdown: deepest peak-to-trough decline
- max_dd_duration: longest consecutive bars in drawdown
- cvar_5pct: expected loss in worst 5% of bars
- win_rate: fraction of profitable trades
- profit_factor: gross profit / gross loss
- alpha: strategy return minus benchmark return
- n_trades: total completed round-trip trades

```python
bootstrap_confidence_intervals(equity) -> (ci_lower, ci_upper)
```

Non-parametric bootstrap with 1000 samples. Resamples bar returns with
replacement, reconstructs synthetic equity curves, computes metrics on
each, and takes quantiles for the 90% CI.

---

### app.py

**Role:** Streamlit dashboard -- user interface, parameter controls,
visualization. No business logic of its own.

**Page configuration:**
- Title: "HMM Regime Terminal"
- Layout: wide mode (uses full browser width)

**Sidebar controls:** 21 interactive widgets organized in 5 sections
(Data, HMM, Strategy, Risk, Backtest) plus a "Run Analysis" button.

**Runtime config construction:** `build_config()` merges base_config
(from YAML) with sidebar widget values using dict spread syntax:
```python
cfg["hmm"] = {**base_config["hmm"], "min_states": min_states, ...}
```

**Tab structure:**

| Tab | Visualizations                                                    |
|-----|-------------------------------------------------------------------|
| 1   | Regime banner, confidence metric, signal metric, position size,   |
|     | confirmation breakdown table, total confirmations counter         |
| 2   | Price chart with regime-colored scatter overlay, transition        |
|     | heatmap (plotly imshow), regime statistics table, return           |
|     | distribution histograms, BIC/AIC line chart                       |
| 3   | 11 metric cards with CI tooltips, equity vs benchmark line chart,  |
|     | drawdown filled area chart                                        |
| 4   | Sortable trade log table                                          |
| 5   | Rolling log-likelihood line chart, dual-axis entropy/confidence    |
|     | chart, feature correlation heatmap                                |

**Regime color map:**

```python
REGIME_COLORS = {
    "crash":    "#d32f2f",  # Material red
    "bear":     "#f57c00",  # Material orange
    "neutral":  "#9e9e9e",  # Material gray
    "bull":     "#388e3c",  # Material green
    "bull_run": "#1565c0",  # Material blue
    "unknown":  "#e0e0e0",  # Light gray
}
```

---

## Data Flow Diagrams

### Full Analysis Pipeline

When the user clicks "Run Analysis", app.py executes the following sequence:

```
[1] fetch_ohlcv(ticker, interval, lookback)
     |
     v
    OHLCV DataFrame (T rows x 5 cols)
     |
[2] compute_features(df, config["data"])
     |
     v
    Feature DataFrame (T' rows x 10 cols, T' < T due to NaN drop)
     |
     +---> [3a] Full-sample path
     |      |
     |      v
     |     standardize(all_data, None, feature_cols)
     |      |
     |      v
     |     get_feature_matrix(train_z, feature_cols)
     |      |
     |      v
     |     X: ndarray (T', 5)
     |      |
     |      v
     |     RegimeDetector(config)
     |      |
     |      v
     |     fit_and_select(X)  -->  bic_scores, aic_scores
     |      |
     |      v
     |     decode(X)          -->  states (T',), posteriors (T', n)
     |      |
     |      v
     |     label_regimes(X)   -->  labels {int: str}
     |      |
     |      v
     |     shannon_entropy(posteriors)  -->  entropy (T',), confidence (T',)
     |      |
     |      v
     |     regime_statistics()  -->  DataFrame (n rows x 6 cols)
     |      |
     |      v
     |     log_likelihood_series(X, 50)  -->  ll_series (T',)
     |      |
     |      v
     |     SignalGenerator(config)
     |      |
     |      v
     |     compute_confirmations(df)  -->  df_conf with 8 bool cols
     |      |
     |      v
     |     generate_signals(df_conf, states, posteriors, labels, confidence)
     |      |
     |      v
     |     signals (T',)
     |      |
     |      v
     |     [Render Tabs 1, 2, 5]
     |
     +---> [3b] Walk-forward path
            |
            v
           WalkForwardBacktester(config)
            |
            v
           run(df)  -->  BacktestResult
            |
            v
           [Render Tabs 3, 4]
```

### Walk-Forward Loop

```
Bar indices:  0          500         600   650         750   800
              |-----------|-----------|     |-----------|     |
Fold 1:       [  TRAIN (500 bars)   ][TEST (100 bars)]
Fold 2:            [  TRAIN (500 bars)   ][TEST (100 bars)]
                   ^ start shifted by step_bars = 50
Fold 3:                 [  TRAIN (500 bars)   ][TEST (100 bars)]
  ...

Within each fold:

  +---------------------------------------------------+
  | train_df = df[start : start + train_window]       |
  | test_df  = df[start + train_window :              |
  |               start + train_window + test_window]  |
  +---------------------------------------------------+
              |
              v
  +---------------------------------------------------+
  | standardize(train_df, test_df, feature_cols)      |
  |   mean, std computed from train_df ONLY            |
  |   applied to both train_df and test_df             |
  +---------------------------------------------------+
              |
              v
  +---------------------------------------------------+
  | X_train = get_feature_matrix(train_z)             |
  | X_test  = get_feature_matrix(test_z)              |
  +---------------------------------------------------+
              |
              v
  +---------------------------------------------------+
  | detector = RegimeDetector(config)     # FRESH      |
  | detector.fit_and_select(X_train)      # FIT        |
  +---------------------------------------------------+
              |
              v
  +---------------------------------------------------+
  | states, posteriors = detector.decode(X_test)       |
  | labels = detector.label_regimes(X_test)            |
  | entropy, confidence = detector.shannon_entropy()   |
  +---------------------------------------------------+
              |
              v
  +---------------------------------------------------+
  | sig_gen = SignalGenerator(config)                  |
  | test_conf = sig_gen.compute_confirmations(test_df) |
  | signals = sig_gen.generate_signals(...)            |
  | sizes = [sig_gen.compute_position_size(c) for c]   |
  +---------------------------------------------------+
              |
              v
  +---------------------------------------------------+
  | Store signals, sizes, regimes, confidence for      |
  | test_df indices in accumulator arrays              |
  +---------------------------------------------------+
              |
              v
  +---------------------------------------------------+
  | start += step_bars                                 |
  | Continue to next fold                              |
  +---------------------------------------------------+

After all folds:

  +---------------------------------------------------+
  | simulate_trades(df, all_signals, all_sizes)        |
  |   -> equity curve, trade records                   |
  +---------------------------------------------------+
              |
              v
  +---------------------------------------------------+
  | compute_metrics(equity, trades, benchmark)         |
  |   -> 11 performance metrics                        |
  +---------------------------------------------------+
              |
              v
  +---------------------------------------------------+
  | bootstrap_confidence_intervals(equity)             |
  |   -> ci_lower, ci_upper dicts                      |
  +---------------------------------------------------+
```

### Signal Generation State Machine

The `generate_signals()` method implements a bar-by-bar state machine.
At each bar t, the following checks occur in priority order:

```
                        Bar t
                          |
                          v
              +------------------------+
              | Is position != 0 AND   |     YES
              | (t - position_start)   |---------> signals[t] = current_pos
              | < min_hold_bars?       |           (maintain position,
              +------------------------+            cannot exit yet)
                     | NO
                     v
              +------------------------+
              | Is current_pos == 0    |     YES
              | AND (t - last_signal)  |---------> signals[t] = 0
              | < cooldown_bars?       |           (in cooldown,
              +------------------------+            stay flat)
                     | NO
                     v
              +------------------------+
              | Is regime_persist[t]   |     YES
              | < hysteresis_bars?     |---------> signals[t] = current_pos
              |                        |           (regime not stable,
              +------------------------+            maintain position)
                     | NO
                     v
              +------------------------+
              | Is confidence[t]       |     YES
              | < min_confidence?      |---------> signals[t] = current_pos
              |                        |           (model too uncertain,
              +------------------------+            maintain position)
                     | NO
                     v
              +------------------------+
              | state in bull_states   |     YES
              | AND n_conf >=          |---------> new_signal = +1 (LONG)
              | min_confirmations?     |
              +------------------------+
                     | NO
                     v
              +------------------------+
              | state in bear_states   |     YES
              | AND n_conf >=          |---------> new_signal = -1 (SHORT)
              | min_confirmations?     |
              +------------------------+
                     | NO
                     v
              new_signal = 0 (FLAT)
                     |
                     v
              +------------------------+
              | new_signal !=          |     YES
              | current_pos?           |---------> Update current_pos,
              |                        |           last_signal_bar,
              +------------------------+           position_start
                     | NO
                     v
              signals[t] = current_pos (no change)
```

**Key state variables:**
- `current_pos`: current position direction (1, -1, or 0)
- `last_signal_bar`: bar index of most recent signal change
- `position_start`: bar index when current position was opened
- `regime_persist[t]`: consecutive bars the current regime has been active

**Regime persistence computation:**
```
regime_persist[0] = 0
for t in 1..T:
    if states[t] == states[t-1]:
        regime_persist[t] = regime_persist[t-1] + 1
    else:
        regime_persist[t] = 0
```

---

## Configuration Cascade

Parameters flow through a three-level cascade:

```
Level 1: config.yaml (file on disk)
  |
  | Loaded once at app startup via @st.cache_data
  | Stored as base_config
  |
  v
Level 2: Streamlit sidebar widgets
  |
  | Widget defaults are set FROM base_config values
  | User adjustments override base_config values
  | Only a subset of config parameters are exposed
  |
  v
Level 3: Runtime config dict (build_config())
  |
  | Merges base_config with widget values via dict spread:
  |   cfg["hmm"] = {**base_config["hmm"], "min_states": widget_val, ...}
  |
  | This means:
  |   - Widget-exposed params use widget values
  |   - Non-widget params retain config.yaml values
  |   - The merge is shallow (one level deep)
  |
  v
Passed to module constructors:
  RegimeDetector(config)   reads config["hmm"]
  SignalGenerator(config)  reads config["strategy"], config["risk"]
  WalkForwardBacktester(config)  reads config["backtest"], config["data"]
```

**Parameters exposed in sidebar vs config.yaml only:**

| Sidebar-exposed                          | Config.yaml only                     |
|------------------------------------------|--------------------------------------|
| ticker, interval, lookback               | features list, rolling_vol_window,   |
| min_states, max_states, model_type,      | rsi_period, n_iter, tol, gmm_n_mix   |
| covariance_type, n_restarts              |                                      |
| min_confirmations, cooldown_bars,        | All 11 confirmation thresholds,      |
| min_hold_bars, min_confidence            | hysteresis_bars                      |
| use_kelly, use_entropy_scaling,          | kelly_fraction, max_position_pct,    |
| max_leverage                             | stop_loss_pct, take_profit_pct       |
| train_window, test_window, step_bars     | initial_capital, commission_pct,     |
|                                          | slippage_pct, bootstrap_samples,     |
|                                          | bootstrap_ci, benchmark_ticker       |

---

## State Management in Streamlit

### Execution Model

Streamlit uses a top-down rerun model: the entire app.py script is
re-executed on every user interaction (slider change, button click, etc.).
This means:

1. All sidebar widgets are re-rendered on every run and return their
   current values as local variables.
2. The `run_btn` variable is True only on the specific run triggered by
   clicking "Run Analysis". On subsequent interactions (e.g., tab
   switching), run_btn is False and no analysis code executes.
3. There is no persistent state between runs. Each analysis is
   self-contained.

### Caching Strategy

The application uses one cache:

```python
@st.cache_data
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)
```

This caches the config.yaml parse so it is only read from disk once per
session (or until Streamlit detects file modification). The cache key is
implicit -- since the function has no arguments, it caches a single value.

**What is NOT cached:**
- Data fetching (fetch_ohlcv) -- intentionally not cached because users
  may want fresh data, and yfinance returns change as new bars arrive.
- HMM fitting -- not cached because the model depends on both data and
  config parameters, making cache invalidation complex.
- Signal generation and backtesting -- depend on all upstream results.

**Potential caching improvements:**
- Cache fetch_ohlcv with (ticker, interval, lookback) as key, with a TTL
  matching the bar interval (e.g., 1 hour for hourly data).
- Cache HMM fitting with a hash of the feature matrix and HMM config as
  key. This would avoid re-fitting when only strategy/risk parameters change.

### Widget State

Sidebar widget values are stored in Streamlit's session state automatically.
They persist across tab switches but NOT across browser refreshes. The
initial value of each widget comes from base_config, so config.yaml
effectively provides the "factory reset" values.

---

## Threading and Performance Model

### What is Slow

The dominant computational cost is HMM fitting:

```
Total fitting time = n_state_candidates * n_restarts * time_per_fit

Where:
  n_state_candidates = max_states - min_states + 1  (default: 7)
  n_restarts        = 20 (default)
  time_per_fit      = O(T * n^2 * d^2 * n_iter)

Default: 7 * 20 = 140 model fits
```

For a typical run (2000 bars, 5 features, up to 8 states):
- Full-sample HMM fitting: 20-120 seconds
- Walk-forward backtest: multiplied by n_folds (10-30x)
  - Each fold fits a fresh HMM
  - Total: 2-10 minutes

### What is Fast

| Operation                  | Typical Time   | Notes                          |
|----------------------------|---------------|--------------------------------|
| Data fetch (yfinance)      | 1-3 seconds   | Network-bound                  |
| Feature engineering        | < 100ms       | Vectorized numpy/pandas        |
| Standardization            | < 10ms        | Simple arithmetic              |
| Viterbi decoding           | < 10ms        | O(T * n^2)                     |
| Forward-backward           | < 10ms        | O(T * n^2)                     |
| Signal generation          | < 50ms        | Python loop but T is small     |
| Trade simulation           | < 50ms        | Python loop                    |
| Metric computation         | < 10ms        | Vectorized                     |
| Bootstrap CIs              | 1-5 seconds   | 1000 resamples, vectorized     |
| Chart rendering            | < 500ms       | Plotly JSON generation         |

### What Could Be Parallelized

1. **HMM fits across state counts.** The 7 candidate state counts are
   independent and could be fitted in parallel using multiprocessing.
   This would give a near-linear speedup (7x) on multi-core machines.

2. **HMM fits across random seeds.** The 20 restarts for each state count
   are independent. Combined with state-count parallelism, this gives
   140-way parallelism, though the overhead of process creation limits
   practical speedup.

3. **Walk-forward folds.** Each fold is independent once the data split
   is determined. However, the overlapping windows mean folds share data,
   so memory usage would increase with parallelism.

4. **Bootstrap samples.** The 1000 bootstrap iterations are embarrassingly
   parallel and already use vectorized numpy operations.

Currently, the application is entirely single-threaded. Streamlit itself
runs in a single Python process. Parallelizing HMM fits would require
multiprocessing (not threading, due to the GIL).

### Memory Usage Estimates

| Component                    | Formula                         | Typical    |
|------------------------------|---------------------------------|------------|
| OHLCV DataFrame              | T * 5 * 8 bytes                 | ~80 KB     |
| Feature DataFrame             | T * 10 * 8 bytes                | ~160 KB    |
| Feature matrix (numpy)       | T * 5 * 8 bytes                 | ~80 KB     |
| Posteriors                   | T * n * 8 bytes                 | ~48 KB     |
| HMM model parameters         | n^2 * d^2 * 8 bytes             | ~2 KB      |
| Equity curve                 | T * 8 bytes                     | ~16 KB     |
| Trade records                | n_trades * ~200 bytes           | ~10 KB     |
| Bootstrap workspace          | n_samples * T * 8 bytes         | ~16 MB     |
| **Total (2000 bars, 3 states)**                                | **< 50 MB**|

---

## Error Handling Strategy

### Per-Module Error Handling

**data_loader.py:**
- `fetch_ohlcv` raises `ValueError` with a descriptive message if
  yfinance returns an empty DataFrame. This surfaces in app.py as a
  user-visible error.
- Lookback clamping is silent (the user is not warned when their
  requested lookback exceeds yfinance limits).
- NaN handling is implicit via `dropna()`.

**hmm_engine.py:**
- Individual HMM fits are wrapped in try/except, catching ALL exceptions.
  This is intentional: hmmlearn can raise various errors (convergence
  failures, singular covariance matrices, etc.) and the retry-with-
  different-seed strategy handles them all.
- If ALL fits for ALL state counts fail, `fit_and_select` raises
  `RuntimeError("All HMM fits failed")`.
- Warnings from hmmlearn (ConvergenceWarning) are suppressed via
  `warnings.catch_warnings()` during fitting.

**strategy.py:**
- `compute_confirmations` handles missing `rolling_vol` column by
  defaulting conf_vol_range to True.
- `generate_signals` handles missing `n_confirmations` column by
  defaulting to min_confirmations (always pass).
- `compute_position_size` handles avg_loss=0 by returning 0.0.

**backtester.py:**
- `run()` raises `ValueError` if the DataFrame has fewer bars than
  train_window + test_window. The error message tells the user exactly
  how many bars are needed and how many were provided.
- Individual fold failures (from RegimeDetector.fit_and_select raising
  RuntimeError) are handled by skipping the fold and advancing the window.
- `bootstrap_confidence_intervals` handles n < 10 returns by returning
  zeroed CI dicts.

**app.py:**
- The backtest section wraps `backtester.run()` in a try/except for
  ValueError, displaying the error with `st.error()`.
- The trade log tab checks for the existence of `bt_result` before
  rendering, showing `st.info()` if no backtest has been run.
- Data fetch and HMM fitting errors propagate as unhandled exceptions,
  which Streamlit displays as red error banners.

### Error Recovery Patterns

```
User error (bad ticker)     -> ValueError in fetch_ohlcv
                            -> Streamlit red banner
                            -> User corrects and re-runs

Insufficient data           -> ValueError in backtester.run()
                            -> st.error() with specific message
                            -> User increases lookback or decreases windows

All HMM fits fail           -> RuntimeError in fit_and_select
                            -> Streamlit red banner
                            -> User tries: fewer states, more data,
                               different covariance type

Individual fold fails       -> RuntimeError caught in backtester.run()
                            -> Fold skipped, next fold attempted
                            -> No user notification (silent degradation)
```

---

## Data Integrity Guarantees

### No Lookahead Bias

| Stage               | Guarantee                                              |
|----------------------|--------------------------------------------------------|
| Standardization      | standardize() computes mean/std from train only,       |
|                      | applies to test using train statistics                 |
| HMM fitting          | fit_and_select() receives X_train only                 |
| HMM decoding         | decode() uses model fit on X_train to decode X_test    |
| Confirmations        | RSI, ADX, MACD, EMA are all causal indicators          |
|                      | (use only past and current data by construction)       |
| Signal generation    | generate_signals() iterates forward in time,           |
|                      | using only past states and confidence values           |

### Trade Simulation Realism

| Cost          | Implementation                                         |
|---------------|--------------------------------------------------------|
| Commission    | capital * commission_pct on each trade open and close  |
| Slippage      | Price shifted adversely by slippage_pct on entry/exit  |
| Execution     | Trades execute at close price of signal bar             |
| No partial    | Full execution assumed at adjusted price               |

### Limitations

- No bid-ask spread modeling (slippage is a rough proxy)
- No market impact (assumes infinite liquidity)
- No funding costs for short positions
- Bar-level execution (no intra-bar fills)
- The annualization factor (sqrt(8760)) assumes hourly data -- this is
  incorrect for other intervals but is used uniformly

---

## Extension Points

### Adding New Features

1. Add computation in `compute_features()` in data_loader.py
2. Add feature name to config.yaml -> data.features list
3. The rest of the pipeline picks it up automatically (feature_cols
   propagates through standardize, get_feature_matrix, and all downstream)

### Adding New Confirmation Conditions

1. Add boolean column computation in `compute_confirmations()` in strategy.py
2. Name it with the `conf_` prefix
3. It is automatically counted in n_confirmations
4. Update min_confirmations default if needed (denominator changed)

### Supporting New Model Types

1. Import the model class in hmm_engine.py
2. Add a branch in fit_and_select() based on self.model_type
3. Adjust _count_params() for the new model's parameter count
4. Ensure decode(), label_regimes(), regime_statistics() work with
   the new model's API (predict, predict_proba, means_, covars_)

### Adding New Metrics

1. Add computation in compute_metrics() in backtester.py
2. Add to the return dict
3. Add display in the relevant tab in app.py
4. Optionally add to bootstrap_confidence_intervals() for CI bounds

### Multi-Asset Support

The current architecture processes one ticker at a time. To support
portfolios:
1. Run fetch_ohlcv() and compute_features() per asset
2. Fit separate HMMs per asset (regime structure is asset-specific)
3. Add a portfolio allocation module that combines signals
4. Modify backtester to track per-asset positions and equity
