# HMM Regime Terminal -- Implementation Guide

This document is a developer-focused deep dive into the implementation.
It covers key design decisions, algorithm internals, performance
characteristics, known limitations, and testing strategy.

## Table of Contents

1. [Module Walkthrough](#module-walkthrough)
   - [data_loader.py](#data_loaderpy)
   - [hmm_engine.py](#hmm_enginepy)
   - [strategy.py](#strategypy)
   - [backtester.py](#backtesterpy)
   - [app.py](#apppy)
2. [How hmmlearn Is Used](#how-hmmlearn-is-used)
3. [The Random Restart Strategy](#the-random-restart-strategy)
4. [Signal Generation: Detailed Walkthrough](#signal-generation-detailed-walkthrough)
5. [Trade Simulation](#trade-simulation)
6. [Bootstrap Implementation](#bootstrap-implementation)
7. [Performance Profiling](#performance-profiling)
8. [Known Limitations and Assumptions](#known-limitations-and-assumptions)
9. [Testing Strategy](#testing-strategy)

---

## Module Walkthrough

### data_loader.py

**Key design decision: stateless pure functions.**

All four functions in data_loader.py are pure -- they take inputs and
return outputs with no side effects. There is no class because there is
no state to encapsulate. This makes the module trivially testable and
reusable.

**fetch_ohlcv: yfinance wrapper with guard rails.**

The main complexity is handling yfinance's per-interval lookback limits.
The `max_days` dict maps each interval to its maximum allowed lookback:

```python
max_days = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
    "1h": 730, "1d": 10000, "1wk": 10000, "1mo": 10000,
}
```

If the user requests a lookback exceeding the limit, the start date is
silently clamped. This is a deliberate UX choice -- an error would be
frustrating when the user just wants "as much data as possible."

The MultiIndex column handling addresses a yfinance version inconsistency:
when downloading a single ticker, some versions return flat columns
`["Open", "High", ...]` while others return a MultiIndex with ticker as
the second level. The code normalizes this by taking `get_level_values(0)`.

**compute_features: why these specific computations.**

Each feature computation is deliberately simple and uses well-known
formulas:

- `log_return`: `np.log(Close / Close.shift(1))` -- shift(1) gives the
  previous bar's close. This is a single vectorized operation.

- `rolling_vol`: `log_return.rolling(window).std()` -- pandas rolling std.
  The window default is 21 (approximately one month of daily bars or
  ~1 day of hourly bars).

- `volume_change`: `Volume / Volume.rolling(20).mean() - 1` -- relative
  to a 20-bar simple moving average. The -1 centers it around zero
  (positive = above average, negative = below).

- `intraday_range`: `(High - Low) / Close` -- normalized by close to
  make it scale-independent across assets with different price levels.

- `rsi`: Delegated to the `ta` library's `RSIIndicator` class. This
  uses the standard Wilder smoothing method.

The final `dropna()` removes rows with NaN values from rolling window
initialization. This means the first ~21 rows of the OHLCV data are
lost. The `reset_index(drop=False)` preserves the DatetimeIndex as a
"Date" column for downstream charting.

**standardize: the anti-lookahead mechanism.**

The function signature reveals the design intent:

```python
def standardize(train, test=None, cols=None):
```

The test argument is optional because the full-sample path (Tabs 1, 2, 5)
standardizes all data together (there is no train/test split in that path).
When test is None, only train_z is returned.

The sigma=0 guard (`if sigma == 0: sigma = 1.0`) handles constant features.
This can occur with very short time windows where a feature (e.g., RSI)
does not vary.

### hmm_engine.py

**Key design decision: class with progressive state accumulation.**

RegimeDetector is a class because it accumulates state across method calls:
fit_and_select sets self.model and self.n_states, decode uses self.model,
label_regimes sets self.labels, regime_statistics uses self.labels and
self.model, etc. The methods form a pipeline that must be called in order.

**_count_params: parameter counting for BIC.**

This is a critical method because an incorrect parameter count leads to
incorrect BIC values and wrong model selection. The implementation
carefully handles all four covariance types:

```python
k = (n - 1) + n * (n - 1) + n * d  # pi + A + means
if self.covariance_type == "full":
    k += n * d * (d + 1) // 2       # upper triangle per state
elif self.covariance_type == "diag":
    k += n * d                       # diagonal per state
elif self.covariance_type == "spherical":
    k += n                            # one scalar per state
elif self.covariance_type == "tied":
    k += d * (d + 1) // 2            # one matrix shared
```

The full covariance formula `d*(d+1)/2` comes from the fact that a
$d \times d$ symmetric matrix has $d$ diagonal and $d(d-1)/2$ unique
off-diagonal elements, totaling $d(d+1)/2$. For $d = 5$, that is 15
parameters per state.

**fit_and_select: the model selection loop.**

The nested loop structure is:

```
for n in [2, 3, 4, 5, 6, 7, 8]:    # outer: state counts
    best_ll_for_this_n = -inf
    for seed in [0, 1, ..., 19]:     # inner: random restarts
        try:
            model = GaussianHMM(n_components=n, random_state=seed)
            model.fit(X_train)
            ll = model.score(X_train)
            if ll > best_ll_for_this_n:
                best_ll_for_this_n = ll
                best_model_for_this_n = model
        except:
            continue  # skip failed fits

    bic = -2 * best_ll_for_this_n + k * log(T)
    if bic < global_best_bic:
        global_best_bic = bic
        self.model = best_model_for_this_n
```

The `warnings.catch_warnings()` context manager suppresses hmmlearn's
ConvergenceWarning, which fires frequently when the EM algorithm does
not converge within n_iter iterations. This is expected behavior -- some
seed/state combinations simply do not converge, and the retry strategy
handles it.

**label_regimes: semantic naming by return ordering.**

The core insight is that HMM state indices are arbitrary (the model does
not know what "bull" or "bear" means). By sorting states by their mean
log-return emission (`model.means_[:, 0]`), we can assign labels in
order from most negative (crash/bear) to most positive (bull/bull_run).

The `np.argsort(means)` returns indices that would sort the means in
ascending order. The `sorted_idx[0]` is the state with the lowest mean
return (crash/bear), and `sorted_idx[-1]` is the highest (bull/bull_run).

**regime_statistics: eigenvector computation.**

The stationary distribution computation uses numpy's general eigenvalue
solver:

```python
eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
stationary = np.real(eigenvectors[:, idx])
stationary = stationary / stationary.sum()
```

`np.linalg.eig` returns complex eigenvalues/eigenvectors for general
matrices. The stochastic matrix A always has eigenvalue 1.0, but floating
point arithmetic may return 0.9999... or 1.0001..., so we find the
eigenvalue closest to 1.0 rather than testing for exact equality.

The `np.real()` call drops the imaginary component, which should be zero
(or negligibly small) for the eigenvector corresponding to a real
eigenvalue of a real matrix.

**shannon_entropy: numerical stability.**

```python
eps = 1e-12
p = np.clip(posteriors, eps, 1.0)
entropy = -np.sum(p * np.log2(p), axis=1)
```

The epsilon clipping prevents `log2(0) = -inf`. Posteriors from
forward-backward are theoretically in [0, 1] but can be exactly 0 for
states that are extremely unlikely. Clipping to 1e-12 introduces
negligible error (log2(1e-12) = -39.86 bits, but multiplied by p=1e-12
gives a contribution of 4.8e-11 bits).

### strategy.py

**Key design decision: explicit state machine over vectorized logic.**

The `generate_signals` method uses a Python for-loop instead of vectorized
pandas operations. This is deliberate: the signal logic involves state
(current position, last signal bar, position start) that depends on the
entire history of prior decisions. Vectorizing this would require complex
conditional cumulative operations that would be harder to understand and
debug than the explicit loop.

**compute_confirmations: independent boolean columns.**

Each confirmation is computed independently and stored as a boolean column
with the `conf_` prefix. The auto-discovery mechanism:

```python
conf_cols = [c for c in out.columns if c.startswith("conf_")]
out["n_confirmations"] = out[conf_cols].sum(axis=1)
```

means you can add new confirmations by just adding a column with the
right prefix -- no other code changes needed.

**compute_position_size: three-stage pipeline.**

The sizing pipeline is intentionally sequential:

1. Kelly gives the theoretically optimal bet size based on edge
2. kelly_fraction (half-Kelly) reduces it for practical safety
3. Entropy scaling reduces it further based on model confidence
4. The cap prevents extreme leverage

Each stage independently makes sense, and they compose multiplicatively:
`final_size = kelly * fraction * confidence`, capped at `max_leverage * max_position_pct`.

### backtester.py

**Key design decision: fresh HMM per fold.**

In the walk-forward loop, a new `RegimeDetector` is instantiated for
each fold:

```python
detector = RegimeDetector(self.config)
detector.fit_and_select(X_train)
```

This means:
- The model adapts to changing market structure over time
- There is no risk of one bad fold contaminating subsequent folds
- Each fold is independent and could be parallelized
- The computational cost is O(n_folds * fitting_cost)

An alternative design would reuse the model and only update it with
new data. This would be faster but would require implementing online
HMM updates, which hmmlearn does not support natively.

**simulate_trades: mark-to-market accounting.**

The trade simulation uses a mark-to-market approach rather than tracking
individual position lots:

```python
if position != 0:
    ret = (price / prev_price - 1) * position * position_size
    capital *= (1 + ret)
```

This means capital is updated every bar based on the actual price change,
not just at entry/exit. The `position` variable (1 or -1) ensures longs
profit from price increases and shorts profit from decreases. The
`position_size` scales the return proportionally.

**Slippage and commission model:**

Slippage is modeled as a fixed percentage of price, applied adversely:
- Entry: `entry_price = price * (1 + slippage * |signal|)` (pay more)
- Exit: `exit_price = price * (1 - slippage * |position|)` (receive less)

Commission is modeled as a percentage of current capital, deducted on
both entry and exit. This is a simplification -- real commissions are
typically per-share or per-contract, not percentage-of-capital. However,
for back-of-envelope analysis, the percentage model is reasonable.

**bootstrap_confidence_intervals: implementation details.**

```python
rng = np.random.default_rng(42)
```

Uses numpy's modern random number generator (PCG64 by default) instead
of the legacy `np.random.RandomState`. The seed 42 ensures deterministic
results -- the same equity curve always produces the same CIs.

The bootstrap reconstructs synthetic equity curves:

```python
sample = rng.choice(returns, size=n, replace=True)
cumulative = np.cumprod(1 + sample)
```

This preserves the return distribution but breaks autocorrelation.
The CIs are **percentile-based** (not bias-corrected). For the 90% CI:

```python
alpha_lo = 0.05  # (1 - 0.90) / 2
alpha_hi = 0.95
ci_lower = np.quantile(boot_stats, 0.05)
ci_upper = np.quantile(boot_stats, 0.95)
```

### app.py

**Key design decision: no persistent state.**

The entire app.py script re-executes on every user interaction. There is
no session state, no database, no cached models. The `run_btn` flag
gates all computation:

```python
if run_btn:
    # ... all analysis code ...
else:
    st.info("Configure parameters and click Run Analysis")
```

This means switching tabs does NOT re-run the analysis. The charts and
tables from the most recent run remain visible until the page is
refreshed or the button is clicked again.

**build_config: shallow merge.**

```python
cfg["hmm"] = {**base_config["hmm"], "min_states": min_states, ...}
```

The dict spread `{**base, key: value}` creates a new dict with all
base keys plus overrides. This is a **shallow merge** -- nested dicts
are not recursively merged. The strategy.confirmations section is
handled explicitly:

```python
"confirmations": {**base_config["strategy"]["confirmations"],
                  "min_confidence": min_regime_conf}
```

**Plotly chart patterns.**

The app uses two Plotly patterns:

1. `go.Figure()` with manual trace addition for full control over
   styling (used for price overlay, equity curves, drawdown charts).

2. `px.imshow()` for heatmaps (transition matrix, correlation matrix).

The `make_subplots(specs=[[{"secondary_y": True}]])` pattern creates
a dual-axis chart for the entropy/confidence visualization.

---

## How hmmlearn Is Used

The application uses three methods from `hmmlearn.hmm.GaussianHMM`:

### GaussianHMM Constructor

```python
model = GaussianHMM(
    n_components=n,           # number of hidden states
    covariance_type="full",   # covariance structure
    n_iter=200,               # max EM iterations
    tol=1e-4,                 # convergence tolerance
    random_state=seed,        # random seed for initialization
)
```

The `random_state` parameter controls initialization of:
- Initial state probabilities (random Dirichlet)
- Transition matrix rows (random Dirichlet)
- Emission means (random from data range)
- Emission covariances (random positive definite)

### model.fit(X)

Runs the Baum-Welch (EM) algorithm:

1. Initialize parameters from random_state
2. E-step: Run forward-backward to compute gamma (state occupancies)
   and xi (transition expectations)
3. M-step: Update pi, A, mu, Sigma from gamma/xi
4. Check convergence: |LL_new - LL_old| < tol
5. Repeat until convergence or n_iter reached

The fit operates on X with shape (T, d). Internally, hmmlearn stores
the fitted parameters as:
- `model.startprob_`: initial state distribution (N,)
- `model.transmat_`: transition matrix (N, N)
- `model.means_`: emission means (N, d)
- `model.covars_`: emission covariances (shape depends on covariance_type)

### model.score(X)

Returns the **total log-likelihood** of the observation sequence under
the model:

$$\text{score}(X) = \log P(X_{1:T} \mid \lambda) = \log \sum_{i=1}^N \alpha_T(i)$$

This is computed using the forward algorithm. The returned value is a
single float (not per-bar). In the rolling log-likelihood method, we
divide by the window size to get per-bar log-likelihood:

```python
ll_series[t - 1] = self.model.score(chunk) / window
```

### model.predict(X)

Runs the **Viterbi algorithm** and returns the most likely state sequence:

```python
states = model.predict(X)  # shape (T,), dtype int
```

Each element is a state index in {0, 1, ..., N-1}.

### model.predict_proba(X)

Runs the **forward-backward algorithm** and returns posterior state
probabilities:

```python
posteriors = model.predict_proba(X)  # shape (T, N), dtype float
```

Each row sums to 1.0. Element [t, i] is P(S_t = i | X_{1:T}).

**Important distinction:** predict() gives a single "hard" assignment
per bar (most likely sequence), while predict_proba() gives "soft"
probabilities per bar. They can disagree: the Viterbi path optimizes
the entire sequence jointly, while posteriors are marginals at each time
step.

---

## The Random Restart Strategy

### Why 20 Restarts

Baum-Welch is a local optimizer -- it converges to a local maximum of
the likelihood surface, not necessarily the global maximum. The likelihood
surface for HMMs is typically multimodal, especially with more states.

The random restart strategy addresses this by:
1. Running the same model configuration (same n_states, same covariance
   type) with 20 different random initializations
2. Keeping the model with the highest log-likelihood

The number 20 is a practical compromise:
- Too few (e.g., 5): high probability of missing the global optimum
- Too many (e.g., 100): diminishing returns with high computational cost
- 20: empirically provides stable results for 2-8 state models with 5
  features (the parameter space explored in this application)

### Interpreting Convergence Failures

A convergence failure occurs when:
- The EM algorithm does not converge within n_iter iterations (200 by
  default). hmmlearn issues a ConvergenceWarning, which we suppress.
- The fit throws an exception (e.g., singular covariance matrix).
  We catch ALL exceptions and skip to the next seed.

Common failure modes:
1. **Singular covariance:** Too few data points for the number of
   parameters. More likely with full covariance and many states.
   Fix: use diag covariance or reduce max_states.
2. **Degenerate state:** A state is assigned zero data points during EM.
   The covariance becomes undefined. hmmlearn handles this internally
   in most cases, but it can still cause numerical errors.
3. **Slow convergence:** Complex models with many states may need more
   than 200 iterations. Fix: increase n_iter in config.yaml.

If all 20 restarts fail for a given state count, that state count is
skipped (its BIC is not computed). If all restarts for ALL state counts
fail, RuntimeError is raised.

---

## Signal Generation: Detailed Walkthrough

The `generate_signals` method in strategy.py is the most complex piece
of logic in the application. Here is a step-by-step walkthrough.

### Initialization

```python
signals = np.zeros(n, dtype=int)
bull_states = {s for s, l in labels.items() if l in ("bull", "bull_run")}
bear_states = {s for s, l in labels.items() if l in ("bear", "crash")}
```

Signals start at 0 (flat). States are classified as bullish or bearish
based on their labels. States labeled "neutral" or "regime_N" are neither
-- they default to producing flat signals.

### Regime Persistence

```python
regime_persist = np.zeros(n, dtype=int)
for t in range(1, n):
    if states[t] == states[t - 1]:
        regime_persist[t] = regime_persist[t - 1] + 1
    else:
        regime_persist[t] = 0
```

This pre-computes a running count of how many consecutive bars the
current regime has been active. A transition resets the counter to 0.
This is used for the hysteresis check -- the regime must persist for
`hysteresis_bars` (default 3) before it can trigger a signal change.

### State Variables

```python
last_signal_bar = -cooldown_bars - 1     # allow immediate first signal
position_start = -min_hold_bars - 1       # allow immediate first signal
current_pos = 0                           # start flat
```

The initial values are set so that cooldown and min_hold checks pass on
the first bar. This prevents the system from being stuck in an artificial
initial wait.

### Main Loop: Edge Cases

**Edge case 1: Conflicting regime and confirmations.**
If the regime is bullish but fewer than `min_confirmations` conditions
are met, the signal is 0 (flat). The system will not enter a position
just because the regime is bullish -- technical conditions must also agree.

**Edge case 2: Neutral regime.**
States labeled "neutral" are not in `bull_states` or `bear_states`.
They produce `new_signal = 0` regardless of confirmations. This means
the system exits any existing position when the regime becomes neutral
(after the min_hold period).

**Edge case 3: Position exit during min_hold.**
If the regime flips from bull to bear within the min_hold period, the
system maintains the existing position until min_hold expires. This
prevents premature exits on regime noise.

**Edge case 4: Cooldown after exit.**
When a position is closed (current_pos changes from non-zero to zero),
no new position can be opened for `cooldown_bars` bars. This prevents
overtrading when the regime is oscillating.

---

## Trade Simulation

### Mark-to-Market Calculation

The simulation tracks capital through a mark-to-market accounting model:

```python
for t in range(1, len(df)):
    price = df["Close"].iloc[t]
    prev_price = df["Close"].iloc[t - 1]

    if position != 0:
        ret = (price / prev_price - 1) * position * position_size
        capital *= (1 + ret)
```

The return `(price / prev_price - 1)` is the simple return for the bar.
Multiplied by `position` (+1 or -1), this correctly handles both long
and short positions. Multiplied by `position_size` (0 to 1+), this
scales the return by the allocation.

The multiplicative update `capital *= (1 + ret)` compounds returns.
This is correct for a levered portfolio -- if position_size is 0.5,
only half the capital participates in the return.

### Position Change Logic

When the signal differs from the current position, both a close and an
open may occur in the same bar:

```python
if signal != position:
    if position != 0:
        # CLOSE existing position
        exit_price = price * (1 - slippage * |position|)
        capital -= capital * commission_pct
        # record TradeRecord

    if signal != 0:
        # OPEN new position
        entry_price = price * (1 + slippage * |signal|)
        capital -= capital * commission_pct
        position_size = sizes.iloc[t]

    position = signal
```

Note that capital is reduced by commission on BOTH the close and the open.
A signal change from long to short involves two commission charges (close
long, open short).

### Slippage Model

Slippage is modeled as a percentage of the execution price, applied
adversely:

- **Long entry:** `price * (1 + slippage)` -- you pay more than the
  close price due to execution delay, market impact, and bid-ask spread.
- **Long exit:** `price * (1 - slippage)` -- you receive less.
- **Short entry:** `price * (1 + slippage)` -- you sell at a lower
  effective price.
- **Short exit:** `price * (1 - slippage)` -- you buy back at a higher
  effective price.

Default slippage is 5 bps (0.05%), which is realistic for liquid assets
like BTC-USD or major equities on hourly bars. For illiquid assets or
minute-level data, increase this in config.yaml.

---

## Bootstrap Implementation

### Why numpy's default_rng

The implementation uses `np.random.default_rng(42)` instead of the
legacy `np.random.RandomState`:

1. **Better statistical properties:** default_rng uses PCG64, which has
   a period of 2^128 and better equidistribution than Mersenne Twister.
2. **Reproducibility:** The fixed seed 42 ensures identical CIs for the
   same equity curve.
3. **Modern API:** default_rng supports `rng.choice()` which is faster
   and more ergonomic than `np.random.choice()`.

### Seed Choice

The seed 42 is fixed (not user-configurable). This is intentional:
bootstrap CIs should be deterministic for reproducibility. If users
want to test sensitivity to bootstrap randomness, they can modify the
seed in backtester.py directly.

### Computational Complexity

For $B$ bootstrap samples and $T$ return observations:

- Resampling: $O(B \times T)$ -- `rng.choice(returns, size=T)` for each
  sample
- Equity reconstruction: $O(B \times T)$ -- `np.cumprod(1 + sample)`
- Sharpe computation: $O(B \times T)$ -- mean and std of each sample
- Max drawdown: $O(B \times T)$ -- `np.maximum.accumulate` and division
- Quantile computation: $O(B \log B)$ -- sorting the bootstrap stats

Total: $O(B \times T)$ which for default values ($B = 1000$, $T \sim 2000$)
is approximately 2 million operations -- well within sub-second territory
for numpy's vectorized operations.

Memory: $O(B + T)$ per iteration (the sample array is overwritten each
iteration, and only the scalar statistics are accumulated in lists).
Peak memory from the bootstrap arrays: $3B \times 8$ bytes = 24 KB for
the three statistics lists.

---

## Performance Profiling

### What Dominates Runtime

For a typical run (BTC-USD, 1h, 90 days, ~2000 bars):

| Operation                  | Time       | % of Total | Notes                     |
|----------------------------|------------|------------|---------------------------|
| yfinance data fetch        | 1-3s       | 3%         | Network bound             |
| Feature engineering        | <100ms     | <1%        | Vectorized pandas         |
| HMM fitting (full-sample)  | 20-120s    | 40-60%     | 140 EM runs               |
| Viterbi + forward-backward | <100ms     | <1%        | O(T * N^2)                |
| Confirmation computation   | <100ms     | <1%        | ta library, vectorized    |
| Signal generation          | <50ms      | <1%        | Python loop, T iterations |
| Walk-forward backtest      | 60-600s    | 30-50%     | HMM fitting * n_folds     |
| Trade simulation           | <50ms      | <1%        | Python loop               |
| Bootstrap CIs              | 1-5s       | 2%         | 1000 samples, vectorized  |
| Plotly chart rendering     | <500ms     | 1%         | JSON generation           |

**The bottleneck is HMM fitting.** The Baum-Welch algorithm's cost per
iteration is $O(T \times N^2 \times d^2)$ for full covariance, and each
fit runs up to 200 iterations. With 140 fits per state selection, this
dominates everything else.

### Reducing Runtime

In order of impact:
1. **Reduce n_restarts** (20 -> 10): 2x speedup, slight risk of worse
   model selection.
2. **Narrow state range** (2-8 -> 2-4): 2.3x speedup, may miss complex
   regime structures.
3. **Use diag covariance**: ~3x speedup per fit (fewer parameters,
   simpler M-step), loses cross-feature correlations.
4. **Reduce lookback** (fewer bars): linear speedup in T, less training
   data.
5. **Increase step_bars**: fewer walk-forward folds, less out-of-sample
   coverage.

### Memory Usage

| Component               | Size (2000 bars, 3 states)    |
|--------------------------|-------------------------------|
| OHLCV DataFrame          | 80 KB (2000 * 5 * 8 bytes)    |
| Feature DataFrame         | 160 KB (2000 * 10 * 8 bytes)  |
| Feature matrix            | 80 KB (2000 * 5 * 8 bytes)    |
| Posteriors                | 48 KB (2000 * 3 * 8 bytes)    |
| HMM model internals      | ~2 KB                         |
| Equity curve              | 16 KB (2000 * 8 bytes)        |
| Trade records             | ~10 KB (varies)               |
| Bootstrap workspace       | ~16 MB (1000 * 2000 * 8 bytes)|
| **Total**                 | **< 20 MB**                   |

The bootstrap workspace is the largest single allocation but is transient
(freed after CIs are computed). The overall memory footprint is small
enough for any modern machine.

---

## Known Limitations and Assumptions

### Statistical Assumptions

1. **Gaussian emissions.** The HMM assumes each regime's features follow
   a multivariate Gaussian distribution. Financial returns have fat tails,
   so this assumption is approximate. Mitigations: the multiple-regime
   structure allows different regimes to capture tail behavior; z-score
   standardization improves normality.

2. **Markov property.** The model assumes the next regime depends only on
   the current regime, not the entire history. In reality, regime
   transitions may depend on how long the current regime has persisted
   (duration dependence). This is a fundamental HMM limitation.

3. **Time-invariant parameters.** Within a walk-forward fold, the
   transition matrix and emission parameters are constant. If the market
   structure changes within a fold, the model cannot adapt until the next
   fold.

4. **Hourly annualization.** The Sharpe/Sortino/Calmar computations use
   `ann_factor = sqrt(8760)`, which assumes hourly bars (~24 * 365 bars
   per year). This is incorrect for daily, minute, or weekly data. The
   metrics will be miscalibrated for non-hourly intervals.

### Implementation Limitations

1. **No bid-ask spread.** The slippage model is a rough proxy for
   execution costs. Real trading involves a bid-ask spread that varies
   with time, liquidity, and order size.

2. **No market impact.** The simulation assumes infinite liquidity --
   trades can be executed at any size without moving the price. For large
   positions or illiquid assets, this is unrealistic.

3. **No funding costs.** Short positions incur borrowing costs in
   practice. The simulation does not model these.

4. **Bar-level execution only.** Trades execute at the close price of
   the signal bar. In practice, the signal would be available only after
   the bar closes, and the trade would execute at the next bar's open
   (or later). This introduces a small lookahead bias.

5. **No intra-bar stop-loss.** The config has stop_loss_pct and
   take_profit_pct, but they are not implemented in the simulation loop.
   Positions are managed purely by the signal state machine.

6. **Single-threaded.** All computation runs in a single Python thread.
   HMM fitting could be parallelized for significant speedup.

7. **No model persistence.** The fitted model is not saved to disk.
   Each "Run Analysis" click fits a fresh model from scratch.

---

## Testing Strategy

The test suite consists of four test files:

### test_data_loader.py

Tests for data_loader.py:
- `fetch_ohlcv` returns a DataFrame with the expected columns
- `compute_features` adds exactly 5 feature columns with no NaNs
- `standardize` produces zero mean and unit variance on the training set
- `standardize` applies training statistics to the test set correctly
- `get_feature_matrix` returns the correct shape

### test_hmm_engine.py

Tests for hmm_engine.py:
- `RegimeDetector` initializes with config values
- `_count_params` returns correct parameter counts for each covariance type
- `fit_and_select` fits a model and returns BIC scores
- `decode` returns states and posteriors with correct shapes
- `label_regimes` assigns labels in the correct order
- `regime_statistics` returns a DataFrame with expected columns
- `shannon_entropy` returns values in the correct range
- `log_likelihood_series` returns NaN for the initial window

### test_strategy.py

Tests for strategy.py:
- `SignalGenerator` initializes with config values
- `compute_confirmations` adds 8 boolean columns plus n_confirmations
- `generate_signals` produces values in {-1, 0, 1}
- Cooldown, min hold, and hysteresis filters work correctly
- `compute_position_size` returns values in [0, max_leverage * max_position_pct]
- Kelly formula produces correct values for known inputs

### test_backtester.py

Tests for backtester.py:
- `WalkForwardBacktester` initializes with config values
- `run` raises ValueError for insufficient data
- `simulate_trades` produces an equity curve with correct length
- `compute_metrics` returns all expected keys
- `bootstrap_confidence_intervals` returns CI dicts with expected keys
- Equity curve starts at initial_capital

### How to Verify Correctness

```bash
# Run all tests
python -m pytest test_*.py -v

# Run a specific test file
python -m pytest test_hmm_engine.py -v

# Run with coverage
python -m pytest test_*.py --cov=. --cov-report=term-missing
```

### Manual Verification

Beyond automated tests, the application supports manual verification through
its diagnostics tab (Tab 5):

1. **Rolling log-likelihood:** If the model is correctly fitted, the
   log-likelihood should be stable (not trending downward) over the
   training period.

2. **Feature correlation matrix:** Features should show low mutual
   correlation (|r| < 0.5), confirming they provide independent information.

3. **Transition matrix:** Diagonal values should be close to 1 (regimes
   are sticky). Off-diagonal values should be small but non-zero (regimes
   do transition).

4. **BIC curve:** Should have a clear minimum. A flat curve suggests
   insufficient data for model selection.

5. **Return distributions:** Regime-specific distributions should be
   visually distinct (different means, different spreads). Overlapping
   distributions suggest the model is splitting regimes that are not
   genuinely different.
