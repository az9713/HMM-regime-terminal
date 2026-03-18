# HMM Regime Terminal -- Mathematical Theory

This document covers all the mathematics used in the HMM Regime Terminal.
Every formula referenced in the codebase is derived or explained here.

## Table of Contents

1. [Hidden Markov Models](#hidden-markov-models)
   - [Formal Definition](#formal-definition)
   - [Graphical Model](#graphical-model)
   - [The Three Fundamental Problems](#the-three-fundamental-problems)
2. [Baum-Welch Algorithm](#baum-welch-algorithm)
   - [Forward Variables](#forward-variables)
   - [Backward Variables](#backward-variables)
   - [E-Step: Xi and Gamma](#e-step-xi-and-gamma)
   - [M-Step: Parameter Updates](#m-step-parameter-updates)
   - [Convergence](#convergence)
3. [Viterbi Algorithm](#viterbi-algorithm)
   - [Dynamic Programming Formulation](#dynamic-programming-formulation)
   - [Worked Example](#worked-example)
4. [BIC Model Selection](#bic-model-selection)
   - [Bayesian Model Comparison](#bayesian-model-comparison)
   - [Laplace Approximation to Marginal Likelihood](#laplace-approximation-to-marginal-likelihood)
   - [Parameter Counting](#parameter-counting)
5. [Shannon Entropy and Regime Confidence](#shannon-entropy-and-regime-confidence)
   - [Information-Theoretic Interpretation](#information-theoretic-interpretation)
   - [Normalized Confidence](#normalized-confidence)
6. [Expected Regime Duration](#expected-regime-duration)
7. [Stationary Distribution](#stationary-distribution)
   - [Eigenvector Computation](#eigenvector-computation)
   - [Existence and Uniqueness](#existence-and-uniqueness)
8. [Kelly Criterion](#kelly-criterion)
   - [Derivation](#kelly-derivation)
   - [Half-Kelly and Practical Considerations](#half-kelly-and-practical-considerations)
9. [CVaR / Expected Shortfall](#cvar--expected-shortfall)
10. [Bootstrap Confidence Intervals](#bootstrap-confidence-intervals)
11. [Walk-Forward Validation](#walk-forward-validation)
12. [Feature Engineering Rationale](#feature-engineering-rationale)
13. [References](#references)

---

## Hidden Markov Models

### Formal Definition

A Hidden Markov Model (HMM) is a doubly stochastic process consisting of:

1. A **hidden state sequence** $S_1, S_2, \ldots, S_T$ taking values in
   a finite set $\{1, 2, \ldots, N\}$.

2. An **observation sequence** $X_1, X_2, \ldots, X_T$ where each $X_t$
   is a $d$-dimensional vector (in our case, $d = 5$ features).

The model is parameterized by $\lambda = (\pi, A, \theta)$:

- **Initial state distribution** $\pi$:
  $$\pi_i = P(S_1 = i), \quad i = 1, \ldots, N$$

- **Transition matrix** $A$:
  $$a_{ij} = P(S_{t+1} = j \mid S_t = i), \quad \sum_{j=1}^N a_{ij} = 1$$

- **Emission parameters** $\theta = \{(\mu_i, \Sigma_i)\}_{i=1}^N$:
  $$X_t \mid S_t = i \sim \mathcal{N}(\mu_i, \Sigma_i)$$

The two key assumptions are:

1. **Markov property:** $P(S_{t+1} \mid S_1, \ldots, S_t) = P(S_{t+1} \mid S_t)$
   -- the future state depends only on the current state, not the history.

2. **Output independence:** $P(X_t \mid S_1, \ldots, S_T, X_1, \ldots, X_T) = P(X_t \mid S_t)$
   -- the observation depends only on the current state.

In the financial context:
- Hidden states represent **market regimes** (bull, bear, crash, etc.)
- Observations are **feature vectors** (log return, volatility, volume, etc.)
- The transition matrix captures **regime persistence and switching**
- The emission distributions capture **how each regime "looks"** in feature space

### Graphical Model

```
    pi
     |
     v
    S_1 ---a_ij---> S_2 ---a_ij---> S_3 ---a_ij---> ... ---a_ij---> S_T
     |                |                |                               |
     | b_i(x)         | b_i(x)        | b_i(x)                       | b_i(x)
     v                v                v                               v
    X_1              X_2              X_3              ...             X_T
```

Where $b_i(x) = P(X_t = x \mid S_t = i)$ is the emission probability
(Gaussian density in our case).

### The Three Fundamental Problems

Given a model $\lambda$ and observation sequence $X_{1:T}$:

**Problem 1: Evaluation.** Compute $P(X_{1:T} \mid \lambda)$ -- how well
does the model explain the data? Solved by the **forward algorithm**.
Used in model selection (BIC) and model diagnostics (rolling log-likelihood).

**Problem 2: Decoding.** Find $\arg\max_{S_{1:T}} P(S_{1:T} \mid X_{1:T}, \lambda)$
-- what is the most likely state sequence? Solved by the **Viterbi algorithm**.
Used for regime assignment.

**Problem 3: Learning.** Find $\arg\max_\lambda P(X_{1:T} \mid \lambda)$
-- what model parameters best explain the data? Solved by the
**Baum-Welch algorithm** (EM for HMMs). Used in model fitting.

---

## Baum-Welch Algorithm

The Baum-Welch algorithm is the Expectation-Maximization (EM) algorithm
specialized for HMMs. It iterates between computing expected state
occupancies (E-step) and updating parameters (M-step).

### Forward Variables

Define the forward variable:

$$\alpha_t(i) = P(X_1, X_2, \ldots, X_t, S_t = i \mid \lambda)$$

This is the joint probability of observing the first $t$ observations AND
being in state $i$ at time $t$.

**Initialization:**
$$\alpha_1(i) = \pi_i \cdot b_i(X_1)$$

**Recursion:**
$$\alpha_t(j) = \left[\sum_{i=1}^N \alpha_{t-1}(i) \cdot a_{ij}\right] \cdot b_j(X_t)$$

**Termination:**
$$P(X_{1:T} \mid \lambda) = \sum_{i=1}^N \alpha_T(i)$$

Complexity: $O(T \cdot N^2)$ -- linear in sequence length, quadratic in
number of states. This is vastly more efficient than the naive approach
of enumerating all $N^T$ state sequences.

In practice, computation is done in log-space using the log-sum-exp trick
to prevent numerical underflow for long sequences.

### Backward Variables

Define the backward variable:

$$\beta_t(i) = P(X_{t+1}, X_{t+2}, \ldots, X_T \mid S_t = i, \lambda)$$

This is the probability of the future observations given that we are in
state $i$ at time $t$.

**Initialization:**
$$\beta_T(i) = 1 \quad \text{for all } i$$

**Recursion:**
$$\beta_t(i) = \sum_{j=1}^N a_{ij} \cdot b_j(X_{t+1}) \cdot \beta_{t+1}(j)$$

### E-Step: Xi and Gamma

**Xi (transition expectations):**

$$\xi_t(i, j) = P(S_t = i, S_{t+1} = j \mid X_{1:T}, \lambda)$$

$$\xi_t(i, j) = \frac{\alpha_t(i) \cdot a_{ij} \cdot b_j(X_{t+1}) \cdot \beta_{t+1}(j)}{P(X_{1:T} \mid \lambda)}$$

where $P(X_{1:T} \mid \lambda) = \sum_{i=1}^N \alpha_T(i)$.

**Gamma (state occupancy expectations):**

$$\gamma_t(i) = P(S_t = i \mid X_{1:T}, \lambda) = \sum_{j=1}^N \xi_t(i, j)$$

Equivalently:

$$\gamma_t(i) = \frac{\alpha_t(i) \cdot \beta_t(i)}{\sum_{j=1}^N \alpha_t(j) \cdot \beta_t(j)}$$

Note: $\gamma_t(i)$ is exactly what `model.predict_proba(X)` returns --
the posterior state probabilities used for entropy computation.

### M-Step: Parameter Updates

Given $\xi$ and $\gamma$, the parameters are updated:

**Initial state distribution:**
$$\hat{\pi}_i = \gamma_1(i)$$

**Transition matrix:**
$$\hat{a}_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$$

This is the expected number of transitions from $i$ to $j$ divided by the
expected number of times in state $i$.

**Emission means:**
$$\hat{\mu}_i = \frac{\sum_{t=1}^T \gamma_t(i) \cdot X_t}{\sum_{t=1}^T \gamma_t(i)}$$

This is the gamma-weighted average of observations assigned to state $i$.

**Emission covariances:**
$$\hat{\Sigma}_i = \frac{\sum_{t=1}^T \gamma_t(i) \cdot (X_t - \hat{\mu}_i)(X_t - \hat{\mu}_i)^\top}{\sum_{t=1}^T \gamma_t(i)}$$

### Convergence

The Baum-Welch algorithm guarantees that the log-likelihood is
non-decreasing at each iteration:

$$\log P(X_{1:T} \mid \lambda^{(k+1)}) \geq \log P(X_{1:T} \mid \lambda^{(k)})$$

However, it converges to a **local maximum**, not necessarily the global
maximum. This is why the application uses multiple random restarts (default
20): each restart initializes with different random parameters, and the
model with the highest log-likelihood across all restarts is kept.

In the implementation, convergence is declared when the log-likelihood
improvement between iterations falls below `tol` (default $10^{-4}$),
or after `n_iter` (default 200) iterations, whichever comes first.

---

## Viterbi Algorithm

### Dynamic Programming Formulation

The Viterbi algorithm finds the most likely state sequence:

$$S_{1:T}^* = \arg\max_{S_{1:T}} P(S_{1:T} \mid X_{1:T}, \lambda)$$

Define:

$$\delta_t(j) = \max_{S_1, \ldots, S_{t-1}} P(S_1, \ldots, S_{t-1}, S_t = j, X_{1:t} \mid \lambda)$$

This is the probability of the best path ending in state $j$ at time $t$.

**Initialization:**
$$\delta_1(j) = \pi_j \cdot b_j(X_1)$$
$$\psi_1(j) = 0$$

**Recursion:**
$$\delta_t(j) = \max_{1 \leq i \leq N} [\delta_{t-1}(i) \cdot a_{ij}] \cdot b_j(X_t)$$
$$\psi_t(j) = \arg\max_{1 \leq i \leq N} [\delta_{t-1}(i) \cdot a_{ij}]$$

**Termination:**
$$S_T^* = \arg\max_{1 \leq i \leq N} \delta_T(i)$$

**Backtracking:**
$$S_t^* = \psi_{t+1}(S_{t+1}^*), \quad t = T-1, T-2, \ldots, 1$$

Complexity: $O(T \cdot N^2)$ time, $O(T \cdot N)$ space.

In the implementation, `model.predict(X)` runs the Viterbi algorithm
and returns the optimal state sequence.

### Worked Example

Consider a 2-state HMM ($N = 2$: bull and bear) with:

$$\pi = [0.6, 0.4], \quad A = \begin{bmatrix} 0.9 & 0.1 \\ 0.2 & 0.8 \end{bmatrix}$$

Suppose we have observations $X_1, X_2, X_3$ and emission probabilities:

| State | $b_i(X_1)$ | $b_i(X_2)$ | $b_i(X_3)$ |
|-------|-----------|-----------|-----------|
| Bull  | 0.4       | 0.1       | 0.5       |
| Bear  | 0.1       | 0.6       | 0.3       |

**t = 1:**
$$\delta_1(\text{bull}) = 0.6 \times 0.4 = 0.24$$
$$\delta_1(\text{bear}) = 0.4 \times 0.1 = 0.04$$

**t = 2:**
$$\delta_2(\text{bull}) = \max(0.24 \times 0.9, 0.04 \times 0.2) \times 0.1 = 0.216 \times 0.1 = 0.0216$$
$$\delta_2(\text{bear}) = \max(0.24 \times 0.1, 0.04 \times 0.8) \times 0.6 = 0.032 \times 0.6 = 0.0192$$

**t = 3:**
$$\delta_3(\text{bull}) = \max(0.0216 \times 0.9, 0.0192 \times 0.2) \times 0.5 = 0.01944 \times 0.5 = 0.00972$$
$$\delta_3(\text{bear}) = \max(0.0216 \times 0.1, 0.0192 \times 0.8) \times 0.3 = 0.01536 \times 0.3 = 0.004608$$

**Backtrack:** $S_3^* = \text{bull}$, $S_2^* = \text{bull}$, $S_1^* = \text{bull}$.

Most likely sequence: bull -> bull -> bull, despite observation $X_2$
having higher bear emission probability. The strong self-transition
probability (0.9) and initial bull probability (0.6) outweigh the single
bearish observation.

---

## BIC Model Selection

### Bayesian Model Comparison

The fundamental question in model selection is: given data $X$ and
candidate models $M_1, M_2, \ldots$ (each with a different number of
states), which model is best?

The Bayesian answer is to compare **marginal likelihoods**:

$$P(X \mid M_k) = \int P(X \mid \theta, M_k) \cdot P(\theta \mid M_k) \, d\theta$$

This integral averages the likelihood over all possible parameter values,
weighted by the prior. More complex models have more parameters to
integrate over, naturally penalizing complexity.

### Laplace Approximation to Marginal Likelihood

The marginal likelihood integral is generally intractable. The Laplace
approximation uses a second-order Taylor expansion around the MLE
$\hat{\theta}$:

$$\log P(X \mid M_k) \approx \log P(X \mid \hat{\theta}, M_k) - \frac{k}{2} \log T + C$$

where $k$ is the number of free parameters and $T$ is the number of
observations. Dropping the constant $C$ and multiplying by $-2$ gives the BIC:

$$\text{BIC} = -2 \log P(X \mid \hat{\theta}) + k \log T$$

**Lower BIC = better model.** The first term rewards fit (higher
likelihood = lower value). The second term penalizes complexity ($k \log T$
grows with both the number of parameters and the data size).

### Parameter Counting

For a Gaussian HMM with $N$ states and $d$ features, the parameter
count $k$ depends on the covariance structure:

**Initial state distribution** $\pi$: $N - 1$ free parameters (they
must sum to 1).

**Transition matrix** $A$: $N \times (N - 1)$ free parameters (each row
sums to 1, so $N - 1$ free per row).

**Emission means** $\mu$: $N \times d$ parameters.

**Emission covariances** $\Sigma$:

| Covariance type | Parameters per state | Total     | Description                    |
|-----------------|---------------------|-----------|--------------------------------|
| full            | $d(d+1)/2$          | $N \cdot d(d+1)/2$ | Full symmetric matrix |
| diag            | $d$                 | $N \cdot d$         | Diagonal only         |
| spherical       | $1$                 | $N$                 | $\sigma^2 I$          |
| tied            | $d(d+1)/2$          | $d(d+1)/2$         | Shared across states  |

For the default configuration ($N = 3$, $d = 5$, full covariance):

$$k = (3-1) + 3 \times (3-1) + 3 \times 5 + 3 \times \frac{5 \times 6}{2} = 2 + 6 + 15 + 45 = 68$$

This is implemented in `RegimeDetector._count_params()`.

---

## Shannon Entropy and Regime Confidence

### Information-Theoretic Interpretation

Given posterior probabilities $p_i(t) = P(S_t = i \mid X_{1:T})$ from the
forward-backward algorithm, the Shannon entropy at bar $t$ is:

$$H_t = -\sum_{i=1}^N p_i(t) \cdot \log_2 p_i(t)$$

Entropy measures **uncertainty** in bits:

- **$H_t = 0$ bits:** One state has probability 1. The model is completely
  certain about the current regime.

- **$H_t = \log_2 N$ bits:** All states are equally likely ($p_i = 1/N$).
  The model has maximum uncertainty -- it cannot distinguish between regimes.

For a 3-state model, $H_{\max} = \log_2 3 \approx 1.585$ bits.

Entropy is related to mutual information: if we observe $X_t$ and use it
to infer $S_t$, the mutual information $I(S_t; X_t)$ tells us how much
the observation reduces our uncertainty about the state. High mutual
information corresponds to low posterior entropy.

### Normalized Confidence

The application defines confidence as:

$$c_t = 1 - \frac{H_t}{\log_2 N}$$

This maps entropy to a $[0, 1]$ confidence scale:

| $c_t$ | Meaning                              |
|-------|--------------------------------------|
| 1.0   | Complete certainty (one state at p=1)|
| 0.5   | Moderate certainty                   |
| 0.0   | Maximum uncertainty (uniform)        |

Confidence is used in two places:
1. **Signal gating:** If $c_t < \text{min\_confidence}$, no signal is
   generated (the model is too uncertain to act on).
2. **Position sizing:** If entropy scaling is enabled, position size is
   multiplied by $c_t$, reducing exposure when the model is uncertain.

---

## Expected Regime Duration

The expected duration of regime $i$ is derived from the geometric
distribution of the self-transition probability $a_{ii}$.

If the system is in state $i$, the probability of remaining in state $i$
for exactly $d$ consecutive bars before transitioning is:

$$P(D_i = d) = a_{ii}^{d-1} \cdot (1 - a_{ii})$$

This is a geometric distribution with parameter $p = 1 - a_{ii}$.
The expected value is:

$$E[D_i] = \frac{1}{1 - a_{ii}}$$

Examples:
- $a_{ii} = 0.95 \Rightarrow E[D_i] = 20$ bars
- $a_{ii} = 0.99 \Rightarrow E[D_i] = 100$ bars
- $a_{ii} = 0.80 \Rightarrow E[D_i] = 5$ bars

In the implementation, this is computed as `1.0 / (1.0 - np.diag(transmat))`.

High expected durations indicate sticky regimes (the market tends to stay
in the same state for extended periods). Low durations indicate transient
regimes (frequent switching).

---

## Stationary Distribution

### Eigenvector Computation

The stationary distribution $\pi^*$ is the long-run fraction of time spent
in each state. It satisfies:

$$\pi^* A = \pi^*, \quad \sum_i \pi^*_i = 1$$

Equivalently, $\pi^*$ is the left eigenvector of $A$ corresponding to
eigenvalue 1. Transposing:

$$A^\top \pi^* = \pi^*$$

So $\pi^*$ is a right eigenvector of $A^\top$ with eigenvalue 1.

In the implementation:

```python
eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
stationary = np.real(eigenvectors[:, idx])
stationary = stationary / stationary.sum()
```

The eigenvector closest to eigenvalue 1 is selected, converted to real
values (the imaginary part should be zero for a proper stochastic matrix
but may have floating-point residuals), and normalized.

### Existence and Uniqueness

The stationary distribution exists and is unique if the Markov chain is:

1. **Irreducible:** Every state can be reached from every other state.
   In the transition matrix, this means there are no absorbing states or
   disconnected components. For financial regime models, this is almost
   always satisfied -- markets can transition between any pair of regimes.

2. **Aperiodic:** The chain does not cycle with a fixed period. Since
   $a_{ii} > 0$ for all states in typical HMM fits (self-transitions are
   always possible), the chain is aperiodic.

Together, these conditions make the chain **ergodic**, guaranteeing a
unique stationary distribution that is independent of the initial state.

---

## Kelly Criterion

### Derivation {#kelly-derivation}

The Kelly criterion maximizes the expected logarithmic growth rate of
wealth. Consider a sequence of bets where:
- We wager fraction $f$ of current wealth on each bet
- With probability $p$, we win and gain $b \cdot f$ (return $f \cdot b$)
- With probability $q = 1 - p$, we lose and lose $f$ (return $-f$)

After one bet, wealth is:
- $W \cdot (1 + fb)$ with probability $p$
- $W \cdot (1 - f)$ with probability $q$

The expected log-growth rate is:

$$G(f) = E[\log(W'/W)] = p \cdot \log(1 + fb) + q \cdot \log(1 - f)$$

Setting the derivative to zero:

$$G'(f) = \frac{pb}{1 + fb} - \frac{q}{1 - f} = 0$$

$$pb(1 - f) = q(1 + fb)$$

$$pb - pbf = q + qfb$$

$$pb - q = f(pb + qb) = fb(p + q) = fb$$

$$f^* = \frac{pb - q}{b}$$

This is the **Kelly fraction** -- the optimal bet size that maximizes
long-run geometric growth rate.

In the implementation:

```python
b = avg_win / avg_loss
p = win_rate
q = 1 - p
kelly = (p * b - q) / b
```

Note: $f^* > 0$ only when $pb > q$, i.e., when the expected value of the
bet is positive. If $f^* \leq 0$, the optimal strategy is not to bet,
and the code clamps to 0.

### Half-Kelly and Practical Considerations

Full Kelly betting is theoretically optimal but practically dangerous:

1. **Estimation error.** The true values of $p$ and $b$ are unknown and
   estimated from data. Small errors in estimation can lead to large
   overbetting. Half-Kelly provides a buffer against estimation error.

2. **Drawdown reduction.** Full Kelly produces severe drawdowns. For a
   Kelly bettor facing a geometric random walk, the expected maximum
   drawdown is approximately 50%. Half-Kelly reduces this to approximately
   25%.

3. **Growth rate tradeoff.** Half-Kelly ($f = 0.5 f^*$) retains
   approximately 75% of the optimal growth rate while reducing variance
   by approximately 50%. This can be derived from the Taylor expansion
   of $G(f)$ around $f^*$:

   $$G(f^*/2) \approx G(f^*) - \frac{1}{8} G''(f^*) (f^*)^2 \approx 0.75 \cdot G(f^*)$$

In the implementation, `kelly_fraction` (default 0.5) scales the Kelly
bet: `size = kelly * kelly_fraction`.

When entropy scaling is enabled, the final position size is further
multiplied by the regime confidence:

$$\text{size} = f^* \times \text{kelly\_fraction} \times c_t$$

This means the position is largest when the model is both confident
about the regime (high $c_t$) and the strategy has a positive edge
(high $f^*$).

---

## CVaR / Expected Shortfall

**Value at Risk (VaR)** at confidence level $\alpha$ is the threshold
such that returns fall below it with probability $\alpha$:

$$\text{VaR}_\alpha = \inf\{x : P(r \leq x) \geq \alpha\}$$

**Conditional Value at Risk (CVaR)**, also called Expected Shortfall (ES),
is the expected loss given that the loss exceeds VaR:

$$\text{CVaR}_\alpha = E[r \mid r \leq \text{VaR}_\alpha]$$

In the implementation (for $\alpha = 5\%$):

```python
var_5 = returns.quantile(0.05)
cvar_5 = returns[returns <= var_5].mean()
```

**Why CVaR over VaR?**

CVaR is a **coherent risk measure**, satisfying four desirable properties:

1. **Monotonicity:** If portfolio A always returns more than B, then
   $\text{CVaR}(A) \geq \text{CVaR}(B)$.
2. **Sub-additivity:** $\text{CVaR}(A + B) \leq \text{CVaR}(A) + \text{CVaR}(B)$.
   Diversification cannot increase risk.
3. **Positive homogeneity:** $\text{CVaR}(\lambda A) = \lambda \cdot \text{CVaR}(A)$.
4. **Translation invariance:** $\text{CVaR}(A + c) = \text{CVaR}(A) + c$.

VaR fails sub-additivity. Two individually safe portfolios can have a
combined VaR worse than the sum of individual VaRs -- a perverse result
that CVaR avoids.

---

## Bootstrap Confidence Intervals

The bootstrap provides confidence intervals for statistics without
assuming a parametric distribution for returns.

### Procedure

Given observed bar returns $r_1, r_2, \ldots, r_T$:

1. Draw a bootstrap sample $r_1^*, r_2^*, \ldots, r_T^*$ by sampling
   with replacement from the original returns.
2. Reconstruct a synthetic equity curve: $\text{equity}_t^* = \prod_{s=1}^t (1 + r_s^*)$.
3. Compute the statistic of interest (Sharpe, total return, max drawdown)
   on the synthetic curve.
4. Repeat $B$ times (default $B = 1000$).
5. The $100(1-\alpha)\%$ CI is $[\hat{\theta}^*_{\alpha/2}, \hat{\theta}^*_{1-\alpha/2}]$
   where $\hat{\theta}^*_q$ is the $q$-th quantile of the bootstrap distribution.

In the implementation:

```python
rng = np.random.default_rng(42)  # deterministic seed for reproducibility
for _ in range(1000):
    sample = rng.choice(returns, size=n, replace=True)
    # compute stats on sample
```

The seed 42 is fixed for reproducibility: the same equity curve always
produces the same CIs.

### When Bootstrap Fails

The bootstrap has known limitations:

1. **Heavy tails.** For distributions with infinite variance (common in
   financial returns), the bootstrap consistency guarantees break down.
   The CIs may be too narrow because bootstrap samples underestimate
   tail probabilities.

2. **Small samples.** With fewer than ~30 returns, the bootstrap
   distribution is too discrete to provide reliable quantile estimates.
   The implementation handles this by returning zeroed CIs when $n < 10$.

3. **Autocorrelation.** Resampling i.i.d. breaks any serial dependence
   in returns. This means the bootstrap CIs do not account for volatility
   clustering, momentum, or mean reversion. A **block bootstrap** would
   preserve serial structure by resampling contiguous blocks, but adds
   complexity (choice of block size).

4. **Non-stationarity.** If the return distribution changes over the
   sample period (e.g., a regime shift), the bootstrap provides CIs for
   the average distribution, not the current one.

---

## Walk-Forward Validation

### Why Time-Series CV Differs from IID CV

Standard $k$-fold cross-validation randomly partitions data into train
and validation sets. This works well for i.i.d. data but is invalid for
time series because:

1. **Temporal dependence.** Financial returns exhibit autocorrelation
   (momentum, mean reversion) and volatility clustering (GARCH effects).
   Random splits place future data in the training set, creating
   **information leakage**.

2. **Non-stationarity.** Market regimes change over time. A model trained
   on bull market data and validated on earlier bear market data gives
   misleading accuracy estimates.

3. **Causal ordering.** In production, the model only has access to past
   data. Validation should mimic this constraint.

### Walk-Forward Protocol

Walk-forward validation respects temporal ordering:

```
Time --->
[===== TRAIN =====][== TEST ==]
          [===== TRAIN =====][== TEST ==]
                    [===== TRAIN =====][== TEST ==]
```

At each fold:
1. Train the model on the train window (past data only)
2. Generate signals on the test window (future data)
3. Record out-of-sample performance
4. Advance the window by `step_bars` and repeat

### Anchored vs Sliding Window

The application uses a **sliding window** (not anchored):

- **Sliding:** Train window starts at `start` and has fixed length.
  As the window advances, old data is dropped. This means the model
  adapts to changing market structure, but may discard useful history.

- **Anchored:** Train window always starts at bar 0 and grows with
  each fold. The model sees all available history, but may be
  dominated by stale data.

The sliding window is preferred for financial data because market structure
changes over time. A model trained on 2020 data may perform poorly on 2024
data because volatility regimes, correlations, and market participants
have changed.

### Overlap and Step Size

The test windows overlap when `step_bars < test_window`:

```
step_bars = 50, test_window = 100:

Fold 1 test: [500, 600)
Fold 2 test: [550, 650)  <-- overlaps fold 1 by 50 bars
```

This overlap means some bars are tested multiple times (with different
models). The implementation resolves conflicts by using the signal from
the last fold that covers each bar (later writes overwrite earlier ones
in the accumulator arrays).

---

## Feature Engineering Rationale

### Why Log Returns (Not Simple Returns)

Simple returns $r_t = (P_t - P_{t-1}) / P_{t-1}$ have undesirable
properties for HMM modeling:

1. **Non-additivity.** Multi-period simple returns are multiplicative,
   not additive: $(1 + r_{1:T}) = \prod (1 + r_t)$.

2. **Asymmetry.** A +50% gain followed by a -50% loss does not return
   to zero: $(1.5)(0.5) = 0.75$.

3. **Non-normality.** Simple returns are bounded below at $-1$ (total
   loss), creating a skewed distribution.

Log returns $r_t = \ln(P_t / P_{t-1})$ fix these issues:

1. **Additivity.** $r_{1:T} = \sum r_t$.
2. **Symmetry.** A +50% log return and a -50% log return exactly cancel.
3. **Approximate normality.** For small returns, log returns are
   approximately normally distributed, which aligns with the Gaussian
   emission assumption of the HMM.

### Why Z-Score Standardization

Raw features have different scales (log returns are $O(10^{-3})$, RSI is
$O(10^1)$). Without standardization, the HMM emission covariance would
be dominated by the highest-variance feature, and the model would
effectively ignore low-variance features.

Z-scoring transforms each feature to have zero mean and unit variance:

$$z_t = \frac{x_t - \hat{\mu}}{\hat{\sigma}}$$

This ensures all features contribute equally to the Gaussian log-likelihood.

Critically, $\hat{\mu}$ and $\hat{\sigma}$ are computed from the
**training set only**, preventing information leakage from the test set.

### Why These Five Features

| Feature         | What it captures                    | Regime differentiation              |
|-----------------|-------------------------------------|--------------------------------------|
| log_return      | Direction of price change           | Bull vs bear: positive vs negative  |
| rolling_vol     | Volatility clustering               | Calm vs turbulent markets           |
| volume_change   | Participation relative to average   | Conviction behind price moves       |
| intraday_range  | Realized intra-bar volatility       | Complements rolling_vol; captures   |
|                 |                                     | instantaneous vs rolling volatility |
| rsi             | Mean-reversion tendency             | Overbought/oversold extremes        |

The features are chosen to be:
1. **Diverse in type:** Returns, volatility, volume, range, momentum.
   This gives the HMM multiple independent signals to distinguish regimes.
2. **Low mutual correlation:** The feature correlation matrix (Tab 5)
   typically shows |r| < 0.3 between most pairs, meaning each feature
   adds independent information.
3. **Stationary (or near-stationary):** Log returns and volume change
   are inherently stationary. Rolling vol and RSI are bounded. This
   helps the Gaussian emission assumption hold approximately.
4. **Computable without lookahead:** All features use only past and
   current data (causal computation).

---

## References

The following papers in the `docs/` folder informed the mathematical
approach:

- `jfallon_hmm_stock.pdf` -- Application of HMMs to stock market
  regime detection with discussion of model selection via BIC.

- `AdrovicCinoProenca.pdf` -- Walk-forward backtesting methodology
  for HMM-based trading strategies.

- `stock_hmm.pdf` -- Feature engineering for HMM-based market models
  including the rationale for log returns and volatility features.

- `djk_20190217.pdf` -- Baum-Welch algorithm tutorial with forward-
  backward derivation.

- `Wisebourt_Shaul.pdf` -- Kelly criterion application in quantitative
  trading with discussion of half-Kelly.

- `Trading_Strategy_for_Market_Situation_Estimation_B.pdf` -- Multi-
  confirmation signal gating approaches for regime-based strategies.

- `p3D_2.pdf`, `resumo.pdf`, `1199600.pdf`, `2310.03775v2.pdf`,
  `2407.19858v7.pdf` -- Additional references on HMM theory,
  information criteria, and financial applications.

- `youtube_transcript_EUSXhJNwRqI.md` -- Transcript of a presentation
  on HMM-based market regime detection that motivated several design
  choices.
