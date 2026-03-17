# Hidden Markov Models for Stock Trading: Research Report

## 1. Executive Summary

Hidden Markov Models (HMMs) are probabilistic models that treat financial markets as systems transitioning between unobservable "regimes" (e.g., bull, bear, sideways), each producing different observable market behavior. This report synthesizes findings from 11 academic papers and extensive literature review to inform the design of an HMM-based stock research tool using yfinance data.

The core value proposition of HMMs in finance is **regime detection** — not price prediction per se, but characterizing the current market environment to inform strategy selection, position sizing, and risk management.

---

## 2. Theoretical Foundation

### 2.1 What is a Hidden Markov Model?

An HMM is a doubly stochastic process consisting of:

1. **Hidden states (Z)**: Unobservable market regimes (e.g., bull, bear, sideways)
2. **Observations (Y)**: Observable market data (returns, volatility, volume)

The model is fully specified by three components:

- **Initial state distribution (π)**: Probability of starting in each regime
- **Transition matrix (A)**: N×N matrix where A[i][j] = P(state j at t+1 | state i at t). Diagonal entries are typically high (regimes are "sticky")
- **Emission probabilities (B)**: For each hidden state, a probability distribution over observations. In a Gaussian HMM, each state emits from N(μ_j, Σ_j)

### 2.2 Key Assumptions

1. **Homogeneous Markov assumption**: P(i_{t+1} | i_t, i_{t-1}, ...) = P(i_{t+1} | i_t) — future state depends only on current state
2. **Observation independence**: P(o_t | i_t, i_{t-1}, ..., o_{t-1}, ...) = P(o_t | i_t) — observations depend only on the current hidden state

### 2.3 Core Algorithms

| Algorithm | Purpose | Method |
|-----------|---------|--------|
| **Forward-Backward** | Compute P(state at time t \| all observations) | Dynamic programming, forward/backward passes |
| **Baum-Welch (EM)** | Estimate model parameters (A, B, π) | Iterative expectation-maximization |
| **Viterbi** | Find most likely state sequence | Dynamic programming (max instead of sum) |

**Baum-Welch** iteratively refines parameters to maximize likelihood of observed data. It converges to a local optimum (not global), making initialization critical.

**Viterbi** finds the single most likely sequence of hidden states. In trading, this gives regime labels for each historical day.

---

## 3. HMM Approaches in Stock Trading (from Literature)

### 3.1 Regime Detection (Most Common Application)

The dominant use case across the literature:

**Paper: "Trading Strategy for Market Situation Estimation Based on HMM" (Chen, Yi, Zhao 2020)**
- Models 3 hidden states: bull, mixed (shock), bear market
- Observable features: price rise, price fluctuation, price fall (discretized)
- Uses technical features (closing price, moving averages, exponential MA) rather than fundamental data
- Tests both 3-state and 5-state (finer-grained) models
- Finding: Finer-grained division (5 states) improves strategy profitability

**Paper: "A Hidden Markov Model of Momentum" (Daniel, Jagannathan, Kim 2019)**
- Two-state HMM: **calm** vs **turbulent** market states
- Applied to cross-sectional momentum strategies
- Key finding: In turbulent states, the short side of momentum has high beta and convexity, making crashes more likely
- HMM-based momentum timing avoids crashes and achieves superior out-of-sample risk-adjusted performance
- HMM turbulent-state probability forecasts momentum losses better than GARCH volatility or past returns

### 3.2 Price Prediction

**Paper: "Hidden Markov Models for Stock Market Prediction" (Catello et al., 2023 — arXiv:2310.03775)**
- Trains HMM to predict closing price from opening price and previous day's prices
- Evaluation metrics: MAPE (Mean Absolute Percentage Error) and DPA (Directional Prediction Accuracy)
- Uses continuous HMM with fractional changes as observations

**Paper: "Predicting Stock Values using HMM" (stock_hmm.pdf, IEEE 2012)**
- Uses fractional change in stock value plus intra-day high/low as features
- Maximum a Posteriori (MAP) decision over all possible next-day values
- Continuous HMM trained on these features
- Compared favorably to ANN approaches using MAPE

**Paper: "Making Profit in the Stock Market Using HMMs" (Fallon)**
- Simple approach: train HMM on 2009-2011 data, test on 2011-2012
- Uses Baum-Welch for training, likelihood-matching for prediction
- Achieved 26% profit on $100,000 over 1 year trading 10 stocks
- Stocks: AMD, BAC, CSCO, C, F, GE, INTC, MSFT, PFE, SIRI

### 3.3 Portfolio Selection

**Paper: "Creating Stock Portfolios Using Hidden Markov Models" (Ji, Neerchal)**
- 2-state HMM (buy/sell) for weekly price changes of individual stocks
- Each stock modeled independently: Y_{k,t} | Z_{k,t}=j ~ N(μ_{k,j}, σ²_{k,j})
- Derives closed-form expressions for E(R_k) and Var(R_k) under HMM
- Portfolio selection balances expected return vs variance (Markowitz E-V framework)
- Out-of-sample testing 2010-2018, compared to S&P 500

### 3.4 Hybrid HMM + Neural Network

**Paper: "AI-Powered Energy Algorithmic Trading: Integrating HMMs with Neural Networks" (Monteiro, 2025 — arXiv:2407.19858)**
- Dual-model alpha system: HMM for regime detection + neural network for pattern learning
- Combined with Black-Litterman portfolio optimization
- Tested on energy sector stocks during COVID (2019-2022)
- Achieved 83% return with Sharpe ratio of 0.77
- Two risk management models for volatile periods
- Uses QuantConnect platform for backtesting

### 3.5 High-Frequency / Order Book Applications

**Paper: "Hierarchical Hidden Markov Model of High-Frequency Market Regimes" (Wisebourt, 2011)**
- Hierarchical HMM applied to tick-by-tick data from Toronto Stock Exchange
- Models run and reversal regimes using limit order book imbalance
- Uses TSX60 index constituents
- Proposes statistical measure of order book imbalance as feature vector

**Paper: "Capturing Order Imbalance with HMM: SET50 and KOSPI50" (Wu, Siwasarit)**
- HMM captures states of order imbalance at intraday frequencies
- Compares results between Thai (SET50) and Korean (KOSPI50) markets
- Tests across different frequencies

### 3.6 Discrete HMM with Technical Indicators

**Paper: "Stock Trading with Discrete HMMs" (resumo.pdf)**
- Discrete HMM (not Gaussian) — discretizes price changes into "rise" and "drop"
- Combines daily and weekly windows
- Uses RSI (Relative Strength Index) to choose between different DHMMs
- Trained on S&P 500 (2003-2009), tested 2009-2017
- Achieved 356% return vs 199% for S&P 500 buy-and-hold
- Key insight: Discrete HMMs are more robust because categorical predictions about direction are more stable than exact price predictions

**Paper: "Stock Trading with HMMs" (Adrovic, Di Cino, Proenca)**
- Educational implementation using Electronic Arts stock data
- Covers Forward-Backward, Baum-Welch, and Viterbi algorithms
- Python implementation with regime prediction

---

## 4. Feature Engineering: What Observations to Use

### 4.1 Common Features from Literature

| Feature Category | Specific Features | Source Papers |
|-----------------|-------------------|---------------|
| **Returns** | Daily log returns, fractional changes, weekly % change | All papers |
| **Volatility** | Rolling std dev (21-day), realized volatility | Daniel et al., Monteiro |
| **Price levels** | Open, High, Low, Close (OHLC) | Catello et al., stock_hmm |
| **Volume** | Trading volume, dollar volume, volume changes | Wisebourt, Wu & Siwasarit |
| **Technical indicators** | RSI, Moving Averages, EMA, Rate of Change | Chen et al., resumo |
| **Order book** | Bid-ask imbalance, limit order book depth | Wisebourt, Wu & Siwasarit |
| **Derived** | Intra-day range (High-Low)/Close, fractional change | stock_hmm, Ji & Neerchal |

### 4.2 Recommended Feature Set for Our Tool

Based on the literature, a practical feature set using yfinance data:

1. **Daily log returns**: ln(Close_t / Close_{t-1})
2. **Rolling volatility**: 21-day rolling standard deviation of log returns
3. **Volume change**: Fractional change in volume
4. **Intra-day range**: (High - Low) / Close — captures daily price spread
5. **RSI**: 14-day Relative Strength Index (optional, for signal filtering)

---

## 5. Model Design Decisions

### 5.1 Number of Hidden States

| States | Interpretation | When to Use |
|--------|---------------|-------------|
| **2** | Bull/Bear or Calm/Turbulent | Simplest, most robust; best starting point |
| **3** | Bull/Sideways/Bear | Good balance of granularity and stability |
| **4-5** | Fine-grained regimes | Only with sufficient data; risk of overfitting |

**Model selection**: Use BIC (Bayesian Information Criterion) and AIC to choose optimal state count. Chen et al. (2020) found that 5 states improved profitability over 3 states, but most literature recommends starting with 2-3.

### 5.2 Gaussian vs Discrete HMM

| Type | Pros | Cons |
|------|------|------|
| **Gaussian HMM** | Natural for continuous returns; captures mean + variance per regime | Assumes normality within regimes; misses fat tails |
| **Discrete HMM** | More robust to noise; directional predictions more stable | Loses information in discretization |
| **GMM-HMM** | Handles fat tails with mixture of Gaussians per state | More parameters; slower to fit |

**Recommendation**: Start with Gaussian HMM (standard in literature, well-supported by hmmlearn). Consider GMM-HMM for robustness.

### 5.3 Training Window

- **Expanding window**: Train on all data up to time t, predict t+1. Most common in literature.
- **Rolling window**: Train on fixed-length window (e.g., 2-3 years). Better for non-stationary markets.
- **Walk-forward validation**: Essential to avoid lookahead bias. Train → predict → advance → retrain.

---

## 6. Trading Strategies Built on HMM

### 6.1 Regime-Based Position Sizing

```
If P(bull) > threshold:
    Go long (full position)
If P(bear) > threshold:
    Go to cash or short
If uncertain (no regime > threshold):
    Reduce position / stay neutral
```

Typical threshold: 50-60% regime probability.

### 6.2 Multi-Asset Rotation (from recent research)

- **Bull regime**: Overweight equities
- **Bear regime**: Rotate to Treasuries, gold, or cash
- **Sideways**: Mean-reversion strategies, reduced exposure

### 6.3 Momentum Timing (Daniel et al. 2019)

- Use HMM turbulent-state probability to scale momentum exposure
- When P(turbulent) is high, reduce or eliminate momentum positions
- Avoids catastrophic momentum crashes

### 6.4 Signal Generation (Chen et al. 2020)

1. Estimate current market state using Viterbi or forward-backward
2. If bull → buy signal; if bear → sell signal; if mixed → hold
3. Combine with technical indicators (RSI, MA crossover) for confirmation

---

## 7. Implementation Plan

### 7.1 Technology Stack

| Component | Tool |
|-----------|------|
| Data source | `yfinance` |
| HMM fitting | `hmmlearn` (GaussianHMM, GMMHMM) |
| Data processing | `pandas`, `numpy` |
| Technical indicators | `ta` or manual calculation |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Backtesting | Custom walk-forward framework |
| Model selection | `sklearn` (BIC/AIC comparison) |

### 7.2 Proposed Workflow

```
1. Data Acquisition (yfinance)
   └── Download OHLCV data for target stock(s)

2. Feature Engineering
   ├── Log returns
   ├── Rolling volatility (21-day)
   ├── Volume change
   ├── Intra-day range
   └── Optional: RSI, momentum indicators

3. Model Fitting
   ├── Fit GaussianHMM with n_components = [2, 3, 4]
   ├── Compare models using BIC/AIC
   ├── Multiple random restarts (handle local optima)
   └── Select best model

4. Regime Analysis
   ├── Decode states (Viterbi)
   ├── Compute regime probabilities (forward-backward)
   ├── Label regimes by mean return & volatility
   └── Visualize regime overlay on price chart

5. Signal Generation
   ├── Map regimes to trading signals
   ├── Apply confidence thresholds
   └── Generate buy/sell/hold recommendations

6. Backtesting (Walk-Forward)
   ├── Rolling/expanding window training
   ├── Out-of-sample prediction
   ├── Track portfolio performance
   └── Compare to buy-and-hold benchmark

7. Risk Analysis
   ├── Sharpe ratio, max drawdown, volatility
   ├── Regime-conditional statistics
   └── Transition probability analysis
```

### 7.3 Key Implementation Details

- **Initialization**: Run `GaussianHMM.fit()` multiple times with different `random_state` values; select model with best log-likelihood
- **Covariance type**: Use `covariance_type="full"` for multivariate features to capture correlations
- **Convergence**: Set `n_iter=1000` with reasonable tolerance (`tol=1e-4`)
- **Label switching**: After fitting, sort states by mean return to ensure consistent labeling (state 0 = lowest return = bear, etc.)
- **Lookahead bias prevention**: Never train on future data. Use strict walk-forward validation.

---

## 8. Strengths and Limitations

### 8.1 Strengths

1. **Principled regime detection**: Formal probabilistic framework, not ad hoc rules
2. **Handles non-stationarity**: Different regimes capture shifting market dynamics
3. **Probabilistic output**: Regime probabilities enable gradual position adjustments
4. **Interpretability**: State means and variances are directly meaningful (e.g., "State 1: μ=+0.04%/day, σ=0.8%")
5. **Reproduces stylized facts**: Excess kurtosis, volatility clustering, dynamic correlations
6. **Proven results**: Multiple papers show outperformance vs buy-and-hold, especially in drawdown reduction

### 8.2 Limitations

1. **Overfitting**: Too many states chases noise. Always validate out-of-sample.
2. **Stationarity assumption**: Transition matrix and emission parameters assumed constant — breaks down with structural changes
3. **Gaussian tails**: Standard HMM underestimates extreme events. GMM-HMM partially addresses this.
4. **Memoryless property**: Only current state matters — ignores multi-day momentum/sentiment
5. **Detection latency**: Regime changes often detected with a lag; significant losses may occur before bear detection
6. **Local optima**: Baum-Welch converges to local optima; multiple restarts required
7. **Label switching**: State labels can change across training runs

### 8.3 Mitigation Strategies

| Risk | Mitigation |
|------|-----------|
| Overfitting | BIC/AIC model selection; walk-forward validation; limit to 2-3 states |
| Latency | Use regime probabilities (not hard labels); combine with leading indicators |
| Local optima | Multiple random restarts; compare log-likelihoods |
| Fat tails | Use GMM-HMM or Student-t emissions |
| Non-stationarity | Rolling window retraining; adaptive/online HMM |

---

## 9. Advanced Directions (from Recent Research)

### 9.1 HMM + LSTM Hybrid
- HMM labels regimes unsupervised; LSTM predicts future regimes from historical patterns
- Entropy-weighted Bayesian model averaging to fuse outputs
- Achieved >50% volatility reduction on semiconductor equities (Kemper, 2025)

### 9.2 HMM + Reinforcement Learning
- HMM identifies regime; RL agent learns optimal allocation conditioned on regime
- Multi-asset (equities, Treasuries, gold) tactical allocation
- Transparent, rules-based approach (2025 study)

### 9.3 Hidden Semi-Markov Models (HSMM)
- Allow arbitrary state duration distributions (vs geometric in standard HMM)
- More realistic regime durations, but may not improve out-of-sample performance (Baitinger & Hoch, 2024)

### 9.4 Hierarchical HMM
- Multiple levels of hidden states (macro regime → micro regime)
- Applied to high-frequency data with order book information (Wisebourt, 2011)

---

## 10. Summary of Literature

| Paper | Approach | Key Finding |
|-------|----------|-------------|
| Ji & Neerchal | 2-state HMM portfolio selection | Closed-form E(R) and Var(R) under HMM; outperformed S&P 500 |
| Catello et al. (2023) | Continuous HMM price prediction | MAPE and DPA metrics for evaluation |
| Monteiro (2025) | HMM + Neural Net + Black-Litterman | 83% return, Sharpe 0.77 on energy stocks |
| Adrovic et al. | Educational HMM implementation | Clear exposition of algorithms |
| Daniel et al. (2019) | 2-state HMM for momentum timing | Avoids momentum crashes; superior risk-adjusted returns |
| Fallon | Simple HMM daily trading | 26% profit over 1 year on 10 stocks |
| Wu & Siwasarit | HMM on order imbalance | Cross-market comparison (SET50 vs KOSPI50) |
| resumo | Discrete HMM + RSI | 356% return vs 199% S&P 500 (2009-2017) |
| stock_hmm (IEEE 2012) | MAP-HMM with fractional changes | Outperformed ANN on MAPE |
| Chen et al. (2020) | 3-5 state HMM trading strategy | Finer-grained states improve profitability |
| Wisebourt (2011) | Hierarchical HMM on LOB | Tick-by-tick regime detection on TSX60 |

---

## 11. Recommended Tool Architecture

Based on this research, the proposed HMM stock research tool should include:

1. **Data Module**: Fetch and cache OHLCV data via yfinance
2. **Feature Module**: Compute log returns, volatility, volume change, intra-day range, RSI
3. **Model Module**: Fit Gaussian HMM with BIC-based state selection, multiple restarts
4. **Analysis Module**: Regime decoding, probability estimation, regime statistics
5. **Signal Module**: Map regimes to buy/sell/hold signals with confidence thresholds
6. **Backtest Module**: Walk-forward validation with performance metrics
7. **Visualization Module**: Price charts with regime overlay, transition diagrams, performance plots
8. **Report Module**: Generate summary of regime analysis and trading signals

This architecture covers the full pipeline from data acquisition to actionable trading signals, incorporating best practices from the literature.
