# HMM Regime Terminal

A Python-based stock research platform using **Hidden Markov Models** for market regime detection, multi-confirmation trading signals, and walk-forward backtesting with bootstrap confidence intervals.

## Features

- **BIC-optimal HMM fitting** -- tests 2-8 states with 20 random restarts, selects by Bayesian Information Criterion
- **5 engineered features** -- log return, rolling volatility, volume change, intraday range, RSI
- **8-condition signal gating** -- RSI bounds, momentum, volatility range, volume, ADX, EMA trend, MACD, regime confidence
- **Walk-forward backtesting** -- no lookahead bias, with commission/slippage modeling
- **Bootstrap confidence intervals** -- 1000-sample CIs on Sharpe, return, and drawdown
- **Kelly + entropy-scaled sizing** -- bet less when the model is uncertain
- **Streamlit dashboard** -- 5 interactive tabs

## Dashboard Tabs

| Tab | Contents |
|-----|----------|
| **Current Signal** | Regime banner, confidence %, confirmation breakdown, position size |
| **Regime Analysis** | Price chart with regime overlay, transition heatmap, return distributions, BIC curve |
| **Backtest Results** | Metric cards with 90% CIs, equity curve vs benchmark, drawdown chart |
| **Trade Log** | Sortable table with entry/exit/PnL/regime/size |
| **Model Diagnostics** | Rolling log-likelihood, entropy time series, feature correlation matrix |

## Quick Start

```bash
git clone https://github.com/az9713/HMM-trading-platform.git
cd HMM-trading-platform

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Git Bash (Windows):
source .venv/Scripts/activate
# cmd / PowerShell (Windows):
# .venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

# Install dependencies and run
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
config.yaml      # All tunable parameters
data_loader.py   # yfinance fetch + feature engineering
hmm_engine.py    # HMM fitting, BIC selection, regime analysis
strategy.py      # Confirmation signals, entry/exit, Kelly sizing
backtester.py    # Walk-forward engine, metrics, bootstrap CIs
app.py           # Streamlit dashboard (5 tabs)
```

## Documentation

| Document | Description |
|----------|-------------|
| [User Guide](docs/USER_GUIDE.md) | Complete onboarding guide from installation to productive use, covering every tab, parameter, workflow, and troubleshooting step |
| [Architecture](docs/ARCHITECTURE.md) | System architecture with data flow diagrams, module dependency graph, configuration cascade, state management, and performance model |
| [Theory](docs/THEORY.md) | Mathematical reference covering HMMs, Baum-Welch, Viterbi, BIC, Shannon entropy, Kelly criterion, CVaR, bootstrap CIs, and walk-forward validation |
| [Implementation](docs/IMPLEMENTATION.md) | Developer deep-dive into code design decisions, hmmlearn API usage, signal generation state machine, trade simulation, and testing strategy |

## Key Math

| Concept | Formula | Purpose |
|---------|---------|---------|
| BIC model selection | `BIC = -2*LL + k*ln(T)` | Data-driven state count |
| Shannon entropy | `H = -sum(p_i * log2(p_i))` | Regime uncertainty per bar |
| Expected duration | `E[d_i] = 1/(1-a_ii)` | How long regimes persist |
| Kelly criterion | `f* = (p*b-q)/b` | Optimal sizing from edge |
| CVaR (5%) | `E[r | r <= VaR_5%]` | Tail risk beyond VaR |
| Entropy scaling | `size * (1 - H/H_max)` | Reduce bets when uncertain |

## Configuration

All parameters are in `config.yaml` -- data settings, HMM hyperparameters, strategy thresholds, risk limits, and backtest windows. The sidebar in the Streamlit app overrides these at runtime. See the [User Guide](docs/USER_GUIDE.md) for a complete parameter reference.

## Research

The `docs/` folder contains 11 academic papers and a YouTube transcript that informed the mathematical approach. See the [Theory](docs/THEORY.md) document for details on how each mathematical concept is applied.
