"""
app.py — Streamlit dashboard with 6 tabs for HMM Regime Terminal.

Tabs: Current Signal, Regime Analysis, Backtest Results, Trade Log,
      Model Diagnostics, Fundamentals.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml

from data_loader import fetch_ohlcv, compute_features, standardize, get_feature_matrix
from hmm_engine import RegimeDetector
from strategy import SignalGenerator
from backtester import WalkForwardBacktester
from fundamentals import FundamentalAnalyzer
from regime_analyzer import RegimeTransitionAnalyzer
from multi_timeframe import run_multi_timeframe_analysis, TIMEFRAME_ORDER, DEFAULT_WEIGHTS
from monte_carlo import MonteCarloEngine
from changepoint import (
    BayesianChangepoint, ChangepointHMMFusion,
    create_bocpd_from_config, create_fusion_from_config,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="HMM Regime Terminal", layout="wide")

# ── Global CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

:root {
    --term-bg: #0a0e17;
    --term-surface: #111827;
    --term-border: #1e293b;
    --term-green: #00e599;
    --term-amber: #f59e0b;
    --term-red: #ef4444;
    --term-blue: #3b82f6;
    --term-cyan: #06b6d4;
    --term-text: #e2e8f0;
    --term-muted: #64748b;
    --term-glow: rgba(0, 229, 153, 0.15);
}

/* Hide default Streamlit title */
.stApp > header { background: transparent; }

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1729 0%, #0a0e17 100%);
    border-right: 1px solid var(--term-border);
}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'JetBrains Mono', monospace;
    color: var(--term-green);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid var(--term-border);
    padding-bottom: 0.4rem;
    margin-top: 1.2rem;
}

/* Sidebar logo area */
.sidebar-brand {
    text-align: center;
    padding: 1.5rem 0.5rem 1rem;
    border-bottom: 1px solid var(--term-border);
    margin-bottom: 0.5rem;
}
.sidebar-brand .logo-icon {
    font-size: 2rem;
    display: block;
    margin-bottom: 0.3rem;
    line-height: 1;
}
.sidebar-brand .logo-wordmark {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 1.05rem;
    color: #f1f5f9;
    letter-spacing: 0.08em;
}
.sidebar-brand .logo-sub {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 300;
    font-size: 0.6rem;
    color: var(--term-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.15rem;
}
.sidebar-brand .version-badge {
    display: inline-block;
    margin-top: 0.5rem;
    background: rgba(0, 229, 153, 0.1);
    border: 1px solid rgba(0, 229, 153, 0.25);
    border-radius: 9999px;
    padding: 0.15rem 0.55rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    color: var(--term-green);
    letter-spacing: 0.08em;
}

/* ── Landing page styles ── */
.landing-hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}
.landing-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(0,229,153,0.06) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}
.hero-overline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--term-green);
    letter-spacing: 0.35em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    position: relative; z-index: 1;
}
.hero-title {
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    font-size: 3.2rem;
    line-height: 1.1;
    color: #f8fafc;
    margin-bottom: 0.5rem;
    position: relative; z-index: 1;
}
.hero-title .accent {
    background: linear-gradient(135deg, var(--term-green) 0%, var(--term-cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 1.15rem;
    color: var(--term-muted);
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
    position: relative; z-index: 1;
}
.hero-divider {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, var(--term-green), var(--term-cyan));
    margin: 2rem auto;
    border-radius: 1px;
}

/* Pipeline steps */
.pipeline-section { margin: 1rem auto 2.5rem; max-width: 1000px; }
.pipeline-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--term-muted);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 1rem;
}
.pipeline-row {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0;
}
.pipe-step {
    display: flex;
    align-items: center;
    gap: 0;
}
.pipe-node {
    background: var(--term-surface);
    border: 1px solid var(--term-border);
    border-radius: 8px;
    padding: 0.65rem 1.1rem;
    text-align: center;
    transition: border-color 0.3s, box-shadow 0.3s;
    min-width: 120px;
}
.pipe-node:hover {
    border-color: var(--term-green);
    box-shadow: 0 0 20px var(--term-glow);
}
.pipe-node .step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    color: var(--term-green);
    letter-spacing: 0.15em;
    margin-bottom: 0.2rem;
}
.pipe-node .step-name {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    color: #f1f5f9;
}
.pipe-node .step-detail {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    color: var(--term-muted);
    margin-top: 0.15rem;
}
.pipe-arrow {
    font-family: 'JetBrains Mono', monospace;
    color: var(--term-border);
    font-size: 1.2rem;
    padding: 0 0.35rem;
    user-select: none;
}

/* Math concept cards */
.concepts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    max-width: 1100px;
    margin: 0 auto 2.5rem;
}
.concept-card {
    background: var(--term-surface);
    border: 1px solid var(--term-border);
    border-radius: 10px;
    padding: 1.25rem 1.1rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s, border-color 0.3s, box-shadow 0.3s;
}
.concept-card:hover {
    transform: translateY(-3px);
    border-color: var(--term-green);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3), 0 0 20px var(--term-glow);
}
.concept-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.concept-card.card-green::before { background: linear-gradient(90deg, var(--term-green), transparent); }
.concept-card.card-amber::before { background: linear-gradient(90deg, var(--term-amber), transparent); }
.concept-card.card-blue::before  { background: linear-gradient(90deg, var(--term-blue), transparent); }
.concept-card.card-cyan::before  { background: linear-gradient(90deg, var(--term-cyan), transparent); }
.concept-card.card-red::before   { background: linear-gradient(90deg, var(--term-red), transparent); }

.concept-icon {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    display: block;
}
.concept-title {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.85rem;
    color: #f1f5f9;
    margin-bottom: 0.3rem;
}
.concept-formula {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--term-green);
    background: rgba(0,229,153,0.06);
    padding: 0.3rem 0.5rem;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 0.4rem;
    border: 1px solid rgba(0,229,153,0.1);
}
.concept-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    color: var(--term-muted);
    line-height: 1.5;
}

/* Getting started section */
.getting-started {
    max-width: 700px;
    margin: 0 auto 3rem;
    text-align: center;
    padding: 2rem 1.5rem;
    background: var(--term-surface);
    border: 1px solid var(--term-border);
    border-radius: 12px;
    position: relative;
    overflow: hidden;
}
.getting-started::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--term-green), var(--term-cyan), var(--term-blue));
}
.gs-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--term-green);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.gs-steps {
    text-align: left;
    display: inline-block;
}
.gs-step {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    margin-bottom: 0.85rem;
}
.gs-step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--term-bg);
    background: var(--term-green);
    width: 20px; height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-weight: 700;
    margin-top: 1px;
}
.gs-step-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    color: var(--term-text);
    line-height: 1.45;
}
.gs-step-text strong {
    color: #f1f5f9;
}

/* Footer */
.landing-footer {
    text-align: center;
    padding: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--term-muted);
    letter-spacing: 0.1em;
}
.landing-footer .dot {
    display: inline-block;
    width: 4px; height: 4px;
    background: var(--term-green);
    border-radius: 50%;
    margin: 0 0.6rem;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

st.title("HMM Regime Terminal")

# ── Load config ──────────────────────────────────────────────────────────────

@st.cache_data
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

base_config = load_config()

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="logo-icon">
            <svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="1" y="1" width="34" height="34" rx="8" stroke="#1e293b" stroke-width="1" fill="#111827"/>
                <path d="M8 24 L12 18 L16 20 L20 12 L24 16 L28 10" stroke="#00e599" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
                <circle cx="12" cy="18" r="2" fill="#00e599" opacity="0.6"/>
                <circle cx="20" cy="12" r="2" fill="#00e599" opacity="0.6"/>
                <circle cx="28" cy="10" r="2" fill="#00e599"/>
            </svg>
        </span>
        <div class="logo-wordmark">HMM REGIME</div>
        <div class="logo-sub">Terminal</div>
        <span class="version-badge">v1.0.0</span>
    </div>
    """, unsafe_allow_html=True)

    st.header("Data Settings")
    ticker = st.text_input("Ticker", base_config["data"]["default_ticker"])
    interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"],
                            index=["1m", "5m", "15m", "1h", "1d"].index(base_config["data"]["default_interval"]))
    lookback = st.slider("Lookback (days)", 7, 730, base_config["data"]["default_lookback_days"])

    st.header("HMM Settings")
    min_states = st.slider("Min states", 2, 6, base_config["hmm"]["min_states"])
    max_states = st.slider("Max states", 3, 10, base_config["hmm"]["max_states"])
    model_type = st.selectbox("Model type", ["gaussian", "gmm"],
                              index=0 if base_config["hmm"]["model_type"] == "gaussian" else 1)
    cov_type = st.selectbox("Covariance", ["full", "diag", "tied", "spherical"],
                            index=["full", "diag", "tied", "spherical"].index(base_config["hmm"]["covariance_type"]))
    n_restarts = st.slider("Random restarts", 5, 50, base_config["hmm"]["n_restarts"])

    st.header("Strategy Settings")
    min_conf = st.slider("Min confirmations", 1, 8, base_config["strategy"]["min_confirmations"])
    cooldown = st.slider("Cooldown bars", 0, 20, base_config["strategy"]["cooldown_bars"])
    min_hold = st.slider("Min hold bars", 1, 50, base_config["strategy"]["min_hold_bars"])
    min_regime_conf = st.slider("Min regime confidence", 0.0, 1.0,
                                base_config["strategy"]["confirmations"]["min_confidence"], 0.05)

    st.header("Risk Settings")
    use_kelly = st.checkbox("Kelly sizing", base_config["risk"]["use_kelly"])
    use_entropy = st.checkbox("Entropy scaling", base_config["risk"]["use_entropy_scaling"])
    max_leverage = st.slider("Max leverage", 0.5, 5.0, base_config["risk"]["max_leverage"], 0.5)

    st.header("Backtest Settings")
    train_window = st.slider("Train window (bars)", 100, 2000,
                             base_config["backtest"]["train_window_bars"], 50)
    test_window = st.slider("Test window (bars)", 20, 500,
                            base_config["backtest"]["test_window_bars"], 10)
    step_bars = st.slider("Step size (bars)", 10, 200,
                          base_config["backtest"]["step_bars"], 10)

    st.header("Changepoint Detection")
    enable_bocpd = st.checkbox("Enable BOCPD", True, help="Bayesian Online Changepoint Detection for faster regime transition signals")
    bocpd_hazard = st.slider("Hazard rate (bars)", 20, 500, base_config.get("changepoint", {}).get("hazard_rate", 100), 10,
                             help="Expected bars between changepoints")

    st.header("Multi-Timeframe")
    enable_mtf = st.checkbox("Enable Multi-Timeframe Fusion", False)
    mtf_intervals_options = ["1m", "5m", "15m", "1h", "1d"]
    mtf_intervals = st.multiselect(
        "Timeframes to fuse",
        mtf_intervals_options,
        default=["1h", "1d"],
        help="Select 2+ timeframes. Base (fastest) is used as primary timeline.",
    )

    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

# ── Build runtime config ─────────────────────────────────────────────────────

def build_config():
    cfg = base_config.copy()
    cfg["data"] = {**base_config["data"], "default_ticker": ticker,
                   "default_interval": interval, "default_lookback_days": lookback}
    cfg["hmm"] = {**base_config["hmm"], "min_states": min_states, "max_states": max_states,
                  "model_type": model_type, "covariance_type": cov_type, "n_restarts": n_restarts}
    cfg["strategy"] = {**base_config["strategy"], "min_confirmations": min_conf,
                       "cooldown_bars": cooldown, "min_hold_bars": min_hold,
                       "confirmations": {**base_config["strategy"]["confirmations"],
                                         "min_confidence": min_regime_conf}}
    cfg["risk"] = {**base_config["risk"], "use_kelly": use_kelly,
                   "use_entropy_scaling": use_entropy, "max_leverage": max_leverage}
    cfg["backtest"] = {**base_config["backtest"], "train_window_bars": train_window,
                       "test_window_bars": test_window, "step_bars": step_bars}
    cfg["changepoint"] = {**base_config.get("changepoint", {}),
                          "enabled": enable_bocpd, "hazard_rate": bocpd_hazard}
    return cfg

# ── Regime color map ─────────────────────────────────────────────────────────

REGIME_COLORS = {
    "crash": "#d32f2f",
    "bear": "#f57c00",
    "neutral": "#9e9e9e",
    "bull": "#388e3c",
    "bull_run": "#1565c0",
    "unknown": "#e0e0e0",
}

def get_regime_color(label: str) -> str:
    for key, color in REGIME_COLORS.items():
        if key in label:
            return color
    return "#e0e0e0"

# ── Main analysis ────────────────────────────────────────────────────────────

if run_btn:
    config = build_config()
    feature_cols = config["data"]["features"]

    with st.spinner("Fetching data..."):
        try:
            df = fetch_ohlcv(ticker, interval, lookback)
            df = compute_features(df, config["data"])
        except (ValueError, Exception) as e:
            st.error(f"Failed to fetch data: {e}")
            st.info("Check that the ticker is valid on Yahoo Finance (e.g., GOOG not GOOGL, BTC-USD not BTC).")
            st.stop()

    st.success(f"Loaded {len(df)} bars for {ticker} ({interval})")

    # ── Fit HMM on full dataset for current signal / regime analysis ──
    with st.spinner("Fitting HMM (BIC selection with random restarts)..."):
        train_z, _, stats = standardize(df, cols=feature_cols)
        X = get_feature_matrix(train_z, feature_cols)

        detector = RegimeDetector(config)
        bic_scores = detector.fit_and_select(X)
        states, posteriors = detector.decode(X)
        labels = detector.label_regimes(X)
        entropy, confidence = detector.shannon_entropy(posteriors)
        regime_stats = detector.regime_statistics()
        transmat = detector.transition_matrix()
        ll_series = detector.log_likelihood_series(X, window=50)

    st.success(f"Selected {detector.n_states} states (BIC)")

    # ── Run BOCPD if enabled ──
    fusion_result = None
    if config.get("changepoint", {}).get("enabled", False):
        with st.spinner("Running Bayesian Changepoint Detection..."):
            bocpd = create_bocpd_from_config(config)
            # Run on log returns (feature 0, the primary signal)
            log_returns = X[:, 0]
            bocpd_result = bocpd.detect(log_returns)

            fusion = create_fusion_from_config(config)
            fusion_result = fusion.fuse(
                bocpd_result, entropy, confidence, posteriors,
                n_states=detector.n_states,
                base_hysteresis=config.get("strategy", {}).get("hysteresis_bars", 3),
            )
        st.success(
            f"BOCPD: {len(bocpd_result.changepoint_indices)} changepoints detected "
            f"(threshold={bocpd.threshold})"
        )

    # ── Compute confirmations and signals ──
    sig_gen = SignalGenerator(config)
    df_conf = sig_gen.compute_confirmations(df)
    adaptive_hyst = fusion_result["adaptive_hysteresis"] if fusion_result else None
    signals = sig_gen.generate_signals(
        df_conf, states, posteriors, labels, confidence,
        adaptive_hysteresis=adaptive_hyst,
    )

    # Add regime labels to df
    df["regime"] = [labels.get(s, "unknown") for s in states]
    df["confidence"] = confidence
    df["entropy"] = entropy
    df["signal"] = signals.values

    # ── Tabs ─────────────────────────────────────────────────────────────

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "Current Signal", "Regime Analysis", "Regime Transitions",
        "Backtest Results", "Trade Log", "Model Diagnostics", "Fundamentals",
        "Multi-Timeframe", "Monte Carlo", "Changepoint Detection",
    ])

    # ── Tab 1: Current Signal ────────────────────────────────────────────

    with tab1:
        last = df.iloc[-1]
        regime = last["regime"]
        conf_pct = last["confidence"] * 100
        signal_val = int(last["signal"])
        signal_text = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(signal_val, "FLAT")
        signal_color = {1: "green", -1: "red", 0: "gray"}.get(signal_val, "gray")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"### Regime: **{regime.upper()}**")
            st.markdown(f"<div style='background:{get_regime_color(regime)};padding:20px;"
                        f"border-radius:10px;text-align:center;color:white;font-size:24px;"
                        f"font-weight:bold'>"
                        f"{regime.upper()}</div>", unsafe_allow_html=True)
        with col2:
            st.metric("Confidence", f"{conf_pct:.1f}%")
            signal_bg = {1: "#388e3c", -1: "#d32f2f", 0: "#616161"}.get(signal_val, "#616161")
            st.markdown(f"<div style='padding:8px 16px;border-radius:8px;background:{signal_bg};"
                        f"color:white;font-size:20px;font-weight:bold;text-align:center;"
                        f"margin-top:4px'>{signal_text}</div>",
                        unsafe_allow_html=True)
        with col3:
            pos_size = sig_gen.compute_position_size(last["confidence"])
            st.metric("Position Size", f"{pos_size:.1%}")

        st.subheader("Confirmation Breakdown")
        conf_cols = [c for c in df_conf.columns if c.startswith("conf_")]
        conf_data = []
        for c in conf_cols:
            name = c.replace("conf_", "").replace("_", " ").title()
            val = bool(df_conf[c].iloc[-1])
            conf_data.append({"Condition": name, "Met": val})
        conf_df = pd.DataFrame(conf_data)
        st.dataframe(conf_df, use_container_width=True, hide_index=True)

        st.metric("Total Confirmations", f"{int(df_conf['n_confirmations'].iloc[-1])} / {len(conf_cols)}")

    # ── Tab 2: Regime Analysis ───────────────────────────────────────────

    with tab2:
        # Price chart with regime overlay
        st.subheader("Price with Regime Overlay")
        fig = go.Figure()

        date_col = "Date" if "Date" in df.columns else "Datetime" if "Datetime" in df.columns else df.index.name
        x_vals = df["Date"] if "Date" in df.columns else df.index

        # Add regime background shading
        unique_regimes = df["regime"].unique()
        for regime_label in unique_regimes:
            mask = df["regime"] == regime_label
            indices = df.index[mask]
            if len(indices) == 0:
                continue
            # Create regime overlay as scatter with fill
            fig.add_trace(go.Scatter(
                x=x_vals[mask], y=df["Close"][mask],
                mode="markers", marker=dict(size=3, color=get_regime_color(regime_label)),
                name=regime_label, legendgroup=regime_label,
            ))

        fig.add_trace(go.Scatter(
            x=x_vals, y=df["Close"], mode="lines",
            line=dict(color="black", width=1), name="Price",
            showlegend=True,
        ))
        fig.update_layout(height=500, xaxis_title="", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        # Transition heatmap
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Transition Matrix")
            state_labels = [labels.get(i, f"S{i}") for i in range(detector.n_states)]
            fig_tm = px.imshow(
                transmat, x=state_labels, y=state_labels,
                color_continuous_scale="Blues", text_auto=".2f",
                labels=dict(x="To", y="From", color="P"),
            )
            fig_tm.update_layout(height=400)
            st.plotly_chart(fig_tm, use_container_width=True)

        with col2:
            st.subheader("Regime Statistics")
            st.dataframe(regime_stats.round(4), use_container_width=True, hide_index=True)

        # Per-regime return distributions
        st.subheader("Return Distributions by Regime")
        fig_dist = go.Figure()
        for regime_label in unique_regimes:
            mask = df["regime"] == regime_label
            rets = df.loc[mask, "log_return"]
            fig_dist.add_trace(go.Histogram(
                x=rets, name=regime_label, opacity=0.6,
                marker_color=get_regime_color(regime_label),
            ))
        fig_dist.update_layout(barmode="overlay", height=400,
                               xaxis_title="Log Return", yaxis_title="Count")
        st.plotly_chart(fig_dist, use_container_width=True)

        # BIC/AIC curve
        st.subheader("BIC / AIC Model Selection")
        bic_df = pd.DataFrame({
            "States": list(bic_scores.keys()),
            "BIC": list(bic_scores.values()),
            "AIC": [detector.aic_scores[k] for k in bic_scores.keys()],
        })
        fig_bic = go.Figure()
        fig_bic.add_trace(go.Scatter(x=bic_df["States"], y=bic_df["BIC"],
                                     mode="lines+markers", name="BIC"))
        fig_bic.add_trace(go.Scatter(x=bic_df["States"], y=bic_df["AIC"],
                                     mode="lines+markers", name="AIC"))
        fig_bic.update_layout(height=350, xaxis_title="Number of States",
                              yaxis_title="Information Criterion")
        st.plotly_chart(fig_bic, use_container_width=True)

    # ── Tab 3: Regime Transitions ────────────────────────────────────────

    with tab3:
        st.subheader("Regime Transition Alpha Engine")

        trans_analyzer = RegimeTransitionAnalyzer()
        transitions = trans_analyzer.detect_transitions(
            states, labels, df["Close"].values, entropy, confidence
        )

        if not transitions:
            st.info("No regime transitions detected in the current data.")
        else:
            # ── Transition timing summary ──
            timing = trans_analyzer.transition_timing_analysis(transitions)
            tc1, tc2, tc3, tc4 = st.columns(4)
            with tc1:
                st.metric("Total Transitions", timing["n_transitions"])
            with tc2:
                avg_gap = timing["avg_bars_between"]
                st.metric("Avg Bars Between", f"{avg_gap:.0f}" if not np.isnan(avg_gap) else "N/A")
            with tc3:
                ent_pct = timing["entropy_precedes_transition"]
                st.metric("Entropy Precedes %", f"{ent_pct:.0%}" if not np.isnan(ent_pct) else "N/A")
            with tc4:
                avg_conf = timing["avg_confidence_at_transition"]
                st.metric("Avg Confidence at Transition", f"{avg_conf:.1%}" if not np.isnan(avg_conf) else "N/A")

            # ── Forward returns by transition type ──
            st.subheader("Transition-Conditional Forward Returns")
            fwd_df = trans_analyzer.transition_forward_returns(transitions)
            if not fwd_df.empty:
                display_fwd = fwd_df.copy()
                display_fwd["transition"] = display_fwd["from_regime"] + " → " + display_fwd["to_regime"]
                display_fwd = display_fwd[["transition", "count", "mean_fwd_5", "mean_fwd_10",
                                            "mean_fwd_20", "hit_rate_5"]]
                for col in ["mean_fwd_5", "mean_fwd_10", "mean_fwd_20", "hit_rate_5"]:
                    display_fwd[col] = display_fwd[col].apply(
                        lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A"
                    )
                display_fwd.columns = ["Transition", "Count", "Fwd 5-bar", "Fwd 10-bar",
                                       "Fwd 20-bar", "Hit Rate (5-bar)"]
                st.dataframe(display_fwd, use_container_width=True, hide_index=True)

                # Bar chart of forward returns
                chart_fwd = fwd_df.copy()
                chart_fwd["transition"] = chart_fwd["from_regime"] + " → " + chart_fwd["to_regime"]
                fig_fwd = go.Figure()
                for w, col, color in [(5, "mean_fwd_5", "#00e599"), (10, "mean_fwd_10", "#3b82f6"),
                                       (20, "mean_fwd_20", "#f59e0b")]:
                    fig_fwd.add_trace(go.Bar(
                        x=chart_fwd["transition"], y=chart_fwd[col],
                        name=f"{w}-bar", marker_color=color,
                    ))
                fig_fwd.update_layout(
                    barmode="group", height=400,
                    yaxis_title="Mean Forward Return",
                    xaxis_title="Transition Type",
                )
                st.plotly_chart(fig_fwd, use_container_width=True)

            # ── Empirical transition frequency matrix ──
            st.subheader("Empirical Transition Frequency")
            unique_labels = sorted(set(labels.values()))
            emp_matrix = trans_analyzer.transition_matrix_empirical(transitions, unique_labels)
            fig_emp = px.imshow(
                emp_matrix.values,
                labels=dict(x="To Regime", y="From Regime", color="Count"),
                x=unique_labels, y=unique_labels,
                color_continuous_scale="Greens",
                text_auto=True,
            )
            fig_emp.update_layout(height=400)
            st.plotly_chart(fig_emp, use_container_width=True)

            # ── Early warning signals ──
            st.subheader("Early Warning Signals")
            warnings_df = trans_analyzer.early_warning_signals(
                posteriors, entropy, states, labels,
            )
            if not warnings_df.empty:
                # Plot entropy gradient and posterior gap with warning highlights
                fig_warn = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=["Entropy Gradient (rising = uncertainty)", "Posterior Gap (falling = transition brewing)"],
                    vertical_spacing=0.08,
                )
                warn_x = warnings_df["bar"].values
                if "Datetime" in df.columns:
                    warn_x_vals = df["Datetime"].iloc[warnings_df["bar"].values].values
                else:
                    warn_x_vals = warnings_df["bar"].values

                # Entropy gradient
                fig_warn.add_trace(go.Scatter(
                    x=warn_x_vals, y=warnings_df["entropy_gradient"],
                    mode="lines", name="Entropy Gradient",
                    line=dict(color="#f59e0b"),
                ), row=1, col=1)

                # Posterior gap
                fig_warn.add_trace(go.Scatter(
                    x=warn_x_vals, y=warnings_df["posterior_gap"],
                    mode="lines", name="Posterior Gap",
                    line=dict(color="#06b6d4"),
                ), row=2, col=1)

                # Mark transition points
                trans_bars = [tr.bar for tr in transitions if tr.bar in warnings_df["bar"].values]
                if trans_bars:
                    if "Datetime" in df.columns:
                        trans_x = df["Datetime"].iloc[trans_bars].values
                    else:
                        trans_x = trans_bars
                    for row_num in [1, 2]:
                        fig_warn.add_trace(go.Scatter(
                            x=trans_x,
                            y=[0] * len(trans_bars),
                            mode="markers",
                            marker=dict(color="red", size=10, symbol="triangle-up"),
                            name="Transition" if row_num == 1 else None,
                            showlegend=(row_num == 1),
                        ), row=row_num, col=1)

                # Highlight high-warning zones
                high_warn = warnings_df[warnings_df["warning_level"] >= 2]
                if not high_warn.empty:
                    if "Datetime" in df.columns:
                        hw_x = df["Datetime"].iloc[high_warn["bar"].values].values
                    else:
                        hw_x = high_warn["bar"].values
                    fig_warn.add_trace(go.Scatter(
                        x=hw_x,
                        y=high_warn["entropy_gradient"],
                        mode="markers",
                        marker=dict(color="red", size=6, opacity=0.5),
                        name="High Warning",
                    ), row=1, col=1)

                fig_warn.update_layout(height=500)
                st.plotly_chart(fig_warn, use_container_width=True)

            # ── Regime P&L Attribution ──
            st.subheader("Regime P&L Attribution")
            returns_arr = df["Close"].pct_change().fillna(0).values
            attr_df = trans_analyzer.regime_attribution(
                returns_arr, states, labels, signals.values,
            )
            if not attr_df.empty:
                # Summary metrics
                attr_cols = st.columns(len(attr_df))
                for i, (_, row) in enumerate(attr_df.iterrows()):
                    with attr_cols[i % len(attr_cols)]:
                        regime_name = row["regime"]
                        color = get_regime_color(regime_name)
                        st.markdown(
                            f"<div style='border-left:4px solid {color};"
                            f"padding:8px 12px;margin-bottom:12px'>"
                            f"<span style='color:#888;font-size:11px'>{regime_name.upper()}</span><br>"
                            f"<span style='font-size:16px;font-weight:bold'>"
                            f"PnL: {row['cumulative_return']:.4f}</span><br>"
                            f"<span style='font-size:12px;color:#aaa'>"
                            f"Sharpe: {row['sharpe']:.2f} | "
                            f"Win: {row['win_rate']:.0%} | "
                            f"Time: {row['pct_time']:.0%}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                # PnL contribution pie chart
                fig_pie = go.Figure(data=[go.Pie(
                    labels=attr_df["regime"],
                    values=attr_df["cumulative_return"].abs(),
                    marker=dict(colors=[get_regime_color(r) for r in attr_df["regime"]]),
                    textinfo="label+percent",
                    hole=0.4,
                )])
                fig_pie.update_layout(height=350, title="PnL Contribution by Regime")
                st.plotly_chart(fig_pie, use_container_width=True)

                # Alpha by regime bar chart
                fig_alpha = go.Figure()
                fig_alpha.add_trace(go.Bar(
                    x=attr_df["regime"], y=attr_df["cumulative_return"],
                    name="Strategy Return",
                    marker_color=[get_regime_color(r) for r in attr_df["regime"]],
                ))
                fig_alpha.add_trace(go.Bar(
                    x=attr_df["regime"], y=attr_df["market_return"],
                    name="Market Return",
                    marker_color="rgba(128,128,128,0.5)",
                ))
                fig_alpha.update_layout(
                    barmode="group", height=350,
                    yaxis_title="Cumulative Return",
                    title="Strategy vs Market Return by Regime",
                )
                st.plotly_chart(fig_alpha, use_container_width=True)

    # ── Tab 4: Backtest Results ──────────────────────────────────────────

    with tab4:
        with st.spinner("Running walk-forward backtest..."):
            backtester = WalkForwardBacktester(config)
            try:
                bt_result = backtester.run(df)

                # Metric cards with CIs
                st.subheader("Performance Metrics (with 90% Bootstrap CIs)")
                m = bt_result.metrics
                ci_lo = bt_result.ci_lower
                ci_hi = bt_result.ci_upper

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Total Return", f"{m['total_return']:.2%}",
                              help=f"CI: [{ci_lo.get('total_return',0):.2%}, {ci_hi.get('total_return',0):.2%}]")
                    st.metric("Win Rate", f"{m['win_rate']:.1%}")
                with c2:
                    st.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}",
                              help=f"CI: [{ci_lo.get('sharpe_ratio',0):.2f}, {ci_hi.get('sharpe_ratio',0):.2f}]")
                    st.metric("Sortino Ratio", f"{m['sortino_ratio']:.2f}")
                with c3:
                    st.metric("Max Drawdown", f"{m['max_drawdown']:.2%}",
                              help=f"CI: [{ci_lo.get('max_drawdown',0):.2%}, {ci_hi.get('max_drawdown',0):.2%}]")
                    st.metric("Max DD Duration", f"{m['max_dd_duration']} bars")
                with c4:
                    st.metric("Calmar Ratio", f"{m['calmar_ratio']:.2f}")
                    st.metric("CVaR (5%)", f"{m['cvar_5pct']:.4f}")

                c5, c6, c7 = st.columns(3)
                with c5:
                    st.metric("Alpha", f"{m['alpha']:.2%}")
                with c6:
                    st.metric("Profit Factor", f"{m['profit_factor']:.2f}")
                with c7:
                    st.metric("Total Trades", m["n_trades"])

                # Equity curve vs benchmark
                st.subheader("Equity Curve vs Benchmark")
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=x_vals, y=bt_result.equity_curve,
                    mode="lines", name="Strategy", line=dict(color="blue"),
                ))
                fig_eq.add_trace(go.Scatter(
                    x=x_vals, y=bt_result.benchmark_curve,
                    mode="lines", name="Buy & Hold", line=dict(color="gray", dash="dash"),
                ))
                fig_eq.update_layout(height=450, yaxis_title="Equity ($)")
                st.plotly_chart(fig_eq, use_container_width=True)

                # Drawdown chart
                st.subheader("Drawdown")
                cummax = bt_result.equity_curve.cummax()
                dd = (bt_result.equity_curve - cummax) / cummax
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=x_vals, y=dd, fill="tozeroy",
                    line=dict(color="red"), name="Drawdown",
                ))
                fig_dd.update_layout(height=300, yaxis_title="Drawdown %")
                st.plotly_chart(fig_dd, use_container_width=True)

            except ValueError as e:
                st.error(str(e))

    # ── Tab 5: Trade Log ─────────────────────────────────────────────────

    with tab5:
        st.subheader("Trade Log")
        if "bt_result" in dir() and bt_result.trades:
            trade_data = []
            for t in bt_result.trades:
                trade_data.append({
                    "Entry Bar": t.entry_bar,
                    "Exit Bar": t.exit_bar,
                    "Direction": "Long" if t.direction == 1 else "Short",
                    "Entry Price": f"{t.entry_price:.2f}",
                    "Exit Price": f"{t.exit_price:.2f}",
                    "PnL ($)": f"{t.pnl:.2f}",
                    "PnL (%)": f"{t.pnl_pct:.4f}",
                    "Position Size": f"{t.position_size:.2%}",
                    "Bars Held": t.exit_bar - t.entry_bar,
                })
            trade_df = pd.DataFrame(trade_data)
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run backtest to see trade log.")

    # ── Tab 6: Model Diagnostics ─────────────────────────────────────────

    with tab6:
        col1, col2 = st.columns(2)

        with col1:
            # Rolling log-likelihood
            st.subheader("Rolling Log-Likelihood")
            fig_ll = go.Figure()
            fig_ll.add_trace(go.Scatter(
                x=x_vals, y=ll_series, mode="lines",
                line=dict(color="purple"), name="LL/bar",
            ))
            fig_ll.update_layout(height=350, yaxis_title="Log-Likelihood per bar")
            st.plotly_chart(fig_ll, use_container_width=True)

        with col2:
            # Entropy time series
            st.subheader("Shannon Entropy / Confidence")
            fig_ent = make_subplots(specs=[[{"secondary_y": True}]])
            fig_ent.add_trace(go.Scatter(
                x=x_vals, y=entropy, mode="lines",
                line=dict(color="orange"), name="Entropy",
            ), secondary_y=False)
            fig_ent.add_trace(go.Scatter(
                x=x_vals, y=confidence, mode="lines",
                line=dict(color="teal"), name="Confidence",
            ), secondary_y=True)
            fig_ent.update_yaxes(title_text="Entropy (bits)", secondary_y=False)
            fig_ent.update_yaxes(title_text="Confidence", secondary_y=True)
            fig_ent.update_layout(height=350)
            st.plotly_chart(fig_ent, use_container_width=True)

        # Feature correlation matrix
        st.subheader("Feature Correlation Matrix")
        corr = df[feature_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Tab 7: Fundamentals ─────────────────────────────────────────────

    with tab7:
        if FundamentalAnalyzer.is_crypto(ticker):
            st.warning("Fundamental analysis is not available for crypto assets. "
                       "Crypto tickers do not have traditional financial statements, "
                       "ratios, or analyst coverage.")
        else:
            fa = FundamentalAnalyzer()

            with st.spinner("Fetching fundamental data..."):
                overview = fa.get_company_overview(ticker)
                ratios = fa.get_financial_ratios(ticker)
                statements = fa.get_financial_statements(ticker)
                analyst = fa.get_analyst_data(ticker)
                score = fa.health_score(ratios)

            # ── Company Overview Card ────────────────────────────────
            st.subheader("Company Overview")
            ov_c1, ov_c2, ov_c3, ov_c4 = st.columns(4)
            with ov_c1:
                st.metric("Company", overview["name"])
            with ov_c2:
                st.metric("Sector", overview["sector"])
            with ov_c3:
                st.metric("Industry", overview["industry"])
            with ov_c4:
                st.metric("Market Cap", overview["market_cap_fmt"])

            price_c1, price_c2, price_c3 = st.columns(3)
            with price_c1:
                cp = overview["current_price"]
                st.metric("Current Price",
                          f"${cp:,.2f}" if cp else "N/A")
            with price_c2:
                high52 = overview["fifty_two_week_high"]
                st.metric("52W High",
                          f"${high52:,.2f}" if high52 else "N/A")
            with price_c3:
                low52 = overview["fifty_two_week_low"]
                st.metric("52W Low",
                          f"${low52:,.2f}" if low52 else "N/A")

            desc = overview.get("description", "")
            if desc and desc != "No description available.":
                with st.expander("Business Description"):
                    st.write(desc)

            # ── Financial Health Score ────────────────────────────────
            st.subheader("Financial Health Score")
            score_color = fa.health_color(score)
            score_label = "Strong" if score >= 70 else "Moderate" if score >= 40 else "Weak"

            sc_c1, sc_c2 = st.columns([1, 3])
            with sc_c1:
                st.markdown(
                    f"<div style='background:{score_color};padding:30px;"
                    f"border-radius:15px;text-align:center;color:white;"
                    f"font-size:36px;font-weight:bold'>{score}/100</div>"
                    f"<p style='text-align:center;font-size:18px;margin-top:8px'>"
                    f"{score_label}</p>",
                    unsafe_allow_html=True,
                )
            with sc_c2:
                # Score gauge using plotly
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={"text": "Health Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": score_color},
                        "steps": [
                            {"range": [0, 40], "color": "#ffcdd2"},
                            {"range": [40, 70], "color": "#fff9c4"},
                            {"range": [70, 100], "color": "#c8e6c9"},
                        ],
                    },
                ))
                fig_gauge.update_layout(height=250, margin=dict(t=40, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Financial Ratios Grid ────────────────────────────────
            st.subheader("Key Financial Ratios")

            ratio_display = {
                "pe_trailing": ("P/E (TTM)", None),
                "pe_forward": ("P/E (Fwd)", None),
                "pb": ("P/B", None),
                "ps": ("P/S", None),
                "peg": ("PEG", None),
                "ev_ebitda": ("EV/EBITDA", None),
                "debt_to_equity": ("D/E Ratio", None),
                "current_ratio": ("Current Ratio", None),
                "roe": ("ROE", "%"),
                "roa": ("ROA", "%"),
                "profit_margin": ("Profit Margin", "%"),
                "operating_margin": ("Operating Margin", "%"),
                "gross_margin": ("Gross Margin", "%"),
                "revenue_growth": ("Revenue Growth", "%"),
                "earnings_growth": ("Earnings Growth", "%"),
                "dividend_yield": ("Dividend Yield", "%"),
                "payout_ratio": ("Payout Ratio", "%"),
                "beta": ("Beta", None),
            }

            # Render in rows of 4
            ratio_keys = list(ratio_display.keys())
            for i in range(0, len(ratio_keys), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(ratio_keys):
                        break
                    key = ratio_keys[idx]
                    label, fmt = ratio_display[key]
                    val = ratios.get(key)
                    color = fa.ratio_color(key, val)

                    if val is not None:
                        if fmt == "%":
                            display_val = f"{val * 100:.2f}%"
                        else:
                            display_val = f"{val:.2f}"
                    else:
                        display_val = "N/A"
                        color = "#9e9e9e"

                    with col:
                        st.markdown(
                            f"<div style='border-left:4px solid {color};"
                            f"padding:8px 12px;margin-bottom:8px'>"
                            f"<span style='color:#888;font-size:12px'>{label}</span><br>"
                            f"<span style='font-size:20px;font-weight:bold'>{display_val}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # ── Income Statement Trend ───────────────────────────────
            inc_summary = statements.get("income_summary", pd.DataFrame())
            if not inc_summary.empty:
                st.subheader("Income Statement Trend")
                fig_inc = go.Figure()

                for row_name in ["Total Revenue", "Net Income", "EBITDA"]:
                    if row_name in inc_summary.index:
                        row = inc_summary.loc[row_name]
                        periods = [str(c.date()) if hasattr(c, "date") else str(c)
                                   for c in row.index]
                        vals = row.values.astype(float)
                        fig_inc.add_trace(go.Bar(
                            x=periods, y=vals, name=row_name,
                        ))

                fig_inc.update_layout(
                    barmode="group", height=400,
                    xaxis_title="Period", yaxis_title="USD",
                    xaxis=dict(categoryorder="array",
                               categoryarray=sorted(periods)),
                )
                st.plotly_chart(fig_inc, use_container_width=True)

            # ── Balance Sheet Composition ────────────────────────────
            bal_summary = statements.get("balance_summary", pd.DataFrame())
            if not bal_summary.empty:
                st.subheader("Balance Sheet Composition")
                fig_bal = go.Figure()

                for row_name, color in [
                    ("Total Assets", "#1565c0"),
                    ("Total Liabilities Net Minority Interest", "#d32f2f"),
                    ("Stockholders Equity", "#388e3c"),
                    ("Total Debt", "#f57c00"),
                    ("Cash And Cash Equivalents", "#00897b"),
                ]:
                    if row_name in bal_summary.index:
                        row = bal_summary.loc[row_name]
                        periods = [str(c.date()) if hasattr(c, "date") else str(c)
                                   for c in row.index]
                        vals = row.values.astype(float)
                        fig_bal.add_trace(go.Bar(
                            x=periods, y=vals,
                            name=row_name.replace("Net Minority Interest", ""),
                            marker_color=color,
                        ))

                fig_bal.update_layout(
                    barmode="group", height=400,
                    xaxis_title="Period", yaxis_title="USD",
                    xaxis=dict(categoryorder="array",
                               categoryarray=sorted(periods) if periods else []),
                )
                st.plotly_chart(fig_bal, use_container_width=True)

            # ── Cash Flow Chart ──────────────────────────────────────
            cf_summary = statements.get("cashflow_summary", pd.DataFrame())
            if not cf_summary.empty:
                st.subheader("Cash Flow Summary")
                fig_cf = go.Figure()

                cf_colors = {
                    "Operating Cash Flow": "#388e3c",
                    "Investing Cash Flow": "#f57c00",
                    "Financing Cash Flow": "#1565c0",
                    "Free Cash Flow": "#7b1fa2",
                }
                for row_name in ["Operating Cash Flow", "Investing Cash Flow",
                                 "Financing Cash Flow", "Free Cash Flow"]:
                    if row_name in cf_summary.index:
                        row = cf_summary.loc[row_name]
                        periods = [str(c.date()) if hasattr(c, "date") else str(c)
                                   for c in row.index]
                        vals = row.values.astype(float)
                        fig_cf.add_trace(go.Bar(
                            x=periods, y=vals, name=row_name,
                            marker_color=cf_colors.get(row_name, "#9e9e9e"),
                        ))

                fig_cf.update_layout(
                    barmode="group", height=400,
                    xaxis_title="Period", yaxis_title="USD",
                    xaxis=dict(categoryorder="array",
                               categoryarray=sorted(periods) if periods else []),
                )
                st.plotly_chart(fig_cf, use_container_width=True)

            # ── Analyst Consensus ────────────────────────────────────
            st.subheader("Analyst Consensus")
            an_c1, an_c2, an_c3, an_c4 = st.columns(4)
            with an_c1:
                rec_key = analyst.get("recommendation_key", "N/A")
                rec_colors = {
                    "buy": "#388e3c", "strong_buy": "#1b5e20",
                    "hold": "#f57c00", "sell": "#d32f2f",
                    "strong_sell": "#b71c1c",
                }
                rec_color = rec_colors.get(rec_key, "#9e9e9e")
                st.markdown(
                    f"<div style='background:{rec_color};padding:15px;"
                    f"border-radius:10px;text-align:center;color:white;"
                    f"font-size:18px;font-weight:bold'>"
                    f"{rec_key.upper().replace('_', ' ')}</div>",
                    unsafe_allow_html=True,
                )
            with an_c2:
                st.metric("Analysts", analyst.get("number_of_analysts") or "N/A")
            with an_c3:
                target = analyst.get("target_mean")
                st.metric("Avg Target",
                          f"${target:,.2f}" if target else "N/A")
            with an_c4:
                curr = analyst.get("current_price")
                target_m = analyst.get("target_mean")
                if curr and target_m and curr > 0:
                    upside = (target_m - curr) / curr * 100
                    st.metric("Upside/Downside", f"{upside:+.1f}%")
                else:
                    st.metric("Upside/Downside", "N/A")

            # Price target range
            tgt_low = analyst.get("target_low")
            tgt_high = analyst.get("target_high")
            tgt_med = analyst.get("target_median")
            curr_price = analyst.get("current_price")
            if tgt_low and tgt_high:
                fig_tgt = go.Figure()
                fig_tgt.add_trace(go.Scatter(
                    x=["Target Range"], y=[tgt_low],
                    mode="markers", marker=dict(size=12, color="#d32f2f"),
                    name=f"Low: ${tgt_low:,.2f}",
                ))
                if tgt_med:
                    fig_tgt.add_trace(go.Scatter(
                        x=["Target Range"], y=[tgt_med],
                        mode="markers", marker=dict(size=14, color="#f57c00",
                                                     symbol="diamond"),
                        name=f"Median: ${tgt_med:,.2f}",
                    ))
                fig_tgt.add_trace(go.Scatter(
                    x=["Target Range"], y=[tgt_high],
                    mode="markers", marker=dict(size=12, color="#388e3c"),
                    name=f"High: ${tgt_high:,.2f}",
                ))
                if curr_price:
                    fig_tgt.add_hline(
                        y=curr_price, line_dash="dash", line_color="blue",
                        annotation_text=f"Current: ${curr_price:,.2f}",
                    )
                fig_tgt.update_layout(height=250, showlegend=True,
                                       yaxis_title="Price ($)")
                st.plotly_chart(fig_tgt, use_container_width=True)

            # Recent recommendations table
            recs_df = analyst.get("recommendations", pd.DataFrame())
            if not recs_df.empty:
                with st.expander("Recent Analyst Recommendations"):
                    st.dataframe(recs_df, use_container_width=True)

            # Earnings dates
            earnings_df = analyst.get("earnings_dates", pd.DataFrame())
            if not earnings_df.empty:
                with st.expander("Earnings Dates"):
                    st.dataframe(earnings_df, use_container_width=True)

    # ── Tab 8: Multi-Timeframe Fusion ────────────────────────────────────

    with tab8:
        if not enable_mtf:
            st.info("Enable **Multi-Timeframe Fusion** in the sidebar and select 2+ timeframes to use this tab.")
        elif len(mtf_intervals) < 2:
            st.warning("Select at least 2 timeframes in the sidebar.")
        else:
            with st.spinner("Fitting HMMs across multiple timeframes..."):
                try:
                    fusion = run_multi_timeframe_analysis(
                        ticker=ticker,
                        intervals=mtf_intervals,
                        lookback_days=lookback,
                        config=config,
                    )
                except Exception as e:
                    st.error(f"Multi-timeframe analysis failed: {e}")
                    fusion = None

            if fusion is not None:
                st.subheader("Cross-Timeframe Regime Confluence")

                # ── Confluence score time series ──
                fig_conf = go.Figure()
                conf_idx = fusion.aligned_regimes.index
                fig_conf.add_trace(go.Scatter(
                    x=conf_idx, y=fusion.confluence_score,
                    mode="lines", name="Confluence Score",
                    line=dict(color="#00e599", width=2),
                    fill="tozeroy", fillcolor="rgba(0,229,153,0.1)",
                ))
                fig_conf.add_hline(y=0.7, line_dash="dash", line_color="#f59e0b",
                                   annotation_text="Strong agreement (0.7)")
                fig_conf.update_layout(
                    title="Regime Confluence Score (higher = more timeframe agreement)",
                    yaxis=dict(title="Confluence", range=[0, 1.05]),
                    xaxis_title="Time", height=350,
                    template="plotly_dark",
                )
                st.plotly_chart(fig_conf, use_container_width=True)

                # ── Per-timeframe regime strips ──
                st.subheader("Regime Alignment Across Timeframes")
                regime_color_map = {
                    "crash": "#d32f2f", "bear": "#f57c00", "neutral": "#9e9e9e",
                    "bull": "#388e3c", "bull_run": "#1565c0", "unknown": "#e0e0e0",
                }
                fig_strips = make_subplots(
                    rows=len(mtf_intervals), cols=1,
                    shared_xaxes=True,
                    subplot_titles=[f"{iv} Regime" for iv in sorted(
                        mtf_intervals,
                        key=lambda x: TIMEFRAME_ORDER.index(x) if x in TIMEFRAME_ORDER else 99,
                    )],
                    vertical_spacing=0.04,
                )
                sorted_ivs = sorted(
                    mtf_intervals,
                    key=lambda x: TIMEFRAME_ORDER.index(x) if x in TIMEFRAME_ORDER else 99,
                )
                for row_i, iv in enumerate(sorted_ivs, 1):
                    col_name = f"regime_{iv}"
                    if col_name in fusion.aligned_regimes.columns:
                        regimes_series = fusion.aligned_regimes[col_name].fillna("unknown")
                        colors = [regime_color_map.get(r, "#e0e0e0") for r in regimes_series]
                        from multi_timeframe import _regime_to_sentiment
                        sentiments = [_regime_to_sentiment(r) for r in regimes_series]
                        fig_strips.add_trace(go.Bar(
                            x=conf_idx, y=[1] * len(conf_idx),
                            marker_color=colors,
                            showlegend=False,
                            hovertext=[f"{iv}: {r}" for r in regimes_series],
                            hoverinfo="text",
                        ), row=row_i, col=1)
                        fig_strips.update_yaxes(visible=False, row=row_i, col=1)

                fig_strips.update_layout(
                    height=120 * len(mtf_intervals) + 50,
                    template="plotly_dark",
                    bargap=0, bargroupgap=0,
                    title_text="Regime State per Timeframe (aligned to base timeline)",
                )
                st.plotly_chart(fig_strips, use_container_width=True)

                # ── Enhanced vs base confidence ──
                st.subheader("Enhanced Confidence (Base × Confluence)")
                base_tf = fusion.base_interval
                base_result = fusion.timeframe_results[base_tf]
                fig_enh = go.Figure()
                fig_enh.add_trace(go.Scatter(
                    x=conf_idx, y=base_result.confidence,
                    mode="lines", name="Base Confidence",
                    line=dict(color="#3b82f6", width=1.5, dash="dot"),
                ))
                fig_enh.add_trace(go.Scatter(
                    x=conf_idx, y=fusion.enhanced_confidence,
                    mode="lines", name="Enhanced Confidence",
                    line=dict(color="#00e599", width=2),
                ))
                fig_enh.update_layout(
                    title="Confidence: Single-TF vs Multi-TF Enhanced",
                    yaxis=dict(title="Confidence", range=[0, 1.05]),
                    xaxis_title="Time", height=300,
                    template="plotly_dark",
                )
                st.plotly_chart(fig_enh, use_container_width=True)

                # ── Dominant regime + conflict summary ──
                st.subheader("Dominant Regime (Weighted Vote)")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    current_dominant = fusion.dominant_regime.iloc[-1]
                    dom_color = regime_color_map.get(current_dominant, "#e0e0e0")
                    st.markdown(
                        f"<div style='background:{dom_color};padding:20px;border-radius:10px;"
                        f"text-align:center;color:white;font-size:20px;font-weight:bold'>"
                        f"{current_dominant.upper()}</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption("Current dominant regime across all timeframes")
                with col_b:
                    avg_conf = float(np.mean(fusion.confluence_score[-20:]))
                    st.metric("Avg Confluence (last 20 bars)", f"{avg_conf:.1%}")
                with col_c:
                    n_conflicts = int(fusion.regime_conflicts["conflict"].sum())
                    total_bars = len(fusion.regime_conflicts)
                    st.metric("Conflict Bars", f"{n_conflicts}/{total_bars}",
                              delta=f"{n_conflicts/max(total_bars,1)*100:.0f}%",
                              delta_color="inverse")

                # ── Per-timeframe HMM summary ──
                st.subheader("Per-Timeframe HMM Summary")
                summary_rows = []
                for iv in sorted_ivs:
                    tf_r = fusion.timeframe_results[iv]
                    summary_rows.append({
                        "Timeframe": iv,
                        "N States (BIC)": tf_r.n_states,
                        "Bars": len(tf_r.df),
                        "Current Regime": tf_r.labels.get(tf_r.states[-1], "unknown"),
                        "Avg Confidence": f"{tf_r.confidence.mean():.2%}",
                        "Weight": f"{DEFAULT_WEIGHTS.get(iv, 0.5):.1f}",
                    })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    # ── Tab 9: Monte Carlo Simulation ──────────────────────────────────

    with tab9:
        st.subheader("Monte Carlo Regime Simulation")
        st.caption(
            "Leverages the HMM's generative model to simulate thousands of "
            "synthetic market paths. Produces forward-looking risk metrics "
            "that historical backtests cannot provide."
        )

        mc_col1, mc_col2, mc_col3 = st.columns(3)
        with mc_col1:
            mc_n_paths = st.number_input("Simulation Paths", 100, 10000, 1000, step=100, key="mc_paths")
        with mc_col2:
            mc_n_steps = st.number_input("Steps per Path", 50, 1000, 252, step=50, key="mc_steps")
        with mc_col3:
            mc_ruin = st.slider("Ruin Threshold", 0.1, 0.9, 0.5, 0.05, key="mc_ruin",
                                help="Fraction of capital remaining that defines 'ruin'")

        mc_run = st.button("Run Monte Carlo Simulation", type="primary", key="mc_run")

        if mc_run:
            with st.spinner(f"Simulating {mc_n_paths} paths x {mc_n_steps} steps..."):
                mc_config = config.copy()
                mc_config["monte_carlo"] = {
                    "n_paths": mc_n_paths,
                    "n_steps": mc_n_steps,
                    "ruin_threshold": mc_ruin,
                    "seed": 42,
                }
                mc_engine = MonteCarloEngine(mc_config)

                mc_result = mc_engine.run(
                    transmat=transmat,
                    means=detector.model.means_,
                    covars=detector.model.covars_,
                    labels=labels,
                    covariance_type=detector.covariance_type,
                    n_paths=mc_n_paths,
                    n_steps=mc_n_steps,
                )

            # ── Risk Metrics Cards ──
            st.markdown("---")
            st.subheader("Forward-Looking Risk Metrics")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("VaR (95%)", f"{mc_result.var_95:+.2%}",
                          help="5th percentile of terminal returns")
            with m2:
                st.metric("CVaR (95%)", f"{mc_result.cvar_95:+.2%}",
                          help="Expected loss beyond VaR (tail risk)")
            with m3:
                st.metric("Ruin Probability", f"{mc_result.ruin_probability:.2%}",
                          help=f"P(equity < {mc_ruin:.0%} of initial)")
            with m4:
                st.metric("Median Return", f"{mc_result.median_return:+.2%}")

            m5, m6, m7, m8 = st.columns(4)
            with m5:
                st.metric("VaR (99%)", f"{mc_result.var_99:+.2%}")
            with m6:
                st.metric("CVaR (99%)", f"{mc_result.cvar_99:+.2%}")
            with m7:
                st.metric("Mean Return", f"{mc_result.mean_return:+.2%}")
            with m8:
                median_sharpe = float(np.median(mc_result.path_sharpes))
                st.metric("Median Sharpe", f"{median_sharpe:.2f}")

            # ── Equity Fan Chart ──
            st.markdown("---")
            st.subheader("Equity Fan Chart")
            st.caption("Percentile bands across all simulated paths")

            bands = mc_result.percentile_bands
            x_axis = list(range(mc_n_steps))

            fig_fan = go.Figure()

            # 5th-95th band
            fig_fan.add_trace(go.Scatter(
                x=x_axis + x_axis[::-1],
                y=list(bands[95]) + list(bands[5])[::-1],
                fill="toself", fillcolor="rgba(0, 229, 153, 0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                name="5th-95th percentile", showlegend=True,
            ))
            # 25th-75th band
            fig_fan.add_trace(go.Scatter(
                x=x_axis + x_axis[::-1],
                y=list(bands[75]) + list(bands[25])[::-1],
                fill="toself", fillcolor="rgba(0, 229, 153, 0.18)",
                line=dict(color="rgba(0,0,0,0)"),
                name="25th-75th percentile", showlegend=True,
            ))
            # Median
            fig_fan.add_trace(go.Scatter(
                x=x_axis, y=list(bands[50]),
                mode="lines", name="Median",
                line=dict(color="#00e599", width=2.5),
            ))
            # Initial capital reference
            fig_fan.add_hline(
                y=mc_engine.initial_capital, line_dash="dot",
                line_color="#64748b", annotation_text="Initial Capital",
            )

            fig_fan.update_layout(
                template="plotly_dark",
                height=450,
                xaxis_title="Simulation Step",
                yaxis_title="Equity ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_fan, use_container_width=True)

            # ── Terminal Wealth Distribution ──
            st.subheader("Terminal Wealth Distribution")
            tw_col1, tw_col2 = st.columns([2, 1])

            with tw_col1:
                terminal_returns = mc_result.terminal_wealth / mc_engine.initial_capital - 1
                fig_tw = go.Figure()
                fig_tw.add_trace(go.Histogram(
                    x=terminal_returns * 100,
                    nbinsx=60,
                    marker_color="#00e599",
                    opacity=0.7,
                    name="Terminal Return %",
                ))
                fig_tw.add_vline(x=0, line_dash="solid", line_color="#ef4444", line_width=2,
                                 annotation_text="Break-even")
                fig_tw.add_vline(x=mc_result.var_95 * 100, line_dash="dash",
                                 line_color="#f59e0b", line_width=1.5,
                                 annotation_text="VaR 95%")
                fig_tw.update_layout(
                    template="plotly_dark", height=350,
                    xaxis_title="Terminal Return (%)",
                    yaxis_title="Frequency",
                )
                st.plotly_chart(fig_tw, use_container_width=True)

            with tw_col2:
                st.markdown("**Distribution Stats**")
                pct_profitable = float((terminal_returns > 0).mean()) * 100
                st.markdown(f"- Profitable paths: **{pct_profitable:.1f}%**")
                st.markdown(f"- Mean return: **{mc_result.mean_return:+.2%}**")
                st.markdown(f"- Median return: **{mc_result.median_return:+.2%}**")
                st.markdown(f"- Std dev: **{terminal_returns.std():.2%}**")
                st.markdown(f"- Skewness: **{float(pd.Series(terminal_returns).skew()):.2f}**")
                st.markdown(f"- Kurtosis: **{float(pd.Series(terminal_returns).kurtosis()):.2f}**")
                st.markdown(f"- Best path: **{terminal_returns.max():+.2%}**")
                st.markdown(f"- Worst path: **{terminal_returns.min():+.2%}**")

            # ── Max Drawdown Distribution ──
            st.subheader("Max Drawdown Distribution")
            dd_col1, dd_col2 = st.columns([2, 1])

            with dd_col1:
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Histogram(
                    x=mc_result.max_drawdowns * 100,
                    nbinsx=50,
                    marker_color="#ef4444",
                    opacity=0.7,
                    name="Max Drawdown %",
                ))
                fig_dd.update_layout(
                    template="plotly_dark", height=300,
                    xaxis_title="Max Drawdown (%)",
                    yaxis_title="Frequency",
                )
                st.plotly_chart(fig_dd, use_container_width=True)

            with dd_col2:
                st.markdown("**Drawdown Stats**")
                st.markdown(f"- Median MDD: **{np.median(mc_result.max_drawdowns):.2%}**")
                st.markdown(f"- Mean MDD: **{np.mean(mc_result.max_drawdowns):.2%}**")
                st.markdown(f"- Worst MDD: **{mc_result.max_drawdowns.min():.2%}**")
                st.markdown(f"- 95th pct MDD: **{np.percentile(mc_result.max_drawdowns, 5):.2%}**")

            # ── Regime Time Allocation ──
            st.subheader("Expected Regime Time Allocation")
            regime_frac_df = pd.DataFrame([
                {"Regime": label, "Time Fraction": frac}
                for label, frac in mc_result.regime_time_fractions.items()
            ]).sort_values("Time Fraction", ascending=False)

            fig_regime = go.Figure(go.Bar(
                x=regime_frac_df["Regime"],
                y=regime_frac_df["Time Fraction"] * 100,
                marker_color=["#ef4444" if "crash" in r or "bear" in r
                              else "#00e599" if "bull" in r
                              else "#f59e0b"
                              for r in regime_frac_df["Regime"]],
            ))
            fig_regime.update_layout(
                template="plotly_dark", height=300,
                yaxis_title="Time in Regime (%)",
                xaxis_title="Regime",
            )
            st.plotly_chart(fig_regime, use_container_width=True)

            # ── Stress Scenarios ──
            st.markdown("---")
            st.subheader("Stress Test Scenarios")
            st.caption(
                "Fixed regime sequences with stochastic returns sampled from "
                "the HMM emissions. Shows strategy resilience under adversarial conditions."
            )

            with st.spinner("Running stress scenarios..."):
                stress_results = mc_engine.run_all_stress_tests(
                    means=detector.model.means_,
                    covars=detector.model.covars_,
                    labels=labels,
                    covariance_type=detector.covariance_type,
                    n_paths=500,
                    n_steps=mc_n_steps,
                )

            if stress_results:
                # Summary table
                stress_rows = []
                for sr in stress_results:
                    stress_rows.append({
                        "Scenario": sr.name,
                        "Description": sr.description,
                        "Median Return": f"{sr.median_return:+.2%}",
                        "VaR (95%)": f"{sr.var_95:+.2%}",
                        "CVaR (95%)": f"{sr.cvar_95:+.2%}",
                        "Median MDD": f"{np.median(sr.max_drawdowns):.2%}",
                    })
                st.dataframe(pd.DataFrame(stress_rows), use_container_width=True)

                # Scenario equity fan charts
                for sr in stress_results:
                    with st.expander(f"{sr.name}: {sr.description}"):
                        # Compute percentile bands for this scenario
                        sc_bands = {}
                        for pct in [5, 25, 50, 75, 95]:
                            sc_bands[pct] = np.percentile(sr.equity_paths, pct, axis=0)

                        sc_x = list(range(sr.equity_paths.shape[1]))
                        fig_sc = go.Figure()
                        fig_sc.add_trace(go.Scatter(
                            x=sc_x + sc_x[::-1],
                            y=list(sc_bands[95]) + list(sc_bands[5])[::-1],
                            fill="toself", fillcolor="rgba(239, 68, 68, 0.08)",
                            line=dict(color="rgba(0,0,0,0)"), name="5th-95th",
                        ))
                        fig_sc.add_trace(go.Scatter(
                            x=sc_x + sc_x[::-1],
                            y=list(sc_bands[75]) + list(sc_bands[25])[::-1],
                            fill="toself", fillcolor="rgba(239, 68, 68, 0.18)",
                            line=dict(color="rgba(0,0,0,0)"), name="25th-75th",
                        ))
                        fig_sc.add_trace(go.Scatter(
                            x=sc_x, y=list(sc_bands[50]),
                            mode="lines", name="Median",
                            line=dict(color="#ef4444", width=2),
                        ))
                        fig_sc.add_hline(y=mc_engine.initial_capital, line_dash="dot",
                                         line_color="#64748b")
                        fig_sc.update_layout(
                            template="plotly_dark", height=300,
                            xaxis_title="Step", yaxis_title="Equity ($)",
                            title=sr.name,
                        )
                        st.plotly_chart(fig_sc, use_container_width=True)

                        sc_col1, sc_col2, sc_col3 = st.columns(3)
                        with sc_col1:
                            st.metric("Median Return", f"{sr.median_return:+.2%}")
                        with sc_col2:
                            st.metric("VaR (95%)", f"{sr.var_95:+.2%}")
                        with sc_col3:
                            st.metric("Worst MDD", f"{sr.max_drawdowns.min():.2%}")

    # ── Tab 10: Changepoint Detection ─────────────────────────────────────

    with tab10:
        if fusion_result is None:
            st.info("Enable **BOCPD** in the sidebar to use Changepoint Detection.")
        else:
            st.subheader("Bayesian Online Changepoint Detection")

            cp_prob = fusion_result["changepoint_prob"]
            urgency = fusion_result["urgency"]
            erl = fusion_result["expected_run_length"]
            adapt_hyst = fusion_result["adaptive_hysteresis"]
            rl_dist = fusion_result["run_length_dist"]
            map_rl = fusion_result["map_run_length"]

            # Use datetime index from df
            x_axis = df["Datetime"].values if "Datetime" in df.columns else (
                df["Date"].values if "Date" in df.columns else list(range(len(df)))
            )

            # ── Price + Changepoint Overlay ──
            fig_cp = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                vertical_spacing=0.04,
                row_heights=[0.4, 0.3, 0.3],
                subplot_titles=("Price + Changepoint Detections", "Changepoint Probability & Transition Urgency", "Expected Run Length"),
            )

            # Price trace
            fig_cp.add_trace(go.Scatter(
                x=x_axis, y=df["Close"].values,
                mode="lines", name="Price",
                line=dict(color="#e2e8f0", width=1),
            ), row=1, col=1)

            # Changepoint markers on price
            cp_mask = cp_prob > bocpd.threshold
            if cp_mask.any():
                fig_cp.add_trace(go.Scatter(
                    x=x_axis[cp_mask], y=df["Close"].values[cp_mask],
                    mode="markers", name="Changepoints",
                    marker=dict(color="#ef4444", size=8, symbol="diamond",
                                line=dict(width=1, color="#ffffff")),
                ), row=1, col=1)

            # Regime background shading on price
            for regime_label, color in REGIME_COLORS.items():
                mask = df["regime"].values == regime_label
                if mask.any():
                    fig_cp.add_trace(go.Scatter(
                        x=x_axis[mask], y=df["Close"].values[mask],
                        mode="markers", name=regime_label,
                        marker=dict(color=color, size=3, opacity=0.4),
                        showlegend=False,
                    ), row=1, col=1)

            # Changepoint probability
            fig_cp.add_trace(go.Scatter(
                x=x_axis, y=cp_prob,
                mode="lines", name="CP Probability",
                line=dict(color="#ef4444", width=1.5),
                fill="tozeroy", fillcolor="rgba(239, 68, 68, 0.15)",
            ), row=2, col=1)

            # Transition urgency overlay
            fig_cp.add_trace(go.Scatter(
                x=x_axis, y=urgency,
                mode="lines", name="Transition Urgency",
                line=dict(color="#f59e0b", width=1.5),
            ), row=2, col=1)

            # Threshold line
            fig_cp.add_hline(y=bocpd.threshold, row=2, col=1,
                             line_dash="dot", line_color="#64748b",
                             annotation_text=f"threshold={bocpd.threshold}")

            # Expected run length
            fig_cp.add_trace(go.Scatter(
                x=x_axis, y=erl,
                mode="lines", name="Expected Run Length",
                line=dict(color="#06b6d4", width=1.5),
                fill="tozeroy", fillcolor="rgba(6, 182, 212, 0.1)",
            ), row=3, col=1)

            # MAP run length
            fig_cp.add_trace(go.Scatter(
                x=x_axis, y=map_rl,
                mode="lines", name="MAP Run Length",
                line=dict(color="#3b82f6", width=1, dash="dot"),
            ), row=3, col=1)

            fig_cp.update_layout(
                template="plotly_dark", height=750,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=60, r=30, t=60, b=30),
            )
            fig_cp.update_yaxes(title_text="Price", row=1, col=1)
            fig_cp.update_yaxes(title_text="Probability", row=2, col=1)
            fig_cp.update_yaxes(title_text="Bars", row=3, col=1)
            st.plotly_chart(fig_cp, use_container_width=True)

            # ── Run-Length Heatmap ──
            st.subheader("Run-Length Distribution Heatmap")
            st.caption(
                "Shows P(run length = r | data) over time. "
                "Bright horizontal bands = stable regimes. "
                "Mass snapping to r=0 = changepoint."
            )

            # Trim to display only the interesting part of the RL distribution
            max_display_rl = min(int(erl.max() * 2) + 10, rl_dist.shape[1], 150)
            rl_display = rl_dist[:, :max_display_rl].T

            fig_rl = go.Figure(data=go.Heatmap(
                z=rl_display,
                x=x_axis,
                y=list(range(max_display_rl)),
                colorscale="Inferno",
                zmin=0,
                zmax=np.percentile(rl_display[rl_display > 0], 95) if (rl_display > 0).any() else 0.1,
                colorbar=dict(title="P(r)"),
            ))
            fig_rl.update_layout(
                template="plotly_dark", height=350,
                xaxis_title="Time",
                yaxis_title="Run Length (bars)",
                margin=dict(l=60, r=30, t=30, b=30),
            )
            st.plotly_chart(fig_rl, use_container_width=True)

            # ── Adaptive Hysteresis ──
            st.subheader("Adaptive Hysteresis")
            base_hyst = config.get("strategy", {}).get("hysteresis_bars", 3)

            col_ah1, col_ah2, col_ah3 = st.columns(3)
            with col_ah1:
                st.metric("Base Hysteresis", f"{base_hyst} bars")
            with col_ah2:
                st.metric("Avg Adaptive Hysteresis", f"{adapt_hyst.mean():.1f} bars")
            with col_ah3:
                pct_reduced = (adapt_hyst < base_hyst).mean() * 100
                st.metric("Time at Reduced Hysteresis", f"{pct_reduced:.1f}%")

            fig_ah = go.Figure()
            fig_ah.add_trace(go.Scatter(
                x=x_axis, y=adapt_hyst,
                mode="lines", name="Adaptive Hysteresis",
                line=dict(color="#00e599", width=1.5),
                fill="tozeroy", fillcolor="rgba(0, 229, 153, 0.1)",
            ))
            fig_ah.add_hline(y=base_hyst, line_dash="dash", line_color="#64748b",
                             annotation_text=f"base={base_hyst}")
            fig_ah.update_layout(
                template="plotly_dark", height=250,
                xaxis_title="Time",
                yaxis_title="Hysteresis (bars)",
                margin=dict(l=60, r=30, t=30, b=30),
            )
            st.plotly_chart(fig_ah, use_container_width=True)

            # ── Transition Alerts Table ──
            alerts = fusion_result["transition_alerts"]
            if len(alerts) > 0:
                st.subheader(f"Transition Alerts ({len(alerts)} detected)")
                alert_rows = []
                for idx in alerts[-20:]:  # show last 20
                    alert_rows.append({
                        "Bar": int(idx),
                        "Time": str(x_axis[idx]) if hasattr(x_axis[idx], '__str__') else idx,
                        "Regime": df["regime"].values[idx] if idx < len(df) else "?",
                        "CP Prob": f"{cp_prob[idx]:.3f}",
                        "Urgency": f"{urgency[idx]:.3f}",
                        "Hysteresis": f"{adapt_hyst[idx]:.0f}",
                    })
                st.dataframe(pd.DataFrame(alert_rows), use_container_width=True)
            else:
                st.info("No transition alerts above urgency threshold.")

else:
    # ── Landing Page ─────────────────────────────────────────────────────

    # Hero section
    st.markdown("""
    <div class="landing-hero">
        <div class="hero-overline">Quantitative Regime Detection</div>
        <div class="hero-title">
            HMM Regime <span class="accent">Terminal</span>
        </div>
        <div class="hero-subtitle">
            Decode hidden market states with Bayesian model selection,
            walk-forward backtesting, entropy-weighted position sizing,
            and fundamental financial analysis.
        </div>
        <div class="hero-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline section
    st.markdown("""
    <div class="pipeline-section">
        <div class="pipeline-label">Analysis Pipeline</div>
        <div class="pipeline-row">
            <div class="pipe-step">
                <div class="pipe-node">
                    <div class="step-num">01</div>
                    <div class="step-name">Fetch Data</div>
                    <div class="step-detail">OHLCV via yfinance</div>
                </div>
                <span class="pipe-arrow">&rarr;</span>
            </div>
            <div class="pipe-step">
                <div class="pipe-node">
                    <div class="step-num">02</div>
                    <div class="step-name">Engineer Features</div>
                    <div class="step-detail">Returns, vol, momentum</div>
                </div>
                <span class="pipe-arrow">&rarr;</span>
            </div>
            <div class="pipe-step">
                <div class="pipe-node">
                    <div class="step-num">03</div>
                    <div class="step-name">Fit HMM</div>
                    <div class="step-detail">BIC model selection</div>
                </div>
                <span class="pipe-arrow">&rarr;</span>
            </div>
            <div class="pipe-step">
                <div class="pipe-node">
                    <div class="step-num">04</div>
                    <div class="step-name">Detect Regimes</div>
                    <div class="step-detail">Viterbi decoding</div>
                </div>
                <span class="pipe-arrow">&rarr;</span>
            </div>
            <div class="pipe-step">
                <div class="pipe-node">
                    <div class="step-num">05</div>
                    <div class="step-name">Generate Signals</div>
                    <div class="step-detail">Confirmed entries</div>
                </div>
                <span class="pipe-arrow">&rarr;</span>
            </div>
            <div class="pipe-step">
                <div class="pipe-node">
                    <div class="step-num">06</div>
                    <div class="step-name">Fundamentals</div>
                    <div class="step-detail">Ratios, health score</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Math concepts section
    st.markdown('<div class="pipeline-label">Core Mathematical Concepts</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="concepts-grid">
        <div class="concept-card card-green">
            <span class="concept-icon">&#x1D6CC;</span>
            <div class="concept-title">Hidden Markov Model</div>
            <div class="concept-formula">P(X|Z) = &prod; P(x_t | z_t) P(z_t | z_{t-1})</div>
            <div class="concept-desc">
                Latent-state time-series model. Observable features are
                generated by unobserved regime states with Markov transitions.
            </div>
        </div>
        <div class="concept-card card-amber">
            <span class="concept-icon">&#x0042;</span>
            <div class="concept-title">Bayesian Info Criterion</div>
            <div class="concept-formula">BIC = -2 ln(L) + k ln(n)</div>
            <div class="concept-desc">
                Penalizes model complexity to prevent overfitting.
                Selects optimal number of hidden states automatically.
            </div>
        </div>
        <div class="concept-card card-cyan">
            <span class="concept-icon">&#x0048;</span>
            <div class="concept-title">Shannon Entropy</div>
            <div class="concept-formula">H = -&sum; p_i log&sub2;(p_i)</div>
            <div class="concept-desc">
                Measures regime uncertainty from posterior probabilities.
                Low entropy = high conviction, used for confidence scaling.
            </div>
        </div>
        <div class="concept-card card-blue">
            <span class="concept-icon">&#x0066;</span>
            <div class="concept-title">Kelly Criterion</div>
            <div class="concept-formula">f* = (bp - q) / b</div>
            <div class="concept-desc">
                Optimal fraction of capital to risk per trade.
                Maximizes long-run geometric growth rate of portfolio.
            </div>
        </div>
        <div class="concept-card card-red">
            <span class="concept-icon">&#x03B1;</span>
            <div class="concept-title">Walk-Forward Validation</div>
            <div class="concept-formula">train &rarr; test &rarr; step &rarr; repeat</div>
            <div class="concept-desc">
                Anchored expanding-window backtest. Prevents look-ahead
                bias by re-fitting the model at each fold boundary.
            </div>
        </div>
        <div class="concept-card card-green">
            <span class="concept-icon">&#x0024;</span>
            <div class="concept-title">Fundamental Analysis</div>
            <div class="concept-formula">P/E, ROE, D/E, Health Score</div>
            <div class="concept-desc">
                Financial ratios, income trends, balance sheet composition,
                analyst consensus, and a composite health score for equities.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Getting started section
    st.markdown("""
    <div class="getting-started">
        <div class="gs-title">Getting Started</div>
        <div class="gs-steps">
            <div class="gs-step">
                <span class="gs-step-num">1</span>
                <span class="gs-step-text">
                    Open the <strong>sidebar</strong> and enter a ticker symbol (e.g. SPY, AAPL, BTC-USD).
                </span>
            </div>
            <div class="gs-step">
                <span class="gs-step-num">2</span>
                <span class="gs-step-text">
                    Choose your <strong>interval</strong> and <strong>lookback period</strong> for the data window.
                </span>
            </div>
            <div class="gs-step">
                <span class="gs-step-num">3</span>
                <span class="gs-step-text">
                    Adjust HMM, strategy, and risk parameters &mdash; or keep the defaults.
                </span>
            </div>
            <div class="gs-step">
                <span class="gs-step-num">4</span>
                <span class="gs-step-text">
                    Click <strong>Run Analysis</strong> to fetch data, fit the model, and generate regime signals.
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="landing-footer">
        Hidden Markov Models <span class="dot"></span> Baum-Welch EM
        <span class="dot"></span> Viterbi Decoding <span class="dot"></span> Walk-Forward Backtesting
    </div>
    """, unsafe_allow_html=True)
