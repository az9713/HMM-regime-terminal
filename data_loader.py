"""
data_loader.py — yfinance data fetching + feature engineering.

Provides OHLCV retrieval, technical feature computation,
z-score standardization (train-only stats), and feature matrix extraction.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from datetime import datetime, timedelta


def fetch_ohlcv(
    ticker: str = "BTC-USD",
    interval: str = "1h",
    lookback_days: int = 90,
) -> pd.DataFrame:
    """Download OHLCV data from yfinance."""
    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    # yfinance interval constraints
    max_days = {
        "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
        "1h": 730, "1d": 10000, "1wk": 10000, "1mo": 10000,
    }
    limit = max_days.get(interval, 730)
    if lookback_days > limit:
        start = end - timedelta(days=limit)

    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} ({interval}, {lookback_days}d)")

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


def compute_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add 5 features to OHLCV DataFrame:
      - log_return: ln(close_t / close_{t-1})
      - rolling_vol: std of log_return over window (default 21)
      - volume_change: volume_t / SMA(volume, 20) - 1
      - intraday_range: (high - low) / close
      - rsi: RSI(14)
    """
    out = df.copy()
    vol_window = config.get("rolling_vol_window", 21) if config else 21
    rsi_period = config.get("rsi_period", 14) if config else 14

    out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))
    out["rolling_vol"] = out["log_return"].rolling(vol_window).std()
    vol_sma = out["Volume"].rolling(20).mean()
    out["volume_change"] = out["Volume"] / vol_sma - 1
    out["intraday_range"] = (out["High"] - out["Low"]) / out["Close"]
    out["rsi"] = RSIIndicator(out["Close"], window=rsi_period).rsi()

    out.dropna(inplace=True)
    out.reset_index(drop=False, inplace=True)
    return out


def standardize(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """
    Z-score standardization using training set statistics only (no lookahead).
    Returns (train_z, test_z, stats_dict).
    """
    if cols is None:
        cols = ["log_return", "rolling_vol", "volume_change", "intraday_range", "rsi"]

    stats = {}
    train_z = train.copy()
    for c in cols:
        mu = train[c].mean()
        sigma = train[c].std()
        if sigma == 0:
            sigma = 1.0
        stats[c] = {"mean": mu, "std": sigma}
        train_z[c] = (train[c] - mu) / sigma

    test_z = None
    if test is not None:
        test_z = test.copy()
        for c in cols:
            test_z[c] = (test[c] - stats[c]["mean"]) / stats[c]["std"]

    return train_z, test_z, stats


def get_feature_matrix(df: pd.DataFrame, cols: list[str] | None = None) -> np.ndarray:
    """Extract feature columns as a numpy array for hmmlearn."""
    if cols is None:
        cols = ["log_return", "rolling_vol", "volume_change", "intraday_range", "rsi"]
    return df[cols].values
