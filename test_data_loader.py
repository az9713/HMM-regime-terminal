"""
TDD tests for data_loader.py — standardize and get_feature_matrix.

Note: fetch_ohlcv and compute_features require yfinance/ta which are
not available in this environment, so we test the pure-logic functions.
"""

import numpy as np
import pandas as pd
import pytest

from data_loader import standardize, get_feature_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_feature_df(n=100, seed=42):
    """Create a DataFrame mimicking compute_features output."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "log_return": rng.normal(0, 0.02, n),
        "rolling_vol": np.abs(rng.normal(0.02, 0.005, n)),
        "volume_change": rng.normal(0, 0.5, n),
        "intraday_range": np.abs(rng.normal(0.02, 0.01, n)),
        "rsi": rng.uniform(20, 80, n),
    })


# ---------------------------------------------------------------------------
# Tests: standardize
# ---------------------------------------------------------------------------

class TestStandardize:
    def test_train_is_zero_mean_unit_var(self):
        train = make_feature_df(200, seed=1)
        train_z, _, stats = standardize(train)
        for col in ["log_return", "rolling_vol", "volume_change",
                     "intraday_range", "rsi"]:
            np.testing.assert_allclose(train_z[col].mean(), 0.0, atol=1e-10)
            np.testing.assert_allclose(train_z[col].std(), 1.0, atol=0.01)

    def test_test_uses_train_stats(self):
        train = make_feature_df(200, seed=1)
        test = make_feature_df(50, seed=2)
        train_z, test_z, stats = standardize(train, test)
        # test_z should be standardized with train's mean/std
        for col in ["log_return", "rolling_vol"]:
            expected = (test[col] - stats[col]["mean"]) / stats[col]["std"]
            pd.testing.assert_series_equal(test_z[col], expected)

    def test_no_test_returns_none(self):
        train = make_feature_df(100)
        train_z, test_z, stats = standardize(train)
        assert test_z is None

    def test_stats_dict_structure(self):
        train = make_feature_df(100)
        _, _, stats = standardize(train)
        for col in ["log_return", "rolling_vol", "volume_change",
                     "intraday_range", "rsi"]:
            assert "mean" in stats[col]
            assert "std" in stats[col]

    def test_custom_columns(self):
        train = make_feature_df(100)
        cols = ["log_return", "rsi"]
        train_z, _, stats = standardize(train, cols=cols)
        assert set(stats.keys()) == set(cols)
        # Non-selected columns should be unchanged
        pd.testing.assert_series_equal(train_z["rolling_vol"], train["rolling_vol"])

    def test_zero_std_column(self):
        """A column with zero variance should get std=1 (no division by zero)."""
        train = make_feature_df(100)
        train["log_return"] = 5.0  # constant column
        train_z, _, stats = standardize(train, cols=["log_return"])
        assert stats["log_return"]["std"] == 1.0
        # All values should be (5.0 - 5.0) / 1.0 = 0
        np.testing.assert_allclose(train_z["log_return"].values, 0.0)

    def test_does_not_modify_original(self):
        train = make_feature_df(100)
        original_values = train["log_return"].values.copy()
        standardize(train)
        np.testing.assert_array_equal(train["log_return"].values, original_values)


# ---------------------------------------------------------------------------
# Tests: get_feature_matrix
# ---------------------------------------------------------------------------

class TestGetFeatureMatrix:
    def test_default_columns(self):
        df = make_feature_df(50)
        X = get_feature_matrix(df)
        assert X.shape == (50, 5)

    def test_custom_columns(self):
        df = make_feature_df(50)
        X = get_feature_matrix(df, cols=["log_return", "rsi"])
        assert X.shape == (50, 2)

    def test_returns_numpy_array(self):
        df = make_feature_df(50)
        X = get_feature_matrix(df)
        assert isinstance(X, np.ndarray)

    def test_values_match(self):
        df = make_feature_df(50)
        X = get_feature_matrix(df, cols=["log_return"])
        np.testing.assert_array_equal(X.flatten(), df["log_return"].values)
