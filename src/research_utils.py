import numpy as np
import pandas as pd


FEATURES = [
    "Volatility_5d",
    "Volatility_20d",
    "Mean_Return_5d",
    "Mean_Return_20d",
    "Mean_Absolute_Return_5d",
]
TARGET = "Target_Future_5d_Volatility"
BASELINE = "Baseline_Past_5d_RSS"


def calculate_daily_returns(prices):
    """Calculate close-to-close simple returns."""
    if "Close" not in prices:
        raise KeyError("Price data must include a 'Close' column.")
    return prices["Close"].pct_change()


def build_feature_dataset(prices):
    """Build trailing features, the past-volatility baseline and future target."""
    data = prices.copy().sort_index()
    returns = calculate_daily_returns(data)
    data["Return"] = returns

    data["Volatility_5d"] = returns.rolling(5).std()
    data["Volatility_20d"] = returns.rolling(20).std()
    data["Mean_Return_5d"] = returns.rolling(5).mean()
    data["Mean_Return_20d"] = returns.rolling(20).mean()
    data["Mean_Absolute_Return_5d"] = returns.abs().rolling(5).mean()
    data[BASELINE] = np.sqrt(returns.pow(2).rolling(5).sum())

    future_squared_returns = returns.pow(2).shift(-1)
    future_squared_sum = (
        future_squared_returns.iloc[::-1].rolling(5).sum().iloc[::-1]
    )
    data[TARGET] = np.sqrt(future_squared_sum)

    return data[FEATURES + [TARGET, BASELINE]].dropna()


def chronological_split(data, train_fraction=0.8, gap=5):
    """Split ordered observations and leave a gap between training and test."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if gap < 0:
        raise ValueError("gap must not be negative.")

    split_index = int(len(data) * train_fraction)
    train_end = split_index - gap
    if train_end <= 0 or split_index >= len(data):
        raise ValueError("Data is too short for the requested split and gap.")
    return data.iloc[:train_end], data.iloc[split_index:]


def rolling_splits(data, initial_train_fraction=0.6, test_fraction=0.1, gap=5):
    """Yield expanding chronological train/test windows with a fixed gap."""
    if not 0 < initial_train_fraction < 1:
        raise ValueError("initial_train_fraction must be between 0 and 1.")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if gap < 0:
        raise ValueError("gap must not be negative.")

    initial_train_size = int(len(data) * initial_train_fraction)
    test_size = int(len(data) * test_fraction)
    if test_size < 1 or initial_train_size <= gap:
        raise ValueError("Data is too short for the requested rolling windows.")

    start = initial_train_size
    while start + test_size <= len(data):
        train_end = start - gap
        yield data.iloc[:train_end], data.iloc[start:start + test_size]
        start += test_size


def clip_nonnegative_predictions(predictions):
    """Clip forecasts at zero because the volatility target is non-negative."""
    return np.maximum(predictions, 0)
