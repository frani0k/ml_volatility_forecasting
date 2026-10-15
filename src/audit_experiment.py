
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/spy_prices.csv")
FEATURES_PATH = Path("data/processed/volatility_features.csv")

prices = pd.read_csv(RAW_PATH, index_col=0, parse_dates=True).sort_index()
df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True).sort_index()

returns = prices["Close"].pct_change()

future_squared_returns = returns.pow(2).shift(-1)
expected_target = np.sqrt(
    future_squared_returns.iloc[::-1].rolling(5).sum().iloc[::-1]
)
expected_target = expected_target.reindex(df.index)

target_error = (
    df["Target_Future_5d_Volatility"] - expected_target
).abs()

expected_baseline = np.sqrt(returns.pow(2).rolling(5).sum())
expected_baseline = expected_baseline.reindex(df.index)

baseline_error = (
    df["Baseline_Past_5d_RSS"] - expected_baseline
).abs()

print("--- EXPERIMENT AUDIT ---")
print(f"Rows in feature dataset: {len(df)}")
print(f"Duplicate dates: {df.index.duplicated().sum()}")
print(f"Dates sorted: {df.index.is_monotonic_increasing}")
print(f"Missing values: {int(df.isna().sum().sum())}")
print(f"Maximum target reconstruction error: {target_error.max():.12f}")
print(f"Maximum baseline reconstruction error: {baseline_error.max():.12f}")

print("\n--- FEATURE / TARGET TIMING ---")
print("Features use returns through the current date.")
print("Target uses returns from the next five sessions.")
print("Baseline uses returns from the previous five sessions.")

difference = (
    df["Target_Future_5d_Volatility"] - df["Baseline_Past_5d_RSS"]
).abs()
print(f"Mean absolute target-baseline difference: {difference.mean():.6f}")

if target_error.max() < 1e-10 and baseline_error.max() < 1e-10:
    print("\nPASS: saved target and baseline match independent calculations.")
else:
    print("\nWARNING: reconstruction mismatch detected.")
