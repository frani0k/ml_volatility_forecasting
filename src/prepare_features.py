
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/raw/spy_prices.csv")
OUTPUT_PATH = Path("data/processed/volatility_features.csv")

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
df = df.sort_index()

df["Return"] = df["Close"].pct_change()

df["Volatility_5d"] = df["Return"].rolling(5).std()
df["Volatility_20d"] = df["Return"].rolling(20).std()
df["Mean_Return_5d"] = df["Return"].rolling(5).mean()
df["Mean_Return_20d"] = df["Return"].rolling(20).mean()
df["Mean_Absolute_Return_5d"] = (
    df["Return"].abs().rolling(5).mean()
)
df["Baseline_Past_5d_RSS"] = np.sqrt(
    df["Return"].pow(2).rolling(5).sum()
)
future_squared_returns = df["Return"].pow(2).shift(-1)
future_squared_sum = future_squared_returns.iloc[::-1].rolling(5).sum().iloc[::-1]

df["Target_Future_5d_Volatility"] = np.sqrt(future_squared_sum)

feature_columns = [
    "Volatility_5d",
    "Volatility_20d",
    "Mean_Return_5d",
    "Mean_Return_20d",
    "Mean_Absolute_Return_5d",
]
target_column = "Target_Future_5d_Volatility"

dataset = df[
    feature_columns + [target_column, "Baseline_Past_5d_RSS"]
].dropna()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
dataset.to_csv(OUTPUT_PATH, index_label="Date")

print("--- PREPARED DATASET ---")
print(f"Rows: {len(dataset)}")
print(f"Features: {feature_columns}")
print(f"Target: {target_column}")
print(f"First date: {dataset.index.min().date()}")
print(f"Last date: {dataset.index.max().date()}")
print("\n--- SUMMARY ---")
print(dataset.describe().round(5))
print(f"\nSaved to: {OUTPUT_PATH}")
