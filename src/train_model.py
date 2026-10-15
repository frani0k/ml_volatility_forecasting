
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/processed/volatility_features.csv")

FEATURES = [
    "Volatility_5d",
    "Volatility_20d",
    "Mean_Return_5d",
    "Mean_Return_20d",
    "Mean_Absolute_Return_5d",
]
TARGET = "Target_Future_5d_Volatility"

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).sort_index()

X = df[FEATURES]
y = df[TARGET]

# Chronological split: train on earlier data, test on later data.
split_idx = int(len(df) * 0.8)

# Remove five observations before the test period to reduce
# overlap between training targets and test-period returns.
train_end = split_idx - 5

X_train, X_test = X.iloc[:train_end], X.iloc[split_idx:]
y_train, y_test = y.iloc[:train_end], y.iloc[split_idx:]

# Baseline: use the previous 5-session realized volatility
# to forecast the next 5-session realized volatility.
baseline = df.loc[X_test.index, "Baseline_Past_5d_RSS"]

# Ridge: scale features using training data only, then fit regression.
model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0),
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Volatility cannot be negative.
predictions = np.maximum(predictions, 0)

baseline_mae = mean_absolute_error(y_test, baseline)
ridge_mae = mean_absolute_error(y_test, predictions)

print("--- EXPERIMENT SETUP ---")
print(f"Total observations: {len(df)}")
print(f"Training observations: {len(X_train)}")
print(f"Test observations: {len(X_test)}")
print(f"Training ends: {X_train.index.max().date()}")
print(f"Test starts: {X_test.index.min().date()}")

print("\n--- TEST RESULTS ---")
print(f"Baseline MAE: {baseline_mae:.6f}")
print(f"Ridge MAE:    {ridge_mae:.6f}")

improvement = (baseline_mae - ridge_mae) / baseline_mae * 100
print(f"Ridge improvement vs baseline: {improvement:.2f}%")

results = pd.DataFrame({
    "Actual": y_test,
    "Baseline": baseline,
    "Ridge": predictions,
})
results.to_csv("reports/test_predictions.csv", index_label="Date")
print("\nSaved predictions to reports/test_predictions.csv")
