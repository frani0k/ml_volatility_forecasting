
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
BASELINE = "Baseline_Past_5d_RSS"

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).sort_index()

# Each test window follows its training window in time.
# A five-row gap reduces overlap between train targets and test returns.
initial_train_size = int(len(df) * 0.60)
test_size = int(len(df) * 0.10)
step_size = test_size

results = []
start = initial_train_size

while start + test_size <= len(df):
    train_end = start - 5
    train = df.iloc[:train_end]
    test = df.iloc[start:start + test_size]

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    model.fit(X_train, y_train)

    predictions = np.maximum(model.predict(X_test), 0)
    baseline_predictions = test[BASELINE]

    ridge_mae = mean_absolute_error(y_test, predictions)
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    improvement = (baseline_mae - ridge_mae) / baseline_mae * 100

    results.append({
        "Train_end": train.index[-1].date(),
        "Test_start": test.index[0].date(),
        "Test_end": test.index[-1].date(),
        "Test_rows": len(test),
        "Baseline_MAE": baseline_mae,
        "Ridge_MAE": ridge_mae,
        "Improvement_pct": improvement,
    })

    start += step_size

results_df = pd.DataFrame(results)

print("--- CHRONOLOGICAL ROBUSTNESS TEST ---")
print(results_df.round(6).to_string(index=False))

print("\n--- SUMMARY ---")
print(f"Test windows: {len(results_df)}")
print(f"Windows where Ridge wins: {(results_df['Improvement_pct'] > 0).sum()}")
print(f"Mean improvement: {results_df['Improvement_pct'].mean():.2f}%")
print(f"Median improvement: {results_df['Improvement_pct'].median():.2f}%")

output_path = Path("reports/rolling_evaluation.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
