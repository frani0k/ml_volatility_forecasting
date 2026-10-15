
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error

results_path = Path("reports/test_predictions.csv")
df = pd.read_csv(results_path, index_col=0, parse_dates=True)

# Define the high-volatility group using the test target.
threshold = df["Actual"].quantile(0.80)
df["Regime"] = df["Actual"].apply(
    lambda value: "High volatility (top 20%)"
    if value >= threshold
    else "Other observations"
)

print("--- OVERALL TEST PERFORMANCE ---")
for model in ["Baseline", "Ridge"]:
    print(f"{model} MAE: {mean_absolute_error(df['Actual'], df[model]):.6f}")

print("\n--- PERFORMANCE BY REALIZED VOLATILITY ---")
for regime, group in df.groupby("Regime"):
    baseline_mae = mean_absolute_error(group["Actual"], group["Baseline"])
    ridge_mae = mean_absolute_error(group["Actual"], group["Ridge"])
    improvement = (baseline_mae - ridge_mae) / baseline_mae * 100

    print(f"\n{regime} (n={len(group)})")
    print(f"Baseline MAE: {baseline_mae:.6f}")
    print(f"Ridge MAE:    {ridge_mae:.6f}")
    print(f"Ridge improvement: {improvement:.2f}%")

# Save a comparison chart.
ax = df[["Actual", "Baseline", "Ridge"]].plot(
    figsize=(12, 6),
    alpha=0.8,
    title="Actual vs Forecast 5-Session Volatility",
)
ax.set_xlabel("Date")
ax.set_ylabel("5-session volatility (daily-return units)")

output_path = Path("reports/figures/test_predictions.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
ax.figure.tight_layout()
ax.figure.savefig(output_path, dpi=150)

print(f"\nChart saved to: {output_path}")
