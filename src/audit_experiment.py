import numpy as np
import pandas as pd

from project_paths import FEATURES_PATH, RAW_PRICES_PATH
from research_utils import BASELINE, FEATURES, TARGET


def reconstruct_expected_features(prices):
    data = prices.sort_index()
    returns = data["Close"].pct_change()
    expected = pd.DataFrame(index=data.index)
    expected["Volatility_5d"] = returns.rolling(5).std()
    expected["Volatility_20d"] = returns.rolling(20).std()
    expected["Mean_Return_5d"] = returns.rolling(5).mean()
    expected["Mean_Return_20d"] = returns.rolling(20).mean()
    expected["Mean_Absolute_Return_5d"] = returns.abs().rolling(5).mean()
    expected[BASELINE] = np.sqrt(returns.pow(2).rolling(5).sum())

    future_returns = pd.concat(
        [returns.shift(-step).pow(2) for step in range(1, 6)],
        axis=1,
    )
    expected[TARGET] = np.sqrt(future_returns.sum(axis=1, min_count=5))
    return expected[FEATURES + [TARGET, BASELINE]].dropna()


def audit_data(prices, saved_features, tolerance=1e-10):
    expected = reconstruct_expected_features(prices)
    saved = saved_features
    checks = {
        "dates match reconstructed dataset": saved.index.equals(expected.index),
        "dates are unique": not saved_features.index.has_duplicates,
        "dates are sorted": saved_features.index.is_monotonic_increasing,
        "saved values are complete": not saved.isna().any().any(),
    }

    if saved.index.equals(expected.index):
        errors = (saved[FEATURES + [TARGET, BASELINE]] - expected).abs()
        max_errors = errors.max()
        for column in FEATURES + [TARGET, BASELINE]:
            checks[f"{column} matches reconstruction"] = (
                np.isfinite(max_errors[column]) and max_errors[column] < tolerance
            )
    else:
        max_errors = pd.Series(dtype=float)

    return checks, max_errors


def main():
    prices = pd.read_csv(
        RAW_PRICES_PATH,
        index_col=0,
        parse_dates=True,
    )
    saved_features = pd.read_csv(
        FEATURES_PATH,
        index_col=0,
        parse_dates=True,
    )
    checks, max_errors = audit_data(prices, saved_features)

    print("--- EXPERIMENT AUDIT ---")
    print(f"Rows in feature dataset: {len(saved_features)}")
    print(f"Duplicate dates: {saved_features.index.duplicated().sum()}")
    print(f"Dates sorted: {saved_features.index.is_monotonic_increasing}")
    print(f"Missing values: {int(saved_features.isna().sum().sum())}")
    for column in (TARGET, BASELINE):
        if column in max_errors:
            print(f"Maximum {column} reconstruction error: {max_errors[column]:.12f}")

    print("\n--- FEATURE / TARGET TIMING ---")
    print("Features use returns through the current date.")
    print("Target uses returns from the next five sessions.")
    print("Baseline uses returns from the previous five sessions.")

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    if not all(checks.values()):
        raise SystemExit("Experiment audit failed.")
    print("\nPASS: saved features, target and baseline match independent calculations.")


if __name__ == "__main__":
    main()
