from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/processed/volatility_features.csv")
REPORTS_DIR = Path("reports")

FEATURES = [
    "Volatility_5d",
    "Volatility_20d",
    "Mean_Return_5d",
    "Mean_Return_20d",
    "Mean_Absolute_Return_5d",
]

TARGET = "Target_Future_5d_Volatility"
BASELINE = "Baseline_Past_5d_RSS"


def evaluate_models(train, test, period):
    """Compare the baseline, linear regression and Ridge."""

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    # Ordinary linear regression.
    linear_model = make_pipeline(
        StandardScaler(),
        LinearRegression(),
    )
    linear_model.fit(X_train, y_train)
    linear_predictions = np.maximum(
        linear_model.predict(X_test),
        0,
    )

    # Ridge regression with fixed regularization.
    ridge_model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    ridge_model.fit(X_train, y_train)
    ridge_predictions = np.maximum(
        ridge_model.predict(X_test),
        0,
    )

    baseline_mae = mean_absolute_error(
        y_test,
        test[BASELINE],
    )
    linear_mae = mean_absolute_error(
        y_test,
        linear_predictions,
    )
    ridge_mae = mean_absolute_error(
        y_test,
        ridge_predictions,
    )

    # Positive values mean Ridge has lower MAE.
    ridge_improvement = (
        (linear_mae - ridge_mae) / linear_mae * 100
    )

    return {
        "Period": period,
        "Train_rows": len(train),
        "Test_rows": len(test),
        "Baseline_MAE": baseline_mae,
        "LinearRegression_MAE": linear_mae,
        "Ridge_MAE": ridge_mae,
        "Ridge_improvement_vs_linear_pct": ridge_improvement,
    }


def main():
    df = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True,
    ).sort_index()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Main chronological split, matching train_model.py.
    split_idx = int(len(df) * 0.8)
    train_end = split_idx - 5

    train = df.iloc[:train_end]
    test = df.iloc[split_idx:]

    main_result = evaluate_models(
        train,
        test,
        "Main test",
    )

    main_df = pd.DataFrame([main_result])
    main_df.to_csv(
        REPORTS_DIR / "model_comparison_main.csv",
        index=False,
    )

    print("--- MAIN TEST ---")
    print(f"Training ends: {train.index[-1].date()}")
    print(f"Test starts:   {test.index[0].date()}")
    print(f"Baseline MAE:  {main_result['Baseline_MAE']:.6f}")
    print(
        "Linear Regression MAE: "
        f"{main_result['LinearRegression_MAE']:.6f}"
    )
    print(f"Ridge MAE: {main_result['Ridge_MAE']:.6f}")
    print(
        "Ridge improvement vs Linear Regression: "
        f"{main_result['Ridge_improvement_vs_linear_pct']:.2f}%"
    )

    # Repeat the comparison on the existing rolling windows.
    initial_train_size = int(len(df) * 0.60)
    test_size = int(len(df) * 0.10)
    step_size = test_size

    rolling_results = []
    start = initial_train_size

    while start + test_size <= len(df):
        train_end = start - 5
        train_window = df.iloc[:train_end]
        test_window = df.iloc[start:start + test_size]

        period = (
            f"{test_window.index[0].date()}_"
            f"{test_window.index[-1].date()}"
        )

        result = evaluate_models(
            train_window,
            test_window,
            period,
        )

        result["Train_end"] = train_window.index[-1].date()
        result["Test_start"] = test_window.index[0].date()
        result["Test_end"] = test_window.index[-1].date()

        rolling_results.append(result)
        start += step_size

    rolling_df = pd.DataFrame(rolling_results)
    rolling_df.to_csv(
        REPORTS_DIR / "model_comparison_rolling.csv",
        index=False,
    )

    print("\n--- ROLLING EVALUATION ---")
    print(
        rolling_df[
            [
                "Period",
                "LinearRegression_MAE",
                "Ridge_MAE",
                "Ridge_improvement_vs_linear_pct",
            ]
        ].round(6).to_string(index=False)
    )

    print("\n--- SUMMARY ---")
    ridge_wins = (
        rolling_df["Ridge_improvement_vs_linear_pct"] > 0
    ).sum()

    print(f"Windows where Ridge wins: {ridge_wins}/{len(rolling_df)}")
    print(
        "Mean Ridge improvement: "
        f"{rolling_df['Ridge_improvement_vs_linear_pct'].mean():.2f}%"
    )

    print("\nSaved:")
    print(REPORTS_DIR / "model_comparison_main.csv")
    print(REPORTS_DIR / "model_comparison_rolling.csv")


if __name__ == "__main__":
    main()