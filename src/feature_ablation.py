from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/processed/volatility_features.csv")
REPORTS_DIR = Path("reports")

TARGET = "Target_Future_5d_Volatility"
BASELINE = "Baseline_Past_5d_RSS"

FEATURES_FULL = [
    "Volatility_5d",
    "Volatility_20d",
    "Mean_Return_5d",
    "Mean_Return_20d",
    "Mean_Absolute_Return_5d",
]

FEATURES_WITHOUT_20D = [
    feature
    for feature in FEATURES_FULL
    if feature != "Volatility_20d"
]


def fit_and_evaluate(train, test, features):
    """Train Ridge on earlier observations and evaluate on later data."""
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )

    model.fit(train[features], train[TARGET])

    predictions = np.maximum(
        model.predict(test[features]),
        0,
    )

    mae = mean_absolute_error(
        test[TARGET],
        predictions,
    )

    return mae


def compare_models(train, test, period_name):
    """Compare both feature sets on exactly the same observations."""
    full_mae = fit_and_evaluate(
        train,
        test,
        FEATURES_FULL,
    )

    without_20d_mae = fit_and_evaluate(
        train,
        test,
        FEATURES_WITHOUT_20D,
    )

    baseline_mae = mean_absolute_error(
        test[TARGET],
        test[BASELINE],
    )

    change_pct = (
        (without_20d_mae - full_mae) / full_mae * 100
    )

    return {
        "Period": period_name,
        "Train_rows": len(train),
        "Test_rows": len(test),
        "Baseline_MAE": baseline_mae,
        "Ridge_all_features_MAE": full_mae,
        "Ridge_without_Volatility_20d_MAE": without_20d_mae,
        "MAE_change_after_removal_pct": change_pct,
    }


def main():
    df = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True,
    ).sort_index()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Main chronological test, matching train_model.py.
    split_idx = int(len(df) * 0.8)
    train_end = split_idx - 5

    train = df.iloc[:train_end]
    test = df.iloc[split_idx:]

    main_result = compare_models(
        train,
        test,
        "Main test",
    )

    print("--- FEATURE ABLATION: MAIN TEST ---")
    print(f"Training ends: {train.index[-1].date()}")
    print(f"Test starts:   {test.index[0].date()}")
    print(f"Baseline MAE:  {main_result['Baseline_MAE']:.6f}")
    print(
        "Ridge, all features: "
        f"{main_result['Ridge_all_features_MAE']:.6f}"
    )
    print(
        "Ridge, without Volatility_20d: "
        f"{main_result['Ridge_without_Volatility_20d_MAE']:.6f}"
    )
    print(
        "MAE change after removing Volatility_20d: "
        f"{main_result['MAE_change_after_removal_pct']:.2f}%"
    )

    main_df = pd.DataFrame([main_result])
    main_df.to_csv(
        REPORTS_DIR / "feature_ablation_main.csv",
        index=False,
    )

    # Expanding-window evaluation, matching rolling_evaluation.py.
    initial_train_size = int(len(df) * 0.60)
    test_size = int(len(df) * 0.10)
    step_size = test_size

    rolling_results = []
    start = initial_train_size

    while start + test_size <= len(df):
        train_end = start - 5
        train_window = df.iloc[:train_end]
        test_window = df.iloc[start:start + test_size]

        period_name = (
            f"{test_window.index[0].date()}_"
            f"{test_window.index[-1].date()}"
        )

        result = compare_models(
            train_window,
            test_window,
            period_name,
        )

        result["Train_end"] = train_window.index[-1].date()
        result["Test_start"] = test_window.index[0].date()
        result["Test_end"] = test_window.index[-1].date()

        rolling_results.append(result)
        start += step_size

    rolling_df = pd.DataFrame(rolling_results)
    rolling_path = REPORTS_DIR / "feature_ablation_rolling.csv"
    rolling_df.to_csv(rolling_path, index=False)

    print("\n--- FEATURE ABLATION: ROLLING EVALUATION ---")
    print(
        rolling_df[
            [
                "Period",
                "Ridge_all_features_MAE",
                "Ridge_without_Volatility_20d_MAE",
                "MAE_change_after_removal_pct",
            ]
        ].round(6).to_string(index=False)
    )

    print("\n--- SUMMARY ---")
    print(
        "Windows where removing Volatility_20d increased MAE: "
        f"{(rolling_df['MAE_change_after_removal_pct'] > 0).sum()}"
        f"/{len(rolling_df)}"
    )
    print(
        "Mean MAE change after removal: "
        f"{rolling_df['MAE_change_after_removal_pct'].mean():.2f}%"
    )

    print("\nSaved:")
    print(REPORTS_DIR / "feature_ablation_main.csv")
    print(rolling_path)


if __name__ == "__main__":
    main()