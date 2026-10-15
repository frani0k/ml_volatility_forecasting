import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from project_paths import FEATURES_PATH, REPORTS_DIR
from research_utils import (
    BASELINE,
    FEATURES as FEATURES_FULL,
    TARGET,
    clip_nonnegative_predictions,
    chronological_split,
    rolling_splits,
)

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

    predictions = clip_nonnegative_predictions(model.predict(test[features]))

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
        FEATURES_PATH,
        index_col=0,
        parse_dates=True,
    ).sort_index()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train, test = chronological_split(df)

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

    rolling_results = []
    for train_window, test_window in rolling_splits(df):
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
