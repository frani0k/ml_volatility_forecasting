import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from project_paths import FEATURES_PATH, REPORTS_DIR
from research_utils import (
    BASELINE,
    FEATURES,
    TARGET,
    chronological_split,
    clip_nonnegative_predictions,
    rolling_splits,
)


def evaluate_models(train, test, period):
    """Compare the baseline, linear regression and Ridge."""

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    linear_model = make_pipeline(
        StandardScaler(),
        LinearRegression(),
    )
    linear_model.fit(X_train, y_train)
    linear_predictions = clip_nonnegative_predictions(
        linear_model.predict(X_test)
    )

    ridge_model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    ridge_model.fit(X_train, y_train)
    ridge_predictions = clip_nonnegative_predictions(
        ridge_model.predict(X_test)
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
        FEATURES_PATH,
        index_col=0,
        parse_dates=True,
    ).sort_index()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train, test = chronological_split(df)

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

    rolling_results = []
    for train_window, test_window in rolling_splits(df):
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
