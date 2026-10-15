import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from project_paths import FEATURES_PATH, REPORTS_DIR
from research_utils import (
    BASELINE,
    FEATURES,
    TARGET,
    clip_nonnegative_predictions,
    rolling_splits,
)


def main():
    df = pd.read_csv(
        FEATURES_PATH,
        index_col=0,
        parse_dates=True,
    ).sort_index()

    results = []
    for train, test in rolling_splits(df):
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
        )
        model.fit(train[FEATURES], train[TARGET])
        predictions = clip_nonnegative_predictions(model.predict(test[FEATURES]))
        baseline_predictions = test[BASELINE]

        ridge_mae = mean_absolute_error(test[TARGET], predictions)
        baseline_mae = mean_absolute_error(test[TARGET], baseline_predictions)
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

    results_df = pd.DataFrame(results)
    print("--- CHRONOLOGICAL ROBUSTNESS TEST ---")
    print(results_df.round(6).to_string(index=False))

    print("\n--- SUMMARY ---")
    print(f"Test windows: {len(results_df)}")
    print(f"Windows where Ridge wins: {(results_df['Improvement_pct'] > 0).sum()}")
    print(f"Mean improvement: {results_df['Improvement_pct'].mean():.2f}%")
    print(f"Median improvement: {results_df['Improvement_pct'].median():.2f}%")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "rolling_evaluation.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
