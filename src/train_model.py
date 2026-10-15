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
    chronological_split,
    clip_nonnegative_predictions,
)


def main():
    df = pd.read_csv(
        FEATURES_PATH,
        index_col=0,
        parse_dates=True,
    ).sort_index()
    train, test = chronological_split(df)

    X_train = train[FEATURES]
    X_test = test[FEATURES]
    y_train = train[TARGET]
    y_test = test[TARGET]
    baseline = test[BASELINE]

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    model.fit(X_train, y_train)
    predictions = clip_nonnegative_predictions(model.predict(X_test))

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
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "test_predictions.csv"
    results.to_csv(output_path, index_label="Date")
    print(f"\nSaved predictions to {output_path}")


if __name__ == "__main__":
    main()
