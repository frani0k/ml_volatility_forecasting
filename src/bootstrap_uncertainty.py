import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from project_paths import FEATURES_PATH, REPORTS_DIR
from research_utils import (
    FEATURES,
    TARGET,
    chronological_split,
    clip_nonnegative_predictions,
)


N_BOOTSTRAPS = 5000
BLOCK_LENGTHS = [5, 10, 20]
RANDOM_SEED = 42


def bootstrap_block_ci(
    linear_errors,
    ridge_errors,
    block_length,
    n_bootstraps=N_BOOTSTRAPS,
    random_seed=RANDOM_SEED,
):
    """Estimate uncertainty in paired MAE differences using block bootstrap."""
    linear_errors = np.asarray(linear_errors)
    ridge_errors = np.asarray(ridge_errors)
    if linear_errors.ndim != 1 or ridge_errors.ndim != 1:
        raise ValueError("Error inputs must be one-dimensional arrays.")
    if linear_errors.size == 0 or linear_errors.shape != ridge_errors.shape:
        raise ValueError("Error inputs must be non-empty arrays of equal length.")
    if not 1 <= block_length <= len(linear_errors):
        raise ValueError("block_length must be between 1 and the number of errors.")
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be at least 1.")

    n = len(linear_errors)
    n_blocks = int(np.ceil(n / block_length))
    rng = np.random.default_rng(random_seed)

    max_start = n - block_length
    starts = rng.integers(
        0,
        max_start + 1,
        size=(n_bootstraps, n_blocks),
    )
    offsets = np.arange(block_length)

    indices = (starts[:, :, None] + offsets).reshape(
        n_bootstraps, -1
    )[:, :n]

    sampled_linear = linear_errors[indices].mean(axis=1)
    sampled_ridge = ridge_errors[indices].mean(axis=1)

    differences = sampled_linear - sampled_ridge
    improvements_pct = (
        differences / sampled_linear * 100
    )

    return {
        "Block_length": block_length,
        "Bootstrap_samples": n_bootstraps,
        "MAE_difference_linear_minus_ridge": (
            linear_errors.mean() - ridge_errors.mean()
        ),
        "MAE_difference_CI_lower": np.quantile(
            differences, 0.025
        ),
        "MAE_difference_CI_upper": np.quantile(
            differences, 0.975
        ),
        "Ridge_improvement_pct": (
            (linear_errors.mean() - ridge_errors.mean())
            / linear_errors.mean()
            * 100
        ),
        "Improvement_pct_CI_lower": np.quantile(
            improvements_pct, 0.025
        ),
        "Improvement_pct_CI_upper": np.quantile(
            improvements_pct, 0.975
        ),
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        FEATURES_PATH,
        index_col=0,
        parse_dates=True,
    ).sort_index()

    train, test = chronological_split(df)

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    linear_model = make_pipeline(
        StandardScaler(),
        LinearRegression(),
    )
    ridge_model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )

    linear_model.fit(X_train, y_train)
    ridge_model.fit(X_train, y_train)

    linear_predictions = clip_nonnegative_predictions(
        linear_model.predict(X_test)
    )
    ridge_predictions = clip_nonnegative_predictions(
        ridge_model.predict(X_test)
    )

    predictions = pd.DataFrame(
        {
            "Actual": y_test,
            "LinearRegression": linear_predictions,
            "Ridge": ridge_predictions,
            "AbsError_Linear": np.abs(
                y_test.to_numpy() - linear_predictions
            ),
            "AbsError_Ridge": np.abs(
                y_test.to_numpy() - ridge_predictions
            ),
        },
        index=y_test.index,
    )
    predictions.index.name = "Date"
    predictions.to_csv(
        REPORTS_DIR / "model_comparison_test_predictions.csv"
    )

    linear_errors = predictions[
        "AbsError_Linear"
    ].to_numpy()
    ridge_errors = predictions[
        "AbsError_Ridge"
    ].to_numpy()

    print("--- MODEL COMPARISON ---")
    print(f"Training observations: {len(train)}")
    print(f"Test observations: {len(test)}")
    print(
        "Linear Regression MAE: "
        f"{mean_absolute_error(y_test, linear_predictions):.9f}"
    )
    print(
        "Ridge MAE:             "
        f"{mean_absolute_error(y_test, ridge_predictions):.9f}"
    )
    print(
        "MAE difference (Linear - Ridge): "
        f"{linear_errors.mean() - ridge_errors.mean():.10f}"
    )

    results = [
        bootstrap_block_ci(
            linear_errors,
            ridge_errors,
            block_length,
        )
        for block_length in BLOCK_LENGTHS
    ]

    results_df = pd.DataFrame(results)
    results_path = REPORTS_DIR / "bootstrap_uncertainty.csv"
    results_df.to_csv(results_path, index=False)

    print("\n--- BLOCK BOOTSTRAP (95% INTERVALS) ---")
    for result in results:
        print(
            f"Block length {result['Block_length']:>2}: "
            f"MAE difference CI = "
            f"[{result['MAE_difference_CI_lower']:.8f}, "
            f"{result['MAE_difference_CI_upper']:.8f}], "
            f"improvement CI = "
            f"[{result['Improvement_pct_CI_lower']:.3f}%, "
            f"{result['Improvement_pct_CI_upper']:.3f}%]"
        )

    print(f"\nSaved predictions: "
          f"{REPORTS_DIR / 'model_comparison_test_predictions.csv'}")
    print(f"Saved intervals: {results_path}")


if __name__ == "__main__":
    main()
