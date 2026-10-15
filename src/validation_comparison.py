import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge
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


def evaluate_model(train, test):
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    model.fit(train[FEATURES], train[TARGET])
    predictions = clip_nonnegative_predictions(
        model.predict(test[FEATURES])
    )
    return mean_absolute_error(test[TARGET], predictions)


def main():
    df = pd.read_csv(
        FEATURES_PATH,
        index_col=0,
        parse_dates=True,
    ).sort_index()

    full_train, test = chronological_split(df)
    full_mae = evaluate_model(full_train, test)

    records = []
    for seed in range(30):
        subset = full_train.sample(
            frac=0.8,
            random_state=seed,
        ).sort_index()

        subset_mae = evaluate_model(subset, test)
        records.append({
            "Seed": seed,
            "Train_rows": len(subset),
            "MAE": subset_mae,
            "Full_train_MAE": full_mae,
            "Difference_pct": (subset_mae - full_mae) / full_mae * 100,
        })

    results = pd.DataFrame(records)

    print("--- TRAINING-SAMPLE SENSITIVITY ---")
    print(f"Full training rows: {len(full_train)}")
    print(f"Fixed test rows: {len(test)}")
    print(f"Test period: {test.index.min().date()} to {test.index.max().date()}")
    print(f"Full-training MAE: {full_mae:.8f}")
    print(f"Random-subset MAE, mean: {results['MAE'].mean():.8f}")
    print(f"Random-subset MAE, std:  {results['MAE'].std():.8f}")
    print(f"Mean relative difference: {results['Difference_pct'].mean():+.3f}%")
    print(f"Median relative difference: {results['Difference_pct'].median():+.3f}%")
    print(
        "Subsets with lower MAE than full training: "
        f"{(results['MAE'] < full_mae).sum()} / {len(results)}"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = REPORTS_DIR / "validation_subset_runs.csv"
    results.to_csv(results_path, index=False)

    summary = pd.DataFrame([{
        "Full_train_rows": len(full_train),
        "Test_rows": len(test),
        "Full_train_MAE": full_mae,
        "Subset_train_fraction": 0.8,
        "Runs": len(results),
        "Subset_MAE_mean": results["MAE"].mean(),
        "Subset_MAE_std": results["MAE"].std(),
        "Mean_difference_pct": results["Difference_pct"].mean(),
        "Median_difference_pct": results["Difference_pct"].median(),
        "Subset_wins": int((results["MAE"] < full_mae).sum()),
    }])
    summary_path = REPORTS_DIR / "validation_comparison.csv"
    summary.to_csv(summary_path, index=False)

    figure_path = REPORTS_DIR / "figures" / "validation_comparison.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(results["MAE"], bins=10, edgecolor="black")
    ax.axvline(
        full_mae,
        linestyle="--",
        label=f"Full training MAE = {full_mae:.6f}",
    )
    ax.set_title("MAE across 30 random training subsets")
    ax.set_xlabel("Test MAE")
    ax.set_ylabel("Number of runs")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print(f"\nSaved individual runs: {results_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved chart: {figure_path}")


if __name__ == "__main__":
    main()