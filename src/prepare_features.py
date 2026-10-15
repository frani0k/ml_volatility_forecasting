import pandas as pd

from project_paths import FEATURES_PATH, RAW_PRICES_PATH
from research_utils import FEATURES, TARGET, build_feature_dataset


def main():
    prices = pd.read_csv(
        RAW_PRICES_PATH,
        index_col=0,
        parse_dates=True,
    )
    dataset = build_feature_dataset(prices)

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(FEATURES_PATH, index_label="Date")

    print("--- PREPARED DATASET ---")
    print(f"Rows: {len(dataset)}")
    print(f"Features: {FEATURES}")
    print(f"Target: {TARGET}")
    print(f"First date: {dataset.index.min().date()}")
    print(f"Last date: {dataset.index.max().date()}")
    print("\n--- SUMMARY ---")
    print(dataset.describe().round(5))
    print(f"\nSaved to: {FEATURES_PATH}")


if __name__ == "__main__":
    main()
