import pandas as pd

from project_paths import RAW_PRICES_PATH


def main():
    df = pd.read_csv(RAW_PRICES_PATH, index_col=0, parse_dates=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print("\n--- FIRST 5 ROWS ---")
    print(df.head().to_string())

    print("\n--- DATA SHAPE ---")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n--- COLUMN TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES ---")
    print(df.isna().sum())

    print("\n--- SUMMARY STATISTICS ---")
    print(df.describe().round(2))

    print("\n--- DATE RANGE ---")
    print(f"First date: {df.index.min().date()}")
    print(f"Last date:  {df.index.max().date()}")
    print("\n--- DATA QUALITY CHECKS ---")
    print("Duplicate dates:", df.index.duplicated().sum())
    print("Dates sorted:", df.index.is_monotonic_increasing)
    print("Non-positive closing prices:", (df["Close"] <= 0).sum())
    print("Missing closing prices:", df["Close"].isna().sum())
    print("Duplicate full rows:", df.duplicated().sum())


if __name__ == "__main__":
    main()
