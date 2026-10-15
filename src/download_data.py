"""Download daily, adjusted SPY prices for the volatility research project."""

from pathlib import Path

import yfinance as yf
import pandas as pd


START_DATE = "2015-01-01"
# yfinance treats `end` as exclusive, so this includes 2025-12-31.
END_DATE = "2026-01-01"
TICKER = "SPY"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "spy_prices.csv"


def flatten_columns(data):
    """Safely reduce yfinance's possible MultiIndex columns to field names."""
    if not isinstance(data.columns, pd.MultiIndex):
        return data

    columns = data.columns
    ticker_level = next(
        (level for level in range(columns.nlevels) if TICKER in columns.get_level_values(level)),
        None,
    )
    if ticker_level is not None:
        data = data.xs(TICKER, axis=1, level=ticker_level, drop_level=True)

    if isinstance(data.columns, pd.MultiIndex):
        # Remaining singleton levels carry no information; drop them first.
        for level in reversed(range(data.columns.nlevels)):
            if len(data.columns.get_level_values(level).unique()) == 1:
                data.columns = data.columns.droplevel(level)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ["_".join(str(part) for part in col if str(part)) for col in data.columns]
    return data


def main():
    prices = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )
    if prices.empty:
        raise ValueError(
            f"No price data was downloaded for {TICKER} from {START_DATE} through 2025-12-31."
        )

    prices = flatten_columns(prices)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(OUTPUT_PATH, index_label="Date")

    print(f"Saved {len(prices)} rows to {OUTPUT_PATH}")
    print(f"Date range: {prices.index.min()} through {prices.index.max()}")
    print(f"Columns: {list(prices.columns)}")


if __name__ == "__main__":
    main()
