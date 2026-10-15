
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

data_path = Path("data/raw/spy_prices.csv")
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
df = df.sort_index()

df["Daily_Return"] = df["Close"].pct_change()

df["Volatility_20d"] = df["Daily_Return"].rolling(window=20).std()

df["Volatility_20d_Annualized"] = (
    df["Volatility_20d"] * (252 ** 0.5)
)

print("\n--- DAILY RETURNS ---")
print(df["Daily_Return"].describe().round(5))

print("\n--- ANNUALIZED 20-DAY VOLATILITY ---")
print(df["Volatility_20d_Annualized"].describe().round(4))

print("\n--- LARGEST DAILY MOVES ---")
largest_moves = df.loc[
    df["Daily_Return"].abs().nlargest(5).index,
    ["Close", "Daily_Return"],
]
print(largest_moves.sort_values("Daily_Return"))

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(df.index, df["Close"])
axes[0].set_title("SPY Adjusted Closing Price")
axes[0].set_ylabel("Price (USD)")

axes[1].plot(
    df.index,
    df["Volatility_20d_Annualized"] * 100,
)
axes[1].set_title("20-Session Annualized Historical Volatility")
axes[1].set_ylabel("Volatility (%)")
axes[1].set_xlabel("Date")

fig.tight_layout()

output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "price_and_volatility.png", dpi=150)

plt.show()

print(f"\nChart saved to: {output_dir / 'price_and_volatility.png'}")
