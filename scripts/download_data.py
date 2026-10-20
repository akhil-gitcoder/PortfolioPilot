"""Download adjusted daily prices for the NSE universe (RUN THIS ON YOUR OWN MACHINE).

Source: Yahoo Finance via the `yfinance` library (unofficial, free, no API key).
Yahoo's terms and data quality are NOT guaranteed, so this script saves a
fixed snapshot (data/prices_real.csv) that every experiment then reuses.
Commit the snapshot (small CSV) so results are reproducible.

Usage:
    python scripts/download_data.py --start 2019-01-01 --end 2025-12-31

The script validates the download (missing tickers, gaps, extreme moves) and
prints a report.  Tickers that fail are listed - fix config/universe.csv.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=str(ROOT / "config" / "universe.csv"))
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", default=str(ROOT / "data" / "prices_real.csv"))
    args = ap.parse_args()

    import yfinance as yf  # imported here so the rest of the project does not need it

    uni = pd.read_csv(args.universe)
    tickers = uni["ticker"].tolist()
    # auto_adjust=True -> prices adjusted for splits and dividends
    raw = yf.download(tickers, start=args.start, end=args.end,
                      auto_adjust=True, progress=False)
    if raw.empty:
        sys.exit("Download returned no data. Check internet access / ticker names.")
    prices = raw["Close"].copy()
    prices.index = pd.to_datetime(prices.index)

    missing = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    if missing:
        print("TICKERS WITH NO DATA (fix config/universe.csv):", missing)
    prices = prices.drop(columns=missing)

    # --- validation report ---
    na_frac = prices.isna().mean()
    print("Rows:", len(prices), "| first:", prices.index[0].date(), "| last:", prices.index[-1].date())
    print("Max missing fraction per ticker:\n", na_frac.sort_values(ascending=False).head(5))
    rets = prices.pct_change()
    extreme = (rets.abs() > 0.25).sum()
    if extreme.any():
        print("Days with |daily return| > 25% (check corporate actions):\n", extreme[extreme > 0])

    # forward-fill short gaps (holidays that differ by ticker), then drop leading NaN rows
    prices = prices.ffill(limit=3).dropna(how="any")
    prices.to_csv(args.out)
    print(f"Saved {prices.shape[1]} tickers x {len(prices)} days -> {args.out}")
    print("REMEMBER: record the download date and source in data/DATA.md")


if __name__ == "__main__":
    main()
