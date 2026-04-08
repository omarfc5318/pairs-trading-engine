import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python data/clean.py` from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

RAW_DIR   = Path(__file__).parent / "raw"
CLEAN_DIR = Path(__file__).parent / "clean"


def load_and_align_prices() -> pd.DataFrame:
    """
    Load all raw Parquet files, align to a common date index, forward-fill
    up to 2 consecutive gaps, drop tickers that are still missing after that,
    and save to data/clean/prices.parquet.
    """
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(RAW_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {RAW_DIR}")

    series: dict[str, pd.Series] = {}
    for f in parquet_files:
        ticker = f.stem
        df = pd.read_parquet(f)
        # Each file has one column named after the ticker
        col = df.columns[0]
        series[ticker] = df[col].rename(ticker)

    print(f"[clean] Loaded {len(series)} tickers: {sorted(series)}")

    # Inner join — keeps only dates all tickers share
    prices = pd.concat(series.values(), axis=1, join="inner")
    print(f"[clean] After inner join: {len(prices)} trading days "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")

    # Forward-fill up to 2 consecutive NaNs, then drop columns still missing
    prices_ffill = prices.ffill(limit=2)

    dropped = [c for c in prices_ffill.columns if prices_ffill[c].isna().any()]
    if dropped:
        warnings.warn(f"[clean] Dropping tickers with remaining NaNs after ffill(2): {dropped}")
        prices_ffill = prices_ffill.drop(columns=dropped)
    else:
        print("[clean] No tickers dropped — all clean after forward-fill.")

    # Drop any remaining rows with NaNs (e.g. leading NaNs before ffill kicks in)
    before = len(prices_ffill)
    prices_ffill = prices_ffill.dropna()
    dropped_rows = before - len(prices_ffill)
    if dropped_rows:
        print(f"[clean] Dropped {dropped_rows} rows with leading NaNs.")

    out_path = CLEAN_DIR / "prices.parquet"
    prices_ffill.to_parquet(out_path)

    print(f"[clean] Saved prices.parquet: {prices_ffill.shape[0]} rows x "
          f"{prices_ffill.shape[1]} tickers -> {out_path}")
    return prices_ffill


def compute_log_prices() -> pd.DataFrame:
    """
    Load data/clean/prices.parquet, apply log transform, and save as
    data/clean/log_prices.parquet.
    """
    prices_path = CLEAN_DIR / "prices.parquet"
    if not prices_path.exists():
        raise FileNotFoundError(f"prices.parquet not found — run load_and_align_prices() first.")

    prices = pd.read_parquet(prices_path)
    log_prices = np.log(prices)

    out_path = CLEAN_DIR / "log_prices.parquet"
    log_prices.to_parquet(out_path)

    print(f"[clean] Saved log_prices.parquet: {log_prices.shape[0]} rows x "
          f"{log_prices.shape[1]} tickers -> {out_path}")
    return log_prices


if __name__ == "__main__":
    load_and_align_prices()
    compute_log_prices()
