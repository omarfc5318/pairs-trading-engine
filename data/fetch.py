import sys
import warnings
from pathlib import Path

import yfinance as yf
import pandas as pd

# Allow `python data/fetch.py` from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import UNIVERSE, START_DATE, END_DATE

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_financials() -> list[str]:
    """
    Scrape the S&P 500 Wikipedia table and return tickers in the
    GICS Sector 'Financials'.

    Returns a sorted list of ticker strings. Dots in tickers are replaced
    with hyphens to match yfinance conventions (e.g. BRK.B → BRK-B).
    """
    import io as _io
    import urllib.request as _req

    # Wikipedia returns 403 for the default Python UA — spoof a browser header
    request = _req.Request(
        SP500_WIKI_URL,
        headers={"User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )},
    )
    with _req.urlopen(request, timeout=20) as resp:
        html = resp.read().decode("utf-8")

    tables = pd.read_html(_io.StringIO(html), header=0)
    sp500  = tables[0]   # first table is the constituents list

    # Column names vary slightly — find sector and ticker columns robustly
    sector_col = next(c for c in sp500.columns if "sector" in c.lower())
    ticker_col = next(c for c in sp500.columns
                      if "symbol" in c.lower() or "ticker" in c.lower())

    financials = sp500[sp500[sector_col] == "Financials"][ticker_col].tolist()

    # yfinance uses hyphens not dots
    financials = [t.replace(".", "-") for t in financials]
    financials = sorted(set(financials))

    print(f"[fetch] S&P 500 Financials: {len(financials)} tickers found")
    return financials

RAW_DIR = Path(__file__).parent / "raw"


def fetch_prices(tickers: list[str], start: str, end: str) -> dict[str, pd.Series]:
    """
    Download adjusted close prices for each ticker via yfinance.

    Saves each ticker as a Parquet file in data/raw/<TICKER>.parquet.
    Returns a dict mapping ticker -> pd.Series of adj close prices.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.Series] = {}

    for ticker in tickers:
        try:
            raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

            if raw.empty:
                warnings.warn(f"[fetch] No data returned for {ticker} — skipping.", stacklevel=2)
                continue

            if "Close" not in raw.columns:
                warnings.warn(f"[fetch] 'Close' column missing for {ticker} — skipping.", stacklevel=2)
                continue

            series: pd.Series = raw["Close"].squeeze().rename(ticker)
            series = series.dropna()

            if series.empty:
                warnings.warn(f"[fetch] All NaN after dropna for {ticker} — skipping.", stacklevel=2)
                continue

            out_path = RAW_DIR / f"{ticker}.parquet"
            series.to_frame().to_parquet(out_path)
            print(f"[fetch] Saved {ticker}: {len(series)} rows -> {out_path}")
            results[ticker] = series

        except Exception as exc:
            warnings.warn(f"[fetch] Failed to fetch {ticker}: {exc}", stacklevel=2)

    print(f"[fetch] Done. {len(results)}/{len(tickers)} tickers saved.")
    return results


if __name__ == "__main__":
    fetch_prices(UNIVERSE, START_DATE, END_DATE)
