import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COINT_PVALUE_THRESHOLD

PAIRS_DIR = Path(__file__).parent.parent / "outputs" / "pairs"

# Correlation pre-filter threshold — only test pairs above this level
CORR_THRESHOLD = 0.70


def run_adf_test(series: pd.Series) -> tuple[float, bool]:
    """
    Run Augmented Dickey-Fuller test on a series.

    Returns
    -------
    p_value : float
    is_stationary : bool  — True if p_value < 0.05
    """
    result = adfuller(series.dropna(), autolag="AIC")
    p_value = float(result[1])
    return p_value, p_value < 0.05


def _corr_prefilter(
    log_prices_df: pd.DataFrame,
    threshold: float = CORR_THRESHOLD,
) -> list[tuple[str, str]]:
    """
    Return only pairs whose Pearson correlation exceeds threshold.
    Reduces O(n²) cointegration tests to a tractable candidate set.
    """
    corr    = log_prices_df.corr()
    tickers = list(log_prices_df.columns)
    pairs   = []
    for a, b in combinations(tickers, 2):
        if abs(corr.loc[a, b]) >= threshold:
            pairs.append((a, b))
    return pairs


def _test_one_pair(
    args: tuple[str, str, pd.Series, pd.Series],
) -> dict | None:
    """
    Worker function for parallel cointegration testing.
    Runs Engle-Granger + OLS hedge ratio for a single pair.
    Returns a result dict or None if skipped.
    """
    ticker_a, ticker_b, series_a, series_b = args

    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    if len(aligned) < 30:
        return None

    a = aligned[ticker_a].values
    b = aligned[ticker_b].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p_value, _ = coint(a, b)
        ols_result    = OLS(a, add_constant(b)).fit()

    hedge_ratio = float(ols_result.params[1])

    return {
        "ticker_a":    ticker_a,
        "ticker_b":    ticker_b,
        "p_value":     round(float(p_value), 6),
        "hedge_ratio": round(hedge_ratio, 6),
    }


def run_cointegration_tests(
    log_prices_df: pd.DataFrame,
    corr_threshold: float = CORR_THRESHOLD,
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Run Engle-Granger cointegration test on pairs pre-filtered by correlation.

    Steps
    -----
    1. Pearson correlation pre-filter (|corr| > corr_threshold) to prune pairs
    2. Parallel cointegration tests via ProcessPoolExecutor (max_workers)
    3. tqdm progress bar over the candidate pairs

    Returns a DataFrame sorted by p_value with columns:
        ticker_a, ticker_b, p_value, hedge_ratio
    """
    tickers     = list(log_prices_df.columns)
    total_pairs = len(tickers) * (len(tickers) - 1) // 2

    # ── Step 1: correlation pre-filter ────────────────────────────────────────
    candidates = _corr_prefilter(log_prices_df, corr_threshold)
    print(f"[coint] {len(tickers)} tickers → {total_pairs} total pairs")
    print(f"[coint] Correlation pre-filter (|r| ≥ {corr_threshold}): "
          f"{len(candidates)} candidate pairs remain")

    if not candidates:
        print("[coint] No pairs passed the correlation filter.")
        return pd.DataFrame(columns=["ticker_a", "ticker_b", "p_value", "hedge_ratio"])

    # Build argument tuples for workers
    args_list = [
        (a, b, log_prices_df[a], log_prices_df[b])
        for a, b in candidates
    ]

    # ── Step 2 & 3: parallel tests with progress bar ───────────────────────────
    records = []

    # ThreadPoolExecutor avoids macOS spawn overhead and works in all environments.
    # numpy/statsmodels release the GIL so threads give real parallelism here.
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_test_one_pair, args): args
                       for args in args_list}
            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               desc="Cointegration tests",
                               unit="pair"):
                result = future.result()
                if result is not None:
                    records.append(result)
    except Exception as exc:
        print(f"[coint] Parallel execution failed ({exc}); falling back to sequential.")
        for args in tqdm(args_list, desc="Cointegration tests", unit="pair"):
            result = _test_one_pair(args)
            if result is not None:
                records.append(result)

    coint_df = (
        pd.DataFrame(records)
        .sort_values("p_value")
        .reset_index(drop=True)
    )

    n_sig = (coint_df["p_value"] < COINT_PVALUE_THRESHOLD).sum()
    print(f"[coint] Tested {len(records)} pairs → "
          f"{n_sig} significant at p < {COINT_PVALUE_THRESHOLD}")
    return coint_df


def filter_pairs(coint_df: pd.DataFrame, pvalue_threshold: float) -> pd.DataFrame:
    """
    Filter cointegration results to significant pairs and save to CSV.

    Saves to outputs/pairs/coint_results.csv.
    Returns the filtered DataFrame.
    """
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    significant = (
        coint_df[coint_df["p_value"] < pvalue_threshold]
        .reset_index(drop=True)
    )

    out_path = PAIRS_DIR / "coint_results.csv"
    significant.to_csv(out_path, index=False)

    print(f"[coint] {len(significant)} significant pairs saved → {out_path}")
    if not significant.empty:
        print(significant.to_string(index=False))
    return significant


if __name__ == "__main__":
    log_prices = pd.read_parquet(
        Path(__file__).parent.parent / "data" / "clean" / "log_prices.parquet"
    )
    coint_df = run_cointegration_tests(log_prices)
    filter_pairs(coint_df, COINT_PVALUE_THRESHOLD)
