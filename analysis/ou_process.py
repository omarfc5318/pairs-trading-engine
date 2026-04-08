import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HALFLIFE_MIN, HALFLIFE_MAX

PAIRS_DIR = Path(__file__).parent.parent / "outputs" / "pairs"


def compute_spread(
    log_prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    hedge_ratio: float,
) -> pd.Series:
    """
    Compute spread = log(A) - hedge_ratio * log(B).
    """
    spread = log_prices[ticker_a] - hedge_ratio * log_prices[ticker_b]
    spread.name = f"{ticker_a}-{ticker_b}"
    return spread.dropna()


def estimate_halflife(spread_series: pd.Series) -> float:
    """
    Fit AR(1) to the spread via OLS on lagged differences:
        delta_s(t) = alpha + (rho - 1) * s(t-1) + epsilon

    Half-life = -log(2) / log(rho), where rho is the AR(1) coefficient.
    Returns half-life in trading days.
    """
    s = spread_series.values
    s_lag = s[:-1]
    delta_s = np.diff(s)

    # OLS: delta_s ~ a + b * s_lag  =>  b = rho - 1
    X = np.column_stack([np.ones_like(s_lag), s_lag])
    coeffs, _, _, _ = np.linalg.lstsq(X, delta_s, rcond=None)
    b = coeffs[1]          # b = rho - 1
    rho = 1 + b            # AR(1) coefficient

    # Clip rho to (0, 1) to ensure a valid mean-reverting process
    rho = float(np.clip(rho, 1e-8, 1 - 1e-8))
    halflife = -np.log(2) / np.log(rho)
    return round(float(halflife), 2)


def select_pairs(
    coint_df: pd.DataFrame,
    log_prices: pd.DataFrame,
    halflife_min: int = HALFLIFE_MIN,
    halflife_max: int = HALFLIFE_MAX,
) -> pd.DataFrame:
    """
    Compute half-life for each cointegrated pair and filter to those within
    [halflife_min, halflife_max] trading days.

    Adds a 'halflife' column to the DataFrame.
    Saves result to outputs/pairs/selected_pairs.csv.
    """
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for _, row in coint_df.iterrows():
        ticker_a    = row["ticker_a"]
        ticker_b    = row["ticker_b"]
        hedge_ratio = row["hedge_ratio"]

        spread   = compute_spread(log_prices, ticker_a, ticker_b, hedge_ratio)
        halflife = estimate_halflife(spread)

        records.append({**row.to_dict(), "halflife": halflife})
        print(f"[ou] {ticker_a}/{ticker_b}: half-life = {halflife:.2f} days  "
              f"(p={row['p_value']:.4f}, hedge={hedge_ratio:.4f})")

    all_pairs = pd.DataFrame(records)

    selected = (
        all_pairs[
            all_pairs["halflife"].between(halflife_min, halflife_max)
        ]
        .sort_values("halflife")
        .reset_index(drop=True)
    )

    out_path = PAIRS_DIR / "selected_pairs.csv"
    selected.to_csv(out_path, index=False)

    print(f"\n[ou] {len(selected)}/{len(all_pairs)} pairs pass half-life filter "
          f"[{halflife_min}, {halflife_max}] days -> {out_path}")
    if not selected.empty:
        print(selected[["ticker_a", "ticker_b", "p_value", "hedge_ratio", "halflife"]].to_string(index=False))

    return selected


class KalmanFilterHedge:
    """
    Estimates a time-varying hedge ratio β using a scalar Kalman Filter.

    State-space model
    -----------------
    State transition  : β_t = β_t-1 + w_t       w_t ~ N(0, Q)
    Observation       : A_t  = β_t · B_t + v_t   v_t ~ N(0, R)

    where A_t = log(price_A)_t, B_t = log(price_B)_t.

    Parameters
    ----------
    Q : process noise covariance  (how fast β is allowed to drift)
    R : observation noise covariance (noise in the price relationship)
    """

    def __init__(self, Q: float = 1e-5, R: float = 1e-3):
        self.Q = Q
        self.R = R

    def fit(
        self,
        log_price_a: pd.Series,
        log_price_b: pd.Series,
    ) -> pd.DataFrame:
        """
        Run the Kalman filter over the full series.

        At each timestep t the hedge ratio is estimated using ONLY data up
        to and including t — no lookahead.

        Returns a DataFrame indexed by date with columns:
            dynamic_hedge_ratio  — β_t (posterior mean)
            spread               — log(A)_t - β_t · log(B)_t
            spread_mean          — running posterior mean of spread (= 0 by construction)
            spread_std           — running std estimated from innovations
        """
        aligned = pd.concat([log_price_a, log_price_b], axis=1).dropna()
        aligned.columns = ["a", "b"]
        dates = aligned.index
        n     = len(aligned)

        Q, R = self.Q, self.R

        # Initialise state with OLS on first 20 observations (or all if < 20)
        init_n  = min(20, n)
        a0      = aligned["a"].iloc[:init_n].values
        b0      = aligned["b"].iloc[:init_n].values
        beta_init = float(np.polyfit(b0, a0, 1)[0])

        # State: β, variance: P
        beta = beta_init
        P    = 1.0           # initial state variance

        betas       = np.empty(n)
        spreads     = np.empty(n)
        innovations = np.empty(n)   # for rolling std estimation

        for i in range(n):
            a_t = aligned["a"].iloc[i]
            b_t = aligned["b"].iloc[i]

            # ── Predict ──────────────────────────────────────────────────
            # β_t|t-1 = β_t-1   (random walk: F = 1)
            # P_t|t-1 = P_t-1 + Q
            P_pred = P + Q

            # ── Update ───────────────────────────────────────────────────
            # H_t = b_t  (time-varying observation matrix)
            H_t = b_t

            # Innovation (prediction error)
            y_hat = H_t * beta          # predicted observation
            innov = a_t - y_hat         # innovation

            # Innovation covariance
            S = H_t * P_pred * H_t + R

            # Kalman gain
            K = P_pred * H_t / S

            # Posterior state and variance
            beta = beta + K * innov
            P    = (1 - K * H_t) * P_pred

            betas[i]       = beta
            spreads[i]     = a_t - beta * b_t
            innovations[i] = innov

        # Rolling spread std: expanding std of the posterior spread.
        # This is the correct normaliser for z-score — causal, no lookahead.
        spread_series = pd.Series(spreads, index=dates)
        spread_mean_s = spread_series.expanding(min_periods=2).mean().fillna(0.0)
        spread_std    = spread_series.expanding(min_periods=2).std(ddof=1).fillna(1.0)

        result = pd.DataFrame({
            "dynamic_hedge_ratio": betas,
            "spread":              spreads,
            "spread_mean":         spread_mean_s.values,
            "spread_std":          spread_std.values,
        }, index=dates)

        return result


if __name__ == "__main__":
    coint_df   = pd.read_csv(Path(__file__).parent.parent / "outputs" / "pairs" / "coint_results.csv")
    log_prices = pd.read_parquet(Path(__file__).parent.parent / "data" / "clean" / "log_prices.parquet")
    select_pairs(coint_df, log_prices)
