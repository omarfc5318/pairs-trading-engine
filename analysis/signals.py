import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ENTRY_ZSCORE, EXIT_ZSCORE, STOP_ZSCORE, ZSCORE_LOOKBACK


def compute_rolling_zscore(spread: pd.Series, lookback: int) -> pd.Series:
    """
    Compute rolling z-score using only past data at each point (no lookahead).

    z = (spread - rolling_mean) / rolling_std

    The first (lookback - 1) values are NaN because a full window is required.
    """
    rolling = spread.rolling(window=lookback, min_periods=lookback)
    mean    = rolling.mean()
    std     = rolling.std(ddof=1)

    zscore = (spread - mean) / std
    zscore.name = f"zscore_{spread.name}"
    return zscore


def generate_signals(
    zscore: pd.Series,
    entry: float = ENTRY_ZSCORE,
    exit:  float = EXIT_ZSCORE,
    stop:  float = STOP_ZSCORE,
) -> pd.Series:
    """
    Generate trading positions from a z-score series.

    Position values:
        +1  long spread  (spread is cheap — expect it to rise)
        -1  short spread (spread is expensive — expect it to fall)
         0  flat

    Rules (evaluated in priority order each bar):
        1. Stop-loss : |z| > stop            -> exit to flat
        2. Exit      : |z| < exit            -> exit to flat
        3. Entry long : z < -entry           -> enter long  (if flat)
        4. Entry short: z > +entry           -> enter short (if flat)
        5. Otherwise  : hold current position
    """
    position = np.zeros(len(zscore), dtype=int)
    current  = 0  # current position: -1, 0, or +1

    for i, z in enumerate(zscore):
        if np.isnan(z):
            position[i] = 0
            current     = 0
            continue

        # Priority 1: stop-loss
        if abs(z) > stop:
            current = 0

        # Priority 2: exit on mean reversion
        elif abs(z) < exit:
            current = 0

        # Priority 3 & 4: new entries only when flat
        elif current == 0:
            if z < -entry:
                current = 1     # spread is low → long
            elif z > entry:
                current = -1    # spread is high → short

        # Priority 5: hold — current unchanged

        position[i] = current

    return pd.Series(position, index=zscore.index, name="position", dtype=int)


def generate_kalman_signals(
    kalman_df: pd.DataFrame,
    entry: float = ENTRY_ZSCORE,
    exit:  float = EXIT_ZSCORE,
    stop:  float = STOP_ZSCORE,
) -> pd.Series:
    """
    Generate trading signals from a KalmanFilterHedge result DataFrame.

    Z-score is computed from the dynamic spread and its running std:
        z_t = (spread_t - spread_mean_t) / spread_std_t

    Positions follow the same rules as generate_signals():
        +1  long spread, -1  short spread, 0  flat
    """
    spread      = kalman_df["spread"]
    spread_mean = kalman_df["spread_mean"]
    spread_std  = kalman_df["spread_std"].replace(0, np.nan)

    zscore = (spread - spread_mean) / spread_std
    zscore.name = "kalman_zscore"

    return generate_signals(zscore, entry=entry, exit=exit, stop=stop)


if __name__ == "__main__":
    from analysis.ou_process import compute_spread
    from config import ZSCORE_LOOKBACK

    log_prices   = pd.read_parquet(
        Path(__file__).parent.parent / "data" / "clean" / "log_prices.parquet"
    )
    selected     = pd.read_csv(
        Path(__file__).parent.parent / "outputs" / "pairs" / "selected_pairs.csv"
    )

    for _, row in selected.iterrows():
        spread  = compute_spread(log_prices, row["ticker_a"], row["ticker_b"], row["hedge_ratio"])
        zscore  = compute_rolling_zscore(spread, ZSCORE_LOOKBACK)
        signals = generate_signals(zscore)
        trades  = signals.diff().abs().gt(0).sum()
        print(f"{row['ticker_a']}/{row['ticker_b']}: "
              f"{(signals == 1).sum()} long bars, "
              f"{(signals == -1).sum()} short bars, "
              f"{(signals == 0).sum()} flat bars, "
              f"{trades} transitions")
