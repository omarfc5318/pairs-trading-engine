import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.signals    import compute_rolling_zscore, generate_signals, generate_kalman_signals
from analysis.ou_process import KalmanFilterHedge
from analysis.regime     import RegimeDetector, fetch_spy_returns

BACKTEST_DIR = Path(__file__).parent.parent / "outputs" / "backtest"


@dataclass
class Config:
    train_window:   int
    test_window:    int
    zscore_lookback: int
    entry_zscore:   float
    exit_zscore:    float
    stop_zscore:    float
    cost_bps:       int
    capital:        float
    position_size:  float


class WalkForwardBacktester:
    """
    Rolling walk-forward backtester for pairs trading.

    For each fold:
      - Training window : re-estimate OLS hedge ratio on log prices
      - Test window     : compute spread, z-score (normalised with train stats),
                          generate signals, calculate daily P&L

    No information from the test window is used during parameter estimation.
    """

    def __init__(
        self,
        log_prices:     pd.DataFrame,
        selected_pairs: pd.DataFrame,
        config:         Config,
    ):
        self.log_prices     = log_prices
        self.selected_pairs = selected_pairs
        self.cfg            = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute walk-forward backtest across all selected pairs.

        Returns
        -------
        daily_pnl : pd.DataFrame  — shape (dates, pairs), net P&L in $ per day
        trade_log : pd.DataFrame  — one row per completed trade
        """
        BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

        dates       = self.log_prices.index
        n           = len(dates)
        all_pnl     = {}
        all_trades  = []

        for _, pair in self.selected_pairs.iterrows():
            ticker_a = pair["ticker_a"]
            ticker_b = pair["ticker_b"]
            pair_label = f"{ticker_a}/{ticker_b}"
            print(f"\n[engine] Running {pair_label} ...")

            pair_pnl, pair_trades = self._backtest_pair(
                ticker_a, ticker_b, dates, n
            )
            all_pnl[pair_label]  = pair_pnl
            all_trades.extend(pair_trades)

        daily_pnl = pd.DataFrame(all_pnl).sort_index()
        trade_log = pd.DataFrame(all_trades)

        self._save(daily_pnl, trade_log)
        return daily_pnl, trade_log

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _backtest_pair(
        self,
        ticker_a: str,
        ticker_b: str,
        dates:    pd.DatetimeIndex,
        n:        int,
    ) -> tuple[pd.Series, list[dict]]:
        """Run walk-forward backtest for a single pair."""
        cfg  = self.cfg
        tw   = cfg.train_window
        sw   = cfg.test_window

        pnl_pieces: list[pd.Series] = []
        trades: list[dict]          = []
        prev_position               = 0   # carry position across fold boundaries

        fold = 0
        start = 0
        while start + tw + sw <= n:
            train_idx = slice(start, start + tw)
            test_idx  = slice(start + tw, start + tw + sw)

            train_dates = dates[train_idx]
            test_dates  = dates[test_idx]

            log_a_train = self.log_prices[ticker_a].iloc[train_idx]
            log_b_train = self.log_prices[ticker_b].iloc[train_idx]
            log_a_test  = self.log_prices[ticker_a].iloc[test_idx]
            log_b_test  = self.log_prices[ticker_b].iloc[test_idx]

            # --- 1. Re-estimate hedge ratio on training data (OLS) ---
            hedge_ratio = self._estimate_hedge(log_a_train, log_b_train)

            # --- 2. Compute spread on training data, get normalisation stats ---
            spread_train = log_a_train - hedge_ratio * log_b_train
            train_mean   = spread_train.mean()
            train_std    = spread_train.std(ddof=1)

            if train_std < 1e-8:
                start += sw
                continue

            # --- 3. Compute spread and z-score on test data using TRAIN stats ---
            spread_test = log_a_test - hedge_ratio * log_b_test
            zscore_test = (spread_test - train_mean) / train_std
            zscore_test = pd.Series(zscore_test.values, index=test_dates)

            # Seed the spread return calculation with the last training value
            # so the first day of the test window is not NaN
            spread_test_seeded = pd.concat([
                spread_train.iloc[[-1]],
                spread_test,
            ])

            # --- 4. Generate signals (carry prev position into this fold) ---
            raw_signals = generate_signals(
                zscore_test,
                entry=cfg.entry_zscore,
                exit=cfg.exit_zscore,
                stop=cfg.stop_zscore,
            )

            # Override first bar if we were carrying a position from prev fold
            signals = raw_signals.copy()
            if prev_position != 0:
                # honour hold-until-exit logic: keep prev position unless the
                # new fold's first bar already triggers exit/stop
                if abs(zscore_test.iloc[0]) < cfg.exit_zscore or \
                   abs(zscore_test.iloc[0]) > cfg.stop_zscore:
                    pass  # signal correctly went to 0
                else:
                    signals.iloc[0] = prev_position

            prev_position = int(signals.iloc[-1])

            # --- 5. Compute daily P&L ---
            dollar_size     = cfg.capital * cfg.position_size
            # diff() on the seeded series; first test-window day is no longer NaN
            spread_returns  = spread_test_seeded.diff().iloc[1:]
            spread_returns.index = test_dates
            # Position is entered at close of signal day; P&L accrues next day
            lagged_pos      = signals.shift(1).fillna(0)
            gross_pnl       = lagged_pos * spread_returns * dollar_size

            # Transaction costs: 2 legs × COST_BPS on every position change
            position_change = signals.diff().abs().fillna(signals.abs())
            cost_per_trade  = dollar_size * 2 * cfg.cost_bps / 10_000
            costs           = position_change * cost_per_trade

            net_pnl = gross_pnl - costs
            pnl_pieces.append(net_pnl)

            # --- 6. Build trade log for this fold ---
            trades.extend(self._extract_trades(
                signals, zscore_test, gross_pnl, costs,
                ticker_a, ticker_b, hedge_ratio, fold,
            ))

            fold  += 1
            start += sw   # roll forward by one test window

        pair_pnl = pd.concat(pnl_pieces).sort_index() if pnl_pieces else pd.Series(dtype=float)
        return pair_pnl, trades

    @staticmethod
    def _estimate_hedge(log_a: pd.Series, log_b: pd.Series) -> float:
        """OLS regression of log_a ~ log_b; returns slope (hedge ratio)."""
        result = OLS(log_a.values, add_constant(log_b.values)).fit()
        return float(result.params[1])

    @staticmethod
    def _extract_trades(
        signals:     pd.Series,
        zscore:      pd.Series,
        gross_pnl:   pd.Series,
        costs:       pd.Series,
        ticker_a:    str,
        ticker_b:    str,
        hedge_ratio: float,
        fold:        int,
    ) -> list[dict]:
        """Extract individual trade records (entry → exit) from a signal series."""
        trades = []
        in_trade   = False
        entry_date = None
        entry_z    = None
        direction  = 0
        trade_pnl  = 0.0
        trade_cost = 0.0

        for date, pos in signals.items():
            if not in_trade and pos != 0:
                in_trade   = True
                entry_date = date
                entry_z    = float(zscore[date])
                direction  = int(pos)
                trade_pnl  = float(gross_pnl.get(date, 0.0))
                trade_cost = float(costs.get(date, 0.0))

            elif in_trade and pos == direction:
                trade_pnl  += float(gross_pnl.get(date, 0.0))
                trade_cost += float(costs.get(date, 0.0))

            elif in_trade and pos != direction:
                # Position closed (exit or stop or flip)
                trades.append({
                    "fold":        fold,
                    "ticker_a":    ticker_a,
                    "ticker_b":    ticker_b,
                    "hedge_ratio": round(hedge_ratio, 6),
                    "direction":   "long" if direction == 1 else "short",
                    "entry_date":  entry_date,
                    "exit_date":   date,
                    "entry_zscore": round(entry_z, 4),
                    "exit_zscore":  round(float(zscore[date]), 4),
                    "gross_pnl":   round(trade_pnl, 2),
                    "cost":        round(trade_cost, 2),
                    "net_pnl":     round(trade_pnl - trade_cost, 2),
                })
                in_trade = False
                if pos != 0:   # immediate re-entry in opposite direction
                    in_trade   = True
                    entry_date = date
                    entry_z    = float(zscore[date])
                    direction  = int(pos)
                    trade_pnl  = float(gross_pnl.get(date, 0.0))
                    trade_cost = float(costs.get(date, 0.0))

        # Close any open trade at end of fold
        if in_trade:
            last_date = signals.index[-1]
            trades.append({
                "fold":        fold,
                "ticker_a":    ticker_a,
                "ticker_b":    ticker_b,
                "hedge_ratio": round(hedge_ratio, 6),
                "direction":   "long" if direction == 1 else "short",
                "entry_date":  entry_date,
                "exit_date":   last_date,
                "entry_zscore": round(entry_z, 4),
                "exit_zscore":  round(float(zscore[last_date]), 4),
                "gross_pnl":   round(trade_pnl, 2),
                "cost":        round(trade_cost, 2),
                "net_pnl":     round(trade_pnl - trade_cost, 2),
            })

        return trades

    def run_kalman(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Walk-forward backtest using a Kalman Filter for dynamic hedge ratio estimation.

        For each fold:
          - Training window : warm up the Kalman filter on log prices
          - Test window     : continue the filter (no lookahead), derive z-scores,
                              generate signals, compute P&L

        The filter state carries forward across folds — it is never reset,
        which means the hedge ratio adapts continuously with each new price.

        Returns
        -------
        daily_pnl : pd.DataFrame  (dates x pairs)
        trade_log : pd.DataFrame  (one row per trade)

        Also saves to outputs/backtest/kalman_daily_pnl.csv and kalman_trade_log.csv.
        """
        BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

        dates      = self.log_prices.index
        n          = len(dates)
        all_pnl    = {}
        all_trades = []

        for _, pair in self.selected_pairs.iterrows():
            ticker_a   = pair["ticker_a"]
            ticker_b   = pair["ticker_b"]
            pair_label = f"{ticker_a}/{ticker_b}"
            print(f"\n[engine/kalman] Running {pair_label} ...")

            pair_pnl, pair_trades = self._backtest_pair_kalman(
                ticker_a, ticker_b, dates, n
            )
            all_pnl[pair_label]  = pair_pnl
            all_trades.extend(pair_trades)

        daily_pnl = pd.DataFrame(all_pnl).sort_index()
        trade_log = pd.DataFrame(all_trades)

        pnl_path   = BACKTEST_DIR / "kalman_daily_pnl.csv"
        trade_path = BACKTEST_DIR / "kalman_trade_log.csv"
        daily_pnl.to_csv(pnl_path)
        trade_log.to_csv(trade_path, index=False)
        print(f"\n[engine/kalman] Saved kalman_daily_pnl  -> {pnl_path} "
              f"({daily_pnl.shape[0]} days x {daily_pnl.shape[1]} pairs)")
        print(f"[engine/kalman] Saved kalman_trade_log  -> {trade_path} "
              f"({len(trade_log)} trades)")

        return daily_pnl, trade_log

    def run_kalman_with_regime(self, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Walk-forward Kalman backtest with HMM regime filter on SPY.

        Regime rules:
          - regime = 0 (mean-reverting): new entries allowed, hold existing positions
          - regime = 1 (trending)       : no new entries; exit any open position immediately

        All Kalman hedge ratio logic is identical to run_kalman().

        Parameters
        ----------
        save : if True (default) persist results to outputs/backtest/.
               Pass False during sensitivity sweeps to avoid overwriting canonical files.

        Saves to outputs/backtest/regime_daily_pnl.csv and regime_trade_log.csv.
        """
        BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

        dates = self.log_prices.index
        n     = len(dates)

        # ── Fit regime detector on SPY ─────────────────────────────────────────
        start_str = str(dates[0].date())
        end_str   = str(dates[-1].date())
        print(f"\n[engine/regime] Fetching SPY returns {start_str} → {end_str} ...")
        spy_returns = fetch_spy_returns(start_str, end_str)

        detector = RegimeDetector()
        detector.fit(spy_returns, train_window=252)
        regimes = detector.predict_regimes(spy_returns)

        mr = (regimes == 0).sum()
        tr = (regimes == 1).sum()
        print(f"[engine/regime] Regime split — mean-reverting: {mr} ({mr/len(regimes):.1%})  "
              f"trending: {tr} ({tr/len(regimes):.1%})")

        # Align regimes to log_prices index (fill gaps with 0 = allow trading)
        regimes_aligned = regimes.reindex(dates).fillna(0).astype(int)

        all_pnl    = {}
        all_trades = []

        for _, pair in self.selected_pairs.iterrows():
            ticker_a   = pair["ticker_a"]
            ticker_b   = pair["ticker_b"]
            pair_label = f"{ticker_a}/{ticker_b}"
            print(f"\n[engine/regime] Running {pair_label} ...")

            pair_pnl, pair_trades = self._backtest_pair_kalman_regime(
                ticker_a, ticker_b, dates, n, regimes_aligned
            )
            all_pnl[pair_label]  = pair_pnl
            all_trades.extend(pair_trades)

        daily_pnl = pd.DataFrame(all_pnl).sort_index()
        trade_log = pd.DataFrame(all_trades)

        if save:
            BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
            pnl_path   = BACKTEST_DIR / "regime_daily_pnl.csv"
            trade_path = BACKTEST_DIR / "regime_trade_log.csv"
            daily_pnl.to_csv(pnl_path)
            trade_log.to_csv(trade_path, index=False)
            print(f"\n[engine/regime] Saved regime_daily_pnl -> {pnl_path} "
                  f"({daily_pnl.shape[0]} days x {daily_pnl.shape[1]} pairs)")
            print(f"[engine/regime] Saved regime_trade_log -> {trade_path} "
                  f"({len(trade_log)} trades)")

        return daily_pnl, trade_log

    def _backtest_pair_kalman_regime(
        self,
        ticker_a:        str,
        ticker_b:        str,
        dates:           pd.DatetimeIndex,
        n:               int,
        regimes_aligned: pd.Series,
    ) -> tuple[pd.Series, list[dict]]:
        """Kalman walk-forward with regime gating for a single pair."""
        cfg = self.cfg
        tw  = cfg.train_window
        sw  = cfg.test_window

        log_a = self.log_prices[ticker_a]
        log_b = self.log_prices[ticker_b]

        # Run Kalman filter over full series (causal — no lookahead)
        kf        = KalmanFilterHedge(Q=1e-5, R=1e-3)
        kalman_df = kf.fit(log_a, log_b)

        pnl_pieces: list[pd.Series] = []
        trades: list[dict]          = []
        prev_position               = 0

        fold  = 0
        start = 0
        while start + tw + sw <= n:
            test_idx   = slice(start + tw, start + tw + sw)
            test_dates = dates[test_idx]

            kf_test            = kalman_df.iloc[test_idx]
            regime_test        = regimes_aligned.iloc[test_idx]
            train_last_spread  = kalman_df["spread"].iloc[start + tw - 1]

            # Raw Kalman signals (without regime filter)
            raw_signals = generate_kalman_signals(
                kf_test,
                entry=cfg.entry_zscore,
                exit=cfg.exit_zscore,
                stop=cfg.stop_zscore,
            )

            # ── Apply regime filter ────────────────────────────────────────────
            signals      = raw_signals.copy()
            current_pos  = prev_position

            for i, (date, raw_pos) in enumerate(raw_signals.items()):
                regime = int(regime_test.iloc[i])

                if regime == 1:
                    # Trending — exit immediately, no new entries
                    current_pos = 0
                else:
                    # Mean-reverting — follow Kalman signals normally
                    # Honour carry-over position on first bar
                    if i == 0 and prev_position != 0:
                        spread_std = kf_test["spread_std"].iloc[0]
                        z0 = kf_test["spread"].iloc[0] / spread_std if spread_std > 1e-8 else 0.0
                        if abs(z0) < cfg.exit_zscore or abs(z0) > cfg.stop_zscore:
                            current_pos = raw_pos   # exit signal honoured
                        else:
                            current_pos = prev_position  # hold carried position
                    else:
                        current_pos = raw_pos

                signals.iloc[i] = current_pos

            prev_position = int(signals.iloc[-1])

            # ── P&L ───────────────────────────────────────────────────────────
            dollar_size     = cfg.capital * cfg.position_size
            spread_seeded   = pd.concat([
                pd.Series([train_last_spread], index=[dates[start + tw - 1]]),
                kf_test["spread"],
            ])
            spread_returns  = spread_seeded.diff().iloc[1:]
            spread_returns.index = test_dates

            lagged_pos      = signals.shift(1).fillna(0)
            gross_pnl       = lagged_pos * spread_returns * dollar_size

            position_change = signals.diff().abs().fillna(signals.abs())
            cost_per_trade  = dollar_size * 2 * cfg.cost_bps / 10_000
            costs           = position_change * cost_per_trade

            pnl_pieces.append(gross_pnl - costs)

            mean_hedge    = float(kf_test["dynamic_hedge_ratio"].mean())
            zscore_series = (kf_test["spread"] - kf_test["spread_mean"]) / \
                            kf_test["spread_std"].replace(0, np.nan)
            trades.extend(self._extract_trades(
                signals, zscore_series, gross_pnl, costs,
                ticker_a, ticker_b, mean_hedge, fold,
            ))

            fold  += 1
            start += sw

        pair_pnl = pd.concat(pnl_pieces).sort_index() if pnl_pieces else pd.Series(dtype=float)
        return pair_pnl, trades

    def _backtest_pair_kalman(
        self,
        ticker_a: str,
        ticker_b: str,
        dates:    pd.DatetimeIndex,
        n:        int,
    ) -> tuple[pd.Series, list[dict]]:
        """Run Kalman walk-forward backtest for a single pair."""
        cfg = self.cfg
        tw  = cfg.train_window
        sw  = cfg.test_window

        log_a = self.log_prices[ticker_a]
        log_b = self.log_prices[ticker_b]

        # Run the Kalman filter over the ENTIRE series once.
        # Each estimate at t uses only data up to t (causal by construction).
        kf        = KalmanFilterHedge(Q=1e-5, R=1e-3)
        kalman_df = kf.fit(log_a, log_b)

        pnl_pieces: list[pd.Series] = []
        trades: list[dict]          = []
        prev_position               = 0

        fold  = 0
        start = 0
        while start + tw + sw <= n:
            test_idx   = slice(start + tw, start + tw + sw)
            test_dates = dates[test_idx]

            # Kalman z-score slice for the test window
            kf_test = kalman_df.iloc[test_idx]

            # Get the last training spread value to seed the diff (avoid NaN)
            train_last_spread = kalman_df["spread"].iloc[start + tw - 1]

            # Signals from Kalman z-scores
            raw_signals = generate_kalman_signals(
                kf_test,
                entry=cfg.entry_zscore,
                exit=cfg.exit_zscore,
                stop=cfg.stop_zscore,
            )

            # Carry previous position
            signals = raw_signals.copy()
            spread_std = kf_test["spread_std"].iloc[0]
            if spread_std > 1e-8:
                z0 = kf_test["spread"].iloc[0] / spread_std
            else:
                z0 = 0.0

            if prev_position != 0:
                if abs(z0) < cfg.exit_zscore or abs(z0) > cfg.stop_zscore:
                    pass
                else:
                    signals.iloc[0] = prev_position

            prev_position = int(signals.iloc[-1])

            # Daily P&L
            dollar_size = cfg.capital * cfg.position_size
            spread_test = kf_test["spread"]

            # Seed diff with last training spread value
            spread_seeded   = pd.concat([
                pd.Series([train_last_spread], index=[dates[start + tw - 1]]),
                spread_test,
            ])
            spread_returns  = spread_seeded.diff().iloc[1:]
            spread_returns.index = test_dates

            lagged_pos     = signals.shift(1).fillna(0)
            gross_pnl      = lagged_pos * spread_returns * dollar_size

            position_change = signals.diff().abs().fillna(signals.abs())
            cost_per_trade  = dollar_size * 2 * cfg.cost_bps / 10_000
            costs           = position_change * cost_per_trade

            pnl_pieces.append(gross_pnl - costs)

            # Trade log — use dynamic hedge ratio (mean over test window)
            mean_hedge = float(kf_test["dynamic_hedge_ratio"].mean())
            zscore_series = (kf_test["spread"] - kf_test["spread_mean"]) / \
                            kf_test["spread_std"].replace(0, np.nan)
            trades.extend(self._extract_trades(
                signals, zscore_series, gross_pnl, costs,
                ticker_a, ticker_b, mean_hedge, fold,
            ))

            fold  += 1
            start += sw

        pair_pnl = pd.concat(pnl_pieces).sort_index() if pnl_pieces else pd.Series(dtype=float)
        return pair_pnl, trades

    def _save(self, daily_pnl: pd.DataFrame, trade_log: pd.DataFrame) -> None:
        pnl_path   = BACKTEST_DIR / "daily_pnl.csv"
        trade_path = BACKTEST_DIR / "trade_log.csv"
        daily_pnl.to_csv(pnl_path)
        trade_log.to_csv(trade_path, index=False)
        print(f"\n[engine] Saved daily_pnl  -> {pnl_path} "
              f"({daily_pnl.shape[0]} days x {daily_pnl.shape[1]} pairs)")
        print(f"[engine] Saved trade_log  -> {trade_path} "
              f"({len(trade_log)} trades)")


if __name__ == "__main__":
    import config as cfg

    log_prices     = pd.read_parquet(
        Path(__file__).parent.parent / "data" / "clean" / "log_prices.parquet"
    )
    selected_pairs = pd.read_csv(
        Path(__file__).parent.parent / "outputs" / "pairs" / "selected_pairs.csv"
    )

    bt_config = Config(
        train_window    = cfg.TRAIN_WINDOW,
        test_window     = cfg.TEST_WINDOW,
        zscore_lookback = cfg.ZSCORE_LOOKBACK,
        entry_zscore    = cfg.ENTRY_ZSCORE,
        exit_zscore     = cfg.EXIT_ZSCORE,
        stop_zscore     = cfg.STOP_ZSCORE,
        cost_bps        = cfg.COST_BPS,
        capital         = cfg.CAPITAL,
        position_size   = cfg.POSITION_SIZE,
    )

    backtester = WalkForwardBacktester(log_prices, selected_pairs, bt_config)
    daily_pnl, trade_log = backtester.run()

    print("\n--- Daily P&L tail ---")
    print(daily_pnl.tail())
    print("\n--- Trade log sample ---")
    print(trade_log.head(10).to_string(index=False))
