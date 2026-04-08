import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from data.fetch                 import fetch_prices
from data.clean                 import load_and_align_prices, compute_log_prices
from analysis.cointegration     import run_cointegration_tests, filter_pairs
from analysis.ou_process        import compute_spread, select_pairs
from analysis.signals           import compute_rolling_zscore, generate_signals
from backtest.engine            import WalkForwardBacktester, Config
from backtest.metrics           import compute_metrics, plot_equity_curve
from viz.charts                 import (
    plot_spread_with_signals,
    plot_drawdown,
    plot_rolling_sharpe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pairs Trading Strategy — Walk-Forward Backtest"
    )
    parser.add_argument(
        "--pairs-only",
        action="store_true",
        help="Run only steps 1–3 (data + cointegration + signal viz) for debugging.",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip data fetch and use existing raw Parquet files.",
    )
    parser.add_argument(
        "--financials",
        action="store_true",
        help="Use S&P 500 Financials sector stocks instead of the ETF universe.",
    )
    return parser.parse_args()


def step_banner(n: int, total: int, msg: str) -> None:
    print(f"\n{'─' * 52}")
    print(f"  Step {n}/{total} — {msg}")
    print(f"{'─' * 52}")


def main() -> None:
    args  = parse_args()
    total = 3 if args.pairs_only else 7

    # ── Apply --financials flag before anything else ───────────────────────────
    if args.financials:
        cfg.USE_FINANCIALS = True

    universe = cfg.get_active_universe()

    print(f"\n{'═' * 52}")
    print(f"  PAIRS TRADING STRATEGY — FULL PIPELINE")
    if args.pairs_only:
        print(f"  Mode: --pairs-only (steps 1–3)")
    if args.financials:
        print(f"  Universe: S&P 500 Financials ({len(universe)} tickers)")
    else:
        print(f"  Universe: Sector ETFs ({len(universe)} tickers)")
    print(f"{'═' * 52}")

    # ── Step 1: Fetch data ─────────────────────────────────────────────────────
    step_banner(1, total, "Fetch & clean price data")
    if not args.no_fetch:
        fetch_prices(universe, cfg.START_DATE, cfg.END_DATE)
    else:
        print("[main] --no-fetch: skipping download, using cached raw files.")

    load_and_align_prices()
    compute_log_prices()
    log_prices = pd.read_parquet("data/clean/log_prices.parquet")
    print(f"[main] Log prices: {log_prices.shape[0]} days × {log_prices.shape[1]} tickers "
          f"({log_prices.index[0].date()} → {log_prices.index[-1].date()})")

    # ── Step 2: Cointegration + pair selection ─────────────────────────────────
    step_banner(2, total, "Cointegration tests & pair selection")
    coint_df       = run_cointegration_tests(log_prices)
    filtered       = filter_pairs(coint_df, cfg.COINT_PVALUE_THRESHOLD)
    selected_pairs = select_pairs(filtered, log_prices,
                                  cfg.HALFLIFE_MIN, cfg.HALFLIFE_MAX)

    if selected_pairs.empty:
        print("\n[!] No pairs passed all filters — nothing to backtest.")
        print("    Try relaxing HALFLIFE_MAX or COINT_PVALUE_THRESHOLD in config.py.")
        sys.exit(0)

    print(f"\n[main] Selected {len(selected_pairs)} pair(s) for backtesting:")
    for _, row in selected_pairs.iterrows():
        print(f"       {row['ticker_a']}/{row['ticker_b']}  "
              f"p={row['p_value']:.4f}  "
              f"hedge={row['hedge_ratio']:.4f}  "
              f"half-life={row['halflife']:.1f}d")

    # ── Step 3: Signal visualisation ──────────────────────────────────────────
    step_banner(3, total, "Generate signal charts for each pair")
    for _, row in selected_pairs.iterrows():
        pair_name = f"{row['ticker_a']}/{row['ticker_b']}"
        spread    = compute_spread(
            log_prices, row["ticker_a"], row["ticker_b"], row["hedge_ratio"]
        )
        zscore    = compute_rolling_zscore(spread, cfg.ZSCORE_LOOKBACK)
        signals   = generate_signals(
            zscore, cfg.ENTRY_ZSCORE, cfg.EXIT_ZSCORE, cfg.STOP_ZSCORE
        )
        plot_spread_with_signals(spread, zscore, signals, pair_name)
        print(f"[main] {pair_name}: {(signals==1).sum()} long / "
              f"{(signals==-1).sum()} short / "
              f"{(signals==0).sum()} flat bars")

    if args.pairs_only:
        print(f"\n{'═' * 52}")
        print("  --pairs-only complete. Steps 4–7 skipped.")
        print(f"{'═' * 52}")
        return

    # ── Step 4: Walk-forward backtest ─────────────────────────────────────────
    step_banner(4, total, "Walk-forward backtest")
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
    backtester            = WalkForwardBacktester(log_prices, selected_pairs, bt_config)
    daily_pnl, trade_log  = backtester.run()

    # ── Step 5: Performance metrics ───────────────────────────────────────────
    step_banner(5, total, "Compute performance metrics")
    metrics = compute_metrics(daily_pnl)

    # ── Step 6: Generate all charts ───────────────────────────────────────────
    step_banner(6, total, "Generate all charts")
    plot_equity_curve(daily_pnl)
    plot_drawdown(daily_pnl)
    plot_rolling_sharpe(daily_pnl, window=cfg.TEST_WINDOW)
    print("[main] All charts saved to outputs/charts/")

    # ── Step 7: Final summary table ───────────────────────────────────────────
    step_banner(7, total, "Final summary")

    print(f"\n{'═' * 60}")
    print(f"  FINAL SUMMARY")
    print(f"{'═' * 60}")
    print(f"  {'Pairs Used':<28} {len(selected_pairs)}")
    for _, row in selected_pairs.iterrows():
        print(f"    · {row['ticker_a']}/{row['ticker_b']}"
              f"  (half-life {row['halflife']:.1f}d, "
              f"hedge {row['hedge_ratio']:.3f})")

    print(f"\n  {'Backtest Period':<28} {metrics['period']}")
    print(f"  {'Trading Days':<28} {metrics['trading_days']}")
    print(f"  {'Total Trades':<28} {len(trade_log)}")

    print(f"\n  {'Metric':<28} {'Value':>10}")
    print(f"  {'─'*40}")
    print(f"  {'Total Return':<28} {metrics['total_return']:>9.1%}")
    print(f"  {'CAGR':<28} {metrics['cagr']:>9.1%}")
    print(f"  {'Sharpe Ratio':<28} {metrics['sharpe_ratio']:>10.2f}")
    print(f"  {'Sortino Ratio':<28} {metrics['sortino_ratio']:>10.2f}")
    print(f"  {'Max Drawdown':<28} {metrics['max_drawdown']:>9.1%}")
    print(f"  {'Win Rate':<28} {metrics['win_rate']:>9.1%}")
    print(f"  {'Alpha vs SPY (ann.)':<28} {metrics['alpha_vs_spy']:>10.4f}")
    print(f"  {'Beta vs SPY':<28} {metrics['beta_vs_spy']:>10.4f}")

    print(f"\n  {'Output Files':<28}")
    outputs = [
        "outputs/pairs/coint_results.csv",
        "outputs/pairs/selected_pairs.csv",
        "outputs/backtest/daily_pnl.csv",
        "outputs/backtest/trade_log.csv",
        "outputs/charts/equity_curve.png",
        "outputs/charts/drawdown.png",
        "outputs/charts/rolling_sharpe.png",
    ] + [
        f"outputs/charts/{row['ticker_a']}_{row['ticker_b']}_spread.png"
        for _, row in selected_pairs.iterrows()
    ]
    for f in outputs:
        exists = "✓" if Path(f).exists() else "✗ MISSING"
        print(f"    {exists}  {f}")

    print(f"\n{'═' * 60}")
    print("  Pipeline complete.")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
