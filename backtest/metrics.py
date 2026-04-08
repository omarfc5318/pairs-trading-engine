import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

sys.path.insert(0, str(Path(__file__).parent.parent))

CHARTS_DIR  = Path(__file__).parent.parent / "outputs" / "charts"
TRADING_DAYS = 252


def _fetch_benchmark(ticker: str, start: str, end: str) -> pd.Series:
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    closes = raw["Close"].squeeze()
    return closes.pct_change().dropna().rename(ticker)


def _portfolio_returns(daily_pnl: pd.DataFrame) -> pd.Series:
    """Sum across pairs then convert $ P&L to a return series (base = initial capital)."""
    total_pnl = daily_pnl.fillna(0).sum(axis=1)
    # Running equity so each day's return is pnl / equity at start of that day
    capital   = 100_000.0
    equity    = capital + total_pnl.cumsum()
    equity_lagged = equity.shift(1).fillna(capital)
    returns   = total_pnl / equity_lagged
    return returns


def _sharpe(returns: pd.Series) -> float:
    if returns.std(ddof=1) == 0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def _sortino(returns: pd.Series) -> float:
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std(ddof=1) == 0:
        return 0.0
    return float(returns.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS))


def _cagr(returns: pd.Series) -> float:
    n_years = len(returns) / TRADING_DAYS
    total   = (1 + returns).prod()
    return float(total ** (1 / n_years) - 1)


def _max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    return float(dd.min())


def _win_rate(returns: pd.Series) -> float:
    active = returns[returns != 0]
    if len(active) == 0:
        return 0.0
    return float((active > 0).sum() / len(active))


def _alpha_beta(port_returns: pd.Series, bench_returns: pd.Series) -> tuple[float, float]:
    aligned = pd.concat([port_returns, bench_returns], axis=1).dropna()
    if len(aligned) < 10:
        return 0.0, 0.0
    y = aligned.iloc[:, 0].values
    X = add_constant(aligned.iloc[:, 1].values)
    res = OLS(y, X).fit()
    alpha_daily = float(res.params[0])
    beta        = float(res.params[1])
    alpha_ann   = alpha_daily * TRADING_DAYS
    return round(alpha_ann, 6), round(beta, 6)


def compute_metrics(
    daily_pnl: pd.DataFrame,
    benchmark_ticker: str = "SPY",
) -> dict:
    """
    Compute performance metrics for the portfolio.

    Parameters
    ----------
    daily_pnl        : DataFrame of daily $ P&L (dates x pairs)
    benchmark_ticker : ticker to use as benchmark (default SPY)

    Returns
    -------
    dict with all metrics; also prints a formatted summary table.
    """
    port_returns = _portfolio_returns(daily_pnl)
    start_date   = str(port_returns.index[0].date())
    end_date     = str(port_returns.index[-1].date())

    bench_returns = _fetch_benchmark(benchmark_ticker, start_date, end_date)
    bench_returns = bench_returns.reindex(port_returns.index).fillna(0)

    sharpe      = _sharpe(port_returns)
    sortino     = _sortino(port_returns)
    cagr        = _cagr(port_returns)
    max_dd      = _max_drawdown(port_returns)
    win_rate    = _win_rate(port_returns)
    total_ret   = float((1 + port_returns).prod() - 1)
    alpha, beta = _alpha_beta(port_returns, bench_returns)

    # Benchmark metrics for comparison
    bench_sharpe = _sharpe(bench_returns)
    bench_cagr   = _cagr(bench_returns)
    bench_dd     = _max_drawdown(bench_returns)

    metrics = {
        "period":           f"{start_date} → {end_date}",
        "trading_days":     len(port_returns),
        "total_return":     round(total_ret, 4),
        "cagr":             round(cagr, 4),
        "sharpe_ratio":     round(sharpe, 4),
        "sortino_ratio":    round(sortino, 4),
        "max_drawdown":     round(max_dd, 4),
        "win_rate":         round(win_rate, 4),
        "alpha_vs_spy":     round(alpha, 4),
        "beta_vs_spy":      round(beta, 4),
        "spy_sharpe":       round(bench_sharpe, 4),
        "spy_cagr":         round(bench_cagr, 4),
        "spy_max_drawdown": round(bench_dd, 4),
    }

    _print_table(metrics)
    return metrics


def _print_table(m: dict) -> None:
    sep = "─" * 46
    print(f"\n{'═' * 46}")
    print(f"  PAIRS TRADING — PERFORMANCE SUMMARY")
    print(f"{'═' * 46}")
    print(f"  Period          {m['period']}")
    print(f"  Trading Days    {m['trading_days']}")
    print(sep)
    print(f"  {'Metric':<22} {'Strategy':>8}  {'SPY':>8}")
    print(sep)
    print(f"  {'Total Return':<22} {m['total_return']:>7.1%}  {' ':>8}")
    print(f"  {'CAGR':<22} {m['cagr']:>7.1%}  {m['spy_cagr']:>7.1%}")
    print(f"  {'Sharpe Ratio':<22} {m['sharpe_ratio']:>8.2f}  {m['spy_sharpe']:>8.2f}")
    print(f"  {'Sortino Ratio':<22} {m['sortino_ratio']:>8.2f}  {' ':>8}")
    print(f"  {'Max Drawdown':<22} {m['max_drawdown']:>7.1%}  {m['spy_max_drawdown']:>7.1%}")
    print(f"  {'Win Rate':<22} {m['win_rate']:>7.1%}  {' ':>8}")
    print(sep)
    print(f"  {'Alpha (ann.) vs SPY':<22} {m['alpha_vs_spy']:>8.4f}")
    print(f"  {'Beta vs SPY':<22} {m['beta_vs_spy']:>8.4f}")
    print(f"{'═' * 46}\n")


def plot_equity_curve(daily_pnl: pd.DataFrame) -> None:
    """
    Plot cumulative equity curve for each pair and the combined portfolio.
    Saves to outputs/charts/equity_curve.png.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    capital   = 100_000.0
    pnl_filled = daily_pnl.fillna(0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    # --- Top panel: cumulative equity ---
    ax1 = axes[0]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for (pair, series), color in zip(pnl_filled.items(), colors):
        equity = capital + series.cumsum()
        ax1.plot(equity.index, equity, linewidth=1.2, label=pair, color=color, alpha=0.8)

    # Combined portfolio
    combined_equity = capital + pnl_filled.sum(axis=1).cumsum()
    ax1.plot(combined_equity.index, combined_equity,
             linewidth=2.0, label="Combined", color="#9C27B0", linestyle="--")

    ax1.axhline(capital, color="grey", linewidth=0.8, linestyle=":")
    ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax1.set_title("Walk-Forward Pairs Trading — Equity Curve", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.grid(True, alpha=0.3)

    # --- Bottom panel: combined daily P&L bar chart ---
    ax2 = axes[1]
    daily_combined = pnl_filled.sum(axis=1)
    colors_bar = ["#4CAF50" if v >= 0 else "#F44336" for v in daily_combined]
    ax2.bar(daily_combined.index, daily_combined, color=colors_bar, width=1, alpha=0.7)
    ax2.axhline(0, color="grey", linewidth=0.8)
    ax2.set_ylabel("Daily P&L ($)", fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()

    plt.tight_layout()
    out_path = CHARTS_DIR / "equity_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[metrics] Saved equity curve -> {out_path}")


def compare_strategies(
    static_pnl:  pd.DataFrame,
    kalman_pnl:  pd.DataFrame,
    regime_pnl:  pd.DataFrame | None = None,
    benchmark_ticker: str = "SPY",
) -> dict:
    """
    Print a side-by-side performance table.

    Columns: Static OLS | Kalman | Kalman+Regime (if regime_pnl provided)

    Returns a dict with metrics for all strategies.
    """
    # Align all strategies to a shared date range
    shared_idx = static_pnl.index.intersection(kalman_pnl.index)
    if regime_pnl is not None:
        shared_idx = shared_idx.intersection(regime_pnl.index)

    static_pnl = static_pnl.loc[shared_idx]
    kalman_pnl = kalman_pnl.loc[shared_idx]
    if regime_pnl is not None:
        regime_pnl = regime_pnl.loc[shared_idx]

    start_date = str(shared_idx[0].date())
    end_date   = str(shared_idx[-1].date())

    bench_ret  = _fetch_benchmark(benchmark_ticker, start_date, end_date)
    static_ret = _portfolio_returns(static_pnl)
    bench_ret  = bench_ret.reindex(static_ret.index).fillna(0)

    def _metrics(pnl: pd.DataFrame) -> dict:
        ret         = _portfolio_returns(pnl)
        alpha, beta = _alpha_beta(ret, bench_ret)
        return {
            "sharpe":   round(_sharpe(ret),        3),
            "sortino":  round(_sortino(ret),       3),
            "cagr":     round(_cagr(ret),          4),
            "max_dd":   round(_max_drawdown(ret),  4),
            "win_rate": round(_win_rate(ret),      4),
            "alpha":    round(alpha,               4),
            "beta":     round(beta,                4),
        }

    s = _metrics(static_pnl)
    k = _metrics(kalman_pnl)
    r = _metrics(regime_pnl) if regime_pnl is not None else None

    has_regime = r is not None
    width      = 72 if has_regime else 58
    sep        = "─" * width

    print(f"\n{'═' * width}")
    title = "STRATEGY COMPARISON — Static OLS | Kalman | Kalman+Regime" \
            if has_regime else "STRATEGY COMPARISON — Static OLS vs Kalman Filter"
    print(f"  {title}")
    print(f"{'═' * width}")
    print(f"  Period: {start_date} → {end_date}")
    print(sep)

    if has_regime:
        print(f"  {'Metric':<22} {'Static OLS':>12}  {'Kalman':>12}  {'Kalman+Regime':>14}")
        print(sep)
        def row(label, key, fmt):
            return (f"  {label:<22} {fmt.format(s[key]):>12}  "
                    f"{fmt.format(k[key]):>12}  {fmt.format(r[key]):>14}")
    else:
        print(f"  {'Metric':<22} {'Static OLS':>12}  {'Kalman':>12}")
        print(sep)
        def row(label, key, fmt):
            return f"  {label:<22} {fmt.format(s[key]):>12}  {fmt.format(k[key]):>12}"

    print(row("Sharpe Ratio",        "sharpe",   "{:.3f}"))
    print(row("Sortino Ratio",       "sortino",  "{:.3f}"))
    print(row("CAGR",                "cagr",     "{:.1%}"))
    print(row("Max Drawdown",        "max_dd",   "{:.1%}"))
    print(row("Win Rate",            "win_rate", "{:.1%}"))
    print(row("Alpha vs SPY (ann.)", "alpha",    "{:.4f}"))
    print(row("Beta vs SPY",         "beta",     "{:.4f}"))
    print(f"{'═' * width}\n")

    result = {"static": s, "kalman": k}
    if has_regime:
        result["kalman_regime"] = r
    return result


def transaction_cost_sensitivity(
    log_prices:     pd.DataFrame,
    selected_pairs: pd.DataFrame,
    cost_bps_range: list[int] = [5, 10, 15, 20],
) -> pd.DataFrame:
    """
    Re-run run_kalman_with_regime() at each cost level and collect metrics.

    Parameters
    ----------
    log_prices      : clean log price DataFrame
    selected_pairs  : selected_pairs.csv as DataFrame
    cost_bps_range  : list of transaction cost levels in basis points

    Returns
    -------
    DataFrame indexed by cost_bps with columns: sharpe, cagr, max_drawdown
    """
    import config as cfg
    from backtest.engine import WalkForwardBacktester, Config

    records = []
    print(f"\n[sensitivity] Running {len(cost_bps_range)} cost levels: {cost_bps_range} bps")

    for cost_bps in cost_bps_range:
        print(f"\n[sensitivity] cost_bps = {cost_bps} ...")

        bt_config = Config(
            train_window    = cfg.TRAIN_WINDOW,
            test_window     = cfg.TEST_WINDOW,
            zscore_lookback = cfg.ZSCORE_LOOKBACK,
            entry_zscore    = cfg.ENTRY_ZSCORE,
            exit_zscore     = cfg.EXIT_ZSCORE,
            stop_zscore     = cfg.STOP_ZSCORE,
            cost_bps        = cost_bps,          # override here
            capital         = cfg.CAPITAL,
            position_size   = cfg.POSITION_SIZE,
        )

        bt        = WalkForwardBacktester(log_prices, selected_pairs, bt_config)
        pnl, _    = bt.run_kalman_with_regime(save=False)  # don't overwrite canonical files
        ret       = _portfolio_returns(pnl)

        records.append({
            "cost_bps":    cost_bps,
            "sharpe":      round(_sharpe(ret),       4),
            "cagr":        round(_cagr(ret),         4),
            "max_drawdown": round(_max_drawdown(ret), 4),
        })
        print(f"  → Sharpe={records[-1]['sharpe']:.3f}  "
              f"CAGR={records[-1]['cagr']:.1%}  "
              f"MaxDD={records[-1]['max_drawdown']:.1%}")

    sensitivity_df = pd.DataFrame(records).set_index("cost_bps")

    print(f"\n[sensitivity] Results:\n{sensitivity_df.to_string()}")
    return sensitivity_df


if __name__ == "__main__":
    daily_pnl = pd.read_csv(
        Path(__file__).parent.parent / "outputs" / "backtest" / "daily_pnl.csv",
        index_col=0, parse_dates=True,
    )
    metrics = compute_metrics(daily_pnl)
    plot_equity_curve(daily_pnl)
