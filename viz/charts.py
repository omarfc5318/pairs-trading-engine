import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))

CHARTS_DIR   = Path(__file__).parent.parent / "outputs" / "charts"
TRADING_DAYS = 252


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out = CHARTS_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[charts] Saved -> {out}")


def _fmt_xaxis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)


# ── 1. Spread / Z-score / Position chart ──────────────────────────────────────

def plot_spread_with_signals(
    spread:    pd.Series,
    zscore:    pd.Series,
    signals:   pd.Series,
    pair_name: str,
) -> None:
    """
    Three-panel chart for a single pair:
      Panel 1 — Spread with entry/exit/stop horizontal lines
      Panel 2 — Z-score with coloured threshold bands
      Panel 3 — Position over time (+1 / 0 / -1)

    Saved to outputs/charts/{pair_name}_spread.png
    """
    safe_name = pair_name.replace("/", "_")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1]})
    fig.suptitle(f"Pairs Trading — {pair_name}", fontsize=14, fontweight="bold", y=0.98)

    # ── Panel 1: Spread ────────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(spread.index, spread, color="#1565C0", linewidth=0.9, label="Spread")
    ax1.set_ylabel("Spread\n(log A − β·log B)", fontsize=9)
    ax1.set_title("Spread", fontsize=10, pad=3)

    spread_mean = spread.mean()
    spread_std  = spread.std(ddof=1)
    for mult, style, label in [
        (0,    ("grey",    "-",  0.9), "Mean"),
        (2.0,  ("#E53935", "--", 0.8), "±2σ (entry)"),
        (-2.0, ("#E53935", "--", 0.8), None),
        (0.5,  ("#43A047", ":",  0.8), "±0.5σ (exit)"),
        (-0.5, ("#43A047", ":",  0.8), None),
    ]:
        color, ls, alpha = style
        ax1.axhline(spread_mean + mult * spread_std,
                    color=color, linestyle=ls, linewidth=0.9,
                    alpha=alpha, label=label)

    ax1.legend(fontsize=8, loc="upper left", ncol=3)
    ax1.grid(True, alpha=0.25)

    # ── Panel 2: Z-score ───────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(zscore.index, zscore, color="#37474F", linewidth=0.8, label="Z-score")
    ax2.set_ylabel("Z-score", fontsize=9)
    ax2.set_title("Z-score with Thresholds", fontsize=10, pad=3)

    # Shade entry/stop zones
    ax2.axhspan( 2.0,  3.5, alpha=0.08, color="#E53935", label="Entry zone")
    ax2.axhspan(-3.5, -2.0, alpha=0.08, color="#1565C0")
    ax2.axhspan( 3.5,  zscore.max() + 0.5 if zscore.max() > 3.5 else 4.5,
                alpha=0.12, color="#B71C1C", label="Stop zone")
    ax2.axhspan( zscore.min() - 0.5 if zscore.min() < -3.5 else -4.5, -3.5,
                alpha=0.12, color="#B71C1C")

    thresholds = [
        ( 2.0, "#E53935", "--", "Entry  ±2.0"),
        (-2.0, "#E53935", "--", None),
        ( 0.5, "#43A047", ":",  "Exit   ±0.5"),
        (-0.5, "#43A047", ":",  None),
        ( 3.5, "#B71C1C", "-.", "Stop   ±3.5"),
        (-3.5, "#B71C1C", "-.", None),
        ( 0.0, "grey",    "-",  "Zero"),
    ]
    for level, color, ls, label in thresholds:
        ax2.axhline(level, color=color, linestyle=ls, linewidth=0.85,
                    alpha=0.85, label=label)

    ax2.legend(fontsize=8, loc="upper left", ncol=4)
    ax2.grid(True, alpha=0.25)

    # ── Panel 3: Position ──────────────────────────────────────────────────────
    ax3 = axes[2]
    pos_colors = signals.map({1: "#1565C0", -1: "#E53935", 0: "#BDBDBD"})
    ax3.bar(signals.index, signals, color=pos_colors, width=1, alpha=0.85)
    ax3.axhline(0, color="grey", linewidth=0.6)
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(["Short", "Flat", "Long"], fontsize=8)
    ax3.set_ylabel("Position", fontsize=9)
    ax3.set_title("Position", fontsize=10, pad=3)
    ax3.set_ylim(-1.4, 1.4)

    long_patch  = mpatches.Patch(color="#1565C0", label="Long (+1)")
    short_patch = mpatches.Patch(color="#E53935", label="Short (−1)")
    flat_patch  = mpatches.Patch(color="#BDBDBD", label="Flat (0)")
    ax3.legend(handles=[long_patch, short_patch, flat_patch],
               fontsize=8, loc="upper left", ncol=3)
    ax3.grid(True, alpha=0.25)

    _fmt_xaxis(ax3)
    fig.align_ylabels(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, f"{safe_name}_spread.png")


# ── 2. Drawdown chart ──────────────────────────────────────────────────────────

def plot_drawdown(daily_pnl: pd.DataFrame) -> None:
    """
    Portfolio drawdown chart — both combined and per-pair.
    Saved to outputs/charts/drawdown.png
    """
    capital    = 100_000.0
    pnl_filled = daily_pnl.fillna(0)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("Portfolio Drawdown Analysis", fontsize=13, fontweight="bold")

    # ── Combined drawdown ──────────────────────────────────────────────────────
    ax1 = axes[0]
    combined  = capital + pnl_filled.sum(axis=1).cumsum()
    roll_max  = combined.cummax()
    drawdown  = (combined - roll_max) / roll_max * 100

    ax1.fill_between(drawdown.index, drawdown, 0,
                     where=(drawdown < 0), color="#E53935", alpha=0.4, label="Drawdown")
    ax1.plot(drawdown.index, drawdown, color="#B71C1C", linewidth=0.9)
    ax1.axhline(0, color="grey", linewidth=0.7)
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    ax1.annotate(f"Max DD: {max_dd:.1f}%",
                 xy=(max_dd_date, max_dd),
                 xytext=(20, -15), textcoords="offset points",
                 fontsize=9, color="#B71C1C",
                 arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=0.8))
    ax1.set_ylabel("Drawdown (%)", fontsize=10)
    ax1.set_title("Combined Portfolio Drawdown", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    # ── Per-pair drawdown ──────────────────────────────────────────────────────
    ax2 = axes[1]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    for (pair, series), color in zip(pnl_filled.items(), colors):
        eq   = capital + series.cumsum()
        rm   = eq.cummax()
        dd   = (eq - rm) / rm * 100
        ax2.plot(dd.index, dd, linewidth=0.8, label=pair, color=color, alpha=0.85)

    ax2.axhline(0, color="grey", linewidth=0.6)
    ax2.set_ylabel("Drawdown (%)", fontsize=9)
    ax2.set_title("Per-Pair Drawdown", fontsize=10)
    ax2.legend(fontsize=8, loc="lower left", ncol=3)
    ax2.grid(True, alpha=0.25)

    _fmt_xaxis(ax2)
    fig.align_ylabels(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "drawdown.png")


# ── 3. Rolling Sharpe ──────────────────────────────────────────────────────────

def plot_rolling_sharpe(daily_pnl: pd.DataFrame, window: int = 63) -> None:
    """
    Rolling annualised Sharpe ratio for each pair and the combined portfolio.
    Saved to outputs/charts/rolling_sharpe.png
    """
    capital    = 100_000.0
    pnl_filled = daily_pnl.fillna(0)

    def rolling_sharpe(pnl_series: pd.Series) -> pd.Series:
        eq      = capital + pnl_series.cumsum()
        eq_lag  = eq.shift(1).fillna(capital)
        returns = pnl_series / eq_lag
        roll    = returns.rolling(window, min_periods=window)
        return roll.mean() / roll.std(ddof=1) * np.sqrt(TRADING_DAYS)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(f"Rolling {window}-Day Sharpe Ratio", fontsize=13, fontweight="bold")

    # ── Combined ───────────────────────────────────────────────────────────────
    ax1 = axes[0]
    combined_rs = rolling_sharpe(pnl_filled.sum(axis=1))
    ax1.plot(combined_rs.index, combined_rs, color="#9C27B0", linewidth=1.4,
             label="Combined")
    ax1.axhline(0,   color="grey",    linewidth=0.8, linestyle="--")
    ax1.axhline(1.0, color="#43A047", linewidth=0.8, linestyle=":",
                label="Sharpe = 1 (target)")
    ax1.fill_between(combined_rs.index, combined_rs, 0,
                     where=(combined_rs >= 0), alpha=0.12, color="#9C27B0")
    ax1.fill_between(combined_rs.index, combined_rs, 0,
                     where=(combined_rs < 0),  alpha=0.12, color="#E53935")
    ax1.set_ylabel("Sharpe Ratio (ann.)", fontsize=10)
    ax1.set_title("Combined Portfolio", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    # ── Per-pair ───────────────────────────────────────────────────────────────
    ax2 = axes[1]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    for (pair, series), color in zip(pnl_filled.items(), colors):
        rs = rolling_sharpe(series)
        ax2.plot(rs.index, rs, linewidth=0.9, label=pair, color=color, alpha=0.85)

    ax2.axhline(0, color="grey", linewidth=0.7, linestyle="--")
    ax2.set_ylabel("Sharpe Ratio (ann.)", fontsize=9)
    ax2.set_title("Per-Pair Rolling Sharpe", fontsize=10)
    ax2.legend(fontsize=8, loc="upper left", ncol=3)
    ax2.grid(True, alpha=0.25)

    _fmt_xaxis(ax2)
    fig.align_ylabels(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "rolling_sharpe.png")


# ── 4. Cost sensitivity chart ─────────────────────────────────────────────────

def plot_cost_sensitivity(sensitivity_df: pd.DataFrame) -> None:
    """
    3-panel figure: Sharpe / CAGR / Max Drawdown vs transaction cost (bps).

    Adds a vertical dashed line at 5 bps (base assumption).
    Saves to outputs/charts/cost_sensitivity.png
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    costs  = sensitivity_df.index.tolist()
    base   = 5   # base cost assumption

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    fig.suptitle(
        "Transaction Cost Sensitivity — Kalman + Regime Strategy",
        fontsize=13, fontweight="bold", y=0.98,
    )

    panels = [
        ("sharpe",       "Sharpe Ratio",    "#1565C0", None),
        ("cagr",         "CAGR",            "#2E7D32", "{:.1%}"),
        ("max_drawdown", "Max Drawdown",    "#C62828", "{:.1%}"),
    ]

    for ax, (col, ylabel, color, fmt) in zip(axes, panels):
        values = sensitivity_df[col].tolist()

        ax.plot(costs, values, color=color, linewidth=2.2,
                marker="o", markersize=7, markerfacecolor="white",
                markeredgewidth=2, markeredgecolor=color)

        # Shade area under/above the line for visual weight
        ax.fill_between(costs, values, min(values) - abs(min(values)) * 0.1,
                        alpha=0.08, color=color)

        # Annotate each point
        for x, y in zip(costs, values):
            label = f"{y:.1%}" if fmt else f"{y:.3f}"
            ax.annotate(
                label, xy=(x, y),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=9, color=color, fontweight="bold",
            )

        # Base cost vertical line
        ax.axvline(base, color="grey", linewidth=1.2, linestyle="--", alpha=0.7,
                   label=f"Base: {base} bps")

        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(costs)
        ax.set_xticklabels([f"{c} bps" for c in costs], fontsize=9)

        # Tighten y-axis padding
        y_range = max(values) - min(values)
        pad     = max(y_range * 0.25, 1e-4)
        ax.set_ylim(min(values) - pad, max(values) + pad)

    axes[-1].set_xlabel("Transaction Cost (bps per leg)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "cost_sensitivity.png")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from analysis.ou_process import compute_spread
    from analysis.signals    import compute_rolling_zscore, generate_signals
    import config as cfg

    log_prices     = pd.read_parquet(
        Path(__file__).parent.parent / "data" / "clean" / "log_prices.parquet"
    )
    selected_pairs = pd.read_csv(
        Path(__file__).parent.parent / "outputs" / "pairs" / "selected_pairs.csv"
    )
    daily_pnl = pd.read_csv(
        Path(__file__).parent.parent / "outputs" / "backtest" / "daily_pnl.csv",
        index_col=0, parse_dates=True,
    )

    # Spread / signal charts for each pair
    for _, row in selected_pairs.iterrows():
        pair_name = f"{row['ticker_a']}/{row['ticker_b']}"
        spread    = compute_spread(log_prices, row["ticker_a"],
                                   row["ticker_b"], row["hedge_ratio"])
        zscore    = compute_rolling_zscore(spread, cfg.ZSCORE_LOOKBACK)
        signals   = generate_signals(zscore, cfg.ENTRY_ZSCORE,
                                     cfg.EXIT_ZSCORE, cfg.STOP_ZSCORE)
        plot_spread_with_signals(spread, zscore, signals, pair_name)

    plot_drawdown(daily_pnl)
    plot_rolling_sharpe(daily_pnl, window=63)
