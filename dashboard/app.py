"""
Pairs Trading Strategy Dashboard
==================================
Run with:  streamlit run dashboard/app.py
All data loaded from outputs/ — no backtest recomputation on load.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Path setup ─────────────────────────────────────────────────────────────────
# Path(__file__) is always absolute, so ROOT resolves correctly regardless of
# the working directory — locally (pairs_trading/) or on Streamlit Cloud
# (/mount/src/<repo>/dashboard/app.py → ROOT = /mount/src/<repo>).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUTS   = ROOT / "outputs"
BACKTEST  = OUTPUTS / "backtest"
CHARTS    = OUTPUTS / "charts"
DATA      = ROOT / "data" / "clean"

# ── Sanity-check required data files at startup ────────────────────────────────
_REQUIRED = [
    BACKTEST / "daily_pnl.csv",
    BACKTEST / "kalman_daily_pnl.csv",
    BACKTEST / "regime_daily_pnl.csv",
    BACKTEST / "trade_log.csv",
    BACKTEST / "kalman_trade_log.csv",
    BACKTEST / "regime_trade_log.csv",
    BACKTEST / "factor_results.json",
    DATA / "log_prices.parquet",
    ROOT / "outputs" / "pairs" / "selected_pairs.csv",
]
_missing = [str(p.relative_to(ROOT)) for p in _REQUIRED if not p.exists()]
if _missing:
    st.error(
        "**Missing output files** — run `python main.py --no-fetch` locally first "
        "and commit the `outputs/` and `data/clean/` directories to your repository.\n\n"
        + "\n".join(f"- `{p}`" for p in _missing)
    )
    st.stop()

DARK = "plotly_dark"
MARGIN = dict(l=40, r=20, t=50, b=40)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pairs Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS tweaks ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stMetric > div { background: #161B22; border-radius: 8px; padding: 10px 16px; }
    .block-container { padding-top: 1.5rem; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (all cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_daily_pnl() -> dict[str, pd.DataFrame]:
    return {
        "Static OLS":   pd.read_csv(BACKTEST / "daily_pnl.csv",         index_col=0, parse_dates=True),
        "Kalman":        pd.read_csv(BACKTEST / "kalman_daily_pnl.csv",  index_col=0, parse_dates=True),
        "Kalman+Regime": pd.read_csv(BACKTEST / "regime_daily_pnl.csv",  index_col=0, parse_dates=True),
    }

@st.cache_data
def load_trade_logs() -> dict[str, pd.DataFrame]:
    return {
        "Static OLS":   pd.read_csv(BACKTEST / "trade_log.csv",         parse_dates=["entry_date", "exit_date"]),
        "Kalman":        pd.read_csv(BACKTEST / "kalman_trade_log.csv",  parse_dates=["entry_date", "exit_date"]),
        "Kalman+Regime": pd.read_csv(BACKTEST / "regime_trade_log.csv",  parse_dates=["entry_date", "exit_date"]),
    }

@st.cache_data
def load_log_prices() -> pd.DataFrame:
    return pd.read_parquet(DATA / "log_prices.parquet")

@st.cache_data
def load_selected_pairs() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / "pairs" / "selected_pairs.csv")

@st.cache_data
def load_factor_results() -> dict:
    with open(BACKTEST / "factor_results.json") as f:
        return json.load(f)

@st.cache_data
def compute_metrics(daily_pnl: dict[str, pd.DataFrame], trade_logs: dict[str, pd.DataFrame], capital: float = 100_000) -> pd.DataFrame:
    rows = []
    for name in ["Static OLS", "Kalman", "Kalman+Regime"]:
        pnl    = daily_pnl[name].sum(axis=1)
        trades = trade_logs[name]
        cum    = pnl.cumsum()
        ret    = pnl / capital
        ann_r  = ret.mean() * 252
        ann_v  = ret.std()  * np.sqrt(252)
        sharpe = ann_r / ann_v if ann_v > 0 else 0
        dd     = cum - cum.cummax()
        win_r  = (trades["net_pnl"] > 0).mean()
        rows.append({
            "Strategy":    name,
            "Total PnL":   f"${cum.iloc[-1]:,.0f}",
            "Ann. Return": f"{ann_r:.1%}",
            "Sharpe":      f"{sharpe:.2f}",
            "Max DD":      f"${dd.min():,.0f}",
            "# Trades":    len(trades),
            "Win Rate":    f"{win_r:.1%}",
        })
    return pd.DataFrame(rows).set_index("Strategy")

@st.cache_data
def compute_equity_curves(daily_pnl: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    return {name: df.sum(axis=1).cumsum() for name, df in daily_pnl.items()}

@st.cache_data
def compute_pair_data(pair_label: str) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Returns (spread, zscore, kalman_hedge_df) for the selected pair."""
    log_prices = load_log_prices()
    pairs_df   = load_selected_pairs()

    a, b = pair_label.split("/")
    row  = pairs_df[(pairs_df["ticker_a"] == a) & (pairs_df["ticker_b"] == b)].iloc[0]
    beta = row["hedge_ratio"]

    spread = log_prices[a] - beta * log_prices[b]
    spread.name = "spread"

    # Rolling z-score (60-day lookback, matching config)
    roll   = spread.rolling(60, min_periods=60)
    zscore = (spread - roll.mean()) / roll.std(ddof=1)
    zscore.name = "zscore"

    # Kalman hedge ratio drift — reconstruct per-fold time series from trade log
    kalman_trades = load_trade_logs()["Kalman"]
    pair_trades   = kalman_trades[
        (kalman_trades["ticker_a"] == a) & (kalman_trades["ticker_b"] == b)
    ].copy()

    if len(pair_trades) > 0:
        hedge_df = (
            pair_trades.groupby("fold")
            .agg(hedge_ratio=("hedge_ratio", "first"), entry_date=("entry_date", "min"))
            .reset_index()
            .sort_values("entry_date")
        )
    else:
        hedge_df = pd.DataFrame(columns=["entry_date", "hedge_ratio"])

    return spread, zscore, hedge_df

@st.cache_data
def compute_cost_sensitivity(daily_pnl: dict[str, pd.DataFrame], trade_logs: dict[str, pd.DataFrame], capital: float = 100_000) -> pd.DataFrame:
    bps_list  = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
    base_bps  = 5
    rows: list[dict] = []

    for name in ["Static OLS", "Kalman", "Kalman+Regime"]:
        pnl    = daily_pnl[name].sum(axis=1)
        trades = trade_logs[name]
        total_cost_base = trades["cost"].sum()
        n_days = len(pnl)

        for bps in bps_list:
            scale         = bps / base_bps
            total_cost_new = total_cost_base * scale
            cost_adj_daily = (total_cost_new - total_cost_base) / n_days
            adj_ret        = (pnl - cost_adj_daily) / capital
            ann_r          = adj_ret.mean() * 252
            ann_v          = adj_ret.std()  * np.sqrt(252)
            sharpe         = ann_r / ann_v if ann_v > 0 else 0
            rows.append({"Strategy": name, "bps": bps, "Sharpe": sharpe})

    return pd.DataFrame(rows)

@st.cache_data
def compute_rolling_sharpe(daily_pnl: dict[str, pd.DataFrame], capital: float = 100_000, window: int = 63) -> pd.Series:
    pnl = daily_pnl["Kalman+Regime"].sum(axis=1) / capital
    rs  = pnl.rolling(window).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() > 1e-10 else 0
    )
    rs.name = "rolling_sharpe"
    return rs

@st.cache_data
def compute_max_drawdowns(daily_pnl: dict[str, pd.DataFrame]) -> dict[str, float]:
    result = {}
    for name, df in daily_pnl.items():
        cum = df.sum(axis=1).cumsum()
        dd  = (cum - cum.cummax()).min()
        result[name] = dd
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## Pairs Trading")
    st.markdown("*XL-sector ETFs · 2019–2024*")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Strategy Overview", "Pair Analysis", "Risk Analysis", "Factor Decomposition"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        """
        **Universe:** XLP, XLU, XLB, XLV
        **Pairs:** 3 cointegrated
        **Method:** Walk-forward (252/63)
        **Costs:** 5 bps round-trip
        """,
        unsafe_allow_html=False,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — STRATEGY OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def page_overview():
    st.title("Strategy Overview")

    st.markdown(
        """
        This strategy implements a **statistical arbitrage pairs trading system** across S&P 500 SPDR sector ETFs,
        exploiting mean-reversion in three cointegrated pairs (XLP/XLU, XLP/XLV, XLB/XLV) identified via
        Engle-Granger cointegration testing over a 2015–2019 formation window. A **walk-forward backtest**
        (252-day train / 63-day test) evaluates three progressively sophisticated models: a static OLS
        hedge ratio baseline, a Kalman Filter that tracks the time-varying hedge ratio, and a Kalman+HMM
        Regime Filter that suppresses trading during trending (high-volatility) regimes identified by a
        2-state Gaussian Hidden Markov Model. All strategies are evaluated net of 5 bps round-trip transaction
        costs on a $100,000 capital base with 10% per-pair position sizing.
        """,
    )

    st.divider()

    # ── Comparison table ───────────────────────────────────────────────────────
    daily_pnl  = load_daily_pnl()
    trade_logs = load_trade_logs()
    metrics_df = compute_metrics(daily_pnl, trade_logs)

    st.subheader("Performance Comparison")
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    colors = {"Static OLS": "#4C9BE8", "Kalman": "#F0A500", "Kalman+Regime": "#3ECF8E"}

    for (strat, row), col in zip(metrics_df.iterrows(), cols):
        with col:
            color = colors[strat]
            st.markdown(
                f"<div style='border-top: 3px solid {color}; padding-top: 6px;'>"
                f"<b style='color:{color}; font-size:1.05em'>{strat}</b></div>",
                unsafe_allow_html=True,
            )
            for metric, val in row.items():
                st.metric(metric, val)

    st.divider()

    # ── Equity curves ──────────────────────────────────────────────────────────
    st.subheader("Equity Curves")
    equity = compute_equity_curves(daily_pnl)

    fig = go.Figure()
    line_styles = {"Static OLS": "dot", "Kalman": "dash", "Kalman+Regime": "solid"}
    line_colors = {"Static OLS": "#4C9BE8", "Kalman": "#F0A500", "Kalman+Regime": "#3ECF8E"}

    for name, curve in equity.items():
        fig.add_trace(go.Scatter(
            x=curve.index, y=curve.values,
            name=name,
            mode="lines",
            line=dict(color=line_colors[name], dash=line_styles[name], width=2),
        ))

    fig.update_layout(
        template=DARK, margin=MARGIN, height=380,
        xaxis_title="Date", yaxis_title="Cumulative PnL ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Regime filter donut ────────────────────────────────────────────────────
    st.subheader("Regime Filter Allocation")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(
            """
            The HMM regime filter identifies two market states from SPY daily returns:

            - **Mean-Reverting (88.3%)** — low-volatility environment; pairs trading signals are executed.
            - **Trending (11.7%)** — elevated volatility or trending regime; all new entries are suppressed to reduce drawdown.
            """,
        )
    with c2:
        donut = go.Figure(go.Pie(
            labels=["Mean-Reverting", "Trending"],
            values=[88.3, 11.7],
            hole=0.6,
            marker_colors=["#3ECF8E", "#E05252"],
            textinfo="label+percent",
            textfont_size=13,
        ))
        donut.update_layout(
            template=DARK, margin=dict(l=10, r=10, t=30, b=10), height=280,
            showlegend=False,
            annotations=[dict(text="Regime\nSplit", x=0.5, y=0.5, font_size=13, showarrow=False, font_color="#E6EDF3")],
        )
        st.plotly_chart(donut, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PAIR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def page_pair_analysis():
    st.title("Pair Analysis")

    pair_label = st.selectbox(
        "Select pair",
        ["XLB/XLV", "XLP/XLV", "XLP/XLU"],
        index=0,
    )

    spread, zscore, hedge_df = compute_pair_data(pair_label)
    a, b = pair_label.split("/")

    # Kalman trade log for entry/exit markers on the selected pair
    kalman_trades = load_trade_logs()["Kalman"]
    pair_trades   = kalman_trades[
        (kalman_trades["ticker_a"] == a) & (kalman_trades["ticker_b"] == b)
    ].copy()

    # ── 3-panel subplot ────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f"Spread — {pair_label}  (log {a} − β·log {b})",
            "Z-Score with Threshold Lines",
            "Kalman Hedge Ratio Drift",
        ],
        vertical_spacing=0.08,
        row_heights=[0.38, 0.38, 0.24],
    )

    # Panel 1: Spread + entry/exit markers
    fig.add_trace(go.Scatter(
        x=spread.index, y=spread.values,
        name="Spread", mode="lines",
        line=dict(color="#4C9BE8", width=1.2),
        showlegend=True,
    ), row=1, col=1)

    # Spread mean/std bands
    s_mean = spread.mean()
    s_std  = spread.std(ddof=1)
    for mult, color, dash, label in [
        (0,    "grey",    "dot",  "Mean"),
        (2.0,  "#E05252", "dash", "±2σ entry"),
        (-2.0, "#E05252", "dash", None),
        (0.5,  "#3ECF8E", "dot",  "±0.5σ exit"),
        (-0.5, "#3ECF8E", "dot",  None),
    ]:
        fig.add_hline(
            y=s_mean + mult * s_std,
            line_dash=dash, line_color=color, line_width=1,
            annotation_text=label if label else "",
            annotation_position="right",
            annotation_font_size=9,
            row=1, col=1,
        )

    # Entry markers (long = green triangle-up, short = red triangle-down)
    if len(pair_trades) > 0:
        long_entries  = pair_trades[pair_trades["direction"] == "long"]
        short_entries = pair_trades[pair_trades["direction"] == "short"]

        for entries, symbol, color, name in [
            (long_entries,  "triangle-up",   "#3ECF8E", "Long entry"),
            (short_entries, "triangle-down",  "#E05252", "Short entry"),
        ]:
            if len(entries) > 0:
                entry_spreads = spread.reindex(entries["entry_date"]).values
                fig.add_trace(go.Scatter(
                    x=entries["entry_date"], y=entry_spreads,
                    name=name, mode="markers",
                    marker=dict(symbol=symbol, size=9, color=color, line=dict(width=1, color="white")),
                ), row=1, col=1)

        # Exit markers (X)
        exit_spreads = spread.reindex(pair_trades["exit_date"]).values
        fig.add_trace(go.Scatter(
            x=pair_trades["exit_date"], y=exit_spreads,
            name="Exit", mode="markers",
            marker=dict(symbol="x", size=7, color="#F0A500", line=dict(width=1.5)),
        ), row=1, col=1)

    # Panel 2: Z-score with threshold lines
    fig.add_trace(go.Scatter(
        x=zscore.index, y=zscore.values,
        name="Z-score", mode="lines",
        line=dict(color="#B39DDB", width=1),
        showlegend=True,
    ), row=2, col=1)

    for level, color, dash, label in [
        ( 2.0, "#E05252", "dash",  "±2.0 entry"),
        (-2.0, "#E05252", "dash",  None),
        ( 3.5, "#FF7043", "dashdot", "±3.5 stop"),
        (-3.5, "#FF7043", "dashdot", None),
        ( 0.0, "grey",    "dot",   "Zero"),
    ]:
        fig.add_hline(
            y=level,
            line_dash=dash, line_color=color, line_width=1.2,
            annotation_text=label if label else "",
            annotation_position="right",
            annotation_font_size=9,
            row=2, col=1,
        )

    # Panel 3: Kalman hedge ratio as step function
    if len(hedge_df) > 0:
        fig.add_trace(go.Scatter(
            x=hedge_df["entry_date"], y=hedge_df["hedge_ratio"],
            name="Kalman β", mode="lines+markers",
            line=dict(color="#F0A500", width=2, shape="hv"),
            marker=dict(size=6),
        ), row=3, col=1)

        pairs_df   = load_selected_pairs()
        static_row = pairs_df[(pairs_df["ticker_a"] == a) & (pairs_df["ticker_b"] == b)]
        if len(static_row) > 0:
            static_beta = static_row.iloc[0]["hedge_ratio"]
            fig.add_hline(
                y=static_beta,
                line_dash="dot", line_color="#4C9BE8", line_width=1.5,
                annotation_text=f"Static OLS β={static_beta:.3f}",
                annotation_position="right",
                annotation_font_size=9,
                row=3, col=1,
            )

    fig.update_layout(
        template=DARK, margin=MARGIN, height=780,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Spread", row=1, col=1)
    fig.update_yaxes(title_text="Z-score", row=2, col=1)
    fig.update_yaxes(title_text="Hedge Ratio β", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ── Trade statistics ───────────────────────────────────────────────────────
    if len(pair_trades) > 0:
        st.subheader(f"Trade Statistics — {pair_label} (Kalman)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total trades",  len(pair_trades))
        c2.metric("Win rate",      f"{(pair_trades['net_pnl']>0).mean():.1%}")
        c3.metric("Avg net PnL",   f"${pair_trades['net_pnl'].mean():.1f}")
        c4.metric("Total net PnL", f"${pair_trades['net_pnl'].sum():.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def page_risk_analysis():
    st.title("Risk Analysis")

    daily_pnl  = load_daily_pnl()
    trade_logs = load_trade_logs()

    # ── Row 1: Cost sensitivity + Rolling Sharpe ───────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Transaction Cost Sensitivity")
        cost_df = compute_cost_sensitivity(daily_pnl, trade_logs)

        fig_cost = go.Figure()
        cost_colors = {"Static OLS": "#4C9BE8", "Kalman": "#F0A500", "Kalman+Regime": "#3ECF8E"}
        cost_dashes = {"Static OLS": "dot", "Kalman": "dash", "Kalman+Regime": "solid"}

        for strat, grp in cost_df.groupby("Strategy"):
            fig_cost.add_trace(go.Scatter(
                x=grp["bps"], y=grp["Sharpe"],
                name=strat, mode="lines+markers",
                line=dict(color=cost_colors[strat], dash=cost_dashes[strat], width=2),
                marker=dict(size=6),
            ))

        fig_cost.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
        fig_cost.add_vline(x=5, line_dash="dash", line_color="#888", line_width=1,
                           annotation_text="Baseline (5 bps)", annotation_position="top right",
                           annotation_font_size=10)
        fig_cost.update_layout(
            template=DARK, margin=MARGIN, height=360,
            xaxis_title="Transaction Cost (bps, one-way)",
            yaxis_title="Annualised Sharpe Ratio",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    with col_right:
        st.subheader("Rolling 63-Day Sharpe — Kalman+Regime")
        rs = compute_rolling_sharpe(daily_pnl)

        fig_rs = go.Figure()
        # Colour fill positive green, negative red
        pos_vals = rs.where(rs >= 0, 0)
        neg_vals = rs.where(rs <  0, 0)

        fig_rs.add_trace(go.Scatter(
            x=rs.index, y=pos_vals.values,
            name="Sharpe ≥ 0", fill="tozeroy",
            mode="none", fillcolor="rgba(62, 207, 142, 0.25)",
        ))
        fig_rs.add_trace(go.Scatter(
            x=rs.index, y=neg_vals.values,
            name="Sharpe < 0", fill="tozeroy",
            mode="none", fillcolor="rgba(224, 82, 82, 0.25)",
        ))
        fig_rs.add_trace(go.Scatter(
            x=rs.index, y=rs.values,
            name="Rolling Sharpe", mode="lines",
            line=dict(color="#3ECF8E", width=1.5),
        ))
        fig_rs.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)

        fig_rs.update_layout(
            template=DARK, margin=MARGIN, height=360,
            xaxis_title="Date", yaxis_title="Sharpe (63-day)",
            showlegend=False,
        )
        st.plotly_chart(fig_rs, use_container_width=True)

    st.divider()

    # ── Row 2: Max drawdown comparison ────────────────────────────────────────
    st.subheader("Max Drawdown Comparison")

    max_dds = compute_max_drawdowns(daily_pnl)
    strats  = list(max_dds.keys())
    dd_vals = [abs(max_dds[s]) for s in strats]

    fig_dd = go.Figure(go.Bar(
        x=strats, y=dd_vals,
        marker_color=["#4C9BE8", "#F0A500", "#3ECF8E"],
        text=[f"${v:,.0f}" for v in dd_vals],
        textposition="outside",
        textfont=dict(size=13, color="#E6EDF3"),
    ))
    fig_dd.update_layout(
        template=DARK, margin=dict(l=40, r=20, t=40, b=40), height=340,
        yaxis_title="Max Drawdown ($, absolute)",
        yaxis=dict(autorange=True),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # ── Drawdown time series ───────────────────────────────────────────────────
    st.subheader("Drawdown Over Time")
    fig_ddt = go.Figure()
    dd_colors = {"Static OLS": "#4C9BE8", "Kalman": "#F0A500", "Kalman+Regime": "#3ECF8E"}
    dd_dashes = {"Static OLS": "dot", "Kalman": "dash", "Kalman+Regime": "solid"}

    for name, df in daily_pnl.items():
        cum = df.sum(axis=1).cumsum()
        dd  = cum - cum.cummax()
        fig_ddt.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            name=name, mode="lines",
            line=dict(color=dd_colors[name], dash=dd_dashes[name], width=1.5),
        ))

    fig_ddt.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
    fig_ddt.update_layout(
        template=DARK, margin=MARGIN, height=320,
        xaxis_title="Date", yaxis_title="Drawdown ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig_ddt, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FACTOR DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

def page_factor_decomposition():
    st.title("Factor Decomposition")
    st.markdown(
        "Fama-French 5-Factor OLS regression of **Kalman+Regime** daily excess returns "
        "against Mkt-RF, SMB, HML, RMW, and CMA. Results pre-computed from Ken French's daily data library."
    )
    st.divider()

    res     = load_factor_results()
    factors = res["factors"]
    names   = list(factors.keys())
    betas   = [factors[f]["beta"]   for f in names]
    pvals   = [factors[f]["pvalue"] for f in names]
    tstats  = [factors[f]["tstat"]  for f in names]
    ci_low  = [factors[f]["ci_low"] for f in names]
    ci_hi   = [factors[f]["ci_high"] for f in names]

    bar_colors = ["#3ECF8E" if p < 0.05 else "#6E7681" for p in pvals]
    err_low  = [b - lo for b, lo in zip(betas, ci_low)]
    err_high = [hi - b for b, hi in zip(betas, ci_hi)]

    col_chart, col_table = st.columns([1, 1])

    with col_chart:
        st.subheader("Factor Loadings (β)")

        fig_fl = go.Figure(go.Bar(
            x=names, y=betas,
            marker_color=bar_colors,
            error_y=dict(
                type="data",
                symmetric=False,
                array=err_high,
                arrayminus=err_low,
                color="#8B949E",
                thickness=1.5,
                width=6,
            ),
            text=[f"{b:.4f}{'*' if p<0.05 else ''}" for b, p in zip(betas, pvals)],
            textposition="outside",
            textfont=dict(size=11, color="#E6EDF3"),
        ))

        fig_fl.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
        fig_fl.update_layout(
            template=DARK, margin=dict(l=30, r=20, t=70, b=40), height=400,
            yaxis_title="Factor Beta",
            title=dict(
                text=(
                    f"FF5 Factor Loadings<br>"
                    f"<sub>R²={res['r_squared']:.4f}  |  N={res['n_obs']} obs  |  * p&lt;0.05</sub>"
                ),
                font=dict(size=14),
            ),
            annotations=[
                dict(
                    x=0.02, y=0.97, xref="paper", yref="paper",
                    text="<b style='color:#3ECF8E'>■</b> Significant (p<0.05)   "
                         "<b style='color:#6E7681'>■</b> Not significant",
                    showarrow=False, font=dict(size=11, color="#C9D1D9"),
                    align="left",
                )
            ],
        )
        st.plotly_chart(fig_fl, use_container_width=True)

    with col_table:
        st.subheader("Regression Table")

        def sig_stars(p: float) -> str:
            if p < 0.001: return "***"
            if p < 0.01:  return "**"
            if p < 0.05:  return "*"
            return ""

        table_rows = []
        for f in names:
            fv = factors[f]
            table_rows.append({
                "Factor":  f,
                "Beta":    f"{fv['beta']:+.4f}",
                "t-stat":  f"{fv['tstat']:+.3f}",
                "p-value": f"{fv['pvalue']:.4f}",
                "95% CI":  f"[{fv['ci_low']:+.4f}, {fv['ci_high']:+.4f}]",
                "Sig":     sig_stars(fv["pvalue"]),
            })

        table_df = pd.DataFrame(table_rows)

        # Highlight significant rows
        def highlight_sig(row):
            p_str = row["p-value"]
            try:
                p = float(p_str)
            except ValueError:
                p = 1.0
            color = "background-color: rgba(62,207,142,0.12)" if p < 0.05 else ""
            return [color] * len(row)

        styled = table_df.style.apply(highlight_sig, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True, height=230)

        st.markdown(
            """
            **Significance levels:** &nbsp; `*` p<0.05 &nbsp; `**` p<0.01 &nbsp; `***` p<0.001
            **Model:** OLS · Newey-West standard errors · 95% CI
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Alpha highlight ────────────────────────────────────────────────────────
    alpha_ann   = res["alpha"]
    alpha_p     = res["alpha_pvalue"]
    alpha_t     = res["alpha_tstat"]
    alpha_sig   = res["alpha_significant"]
    sig_label   = "statistically significant" if alpha_sig else "not statistically significant"
    sig_color   = "#3ECF8E" if alpha_sig else "#E05252"
    star_str    = sig_stars(alpha_p)

    st.markdown(
        f"""
        <div style="
            background: {'rgba(62,207,142,0.10)' if alpha_sig else 'rgba(224,82,82,0.10)'};
            border-left: 4px solid {sig_color};
            border-radius: 6px;
            padding: 16px 20px;
            margin-top: 4px;
        ">
            <p style="margin:0; font-size:1.1em; color:#E6EDF3;">
                <b>Annualised Alpha:</b>
                <span style="font-size:1.4em; color:{sig_color}; font-weight:700;">
                    {alpha_ann:.2%}{star_str}
                </span>
                &nbsp;
                <span style="font-size:0.9em; color:#8B949E;">
                    (t={alpha_t:.3f}, p={alpha_p:.4f})
                </span>
            </p>
            <p style="margin:6px 0 0 0; color:#8B949E; font-size:0.9em;">
                Alpha is <b style="color:{sig_color};">{sig_label}</b> at the 5% level.
                R²={res['r_squared']:.4f} — the strategy's returns are largely independent
                of common factor exposures.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

if page == "Strategy Overview":
    page_overview()
elif page == "Pair Analysis":
    page_pair_analysis()
elif page == "Risk Analysis":
    page_risk_analysis()
elif page == "Factor Decomposition":
    page_factor_decomposition()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#6E7681; font-size:0.85em;'>"
    "Pairs Trading Research Dashboard · "
    "<a href='https://github.com/your-username/pairs-trading' style='color:#4C9BE8;'>"
    "GitHub</a> · Built with Streamlit + Plotly"
    "</div>",
    unsafe_allow_html=True,
)
