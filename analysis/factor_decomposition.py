import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent))

FF5_URL     = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
               "F-F_Research_Data_5_Factors_2x3_daily_TXT.zip")
CHARTS_DIR  = Path(__file__).parent.parent / "outputs" / "charts"
TRADING_DAYS = 252

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


# ── 1. Data fetch ──────────────────────────────────────────────────────────────

def fetch_ff5_factors(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download Fama-French 5-Factor daily data from Ken French's data library.

    Parses the raw text file, converts returns from percent to decimal,
    and clips to [start_date, end_date].

    Returns a DataFrame indexed by date with columns:
        Mkt-RF, SMB, HML, RMW, CMA, RF
    """
    try:
        df = _fetch_ff5_direct(start_date, end_date)
    except Exception as e:
        print(f"[ff5] Direct download failed ({e}). Falling back to pandas-datareader ...")
        df = _fetch_ff5_datareader(start_date, end_date)

    print(f"[ff5] Retrieved {len(df)} daily observations "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    print(f"[ff5] Columns: {list(df.columns)}")
    return df


def _fetch_ff5_direct(start_date: str, end_date: str) -> pd.DataFrame:
    """Primary: download zip directly from Ken French's data library."""
    print("[ff5] Trying direct download from Ken French's data library ...")
    with urllib.request.urlopen(FF5_URL, timeout=30) as resp:
        raw_bytes = resp.read()

    zf  = zipfile.ZipFile(io.BytesIO(raw_bytes))
    txt = zf.read(zf.namelist()[0]).decode("utf-8", errors="ignore")

    rows = []
    for line in txt.splitlines():
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            continue
        parts = stripped.split()
        if len(parts) != 7:
            continue
        try:
            rows.append([parts[0]] + [float(v) for v in parts[1:]])
        except ValueError:
            continue

    df = pd.DataFrame(rows, columns=["date"] + FACTORS + ["RF"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").sort_index()
    df = df / 100.0                          # percent → decimal
    return df.loc[start_date:end_date]


def _fetch_ff5_datareader(start_date: str, end_date: str) -> pd.DataFrame:
    """Fallback: fetch via pandas-datareader Fama-French source."""
    import pandas_datareader.data as web

    print("[ff5] Fetching via pandas-datareader (famafrench) ...")
    ds  = web.DataReader(
        "F-F_Research_Data_5_Factors_2x3_Daily",
        "famafrench",
        start=start_date,
        end=end_date,
    )
    # DataReader returns a dict; key 0 is the daily factor table
    df  = ds[0].copy()
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"

    # Rename to standard column names if needed
    rename = {"Mkt-RF": "Mkt-RF", "SMB": "SMB", "HML": "HML",
              "RMW": "RMW", "CMA": "CMA", "RF": "RF"}
    df = df.rename(columns=rename)[FACTORS + ["RF"]]
    df = df / 100.0                          # percent → decimal
    return df.loc[start_date:end_date]


# ── 2. Factor regression ───────────────────────────────────────────────────────

def run_factor_regression(
    strategy_returns: pd.Series,
    ff5_factors:      pd.DataFrame,
) -> dict:
    """
    OLS regression of (strategy_returns - RF) on the 5 Fama-French factors.

    Parameters
    ----------
    strategy_returns : daily strategy return series (decimal, not %)
    ff5_factors      : DataFrame from fetch_ff5_factors() with RF column

    Returns
    -------
    dict with keys:
        alpha             — annualised Jensen's alpha (daily intercept × 252)
        alpha_pvalue      — p-value of the intercept
        alpha_significant — bool, True if p-value < 0.05
        r_squared         — R² of the regression
        factors           — dict[factor_name -> {beta, tstat, pvalue, ci_low, ci_high}]
        n_obs             — number of observations used
    """
    # Align on common dates
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), ff5_factors],
        axis=1, join="inner"
    ).dropna()

    if len(aligned) < 30:
        raise ValueError(f"Too few aligned observations: {len(aligned)}")

    excess_returns = aligned["strategy"] - aligned["RF"]
    X = sm.add_constant(aligned[FACTORS])
    model = sm.OLS(excess_returns, X).fit()

    alpha_daily = float(model.params["const"])
    alpha_ann   = alpha_daily * TRADING_DAYS
    ci          = model.conf_int(alpha=0.05)   # 95 % CI

    factor_results = {}
    for f in FACTORS:
        factor_results[f] = {
            "beta":    round(float(model.params[f]),     4),
            "tstat":   round(float(model.tvalues[f]),    4),
            "pvalue":  round(float(model.pvalues[f]),    4),
            "ci_low":  round(float(ci.loc[f, 0]),        4),
            "ci_high": round(float(ci.loc[f, 1]),        4),
        }

    results = {
        "alpha":             round(alpha_ann, 6),
        "alpha_daily":       round(alpha_daily, 8),
        "alpha_pvalue":      round(float(model.pvalues["const"]), 4),
        "alpha_tstat":       round(float(model.tvalues["const"]), 4),
        "alpha_significant": bool(float(model.pvalues["const"]) < 0.05),
        "r_squared":         round(float(model.rsquared), 4),
        "n_obs":             int(len(aligned)),
        "factors":           factor_results,
    }

    return results


# ── 3. Factor loading chart ────────────────────────────────────────────────────

def plot_factor_loadings(regression_results: dict) -> None:
    """
    Bar chart of factor betas with 95% CI error bars.
    Green bars = significant (p < 0.05), grey bars = not significant.
    Saves to outputs/charts/factor_loadings.png
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    fr     = regression_results["factors"]
    names  = list(fr.keys())
    betas  = [fr[f]["beta"]   for f in names]
    pvals  = [fr[f]["pvalue"] for f in names]
    ci_low = [fr[f]["ci_low"] for f in names]
    ci_hi  = [fr[f]["ci_high"] for f in names]

    err_low  = [b - lo for b, lo in zip(betas, ci_low)]
    err_high = [hi - b for b, hi in zip(betas, ci_hi)]
    colors   = ["#2E7D32" if p < 0.05 else "#9E9E9E" for p in pvals]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(names, betas, color=colors, alpha=0.85, width=0.5,
                  yerr=[err_low, err_high], capsize=5,
                  error_kw={"elinewidth": 1.5, "ecolor": "#424242", "capthick": 1.5})

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

    # Annotate each bar
    for i, (name, beta, pval) in enumerate(zip(names, betas, pvals)):
        sig  = "**" if pval < 0.01 else ("*" if pval < 0.05 else "")
        col  = "#1B5E20" if pval < 0.05 else "#616161"
        ax.text(i, beta + (max(err_high) * 0.15 if beta >= 0 else -max(err_high) * 0.15),
                f"{beta:.3f}{sig}", ha="center", va="bottom" if beta >= 0 else "top",
                fontsize=9, fontweight="bold", color=col)

    # Legend patches
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#2E7D32", label="Significant (p < 0.05)"),
        Patch(color="#9E9E9E", label="Not significant"),
    ]
    ax.legend(handles=legend, fontsize=9, loc="upper right")

    alpha_ann = regression_results["alpha"]
    alpha_p   = regression_results["alpha_pvalue"]
    sig_str   = " *" if alpha_p < 0.05 else ""
    ax.set_title(
        f"Fama-French 5-Factor Loadings — Kalman+Regime Strategy\n"
        f"Annualised Alpha: {alpha_ann:.2%}{sig_str}  |  "
        f"R²: {regression_results['r_squared']:.3f}  |  "
        f"N={regression_results['n_obs']}",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylabel("Factor Beta", fontsize=10)
    ax.set_xlabel("Fama-French Factor", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=10)

    plt.tight_layout()
    out_path = CHARTS_DIR / "factor_loadings.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ff5] Saved factor_loadings.png -> {out_path}")


# ── 4. Print summary table ─────────────────────────────────────────────────────

def print_factor_summary(regression_results: dict) -> None:
    """
    Formatted table: factor, beta, t-stat, p-value, significance.
    Annualised alpha printed at the bottom.
    """
    fr = regression_results["factors"]

    sep = "─" * 58
    print(f"\n{'═' * 58}")
    print(f"  FAMA-FRENCH 5-FACTOR REGRESSION — Kalman+Regime Strategy")
    print(f"{'═' * 58}")
    print(f"  Observations : {regression_results['n_obs']}")
    print(f"  R-squared    : {regression_results['r_squared']:.4f}")
    print(sep)
    print(f"  {'Factor':<10} {'Beta':>9} {'t-stat':>9} {'p-value':>9}  {'Sig':>4}")
    print(sep)

    for factor, vals in fr.items():
        sig = ("***" if vals["pvalue"] < 0.001 else
               "**"  if vals["pvalue"] < 0.01  else
               "*"   if vals["pvalue"] < 0.05  else "")
        print(f"  {factor:<10} {vals['beta']:>9.4f} {vals['tstat']:>9.3f} "
              f"{vals['pvalue']:>9.4f}  {sig:>4}")

    print(sep)
    alpha_ann  = regression_results["alpha"]
    alpha_p    = regression_results["alpha_pvalue"]
    alpha_t    = regression_results["alpha_tstat"]
    alpha_sig  = "***" if alpha_p < 0.001 else "**" if alpha_p < 0.01 else "*" if alpha_p < 0.05 else "n.s."
    sig_label  = "SIGNIFICANT" if regression_results["alpha_significant"] else "not significant"

    print(f"  {'Alpha (ann.)':<10} {alpha_ann:>9.4f} {alpha_t:>9.3f} {alpha_p:>9.4f}  {alpha_sig:>4}")
    print(f"{'═' * 58}")
    print(f"  Alpha is {sig_label} at the 5% level.")
    print(f"  Significance: *** p<0.001  ** p<0.01  * p<0.05")
    print(f"{'═' * 58}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from backtest.metrics import _portfolio_returns

    # Load Kalman+Regime P&L
    regime_pnl = pd.read_csv(
        Path(__file__).parent.parent / "outputs" / "backtest" / "regime_daily_pnl.csv",
        index_col=0, parse_dates=True,
    )
    strategy_returns = _portfolio_returns(regime_pnl)

    start = str(strategy_returns.index[0].date())
    end   = str(strategy_returns.index[-1].date())

    ff5   = fetch_ff5_factors(start, end)
    res   = run_factor_regression(strategy_returns, ff5)

    print_factor_summary(res)
    plot_factor_loadings(res)
