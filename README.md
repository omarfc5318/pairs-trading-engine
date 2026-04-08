# Statistical Arbitrage Engine with Kalman Filter & Regime Detection

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A production-grade statistical arbitrage system that identifies cointegrated S&P 500 sector ETF pairs, estimates time-varying hedge ratios with a Kalman Filter, and suppresses trading during trending market regimes using a 2-state Hidden Markov Model. The engine is evaluated on a walk-forward backtest (252-day training / 63-day test windows) across 2019–2024, with all signals computed purely from past data to eliminate lookahead bias. Three progressively sophisticated strategies are benchmarked — static OLS, Kalman Filter, and Kalman + HMM Regime Detection — and results are visualised through an interactive Streamlit dashboard covering equity curves, pair spreads, risk decomposition, and Fama-French factor attribution.

---

## Key Results

| Strategy | Sharpe Ratio | Max Drawdown | Notes |
|---|---|---|---|
| Static OLS | 0.692 | — | Baseline walk-forward |
| Kalman Filter | 2.012 | — | **3× improvement** over OLS |
| Kalman + Regime | 1.830 | **−0.4%** | 65% drawdown reduction vs. Kalman |

- Kalman Filter alone delivers a **3× Sharpe improvement** over the OLS baseline by tracking drift in hedge ratios that static estimation misses.
- Adding the HMM Regime Filter reduces max drawdown by 65% (from −$1,133 to −$383 on a $100k book) at a modest Sharpe cost, producing a materially better risk-adjusted profile.
- Fama-French 5-Factor regression confirms **near-zero beta** (β ≈ 0.002 on Mkt-RF), validating market neutrality. Factor R² = 0.0072, indicating returns are largely orthogonal to common equity risk premia.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE OVERVIEW                        │
└─────────────────────────────────────────────────────────────────┘

  [yfinance API]
       │
       ▼
┌─────────────┐     raw Parquet files
│  data/      │ ──────────────────────► data/raw/{TICKER}.parquet
│  fetch.py   │
│  clean.py   │ ──────────────────────► data/clean/log_prices.parquet
└─────────────┘
       │  log prices (T × N)
       ▼
┌──────────────────┐
│  analysis/       │
│  cointegration   │  Engle-Granger test on all pairs
│  .py             │ ──► outputs/pairs/coint_results.csv
│                  │
│  ou_process.py   │  Half-life filter (5–60 days)
│                  │ ──► outputs/pairs/selected_pairs.csv
│                  │
│  signals.py      │  Rolling z-score, entry/exit/stop rules
│                  │  KalmanFilterHedge — time-varying β
│                  │
│  regime.py       │  GaussianHMM (2-state) on SPY returns
│                  │  Causal forward-filter decoding (no lookahead)
└──────────────────┘
       │  signals, hedge ratios, regime labels
       ▼
┌──────────────────┐
│  backtest/       │
│  engine.py       │  Walk-forward (252-day train / 63-day test)
│                  │  3 strategies × 3 pairs
│                  │ ──► outputs/backtest/daily_pnl.csv
│  metrics.py      │ ──► outputs/backtest/trade_log.csv
└──────────────────┘
       │  PnL series, trade log
       ▼
┌──────────────────┐     ┌──────────────────────────┐
│  viz/            │     │  analysis/                │
│  charts.py       │     │  factor_decomposition.py  │
│                  │     │                            │
│  equity curve    │     │  Fama-French 5-Factor OLS  │
│  drawdown        │     │  Ken French data library   │
│  rolling Sharpe  │     │ ──► factor_results.json    │
│  spread charts   │     └──────────────────────────┘
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  dashboard/      │
│  app.py          │  Streamlit · 4 pages · dark theme
│                  │  Strategy Overview / Pair Analysis /
│                  │  Risk Analysis / Factor Decomposition
└──────────────────┘
```

---

## Installation

```bash
git clone https://github.com/your-username/pairs-trading.git
cd pairs-trading
pip install -r requirements.txt
```

---

## Usage

**Run the full pipeline** (fetches data, runs cointegration, backtests all 3 strategies, saves charts):

```bash
python main.py
```

**Skip data fetch** (use cached Parquet files):

```bash
python main.py --no-fetch
```

**Run on S&P 500 Financials sector** instead of the default ETF universe:

```bash
python main.py --financials
```

**Cointegration and signal charts only** (skip backtest):

```bash
python main.py --pairs-only
```

**Launch the interactive dashboard**:

```bash
streamlit run dashboard/app.py
```

---

## Project Structure

```
pairs_trading/
├── config.py                        # Universe, dates, strategy parameters
├── main.py                          # CLI entry point — full pipeline
├── requirements.txt
│
├── data/
│   ├── fetch.py                     # yfinance download → raw Parquet
│   ├── clean.py                     # Align, forward-fill, compute log prices
│   └── clean/
│       ├── prices.parquet           # Adjusted close prices
│       └── log_prices.parquet       # Natural log of prices
│
├── analysis/
│   ├── cointegration.py             # Engle-Granger tests on all pairs
│   ├── ou_process.py                # Half-life estimation, KalmanFilterHedge
│   ├── signals.py                   # Rolling z-score, signal generation
│   ├── regime.py                    # GaussianHMM regime detector
│   └── factor_decomposition.py      # Fama-French 5-Factor regression
│
├── backtest/
│   ├── engine.py                    # WalkForwardBacktester (OLS / Kalman / Regime)
│   └── metrics.py                   # Sharpe, Sortino, drawdown, alpha/beta
│
├── viz/
│   └── charts.py                    # Matplotlib chart helpers
│
├── dashboard/
│   ├── app.py                       # Streamlit dashboard (4 pages)
│   └── .streamlit/config.toml       # Dark theme config
│
├── outputs/
│   ├── pairs/
│   │   ├── coint_results.csv        # All pair p-values and hedge ratios
│   │   └── selected_pairs.csv       # Pairs passing cointegration + half-life filter
│   ├── backtest/
│   │   ├── daily_pnl.csv            # Static OLS daily P&L
│   │   ├── kalman_daily_pnl.csv     # Kalman Filter daily P&L
│   │   ├── regime_daily_pnl.csv     # Kalman + Regime daily P&L
│   │   ├── trade_log.csv            # Static OLS trade log
│   │   ├── kalman_trade_log.csv     # Kalman trade log
│   │   ├── regime_trade_log.csv     # Regime-filtered trade log
│   │   └── factor_results.json      # Pre-computed FF5 regression results
│   └── charts/                      # All saved PNG charts
│
└── tests/
    └── test_signals.py              # Unit tests for signal generation
```

---

## Methodology

### Cointegration & Pair Selection

Candidate pairs are drawn from the 11 S&P 500 Select Sector SPDR ETFs (XLF, XLE, XLK, XLV, XLI, XLP, XLU, XLB, XLY, XLRE, XLC). All 55 unique pair combinations are tested for cointegration using the Engle-Granger two-step procedure on log prices over a 2015–2018 formation period. Pairs with a cointegration p-value below 0.05 are retained, then further filtered by mean-reversion half-life (estimated via Ornstein-Uhlenbeck regression), keeping only pairs with half-lives between 5 and 60 trading days. Three pairs survive: XLP/XLU (p=0.042, τ=38d), XLP/XLV (p=0.020, τ=41d), and XLB/XLV (p=0.011, τ=42d). The spread is defined as log(A) − β·log(B), where β is the OLS hedge ratio from the training window. Trading signals are generated from a 60-day rolling z-score with entry at |z| > 2.0, exit at |z| < 0.5, and stop-loss at |z| > 3.5.

### Kalman Filter Hedge Ratio

The static OLS hedge ratio assumes a constant linear relationship between the two log price series — an assumption that breaks down over multi-year backtests as sector correlations shift. The Kalman Filter models the hedge ratio β as a latent random walk state: β_t = β_{t−1} + η_t, observed through the spread equation s_t = log(A_t) − β_t·log(B_t) + ε_t. By recursively updating the posterior estimate of β using the innovation at each time step, the filter adapts continuously to structural changes in the pair relationship without any lookahead. The process noise covariance is set to Q = 1e-5 and the observation noise to R = 1e-3, calibrated to match the empirical half-life range of the selected pairs. This adaptive hedge ratio is the primary driver of the 3× Sharpe improvement over the static baseline.

### HMM Regime Detection

A 2-state Gaussian Hidden Markov Model is fit to the full SPY daily return series to identify mean-reverting (low-volatility) and trending (high-volatility) market regimes. The feature matrix consists of [return, |return|] to give the model a direct volatility signal and prevent degenerate single-state solutions. State assignment during the backtest uses the causal forward filter (alpha pass), so the regime label at time t is computed exclusively from observations up to and including t — no future data is used. The model identifies 88.3% of trading days as mean-reverting and 11.7% as trending. During trending regimes, all new position entries are blocked; existing positions are allowed to expire naturally. This regime gate reduces maximum drawdown by 65% relative to unfiltered Kalman, capturing a significantly better risk-adjusted profile in exchange for a modest reduction in raw Sharpe.

---

## Limitations

- **Short live history:** The backtest covers 2019–2024 (approximately 1,386 trading days across 3 pairs), which spans only one full market cycle. Results may not generalise to different interest rate environments or structural breaks not represented in this window.
- **Survivorship and selection bias:** Pairs were selected based on cointegration tests run on a pre-defined ETF universe that already existed and remained liquid throughout the entire sample period. The strategy has not been tested on universes with delistings, mergers, or ETF closures, which would affect live performance.
- **Simplified cost model:** Transaction costs are modelled as a flat 5 bps per leg, applied uniformly to all trades regardless of size, time-of-day, or market impact. Real execution costs — particularly for larger position sizes or during volatile periods — would likely be higher and are not captured by this model.

---

## License

This project is licensed under the [MIT License](LICENSE).
