# --- Universe ---
UNIVERSE = ['XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLU', 'XLB', 'XLY', 'XLRE', 'XLC']

# --- Financials expansion ---
# Set USE_FINANCIALS = True (or pass --financials CLI flag) to run on
# all S&P 500 Financials sector stocks instead of the ETF universe.
USE_FINANCIALS = False

def get_active_universe() -> list[str]:
    """Return the universe to use based on USE_FINANCIALS flag."""
    if USE_FINANCIALS:
        from data.fetch import get_sp500_financials
        return get_sp500_financials()
    return UNIVERSE

# UNIVERSE_FINANCIALS is resolved lazily (only when USE_FINANCIALS is True)
# to avoid a Wikipedia scrape on every import.
UNIVERSE_FINANCIALS = None   # populated at runtime by get_active_universe()

# --- Date Range ---
START_DATE = '2015-01-01'
END_DATE   = '2024-12-31'

# --- Walk-Forward Windows ---
TRAIN_WINDOW = 252   # trading days
TEST_WINDOW  = 63    # trading days (~1 quarter)

# --- Cointegration ---
COINT_PVALUE_THRESHOLD = 0.05

# --- OU Process / Half-Life ---
HALFLIFE_MIN = 5
HALFLIFE_MAX = 60

# --- Z-Score Thresholds ---
ENTRY_ZSCORE   = 2.0
EXIT_ZSCORE    = 0.5
STOP_ZSCORE    = 3.5
ZSCORE_LOOKBACK = 60

# --- Execution & Sizing ---
COST_BPS       = 5
CAPITAL        = 100_000
POSITION_SIZE  = 0.10
