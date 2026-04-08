import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.signals import compute_rolling_zscore, generate_signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_zscore_series(values: list[float], name: str = "spread") -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, name=name, dtype=float)


# ---------------------------------------------------------------------------
# compute_rolling_zscore tests
# ---------------------------------------------------------------------------

class TestComputeRollingZscore:

    def test_flat_before_full_lookback_window(self):
        """First (lookback-1) values must be NaN — no lookahead."""
        spread  = pd.Series(np.random.randn(50), dtype=float)
        lookback = 10
        zscore  = compute_rolling_zscore(spread, lookback)
        assert zscore.iloc[:lookback - 1].isna().all(), \
            "Z-score should be NaN before a full lookback window is available"

    def test_first_valid_value_at_lookback(self):
        """Value at index [lookback-1] must be valid (not NaN)."""
        spread  = pd.Series(np.random.randn(50), dtype=float)
        lookback = 10
        zscore  = compute_rolling_zscore(spread, lookback)
        assert not np.isnan(zscore.iloc[lookback - 1]), \
            "First valid z-score should appear at index lookback-1"

    def test_no_lookahead(self):
        """Inserting a spike at position i must not affect z-scores before i."""
        base   = np.ones(40)
        spike  = base.copy()
        spike[30] = 100.0  # large spike at t=30
        zbase  = compute_rolling_zscore(pd.Series(base),  lookback=10)
        zspike = compute_rolling_zscore(pd.Series(spike), lookback=10)
        # z-scores before the spike should be identical
        pd.testing.assert_series_equal(
            zbase.iloc[:30], zspike.iloc[:30],
            check_names=False,
        )

    def test_zscore_mean_zero_std_one(self):
        """Over a stationary window the rolling z-score should be near 0 mean and 1 std."""
        np.random.seed(42)
        spread = pd.Series(np.random.randn(500))
        zscore = compute_rolling_zscore(spread, lookback=60).dropna()
        assert abs(zscore.mean()) < 0.2
        assert abs(zscore.std() - 1.0) < 0.1


# ---------------------------------------------------------------------------
# generate_signals tests
# ---------------------------------------------------------------------------

class TestGenerateSignals:

    def test_flat_while_nan(self):
        """Positions must be 0 wherever z-score is NaN."""
        z = make_zscore_series([np.nan] * 5 + [0.0] * 5)
        sig = generate_signals(z)
        assert (sig.iloc[:5] == 0).all(), "Position must be flat during NaN z-scores"

    def test_enter_long_on_negative_zscore(self):
        """z < -entry should trigger a long position (+1)."""
        # Start flat, then z goes very negative
        z   = make_zscore_series([0.0, 0.0, -2.5, -2.5, -2.5])
        sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5)
        assert sig.iloc[2] == 1, "Should enter long when z < -entry"
        assert sig.iloc[3] == 1, "Should hold long"
        assert sig.iloc[4] == 1, "Should hold long"

    def test_enter_short_on_positive_zscore(self):
        """z > +entry should trigger a short position (-1)."""
        z   = make_zscore_series([0.0, 0.0, 2.5, 2.5, 2.5])
        sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5)
        assert sig.iloc[2] == -1, "Should enter short when z > +entry"
        assert sig.iloc[3] == -1, "Should hold short"
        assert sig.iloc[4] == -1, "Should hold short"

    def test_exit_long_on_mean_reversion(self):
        """Long position should exit to flat when |z| < exit threshold."""
        z   = make_zscore_series([0.0, -2.5, -2.5, 0.3, 0.3])
        sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5)
        assert sig.iloc[1] == 1,  "Enter long"
        assert sig.iloc[2] == 1,  "Hold long"
        assert sig.iloc[3] == 0,  "Exit long when |z| < 0.5"
        assert sig.iloc[4] == 0,  "Stay flat"

    def test_exit_short_on_mean_reversion(self):
        """Short position should exit to flat when |z| < exit threshold."""
        z   = make_zscore_series([0.0, 2.5, 2.5, 0.3, 0.3])
        sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5)
        assert sig.iloc[1] == -1, "Enter short"
        assert sig.iloc[2] == -1, "Hold short"
        assert sig.iloc[3] == 0,  "Exit short when |z| < 0.5"
        assert sig.iloc[4] == 0,  "Stay flat"

    def test_stop_loss_long(self):
        """Stop-loss should trigger on a long position when |z| > stop."""
        z   = make_zscore_series([0.0, -2.5, -3.6, -3.6])
        sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5)
        assert sig.iloc[1] == 1,  "Enter long"
        assert sig.iloc[2] == 0,  "Stop-loss: |z| > 3.5 exits to flat"
        assert sig.iloc[3] == 0,  "Stay flat after stop"

    def test_stop_loss_short(self):
        """Stop-loss should trigger on a short position when |z| > stop."""
        z   = make_zscore_series([0.0, 2.5, 3.6, 3.6])
        sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5)
        assert sig.iloc[1] == -1, "Enter short"
        assert sig.iloc[2] == 0,  "Stop-loss: |z| > 3.5 exits to flat"
        assert sig.iloc[3] == 0,  "Stay flat after stop"

    def test_no_simultaneous_long_and_short(self):
        """Position must never be both long and short simultaneously."""
        np.random.seed(0)
        z   = make_zscore_series(list(np.random.randn(200) * 2))
        sig = generate_signals(z)
        assert set(sig.unique()).issubset({-1, 0, 1}), \
            "Positions must only be -1, 0, or +1"

    def test_no_re_entry_while_in_position(self):
        """Should not flip directly from long to short without going flat first."""
        # Stays long during the negative z region, then exits, then enters short
        values  = [0.0, -2.5, -2.5, 0.3, 2.5, 2.5]
        z       = make_zscore_series(values)
        sig     = generate_signals(z, entry=2.0, exit=0.5, stop=3.5)
        # Must pass through flat (0) before entering short
        pos     = sig.tolist()
        for i in range(1, len(pos)):
            assert not (pos[i - 1] == 1 and pos[i] == -1), \
                "Should not flip long→short without flattening first"
            assert not (pos[i - 1] == -1 and pos[i] == 1), \
                "Should not flip short→long without flattening first"
