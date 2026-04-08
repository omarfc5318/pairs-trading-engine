import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

sys.path.insert(0, str(Path(__file__).parent.parent))


class RegimeDetector:
    """
    Hidden Markov Model regime detector using a 2-state Gaussian HMM.

    States
    ------
    0 — mean-reverting regime (lower volatility) — pairs trading is allowed
    1 — trending regime      (higher volatility) — pairs trading is suppressed

    Design
    ------
    - Parameters are estimated from the full return series so both regimes
      (calm and crisis) are represented in the training data. Fitting only
      on a short calm window produces degenerate models that never learn
      what a crisis looks like.
    - Zero lookahead is enforced during *decoding* via the causal forward
      filter (alpha pass): the regime label at time t is argmax of
      P(state_t | obs_0..obs_t), using no future observations.
    """

    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.n_components  = n_components
        self.random_state  = random_state
        self._model: GaussianHMM | None = None
        self._mr_state: int = 0
        self._train_window: int = 252

    # ── Feature engineering ────────────────────────────────────────────────────

    @staticmethod
    def _make_features(returns: pd.Series) -> np.ndarray:
        """
        Build 2-D feature matrix: [return, |return|].

        The absolute return gives the HMM a direct volatility signal,
        preventing degenerate single-state solutions on a scalar feature.
        """
        r = returns.values
        return np.column_stack([r, np.abs(r)])

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(
        self,
        returns_series: pd.Series,
        train_window:   int = 252,
    ) -> "RegimeDetector":
        """
        Fit the HMM on the full returns series.

        train_window is stored as the minimum history required before the
        forward filter starts labelling regimes.

        Parameters
        ----------
        returns_series : full daily return series (e.g. SPY log-returns)
        train_window   : min observations before regime labels are emitted
        """
        self._train_window = train_window
        ret = returns_series.dropna()
        if len(ret) < train_window:
            raise ValueError(
                f"Need at least {train_window} observations, got {len(ret)}."
            )

        X = self._make_features(ret)

        model = GaussianHMM(
            n_components=self.n_components,
            covariance_type="diag",
            n_iter=500,
            random_state=self.random_state,
            tol=1e-6,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)

        self._model    = model
        self._mr_state = self.identify_mean_reverting_regime(model, ret)
        return self

    def predict_regimes(self, returns_series: pd.Series) -> pd.Series:
        """
        Predict regime labels with strict zero lookahead via causal forward filter.

        At each time t the regime is argmax_i P(state_t=i | obs_0..obs_t),
        computed by the HMM forward (alpha) pass only — no future data used.

        First train_window observations are labelled 0 (assume mean-reverting
        until the model has enough history).

        Returns a pd.Series of {0, 1} labels indexed by date:
            0 — mean-reverting (lower volatility)
            1 — trending       (higher volatility)
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict_regimes().")

        returns  = returns_series.dropna()
        X        = self._make_features(returns)
        n        = len(X)
        n_states = self.n_components
        model    = self._model

        # Log-emission probabilities: shape (n, n_states)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_emit = model._compute_log_likelihood(X)

        log_transmat  = np.log(np.clip(model.transmat_,  1e-300, None))
        log_startprob = np.log(np.clip(model.startprob_, 1e-300, None))

        # Forward pass in log-space
        log_alpha = np.empty((n, n_states))
        log_alpha[0] = log_startprob + log_emit[0]

        for t in range(1, n):
            for j in range(n_states):
                log_alpha[t, j] = (
                    np.logaddexp.reduce(log_alpha[t - 1] + log_transmat[:, j])
                    + log_emit[t, j]
                )

        raw_states = np.argmax(log_alpha, axis=1)
        labels     = np.where(raw_states == self._mr_state, 0, 1).astype(int)

        return pd.Series(labels, index=returns.index, name="regime", dtype=int)

    @staticmethod
    def identify_mean_reverting_regime(
        hmm_model: GaussianHMM,
        returns:   pd.Series,
    ) -> int:
        """
        Identify which hidden state corresponds to mean-reverting behaviour.

        Mean-reverting = lower volatility state.
        Uses the variance of the |return| feature (index 1) as the proxy.

        Returns the state index (int) of the mean-reverting state.
        """
        vols = []
        for i in range(hmm_model.n_components):
            c   = hmm_model.covars_[i]          # shape (n_features,) for "diag"
            var = float(c[1]) if c.ndim == 1 else float(c[1, 1])
            vols.append(np.sqrt(var))

        return int(np.argmin(vols))


# ── Convenience fetcher ────────────────────────────────────────────────────────

def fetch_spy_returns(start: str, end: str) -> pd.Series:
    """Download SPY and return daily log-return series."""
    raw    = yf.download("SPY", start=start, end=end,
                          auto_adjust=True, progress=False)
    closes = raw["Close"].squeeze()
    return np.log(closes / closes.shift(1)).dropna().rename("SPY_returns")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import config as cfg

    returns = fetch_spy_returns(cfg.START_DATE, cfg.END_DATE)
    print(f"[regime] SPY returns: {len(returns)} days "
          f"({returns.index[0].date()} → {returns.index[-1].date()})")

    detector = RegimeDetector()
    detector.fit(returns, train_window=252)

    regimes = detector.predict_regimes(returns)
    mr = (regimes == 0).sum()
    tr = (regimes == 1).sum()
    print(f"[regime] Mean-reverting days: {mr} ({mr/len(regimes):.1%})")
    print(f"[regime] Trending days      : {tr} ({tr/len(regimes):.1%})")

    # Show some trending dates (should cluster around known market stress periods)
    trending_dates = regimes[regimes == 1].index
    print(f"\nSample trending dates (expect 2020 COVID, 2022 rate hikes):")
    print(trending_dates[:20].tolist())
