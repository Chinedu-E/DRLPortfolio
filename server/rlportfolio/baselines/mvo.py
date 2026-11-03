"""Mean-Variance Optimization baseline with Ledoit–Wolf shrinkage (Section 5.3)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


def psd_repair(cov: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """
    Enforce positive semi-definiteness by zeroing negative eigenvalues.

    As described in the paper: set negative eigenvalues to 0 and rebuild.
    """
    cov = np.asarray(cov, dtype=np.float64)
    # Symmetrize against numerical noise.
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, eps)
    repaired = (eigvecs * eigvals) @ eigvecs.T
    return 0.5 * (repaired + repaired.T)


def equal_weights(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    return np.full(n, 1.0 / n, dtype=np.float64)


class MVOStrategy:
    """
    Daily max-Sharpe MVO using a rolling lookback of log returns.

    Covariance uses Ledoit–Wolf shrinkage with PSD repair. Weights are long-only
    over the ``n`` sector assets (no dedicated cash sleeve).
    """

    def __init__(self, lookback: int = 60, risk_free_rate: float = 0.0) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        self.lookback = int(lookback)
        self.risk_free_rate = float(risk_free_rate)

    def weights_on_day(self, log_returns: np.ndarray | pd.DataFrame, t_index: int) -> np.ndarray:
        """
        Estimate weights for day ``t_index`` using returns in ``[t-lookback, t)``.

        ``t_index`` is the day being traded; the lookback window ends the day before.
        """
        if isinstance(log_returns, pd.DataFrame):
            panel = log_returns.to_numpy(dtype=np.float64)
            n_assets = panel.shape[1]
        else:
            panel = np.asarray(log_returns, dtype=np.float64)
            if panel.ndim != 2:
                raise ValueError("log_returns must be 2-D")
            n_assets = panel.shape[1]

        if t_index < self.lookback:
            return equal_weights(n_assets)

        window = panel[t_index - self.lookback : t_index]
        if not np.isfinite(window).all():
            return equal_weights(n_assets)

        return self._optimize(window)

    def _optimize(self, window: np.ndarray) -> np.ndarray:
        n_assets = window.shape[1]
        mu = window.mean(axis=0) * TRADING_DAYS

        try:
            lw = LedoitWolf().fit(window)
            cov = psd_repair(lw.covariance_ * TRADING_DAYS)
        except Exception:
            return equal_weights(n_assets)

        # PyPortfolioOpt expects a dict / Series for mu and DataFrame for cov.
        tickers = [f"a{i}" for i in range(n_assets)]
        mu_series = pd.Series(mu, index=tickers)
        cov_df = pd.DataFrame(cov, index=tickers, columns=tickers)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ef = EfficientFrontier(mu_series, cov_df, weight_bounds=(0.0, 1.0))
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
                cleaned = ef.clean_weights()
            weights = np.array([cleaned[t] for t in tickers], dtype=np.float64)
            weights = np.clip(weights, 0.0, None)
            total = weights.sum()
            if total <= 1e-12:
                return equal_weights(n_assets)
            return weights / total
        except Exception:
            return equal_weights(n_assets)

    def generate_weight_path(
        self,
        log_returns: pd.DataFrame,
        dates: pd.DatetimeIndex | None = None,
    ) -> pd.DataFrame:
        """
        Compute daily MVO weights for each date in ``log_returns``.

        Rows before ``lookback`` receive equal weights. Index matches ``dates``
        or ``log_returns.index``.
        """
        if dates is None:
            dates = pd.DatetimeIndex(log_returns.index)
        panel = log_returns.to_numpy(dtype=np.float64)
        rows = [
            self.weights_on_day(panel, t) for t in range(len(panel))
        ]
        return pd.DataFrame(rows, index=dates, columns=list(log_returns.columns))
