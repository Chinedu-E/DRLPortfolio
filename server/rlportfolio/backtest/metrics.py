"""Pyfolio-compatible performance stats and portfolio turnover Δp_w."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from pyfolio import timeseries as pf_timeseries
except ImportError:  # pragma: no cover
    pf_timeseries = None  # type: ignore[assignment]


def delta_pw(weights_prev: np.ndarray, weights_next: np.ndarray) -> float:
    """
    Absolute portfolio change ignoring cash.

    ``Δp_w = sum_i |w_i,t - w_i,t-1|`` over non-cash assets; range ``[0, 2]``.
    """
    a = np.asarray(weights_prev, dtype=np.float64).reshape(-1)
    b = np.asarray(weights_next, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("weight vectors must match and include a cash slot")
    return float(np.abs(a[:-1] - b[:-1]).sum())


def delta_pw_series(weights: np.ndarray) -> np.ndarray:
    """Per-step Δp_w for a ``(T, n_slots)`` weight path; first step is 0."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    out = np.zeros(w.shape[0], dtype=np.float64)
    for t in range(1, w.shape[0]):
        out[t] = delta_pw(w[t - 1], w[t])
    return out


def evaluate_returns(
    returns: pd.Series | np.ndarray,
    *,
    dates: pd.DatetimeIndex | None = None,
    delta_pw_values: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute Table-2 style metrics via ``pyfolio.timeseries.perf_stats``.

    Also attaches ``Mean Δp_w`` when a turnover series is provided.
    """
    if isinstance(returns, np.ndarray):
        if dates is None:
            raise ValueError("dates required when returns is a numpy array")
        series = pd.Series(returns, index=pd.DatetimeIndex(dates), dtype=float)
    else:
        series = returns.astype(float).copy()
        series.index = pd.DatetimeIndex(series.index)

    # Pyfolio expects a DatetimeIndex without tz.
    if getattr(series.index, "tz", None) is not None:
        series.index = series.index.tz_localize(None)

    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        raise ValueError("returns series is empty")

    if pf_timeseries is None:
        raise ImportError("pyfolio-reloaded is required for evaluate_returns")

    stats = pf_timeseries.perf_stats(series)
    # Normalize to a plain dict with stable keys.
    out: dict[str, float] = {}
    for key, value in stats.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    if delta_pw_values is not None and len(delta_pw_values) > 0:
        out["Mean Δp_w"] = float(np.mean(delta_pw_values))
    return out


def aggregate_window_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """
    Average metrics across windows; Max Drawdown is the worst (most negative).

    ``rows`` are per-window metric dicts (possibly with non-numeric extras skipped).
    """
    if not rows:
        return {}

    numeric_keys: set[str] = set()
    for row in rows:
        for k, v in row.items():
            if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v)):
                numeric_keys.add(k)

    agg: dict[str, float] = {}
    for key in sorted(numeric_keys):
        vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
        if not vals:
            continue
        key_l = key.lower()
        if "max drawdown" in key_l or key_l.endswith("max_drawdown"):
            agg[key] = float(min(vals))  # most negative
        else:
            agg[key] = float(np.mean(vals))
    return agg


def metrics_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Stack per-window metric dicts into a DataFrame."""
    return pd.DataFrame(rows)


def rolling_sharpe(
    returns: np.ndarray | pd.Series,
    *,
    window: int = 63,
    trading_days: int = 252,
) -> np.ndarray:
    """Annualized rolling Sharpe; leading values are NaN until ``window`` fills."""
    series = pd.Series(np.asarray(returns, dtype=float))
    roll_mean = series.rolling(window, min_periods=window).mean()
    roll_std = series.rolling(window, min_periods=window).std(ddof=1)
    sharpe = (roll_mean / roll_std) * np.sqrt(trading_days)
    return sharpe.to_numpy(dtype=np.float64)


def normalize_equity(values: np.ndarray) -> np.ndarray:
    """Normalize a portfolio-value path to 1.0 at the first finite point."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return v
    base = float(v[0])
    if not np.isfinite(base) or abs(base) < 1e-12:
        return np.ones_like(v)
    return v / base


def spx_buy_and_hold_equity(
    spx_prices: np.ndarray | pd.Series,
    *,
    dates: pd.DatetimeIndex,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Continuous S&P 500 buy-and-hold equity normalized to 1.0 at start."""
    px = np.asarray(spx_prices, dtype=np.float64)
    if px.size == 0:
        return pd.DatetimeIndex([]), np.array([], dtype=np.float64)
    equity = normalize_equity(px)
    return pd.DatetimeIndex(dates[: len(equity)]), equity


def regime_drawdown_bands(
    spx_prices: np.ndarray | pd.Series,
    dates: pd.DatetimeIndex,
    *,
    threshold: float = 0.15,
) -> list[dict[str, str]]:
    """
    Contiguous bands where SPX drawdown from running peak exceeds ``threshold``.

    Returns ``[{start, end}, ...]`` as ISO date strings for chart overlays.
    """
    px = np.asarray(spx_prices, dtype=np.float64)
    if px.size == 0:
        return []
    peak = np.maximum.accumulate(px)
    dd = 1.0 - px / np.maximum(peak, 1e-12)
    in_stress = dd >= threshold

    bands: list[dict[str, str]] = []
    start_i: int | None = None
    for i, flag in enumerate(in_stress):
        if flag and start_i is None:
            start_i = i
        elif not flag and start_i is not None:
            bands.append(
                {
                    "start": str(pd.Timestamp(dates[start_i]).date()),
                    "end": str(pd.Timestamp(dates[i - 1]).date()),
                }
            )
            start_i = None
    if start_i is not None:
        bands.append(
            {
                "start": str(pd.Timestamp(dates[start_i]).date()),
                "end": str(pd.Timestamp(dates[len(dates) - 1]).date()),
            }
        )
    return bands
