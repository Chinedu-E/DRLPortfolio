"""Helpers to build tradeable price panels for MarketReplayEnv."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rlportfolio.data_engine import SECTORS


@dataclass(frozen=True)
class EnvArrays:
    """NumPy panels ready for ``MarketReplayEnv`` over one date window."""

    asset_names: list[str]
    prices: np.ndarray  # (T_days, n)
    log_returns: np.ndarray  # (T_days, n)
    vol_20: np.ndarray  # (T_days,)
    vol_ratio: np.ndarray  # (T_days,)
    vix: np.ndarray  # (T_days,)
    dates: pd.DatetimeIndex


def tradeable_assets(
    prices: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    candidates: list[str] | None = None,
) -> list[str]:
    """
    Return sector columns with no NaNs in ``[start, end]``.

    Drops incomplete series (e.g. CommunicationServices before XLC inception).
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if candidates is None:
        candidates = list(SECTORS.keys())

    window = prices.loc[(prices.index >= start_ts) & (prices.index <= end_ts)]
    if window.empty:
        return []

    keep: list[str] = []
    for name in candidates:
        if name not in window.columns:
            continue
        if window[name].isna().any():
            continue
        keep.append(name)
    return keep


def make_env_arrays(
    prices: pd.DataFrame,
    log_returns: pd.DataFrame,
    features: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    asset_names: list[str] | None = None,
) -> EnvArrays:
    """Slice dataset frames into contiguous NumPy arrays for one episode."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    if asset_names is None:
        asset_names = tradeable_assets(prices, start_ts, end_ts)
    if not asset_names:
        raise ValueError(f"No tradeable assets in [{start_ts.date()}, {end_ts.date()}]")

    mask = (prices.index >= start_ts) & (prices.index <= end_ts)
    px = prices.loc[mask, asset_names]
    rets = log_returns.loc[mask, asset_names]
    feat = features.loc[mask]
    vix = prices.loc[mask, "VIX"]

    if px.isna().any().any():
        raise ValueError("Selected asset prices contain NaNs in the episode window")
    if feat[["vol_20", "vol_ratio"]].isna().any().any():
        raise ValueError("vol_20 / vol_ratio contain NaNs in the episode window")
    if vix.isna().any():
        raise ValueError("VIX contains NaNs in the episode window")

    return EnvArrays(
        asset_names=list(asset_names),
        prices=px.to_numpy(dtype=np.float64),
        log_returns=rets.to_numpy(dtype=np.float64),
        vol_20=feat["vol_20"].to_numpy(dtype=np.float64),
        vol_ratio=feat["vol_ratio"].to_numpy(dtype=np.float64),
        vix=vix.to_numpy(dtype=np.float64),
        dates=pd.DatetimeIndex(px.index),
    )
