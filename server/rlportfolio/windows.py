"""Sliding train / validation / test windows from the paper (Section 5.2)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rlportfolio.data_engine import MarketDataset
from rlportfolio.env.panel import EnvArrays, make_env_arrays, tradeable_assets

LOOKBACK_DAYS = 60


@dataclass(frozen=True)
class SlidingWindow:
    """One 7-year group: 5y train, 1y val (burn), 1y test."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # exclusive calendar bound
    val_start: pd.Timestamp
    val_end: pd.Timestamp  # exclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusive


def sliding_windows() -> list[SlidingWindow]:
    """
    Ten windows shifted by one year.

    Window 0: train [2006, 2011), val 2011, test 2012
    Window 9: train [2015, 2020), val 2020, test 2021
    """
    windows: list[SlidingWindow] = []
    for i in range(10):
        train_start_year = 2006 + i
        windows.append(
            SlidingWindow(
                index=i,
                train_start=pd.Timestamp(f"{train_start_year}-01-01"),
                train_end=pd.Timestamp(f"{train_start_year + 5}-01-01"),
                val_start=pd.Timestamp(f"{train_start_year + 5}-01-01"),
                val_end=pd.Timestamp(f"{train_start_year + 6}-01-01"),
                test_start=pd.Timestamp(f"{train_start_year + 6}-01-01"),
                test_end=pd.Timestamp(f"{train_start_year + 7}-01-01"),
            )
        )
    return windows


def get_window(index: int) -> SlidingWindow:
    windows = sliding_windows()
    if index < 0 or index >= len(windows):
        raise IndexError(f"window index must be in [0, {len(windows) - 1}], got {index}")
    return windows[index]


def _inclusive_end(exclusive: pd.Timestamp) -> pd.Timestamp:
    """Last calendar day still inside an exclusive end bound."""
    return exclusive - pd.Timedelta(days=1)


def pad_start(
    index: pd.DatetimeIndex,
    period_start: pd.Timestamp,
    lookback: int = LOOKBACK_DAYS,
) -> pd.Timestamp:
    """Earliest trading date so ``period_start`` has ``lookback`` prior sessions."""
    locs = index.get_indexer([period_start], method="bfill")
    if locs[0] < 0:
        raise ValueError(f"period_start {period_start.date()} not in price index")
    pos = int(locs[0])
    pad_pos = max(0, pos - lookback)
    return pd.Timestamp(index[pad_pos])


def window_asset_names(
    prices: pd.DataFrame,
    window: SlidingWindow,
) -> list[str]:
    """Tradeable sectors with complete history over train_start..test_end."""
    return tradeable_assets(
        prices,
        window.train_start,
        _inclusive_end(window.test_end),
    )


def make_padded_env_arrays(
    dataset: MarketDataset,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    asset_names: list[str],
    lookback: int = LOOKBACK_DAYS,
) -> EnvArrays:
    """
    Build env panels for ``[start, end_exclusive)`` with ``lookback`` prior days.

    The pad lets ``MarketReplayEnv`` fill the return window from the first
    actionable day at index ``lookback``.
    """
    end_inclusive = _inclusive_end(end_exclusive)
    padded_start = pad_start(dataset.prices.index, start, lookback=lookback)
    return make_env_arrays(
        dataset.prices,
        dataset.log_returns,
        dataset.features,
        start=padded_start,
        end=end_inclusive,
        asset_names=asset_names,
    )


def arrays_for_window(
    dataset: MarketDataset,
    window: SlidingWindow,
    *,
    lookback: int = LOOKBACK_DAYS,
) -> tuple[list[str], EnvArrays, EnvArrays, EnvArrays]:
    """Return asset names and padded train / val / test EnvArrays."""
    assets = window_asset_names(dataset.prices, window)
    if not assets:
        raise ValueError(f"No tradeable assets for window {window.index}")

    train = make_padded_env_arrays(
        dataset, window.train_start, window.train_end, assets, lookback=lookback
    )
    val = make_padded_env_arrays(
        dataset, window.val_start, window.val_end, assets, lookback=lookback
    )
    test = make_padded_env_arrays(
        dataset, window.test_start, window.test_end, assets, lookback=lookback
    )
    return assets, train, val, test
