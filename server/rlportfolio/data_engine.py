"""Market data engine: Yahoo Finance pulls and Section 5.1 volatility features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_START = "2006-01-01"
WARMUP_START = "2005-01-01"
SPX_TICKER = "^GSPC"
VIX_TICKER = "^VIX"

# Logical sector name -> (preferred GICS index ticker, SPDR ETF fallback).
SECTORS: dict[str, tuple[str, str]] = {
    "Energy": ("^GSPE", "XLE"),
    "Materials": ("^SP500-15", "XLB"),
    "Industrials": ("^SP500-20", "XLI"),
    "ConsumerDiscretionary": ("^SP500-25", "XLY"),
    "ConsumerStaples": ("^SP500-30", "XLP"),
    "HealthCare": ("^SP500-35", "XLV"),
    "Financials": ("^SP500-40", "XLF"),
    "InformationTechnology": ("^SP500-45", "XLK"),
    "CommunicationServices": ("^SP500-50", "XLC"),
    "Utilities": ("^SP500-55", "XLU"),
    "RealEstate": ("^SP500-60", "XLRE"),
}

MARKET: dict[str, str] = {
    "SPX": SPX_TICKER,
    "VIX": VIX_TICKER,
}

LOGICAL_COLUMNS: list[str] = [*SECTORS.keys(), "SPX", "VIX"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICE_CACHE_PATH = DATA_DIR / "adjusted_close.csv"
SOURCES_CACHE_PATH = DATA_DIR / "sources.csv"

# Prefer at least this many daily bars before treating a GICS index as usable.
_MIN_HISTORY_BARS = 60


@dataclass(frozen=True)
class MarketDataset:
    """Prices, log returns, SPX volatility features, and Yahoo source map."""

    prices: pd.DataFrame
    log_returns: pd.DataFrame
    features: pd.DataFrame
    sources: dict[str, str]


def _normalize_end(end: str | pd.Timestamp | None) -> pd.Timestamp:
    if end is None:
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(end).normalize()


def _extract_adj_close(raw: pd.DataFrame | pd.Series, ticker: str) -> pd.Series:
    """Pull Adj Close for one ticker; fall back to Close if Adj Close is missing."""
    if isinstance(raw, pd.Series):
        series = raw.copy()
        series.name = ticker
        return series.astype(float)

    if isinstance(raw.columns, pd.MultiIndex):
        fields = raw.columns.get_level_values(0)
        if "Adj Close" in fields:
            series = raw["Adj Close"]
        elif "Close" in fields:
            series = raw["Close"]
        else:
            raise KeyError(f"No Adj Close/Close column for {ticker}")
        if isinstance(series, pd.DataFrame):
            if ticker in series.columns:
                series = series[ticker]
            else:
                series = series.iloc[:, 0]
    else:
        if "Adj Close" in raw.columns:
            series = raw["Adj Close"]
        elif "Close" in raw.columns:
            series = raw["Close"]
        else:
            raise KeyError(f"No Adj Close/Close column for {ticker}")

    series = series.copy()
    series.name = ticker
    return series.astype(float).dropna(how="all")


def _download_symbol(ticker: str, start: str, end: str | None) -> pd.Series:
    """Download one symbol's adjusted close from Yahoo Finance."""
    # yfinance end is exclusive for daily bars; bump by one day when provided.
    download_end = None
    if end is not None:
        download_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    raw = yf.download(
        ticker,
        start=start,
        end=download_end,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or len(raw) == 0:
        return pd.Series(dtype=float, name=ticker)

    return _extract_adj_close(raw, ticker)


def _has_usable_history(series: pd.Series, sample_start: pd.Timestamp) -> bool:
    """True when the series has enough non-null bars at or after the sample start."""
    if series.empty:
        return False
    in_sample = series.loc[series.index >= sample_start].dropna()
    return len(in_sample) >= _MIN_HISTORY_BARS


def resolve_sector_sources(
    start: str = DEFAULT_START,
    end: str | None = None,
) -> dict[str, str]:
    """Prefer each GICS index; fall back to the matching SPDR ETF if needed."""
    sample_start = pd.Timestamp(start)
    end_ts = _normalize_end(end)
    end_str = end_ts.strftime("%Y-%m-%d")

    sources: dict[str, str] = {}
    for name, (index_ticker, etf_ticker) in SECTORS.items():
        index_series = _download_symbol(index_ticker, WARMUP_START, end_str)
        if _has_usable_history(index_series, sample_start):
            sources[name] = index_ticker
        else:
            sources[name] = etf_ticker
    sources["SPX"] = MARKET["SPX"]
    sources["VIX"] = MARKET["VIX"]
    return sources


def _cache_covers(
    cached: pd.DataFrame,
    warmup_start: pd.Timestamp,
    end: pd.Timestamp,
    columns: list[str],
) -> bool:
    if cached.empty or not set(columns).issubset(cached.columns):
        return False
    idx = pd.DatetimeIndex(cached.index)
    if idx.empty:
        return False
    return idx.min() <= warmup_start and idx.max() >= end - pd.Timedelta(days=5)


def _load_cache() -> tuple[pd.DataFrame | None, dict[str, str] | None]:
    if not PRICE_CACHE_PATH.exists() or not SOURCES_CACHE_PATH.exists():
        return None, None
    prices = pd.read_csv(PRICE_CACHE_PATH, index_col=0, parse_dates=True)
    sources_df = pd.read_csv(SOURCES_CACHE_PATH)
    sources = dict(zip(sources_df["column"], sources_df["ticker"], strict=True))
    return prices, sources


def _save_cache(prices: pd.DataFrame, sources: Mapping[str, str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_csv(PRICE_CACHE_PATH)
    pd.DataFrame(
        {"column": list(sources.keys()), "ticker": list(sources.values())}
    ).to_csv(SOURCES_CACHE_PATH, index=False)


def _download_warmup(
    start: str,
    end_ts: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Download and cache the warmup-inclusive price panel aligned to SPX sessions."""
    end_str = end_ts.strftime("%Y-%m-%d")
    sources = resolve_sector_sources(start=start, end=end_str)
    series_map: dict[str, pd.Series] = {}

    for name in SECTORS:
        series = _download_symbol(sources[name], WARMUP_START, end_str)
        series.name = name
        series_map[name] = series

    for name, ticker in MARKET.items():
        series = _download_symbol(ticker, WARMUP_START, end_str)
        series.name = name
        series_map[name] = series

    if series_map["SPX"].empty:
        raise RuntimeError(
            "Failed to download S&P 500 (^GSPC) history from Yahoo Finance"
        )

    spx_index = series_map["SPX"].dropna().index
    prices_warmup = pd.DataFrame(
        {name: series_map[name].reindex(spx_index) for name in LOGICAL_COLUMNS},
        index=spx_index,
    )
    prices_warmup.index.name = "Date"
    _save_cache(prices_warmup, sources)
    return prices_warmup, sources


def _load_or_download_warmup(
    start: str,
    end_ts: pd.Timestamp,
    refresh: bool,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if not refresh:
        cached_prices, cached_sources = _load_cache()
        if (
            cached_prices is not None
            and cached_sources is not None
            and _cache_covers(
                cached_prices, pd.Timestamp(WARMUP_START), end_ts, LOGICAL_COLUMNS
            )
        ):
            prices_warmup = cached_prices.loc[
                (cached_prices.index >= pd.Timestamp(WARMUP_START))
                & (cached_prices.index <= end_ts),
                LOGICAL_COLUMNS,
            ]
            return prices_warmup, dict(cached_sources)
    return _download_warmup(start, end_ts)


def fetch_adjusted_close(
    start: str = DEFAULT_START,
    end: str | None = None,
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Pull daily adjusted closes for the 11 sectors, SPX, and VIX.

    Downloads from ``WARMUP_START`` so rolling windows are warm by ``start``.
    The returned frame is trimmed to ``[start, end]`` and indexed to SPX sessions.
    """
    sample_start = pd.Timestamp(start)
    end_ts = _normalize_end(end)
    prices_warmup, sources = _load_or_download_warmup(start, end_ts, refresh)
    prices = prices_warmup.loc[
        (prices_warmup.index >= sample_start) & (prices_warmup.index <= end_ts)
    ]
    return prices, sources


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns ``r_t = ln(P_t / P_{t-1})`` for every price column."""
    return np.log(prices / prices.shift(1))


def market_volatility(spx_log_returns: pd.Series) -> pd.DataFrame:
    """
    Rolling SPX volatility features from Section 5.1.

    ``vol_20`` / ``vol_60`` are sample standard deviations (ddof=1) of daily
    S&P 500 log returns. ``vol_ratio = vol_20 / vol_60``.
    """
    vol_20 = spx_log_returns.rolling(window=20, min_periods=20).std(ddof=1)
    vol_60 = spx_log_returns.rolling(window=60, min_periods=60).std(ddof=1)
    vol_ratio = vol_20 / vol_60
    return pd.DataFrame(
        {"vol_20": vol_20, "vol_60": vol_60, "vol_ratio": vol_ratio},
        index=spx_log_returns.index,
    )


def build_dataset(
    start: str = DEFAULT_START,
    end: str | None = None,
    refresh: bool = False,
) -> MarketDataset:
    """
    Build prices, log returns, and SPX volatility features for ``[start, end]``.

    Volatility windows are computed on the warmup history before trimming so
    the first kept session has finite ``vol_60``.
    """
    sample_start = pd.Timestamp(start)
    end_ts = _normalize_end(end)
    prices_warmup, sources = _load_or_download_warmup(start, end_ts, refresh)

    returns_warmup = log_returns(prices_warmup)
    features_warmup = market_volatility(returns_warmup["SPX"])

    mask = (prices_warmup.index >= sample_start) & (prices_warmup.index <= end_ts)
    return MarketDataset(
        prices=prices_warmup.loc[mask],
        log_returns=returns_warmup.loc[mask],
        features=features_warmup.loc[mask],
        sources=sources,
    )
