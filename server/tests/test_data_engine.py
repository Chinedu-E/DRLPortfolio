"""Synthetic unit tests for log returns and SPX volatility features."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rlportfolio.data_engine import build_dataset, log_returns, market_volatility


def test_log_returns_known_ratio() -> None:
    prices = pd.DataFrame(
        {"SPX": [100.0, 110.0, 121.0]},
        index=pd.date_range("2005-01-03", periods=3, freq="B"),
    )
    returns = log_returns(prices)
    assert math.isnan(returns["SPX"].iloc[0])
    assert returns["SPX"].iloc[1] == math.log(1.1)
    assert returns["SPX"].iloc[2] == math.log(1.1)


def test_market_volatility_matches_independent_std() -> None:
    rng = np.random.default_rng(0)
    index = pd.date_range("2005-01-03", periods=120, freq="B")
    returns = pd.Series(rng.normal(0.0, 0.01, size=len(index)), index=index, name="SPX")
    features = market_volatility(returns)

    window = returns.iloc[40:60]
    expected_vol20 = float(window.std(ddof=1))
    assert features["vol_20"].iloc[59] == pytest.approx(expected_vol20)

    window60 = returns.iloc[:60]
    expected_vol60 = float(window60.std(ddof=1))
    assert features["vol_60"].iloc[59] == pytest.approx(expected_vol60)
    assert features["vol_ratio"].iloc[59] == pytest.approx(expected_vol20 / expected_vol60)

    assert features["vol_20"].isna().iloc[18]
    assert not features["vol_20"].isna().iloc[19]
    assert features["vol_60"].isna().iloc[58]
    assert not features["vol_60"].isna().iloc[59]


def test_warmup_trim_yields_finite_vol60(monkeypatch) -> None:
    """build_dataset should keep the first sample date with finite vol_60."""
    n = 320  # ~1 year warmup + sample
    index = pd.date_range("2005-01-03", periods=n, freq="B")
    rng = np.random.default_rng(1)
    shocks = rng.normal(0.0, 0.01, size=n)
    spx = 1000.0 * np.exp(np.cumsum(shocks))
    prices = pd.DataFrame(
        {
            "Energy": spx * 0.9,
            "Materials": spx * 0.91,
            "Industrials": spx * 0.92,
            "ConsumerDiscretionary": spx * 0.93,
            "ConsumerStaples": spx * 0.94,
            "HealthCare": spx * 0.95,
            "Financials": spx * 0.96,
            "InformationTechnology": spx * 0.97,
            "CommunicationServices": spx * 0.98,
            "Utilities": spx * 0.99,
            "RealEstate": spx * 1.01,
            "SPX": spx,
            "VIX": 15.0 + rng.normal(0.0, 0.5, size=n),
        },
        index=index,
    )
    sources = {col: col for col in prices.columns}

    def fake_load(start, end_ts, refresh):
        return prices.loc[prices.index <= end_ts], sources

    monkeypatch.setattr(
        "rlportfolio.data_engine._load_or_download_warmup", fake_load
    )

    dataset = build_dataset(start="2006-01-01", end="2006-06-30", refresh=False)
    assert dataset.prices.index.min() >= pd.Timestamp("2006-01-01")
    first = dataset.features.iloc[0]
    assert np.isfinite(first["vol_20"])
    assert np.isfinite(first["vol_60"])
    assert np.isfinite(first["vol_ratio"])
    assert first["vol_ratio"] == first["vol_20"] / first["vol_60"]
