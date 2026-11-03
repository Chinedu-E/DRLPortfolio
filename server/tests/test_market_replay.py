"""Synthetic tests for MarketReplayEnv broker mechanics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlportfolio.env.market_replay import MarketReplayEnv, softmax
from rlportfolio.env.panel import make_env_arrays, tradeable_assets


def _synthetic_panels(
    n_days: int = 80,
    n_assets: int = 3,
    lookback: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    # Constant then a known +10% jump on one day for MTM checks.
    prices = np.full((n_days, n_assets), 100.0, dtype=np.float64)
    for i in range(n_assets):
        prices[:, i] = 100.0 + i
    log_returns = np.zeros_like(prices)
    log_returns[1:] = np.log(prices[1:] / prices[:-1])
    vol_20 = np.full(n_days, 0.01, dtype=np.float64)
    vol_ratio = np.full(n_days, 1.0, dtype=np.float64)
    vix = 15.0 + rng.normal(0.0, 0.1, size=n_days)
    assert n_days >= lookback + 2
    return prices, log_returns, vol_20, vol_ratio, vix


def test_softmax_long_only_fully_invested() -> None:
    w = softmax(np.array([1.0, 2.0, -3.0, 0.5]))
    assert w.shape == (4,)
    assert np.all(w >= 0.0)
    assert w.sum() == pytest.approx(1.0)


def test_rebalance_whole_shares_and_cash_residual() -> None:
    prices, log_returns, vol_20, vol_ratio, vix = _synthetic_panels()
    env = MarketReplayEnv(
        prices, log_returns, vol_20, vol_ratio, vix, lookback=60, initial_cash=10_000.0
    )
    obs, info = env.reset()
    assert info["weights"][-1] == pytest.approx(1.0)

    # Strong preference for first asset only (logits).
    action = np.array([10.0, -10.0, -10.0, -10.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert np.all(info["weights"] >= -1e-9)
    assert info["weights"].sum() == pytest.approx(1.0)
    # Whole shares only.
    assert np.all(env.shares == np.floor(env.shares))
    # Cash residual non-negative and absorbs leftovers.
    assert env.cash >= 0.0
    px = prices[env.current_index]
    assert env.cash + float(np.dot(env.shares, px)) == pytest.approx(
        env.portfolio_value, rel=1e-9
    )


def test_mark_to_market_after_price_jump() -> None:
    prices, log_returns, vol_20, vol_ratio, vix = _synthetic_panels(n_days=80)
    # After reset at t=60, step rebalances at t=60 then advances to t=61.
    # Make day 61 prices +10% vs day 60 for asset 0; hold others flat.
    prices[61, 0] = prices[60, 0] * 1.10
    log_returns[61, 0] = np.log(1.10)

    env = MarketReplayEnv(
        prices, log_returns, vol_20, vol_ratio, vix, lookback=60, initial_cash=10_000.0
    )
    env.reset()
    # Allocate ~100% to asset 0.
    action = np.array([20.0, -20.0, -20.0, -20.0], dtype=np.float32)
    _, _, _, _, info = env.step(action)

    # Hand calculation: shares bought at t=60 price, valued at t=61.
    px60 = prices[60]
    port_before = 10_000.0
    target = softmax(action)
    shares0 = np.floor(target[0] * port_before / px60[0])
    cash = port_before - shares0 * px60[0]
    expected = shares0 * prices[61, 0] + cash
    assert info["portfolio_value"] == pytest.approx(expected)
    assert info["simple_return"] == pytest.approx(expected / port_before - 1.0)


def test_episode_terminates_on_last_day() -> None:
    lookback = 5
    n_days = lookback + 3  # start at 5; steps at 5->6, 6->7 then terminate
    prices, log_returns, vol_20, vol_ratio, vix = _synthetic_panels(
        n_days=n_days, lookback=lookback
    )
    env = MarketReplayEnv(
        prices, log_returns, vol_20, vol_ratio, vix, lookback=lookback, initial_cash=5_000.0
    )
    env.reset()
    terminated = False
    steps = 0
    while not terminated:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert not truncated
        steps += 1
        assert steps < 10
    assert steps == n_days - lookback - 1
    assert env.current_index == n_days - 1


def test_observation_shape_and_weight_column() -> None:
    lookback = 60
    prices, log_returns, vol_20, vol_ratio, vix = _synthetic_panels(lookback=lookback)
    env = MarketReplayEnv(
        prices, log_returns, vol_20, vol_ratio, vix, lookback=lookback, initial_cash=10_000.0
    )
    obs, info = env.reset()
    n_slots = 4
    assert obs.shape == (n_slots * lookback,)
    matrix = obs.reshape(n_slots, lookback)
    assert matrix[:, 0] == pytest.approx(info["weights"].astype(np.float32))

    action = np.ones(n_slots, dtype=np.float32)
    obs2, _, _, _, info2 = env.step(action)
    matrix2 = obs2.reshape(n_slots, lookback)
    assert matrix2[:, 0] == pytest.approx(info2["weights"].astype(np.float32))


def test_tradeable_assets_drops_nan_sector() -> None:
    idx = pd.date_range("2010-01-01", periods=10, freq="B")
    prices = pd.DataFrame(
        {
            "Energy": np.linspace(50, 55, 10),
            "CommunicationServices": [np.nan] * 5 + list(np.linspace(40, 45, 5)),
            "VIX": np.full(10, 15.0),
        },
        index=idx,
    )
    keep = tradeable_assets(
        prices, "2010-01-01", "2010-01-15", candidates=["Energy", "CommunicationServices"]
    )
    assert keep == ["Energy"]


def test_make_env_arrays_builds_numpy_panels() -> None:
    idx = pd.date_range("2010-01-01", periods=70, freq="B")
    prices = pd.DataFrame(
        {
            "Energy": np.linspace(50, 60, 70),
            "Materials": np.linspace(40, 50, 70),
            "SPX": np.linspace(1000, 1100, 70),
            "VIX": np.full(70, 16.0),
        },
        index=idx,
    )
    log_returns = np.log(prices / prices.shift(1))
    features = pd.DataFrame(
        {
            "vol_20": np.full(70, 0.01),
            "vol_60": np.full(70, 0.012),
            "vol_ratio": np.full(70, 0.01 / 0.012),
        },
        index=idx,
    )
    arrays = make_env_arrays(
        prices,
        log_returns,
        features,
        start="2010-01-01",
        end=idx[-1],
        asset_names=["Energy", "Materials"],
    )
    assert arrays.prices.shape == (70, 2)
    assert arrays.asset_names == ["Energy", "Materials"]
    assert arrays.vix.shape == (70,)
