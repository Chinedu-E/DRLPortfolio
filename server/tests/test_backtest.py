"""Synthetic tests for backtest broker, metrics, and runners."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rlportfolio.agents.ppo_train import train_ppo_window
from rlportfolio.backtest.broker import env_from_arrays
from rlportfolio.backtest.metrics import (
    aggregate_window_metrics,
    delta_pw,
    evaluate_returns,
)
from rlportfolio.backtest.runner import run_drl_backtest, run_mvo_backtest
from rlportfolio.env.panel import EnvArrays


def _synthetic_arrays(n_days: int = 100, n_assets: int = 2, seed: int = 0) -> EnvArrays:
    rng = np.random.default_rng(seed)
    prices = 50.0 + np.cumsum(rng.normal(0.0, 0.3, size=(n_days, n_assets)), axis=0)
    prices = np.maximum(prices, 5.0)
    log_returns = np.zeros_like(prices)
    log_returns[1:] = np.log(prices[1:] / prices[:-1])
    return EnvArrays(
        asset_names=[f"a{i}" for i in range(n_assets)],
        prices=prices,
        log_returns=log_returns,
        vol_20=np.full(n_days, 0.01),
        vol_ratio=np.full(n_days, 1.0),
        vix=np.full(n_days, 15.0),
        dates=pd.date_range("2012-01-01", periods=n_days, freq="B"),
    )


def test_step_weights_whole_shares() -> None:
    arrays = _synthetic_arrays()
    env = env_from_arrays(arrays, lookback=60, initial_cash=10_000.0)
    env.reset()
    t = env.current_index
    px = arrays.prices[t]
    # 100% first asset.
    target = np.array([1.0, 0.0, 0.0])
    _, _, _, _, info = env.step_weights(target)
    shares0 = np.floor(10_000.0 / px[0])
    cash = 10_000.0 - shares0 * px[0]
    # After step, portfolio is marked at t+1.
    expected = shares0 * arrays.prices[t + 1, 0] + cash
    assert info["portfolio_value"] == pytest.approx(expected)
    assert env.shares[0] == shares0
    assert env.shares[1] == 0.0


def test_delta_pw_full_rotation() -> None:
    # All in A (cash trailing), then all in B.
    w0 = np.array([1.0, 0.0, 0.0])
    w1 = np.array([0.0, 1.0, 0.0])
    assert delta_pw(w0, w1) == pytest.approx(2.0)


def test_evaluate_returns_constant() -> None:
    # Constant positive daily return -> positive Sharpe / annual return.
    n = 252
    dates = pd.date_range("2012-01-01", periods=n, freq="B")
    r = pd.Series(np.full(n, 0.001), index=dates)
    metrics = evaluate_returns(r, delta_pw_values=np.zeros(n))
    assert "Sharpe ratio" in metrics or "Sharpe Ratio" in metrics
    sharpe_key = "Sharpe ratio" if "Sharpe ratio" in metrics else "Sharpe Ratio"
    assert metrics[sharpe_key] > 0
    assert metrics["Mean Δp_w"] == pytest.approx(0.0)
    ann_key = "Annual return" if "Annual return" in metrics else None
    if ann_key:
        # (1.001**252 - 1) ≈ 0.286
        assert metrics[ann_key] == pytest.approx(1.001**252 - 1, rel=0.05)


def test_aggregate_max_drawdown_is_worst() -> None:
    rows = [
        {"Sharpe ratio": 1.0, "Max drawdown": -0.1},
        {"Sharpe ratio": 0.5, "Max drawdown": -0.3},
    ]
    agg = aggregate_window_metrics(rows)
    assert agg["Sharpe ratio"] == pytest.approx(0.75)
    assert agg["Max drawdown"] == pytest.approx(-0.3)


def test_mvo_backtest_runs() -> None:
    arrays = _synthetic_arrays(n_days=100)
    result = run_mvo_backtest(arrays, lookback=60, initial_cash=50_000.0, window_index=0)
    assert len(result.trajectory.returns) > 0
    assert np.isfinite(result.trajectory.returns).all()
    assert "Mean Δp_w" in result.metrics
    assert result.strategy == "mvo"


def test_drl_backtest_with_tiny_model(tmp_path: Path) -> None:
    train = _synthetic_arrays(n_days=100, seed=0)
    val = _synthetic_arrays(n_days=100, seed=1)
    train_result = train_ppo_window(
        train,
        val,
        seeds=[0],
        n_envs=1,
        total_timesteps=512,
        lookback=60,
        models_dir=tmp_path,
        window_index=0,
        use_subproc=False,
        device="cpu",
    )
    test = _synthetic_arrays(n_days=100, seed=2)
    result = run_drl_backtest(
        test,
        [train_result.best_model_path],
        lookback=60,
        initial_cash=50_000.0,
        window_index=0,
        device="cpu",
    )
    sharpe_key = "Sharpe ratio" if "Sharpe ratio" in result.metrics else "Sharpe Ratio"
    assert np.isfinite(result.metrics[sharpe_key])
    assert result.strategy == "drl"
