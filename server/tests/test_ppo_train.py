"""Tests for PPO helpers and a tiny learn() smoke run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rlportfolio.agents.ppo_train import (
    linear_schedule,
    make_subproc_vec_env,
    train_ppo_window,
)
from rlportfolio.env.panel import EnvArrays


def _synthetic_arrays(n_days: int = 90, n_assets: int = 2) -> EnvArrays:
    rng = np.random.default_rng(0)
    prices = 100.0 + np.cumsum(rng.normal(0.0, 0.5, size=(n_days, n_assets)), axis=0)
    prices = np.maximum(prices, 1.0)
    log_returns = np.zeros_like(prices)
    log_returns[1:] = np.log(prices[1:] / prices[:-1])
    return EnvArrays(
        asset_names=[f"a{i}" for i in range(n_assets)],
        prices=prices,
        log_returns=log_returns,
        vol_20=np.full(n_days, 0.01),
        vol_ratio=np.full(n_days, 1.0),
        vix=np.full(n_days, 15.0),
        dates=pd_date_index(n_days),
    )


def pd_date_index(n_days: int):
    import pandas as pd

    return pd.date_range("2010-01-01", periods=n_days, freq="B")


def test_linear_schedule_endpoints() -> None:
    sched = linear_schedule(3e-4, 1e-5)
    assert sched(1.0) == pytest.approx(3e-4)
    assert sched(0.0) == pytest.approx(1e-5)
    assert sched(0.5) == pytest.approx(0.5 * (3e-4 + 1e-5))


def test_dummy_vec_env_reset() -> None:
    arrays = _synthetic_arrays()
    vec = make_subproc_vec_env(arrays, n_envs=2, lookback=60, use_subproc=False)
    try:
        obs = vec.reset()
        assert obs.shape[0] == 2
        assert obs.shape[1] == (2 + 1) * 60
    finally:
        vec.close()


def test_tiny_ppo_learn(tmp_path: Path) -> None:
    train = _synthetic_arrays(n_days=90)
    val = _synthetic_arrays(n_days=90)
    result = train_ppo_window(
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
    assert result.best_model_path.exists()
    assert 0 in result.seed_rewards
    assert np.isfinite(result.best_val_reward)
