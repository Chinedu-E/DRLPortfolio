"""Shared helpers for recording backtest trajectories from MarketReplayEnv."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rlportfolio.env.market_replay import MarketReplayEnv
from rlportfolio.env.panel import EnvArrays


@dataclass
class Trajectory:
    """Daily path produced by a backtest episode."""

    dates: pd.DatetimeIndex
    portfolio_values: np.ndarray
    returns: np.ndarray
    weights: np.ndarray  # (T_steps, n_slots) rebalanced weights after each step
    delta_pw: np.ndarray  # length T_steps; first entry 0


def env_from_arrays(
    arrays: EnvArrays,
    *,
    lookback: int = 60,
    initial_cash: float = 100_000.0,
) -> MarketReplayEnv:
    return MarketReplayEnv(
        arrays.prices,
        arrays.log_returns,
        arrays.vol_20,
        arrays.vol_ratio,
        arrays.vix,
        lookback=lookback,
        initial_cash=initial_cash,
        asset_names=list(arrays.asset_names),
    )


def record_step(
    *,
    dates: list[pd.Timestamp],
    values: list[float],
    returns: list[float],
    weights: list[np.ndarray],
    env: MarketReplayEnv,
    info: dict,
    arrays: EnvArrays,
) -> None:
    """Append one post-step observation to the trajectory buffers."""
    dates.append(pd.Timestamp(arrays.dates[info["t"]]))
    values.append(float(info["portfolio_value"]))
    returns.append(float(info["simple_return"]))
    weights.append(np.asarray(info["weights"], dtype=np.float64))


def finalize_trajectory(
    dates: list[pd.Timestamp],
    values: list[float],
    returns: list[float],
    weights: list[np.ndarray],
) -> Trajectory:
    from rlportfolio.backtest.metrics import delta_pw_series

    w = np.vstack(weights) if weights else np.zeros((0, 0))
    return Trajectory(
        dates=pd.DatetimeIndex(dates),
        portfolio_values=np.asarray(values, dtype=np.float64),
        returns=np.asarray(returns, dtype=np.float64),
        weights=w,
        delta_pw=delta_pw_series(w),
    )


@dataclass
class TrajectoryBuilder:
    """Convenience accumulator used by the runners."""

    arrays: EnvArrays
    dates: list[pd.Timestamp] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    weights: list[np.ndarray] = field(default_factory=list)

    def add(self, env: MarketReplayEnv, info: dict) -> None:
        record_step(
            dates=self.dates,
            values=self.values,
            returns=self.returns,
            weights=self.weights,
            env=env,
            info=info,
            arrays=self.arrays,
        )

    def build(self) -> Trajectory:
        return finalize_trajectory(self.dates, self.values, self.returns, self.weights)
