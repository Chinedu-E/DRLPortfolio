"""Date-range backtests for the research dashboard API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from rlportfolio.backtest.metrics import (
    evaluate_returns,
    normalize_equity,
    regime_drawdown_bands,
    rolling_sharpe,
    spx_buy_and_hold_equity,
)
from rlportfolio.backtest.runner import (
    MODELS_DIR,
    discover_window_models,
    run_drl_backtest,
    run_mvo_backtest,
)
from rlportfolio.data_engine import MarketDataset, SECTORS
from rlportfolio.env.panel import tradeable_assets
from rlportfolio.windows import LOOKBACK_DAYS, make_padded_env_arrays, sliding_windows


@dataclass
class RangeBacktestPayload:
    """JSON-serializable dashboard payload for one date range."""

    start: str
    end: str
    asset_names: list[str]
    dates: list[str]
    equity: dict[str, list[float | None]]
    rolling_sharpe: dict[str, list[float | None]]
    weights: list[dict[str, float]]
    weight_columns: list[str]
    metrics: dict[str, dict[str, float]]
    regime_hints: list[dict[str, str]]
    drl_status: str
    model_paths: list[str] = field(default_factory=list)
    window_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "asset_names": self.asset_names,
            "dates": self.dates,
            "equity": self.equity,
            "rolling_sharpe": self.rolling_sharpe,
            "weights": self.weights,
            "weight_columns": self.weight_columns,
            "metrics": self.metrics,
            "regime_hints": self.regime_hints,
            "drl_status": self.drl_status,
            "model_paths": self.model_paths,
            "window_index": self.window_index,
        }


def _window_for_test_year(year: int) -> int | None:
    for w in sliding_windows():
        if int(w.test_start.year) == year:
            return w.index
    return None


def _model_action_dim(path: Path) -> int | None:
    try:
        model = PPO.load(str(path), device="cpu")
        space = model.action_space
        shape = getattr(space, "shape", None)
        if shape is None or len(shape) == 0:
            return None
        return int(shape[0])
    except Exception:
        return None


def select_models_for_range(
    *,
    end: pd.Timestamp,
    n_assets: int,
    models_dir: Path | str = MODELS_DIR,
) -> tuple[list[Path], int | None, str]:
    """
    Prefer checkpoints for the paper window whose test year matches ``end.year``.

    Otherwise pick the latest window_* zip whose action dim equals ``n_assets + 1``.
    """
    root = Path(models_dir)
    n_slots = n_assets + 1
    preferred = _window_for_test_year(int(end.year))
    if preferred is not None:
        paths = discover_window_models(root, preferred)
        if paths:
            return paths, preferred, "ok"

    # Fallback: newest window whose action dimension matches n_assets+1.
    all_zips = sorted(root.glob("window*_seed*.zip"), key=lambda p: p.stat().st_mtime)
    by_window: dict[int, list[Path]] = {}
    for path in all_zips:
        stem = path.stem  # window0_seed0
        try:
            wid = int(stem.split("_seed")[0].replace("window", ""))
        except ValueError:
            continue
        by_window.setdefault(wid, []).append(path)

    for wid in sorted(by_window.keys(), reverse=True):
        paths = sorted(by_window[wid])
        dim = _model_action_dim(paths[0])
        if dim == n_slots:
            return paths, wid, "ok"

    return [], None, "unavailable"


def run_range_backtest(
    dataset: MarketDataset,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    models_dir: Path | str = MODELS_DIR,
    lookback: int = LOOKBACK_DAYS,
    initial_cash: float = 100_000.0,
    device: str = "cpu",
    rolling_window: int = 63,
) -> RangeBacktestPayload:
    """Run MVO + optional DRL + SPX buy-and-hold over ``[start, end]`` inclusive."""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if end_ts < start_ts:
        raise ValueError("end must be on or after start")

    # Inclusive end -> exclusive bound for pad helper.
    end_exclusive = end_ts + pd.Timedelta(days=1)
    assets = tradeable_assets(dataset.prices, start_ts, end_ts)
    if not assets:
        raise ValueError("No tradeable assets in the selected range")

    arrays = make_padded_env_arrays(
        dataset, start_ts, end_exclusive, assets, lookback=lookback
    )

    mvo = run_mvo_backtest(
        arrays,
        lookback=lookback,
        initial_cash=initial_cash,
        window_index=-1,
    )

    model_paths, window_index, drl_status = select_models_for_range(
        end=end_ts, n_assets=len(assets), models_dir=models_dir
    )
    drl_traj = None
    drl_metrics: dict[str, float] = {}
    if model_paths and drl_status == "ok":
        try:
            drl = run_drl_backtest(
                arrays,
                model_paths,
                lookback=lookback,
                initial_cash=initial_cash,
                window_index=window_index if window_index is not None else -1,
                device=device,
            )
            drl_traj = drl.trajectory
            drl_metrics = drl.metrics
        except Exception:
            drl_status = "unavailable"
            model_paths = []
            window_index = None

    # Align SPX to the MVO trajectory dates (post-lookback trading days).
    dates = mvo.trajectory.dates
    spx_slice = dataset.prices.loc[dates, "SPX"].to_numpy(dtype=np.float64)
    _, spx_eq = spx_buy_and_hold_equity(spx_slice, dates=dates)

    mvo_eq = normalize_equity(mvo.trajectory.portfolio_values)
    equity: dict[str, list[float | None]] = {
        "mvo": [float(x) for x in mvo_eq],
        "spx": [float(x) for x in spx_eq],
        "ppo": [],
    }
    roll: dict[str, list[float | None]] = {
        "mvo": _nan_to_none(rolling_sharpe(mvo.trajectory.returns, window=rolling_window)),
        "ppo": [],
    }

    weight_source = mvo.trajectory
    if drl_traj is not None:
        # Align lengths.
        n = min(len(dates), len(drl_traj.dates), len(drl_traj.portfolio_values))
        dates = dates[:n]
        equity["mvo"] = equity["mvo"][:n]
        equity["spx"] = equity["spx"][:n]
        roll["mvo"] = roll["mvo"][:n]
        ppo_eq = normalize_equity(drl_traj.portfolio_values[:n])
        equity["ppo"] = [float(x) for x in ppo_eq]
        roll["ppo"] = _nan_to_none(
            rolling_sharpe(drl_traj.returns[:n], window=rolling_window)
        )
        weight_source = TrajectorySlice(drl_traj, n)
    else:
        equity["ppo"] = [None] * len(dates)
        roll["ppo"] = [None] * len(dates)

    weight_columns = [*assets, "Cash"]
    weights_rows = _weights_as_rows(weight_source.weights, weight_columns)

    regime = regime_drawdown_bands(spx_slice[: len(dates)], dates)

    metrics = {"mvo": mvo.metrics, "ppo": drl_metrics, "spx": {}}
    # Simple SPX metrics from buy-and-hold returns.
    if len(spx_slice) > 1:
        spx_rets = np.diff(spx_slice[: len(dates)]) / spx_slice[: len(dates) - 1]
        spx_dates = dates[1:]
        try:
            metrics["spx"] = evaluate_returns(spx_rets, dates=spx_dates)
        except Exception:
            metrics["spx"] = {}

    return RangeBacktestPayload(
        start=str(start_ts.date()),
        end=str(end_ts.date()),
        asset_names=list(assets),
        dates=[str(pd.Timestamp(d).date()) for d in dates],
        equity=equity,
        rolling_sharpe=roll,
        weights=weights_rows,
        weight_columns=weight_columns,
        metrics=metrics,
        regime_hints=regime,
        drl_status=drl_status,
        model_paths=[str(p) for p in model_paths],
        window_index=window_index,
    )


@dataclass
class TrajectorySlice:
    weights: np.ndarray

    def __init__(self, traj: Any, n: int) -> None:
        self.weights = traj.weights[:n]


def _nan_to_none(arr: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for x in arr:
        if x is None or not np.isfinite(x):
            out.append(None)
        else:
            out.append(float(x))
    return out


def _weights_as_rows(
    weights: np.ndarray, columns: list[str]
) -> list[dict[str, float]]:
    w = np.asarray(weights, dtype=np.float64)
    rows: list[dict[str, float]] = []
    if w.ndim != 2 or w.shape[0] == 0:
        return rows
    n_cols = min(w.shape[1], len(columns))
    for i in range(w.shape[0]):
        rows.append({columns[j]: float(w[i, j]) for j in range(n_cols)})
    return rows


def dataset_meta(dataset: MarketDataset, models_dir: Path | str = MODELS_DIR) -> dict[str, Any]:
    """Metadata for the dashboard bootstrap."""
    prices = dataset.prices
    models_dir = Path(models_dir)
    available = sorted(p.name for p in models_dir.glob("window*_seed*.zip"))
    windows = [
        {
            "index": w.index,
            "train_start": str(w.train_start.date()),
            "val_start": str(w.val_start.date()),
            "test_start": str(w.test_start.date()),
            "test_end": str((w.test_end - pd.Timedelta(days=1)).date()),
        }
        for w in sliding_windows()
    ]
    return {
        "min_date": str(prices.index.min().date()),
        "max_date": str(prices.index.max().date()),
        "sectors": list(SECTORS.keys()),
        "windows": windows,
        "models": available,
        "presets": {
            "2012": {"start": "2012-01-01", "end": "2012-12-31"},
            "2020": {"start": "2020-01-01", "end": "2020-12-31"},
            # Capped to ~5y for interactive PPO rollouts (API enforces the same).
            "Full OOS": {"start": "2017-01-01", "end": "2021-12-31"},
        },
    }
