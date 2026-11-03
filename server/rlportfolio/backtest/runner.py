"""DRL and MVO out-of-sample backtest runners (Section 5.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from rlportfolio.backtest.broker import Trajectory, TrajectoryBuilder, env_from_arrays
from rlportfolio.backtest.metrics import (
    aggregate_window_metrics,
    evaluate_returns,
    metrics_to_frame,
)
from rlportfolio.baselines.mvo import MVOStrategy
from rlportfolio.data_engine import MarketDataset, build_dataset
from rlportfolio.env.panel import EnvArrays
from rlportfolio.windows import arrays_for_window, get_window, sliding_windows

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


@dataclass
class BacktestResult:
    """One strategy backtest on a single test window."""

    window_index: int
    strategy: str  # "drl" | "mvo"
    trajectory: Trajectory
    metrics: dict[str, float]
    seeds_used: list[int] = field(default_factory=list)


def _pad_sector_weights(sector_w: np.ndarray) -> np.ndarray:
    """Append cash=0 so weights length is n+1."""
    w = np.asarray(sector_w, dtype=np.float64).reshape(-1)
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total > 0:
        w = w / total
    return np.concatenate([w, np.array([0.0])])


def run_mvo_backtest(
    arrays: EnvArrays,
    *,
    lookback: int = 60,
    initial_cash: float = 100_000.0,
    window_index: int = 0,
) -> BacktestResult:
    """Execute MVO weights day-by-day on the padded test panel."""
    env = env_from_arrays(arrays, lookback=lookback, initial_cash=initial_cash)
    strategy = MVOStrategy(lookback=lookback)
    builder = TrajectoryBuilder(arrays=arrays)

    obs, info = env.reset()
    terminated = False
    while not terminated:
        t = env.current_index
        sector_w = strategy.weights_on_day(arrays.log_returns, t)
        weights = _pad_sector_weights(sector_w)
        obs, _, terminated, _, info = env.step_weights(weights)
        builder.add(env, info)

    traj = builder.build()
    metrics = evaluate_returns(
        traj.returns, dates=traj.dates, delta_pw_values=traj.delta_pw
    )
    return BacktestResult(
        window_index=window_index,
        strategy="mvo",
        trajectory=traj,
        metrics=metrics,
    )


def discover_window_models(models_dir: Path | str, window_index: int) -> list[Path]:
    """Find ``window{i}_seed*.zip`` checkpoints for a window."""
    root = Path(models_dir)
    pattern = f"window{window_index}_seed*.zip"
    return sorted(root.glob(pattern))


def _run_single_drl(
    arrays: EnvArrays,
    model_path: Path,
    *,
    lookback: int,
    initial_cash: float,
    device: str,
) -> Trajectory:
    env = env_from_arrays(arrays, lookback=lookback, initial_cash=initial_cash)
    model = PPO.load(str(model_path), device=device)
    builder = TrajectoryBuilder(arrays=arrays)

    obs, info = env.reset()
    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, info = env.step(action)
        builder.add(env, info)
    return builder.build()


def _average_trajectories(trajectories: list[Trajectory]) -> Trajectory:
    """Average daily returns / values / Δp_w across seeds (aligned by length)."""
    if len(trajectories) == 1:
        return trajectories[0]
    min_len = min(len(t.returns) for t in trajectories)
    dates = trajectories[0].dates[:min_len]
    returns = np.mean([t.returns[:min_len] for t in trajectories], axis=0)
    values = np.mean([t.portfolio_values[:min_len] for t in trajectories], axis=0)
    delta = np.mean([t.delta_pw[:min_len] for t in trajectories], axis=0)
    # Weights: average across seeds.
    w_dim = trajectories[0].weights.shape[1]
    weights = np.mean(
        [t.weights[:min_len, :w_dim] for t in trajectories], axis=0
    )
    return Trajectory(
        dates=dates,
        portfolio_values=values,
        returns=returns,
        weights=weights,
        delta_pw=delta,
    )


def run_drl_backtest(
    arrays: EnvArrays,
    model_paths: Sequence[Path | str],
    *,
    lookback: int = 60,
    initial_cash: float = 100_000.0,
    window_index: int = 0,
    device: str = "cpu",
) -> BacktestResult:
    """Run deterministic PPO backtest; average across seeds if multiple paths."""
    paths = [Path(p) for p in model_paths]
    if not paths:
        raise FileNotFoundError("No DRL model paths provided")

    trajs = [
        _run_single_drl(
            arrays, p, lookback=lookback, initial_cash=initial_cash, device=device
        )
        for p in paths
    ]
    traj = _average_trajectories(trajs)
    metrics = evaluate_returns(
        traj.returns, dates=traj.dates, delta_pw_values=traj.delta_pw
    )
    seeds: list[int] = []
    for p in paths:
        # window0_seed3.zip -> 3
        stem = p.stem
        if "_seed" in stem:
            try:
                seeds.append(int(stem.split("_seed")[-1]))
            except ValueError:
                pass
    return BacktestResult(
        window_index=window_index,
        strategy="drl",
        trajectory=traj,
        metrics=metrics,
        seeds_used=seeds,
    )


def write_daily_csv(result: BacktestResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"backtest_window{result.window_index}_{result.strategy}.csv"
    df = pd.DataFrame(
        {
            "portfolio_value": result.trajectory.portfolio_values,
            "return": result.trajectory.returns,
            "delta_pw": result.trajectory.delta_pw,
        },
        index=result.trajectory.dates,
    )
    df.index.name = "Date"
    df.to_csv(path)
    return path


def run_backtests(
    *,
    windows: Sequence[int] | None = None,
    dataset: MarketDataset | None = None,
    models_dir: Path | str = MODELS_DIR,
    results_dir: Path | str = RESULTS_DIR,
    lookback: int = 60,
    initial_cash: float = 100_000.0,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Run MVO (always) and DRL (if checkpoints exist) for each requested window.

    Returns a summary DataFrame with per-window rows plus aggregate rows.
    """
    if dataset is None:
        dataset = build_dataset(refresh=False)
    models_dir = Path(models_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if windows is None:
        window_ids = list(range(len(sliding_windows())))
    else:
        window_ids = list(windows)

    summary_rows: list[dict[str, Any]] = []
    mvo_metric_rows: list[dict[str, float]] = []
    drl_metric_rows: list[dict[str, float]] = []

    for wid in window_ids:
        window = get_window(int(wid))
        assets, _, _, test_arr = arrays_for_window(
            dataset, window, lookback=lookback
        )
        print(
            f"Window {window.index} test {window.test_start.date()}->"
            f"{window.test_end.date()} assets={len(assets)}"
        )

        mvo = run_mvo_backtest(
            test_arr,
            lookback=lookback,
            initial_cash=initial_cash,
            window_index=window.index,
        )
        write_daily_csv(mvo, results_dir)
        mvo_row = {
            "window": window.index,
            "strategy": "mvo",
            "test_year": int(window.test_start.year),
            "n_days": len(mvo.trajectory.returns),
            **mvo.metrics,
        }
        summary_rows.append(mvo_row)
        mvo_metric_rows.append(mvo.metrics)
        _print_brief("MVO", mvo.metrics)

        model_paths = discover_window_models(models_dir, window.index)
        if model_paths:
            drl = run_drl_backtest(
                test_arr,
                model_paths,
                lookback=lookback,
                initial_cash=initial_cash,
                window_index=window.index,
                device=device,
            )
            write_daily_csv(drl, results_dir)
            drl_row = {
                "window": window.index,
                "strategy": "drl",
                "test_year": int(window.test_start.year),
                "n_days": len(drl.trajectory.returns),
                "seeds": ",".join(str(s) for s in drl.seeds_used),
                **drl.metrics,
            }
            summary_rows.append(drl_row)
            drl_metric_rows.append(drl.metrics)
            _print_brief("DRL", drl.metrics)
        else:
            print(f"  DRL: skipped (no models matching window{window.index}_seed*.zip)")

    # Aggregate rows.
    if mvo_metric_rows:
        agg_mvo = aggregate_window_metrics(mvo_metric_rows)
        summary_rows.append({"window": "agg", "strategy": "mvo", **agg_mvo})
    if drl_metric_rows:
        agg_drl = aggregate_window_metrics(drl_metric_rows)
        summary_rows.append({"window": "agg", "strategy": "drl", **agg_drl})

    summary = metrics_to_frame(summary_rows)
    summary_path = results_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    return summary


def _print_brief(label: str, metrics: dict[str, float]) -> None:
    sharpe = metrics.get("Sharpe ratio", metrics.get("Sharpe Ratio", float("nan")))
    mdd = metrics.get("Max drawdown", metrics.get("Max Drawdown", float("nan")))
    turn = metrics.get("Mean Δp_w", float("nan"))
    print(f"  {label}: Sharpe={sharpe:.4f}  MaxDD={mdd:.4f}  mean Δp_w={turn:.4f}")
