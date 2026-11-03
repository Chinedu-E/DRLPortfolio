"""CLI: data smoke, PPO training, MVO baseline, and backtests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rlportfolio.agents.ppo_train import train_ppo_window
from rlportfolio.backtest.runner import run_backtests
from rlportfolio.baselines.mvo import MVOStrategy
from rlportfolio.data_engine import build_dataset
from rlportfolio.env import MarketReplayEnv, make_env_arrays, tradeable_assets
from rlportfolio.windows import arrays_for_window, get_window, pad_start


def cmd_data(_: argparse.Namespace) -> None:
    dataset = build_dataset(refresh=False)
    prices = dataset.prices
    features = dataset.features
    print(f"Date span: {prices.index.min().date()} -> {prices.index.max().date()}")
    print(f"Rows: {len(prices)}")
    print("Sources:")
    for name, ticker in dataset.sources.items():
        print(f"  {name}: {ticker}")
    print("Null counts (prices):")
    print(prices.isna().sum().to_string())

    start, end = "2019-01-01", "2019-06-30"
    assets = tradeable_assets(prices, start, end)
    arrays = make_env_arrays(
        prices,
        dataset.log_returns,
        features,
        start=start,
        end=end,
        asset_names=assets,
    )
    env = MarketReplayEnv(
        arrays.prices,
        arrays.log_returns,
        arrays.vol_20,
        arrays.vol_ratio,
        arrays.vix,
        asset_names=arrays.asset_names,
    )
    obs, info = env.reset(seed=0)
    print(f"Env assets={len(assets)} reset_value={info['portfolio_value']:.2f} obs={obs.shape}")


def cmd_train(args: argparse.Namespace) -> None:
    dataset = build_dataset(refresh=False)
    window = get_window(args.window)
    assets, train_arr, val_arr, _ = arrays_for_window(dataset, window)
    print(
        f"Window {window.index}: train {window.train_start.date()}->"
        f"{window.train_end.date()} val {window.val_start.date()} "
        f"assets={assets}"
    )

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    result = train_ppo_window(
        train_arr,
        val_arr,
        seeds=seeds,
        n_envs=args.n_envs,
        total_timesteps=args.timesteps,
        pretrained_path=args.pretrained,
        models_dir=args.models_dir,
        window_index=window.index,
        use_subproc=not args.dummy_vec,
        device=args.device,
    )
    print(f"Best seed={result.best_seed} val_reward={result.best_val_reward:.6f}")
    print(f"Saved: {result.best_model_path}")
    print(f"Seed rewards: {result.seed_rewards}")


def cmd_mvo(args: argparse.Namespace) -> None:
    dataset = build_dataset(refresh=False)
    year = int(args.year)
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    padded = pad_start(dataset.prices.index, start, lookback=60)
    assets = tradeable_assets(dataset.prices, padded, end)
    if not assets:
        raise SystemExit(f"No tradeable assets for year {year}")

    rets = dataset.log_returns.loc[
        (dataset.log_returns.index >= padded) & (dataset.log_returns.index <= end),
        assets,
    ]
    strategy = MVOStrategy(lookback=60)
    weights = strategy.generate_weight_path(rets)

    year_mask = (weights.index >= start) & (weights.index <= end)
    year_w = weights.loc[year_mask]
    print(f"MVO year={year} assets={assets} days={len(year_w)}")
    print("First row:")
    print(year_w.head(1).to_string())
    print("Last row:")
    print(year_w.tail(1).to_string())

    diffs = year_w.diff().abs().sum(axis=1).dropna()
    print(f"Mean daily |Δw| (all assets): {float(diffs.mean()):.4f}")
    print(f"Weight sum check (last): {float(year_w.iloc[-1].sum()):.6f}")


def _parse_windows(spec: str) -> list[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(10))
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def cmd_backtest(args: argparse.Namespace) -> None:
    windows = _parse_windows(args.windows)
    summary = run_backtests(
        windows=windows,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        device=args.device,
    )
    cols = [
        c
        for c in (
            "window",
            "strategy",
            "test_year",
            "Sharpe ratio",
            "Max drawdown",
            "Mean Δp_w",
            "Annual return",
        )
        if c in summary.columns
    ]
    print("\nSummary (key columns):")
    print(summary[cols].to_string(index=False))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run(
        "rlportfolio.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RL portfolio allocation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="Data engine + env smoke")
    p_data.set_defaults(func=cmd_data)

    p_train = sub.add_parser("train", help="Train PPO on a sliding window")
    p_train.add_argument("--window", type=int, default=0)
    p_train.add_argument("--timesteps", type=int, default=4096)
    p_train.add_argument("--n-envs", type=int, default=2)
    p_train.add_argument("--seeds", type=str, default="0")
    p_train.add_argument("--pretrained", type=str, default=None)
    p_train.add_argument("--models-dir", type=str, default=str(Path("models")))
    p_train.add_argument(
        "--dummy-vec",
        action="store_true",
        help="Use DummyVecEnv instead of SubprocVecEnv",
    )
    p_train.add_argument("--device", type=str, default="cpu")
    p_train.set_defaults(func=cmd_train)

    p_mvo = sub.add_parser("mvo", help="Run MVO baseline for a calendar year")
    p_mvo.add_argument("--year", type=int, default=2012)
    p_mvo.set_defaults(func=cmd_mvo)

    p_bt = sub.add_parser("backtest", help="OOS backtests for DRL and/or MVO")
    p_bt.add_argument(
        "--windows",
        type=str,
        default="0",
        help="Comma list of window indices, or 'all'",
    )
    p_bt.add_argument("--models-dir", type=str, default=str(Path("models")))
    p_bt.add_argument("--results-dir", type=str, default=str(Path("results")))
    p_bt.add_argument("--device", type=str, default="cpu")
    p_bt.set_defaults(func=cmd_backtest)

    p_serve = sub.add_parser("serve", help="Run FastAPI research dashboard API")
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
