"""PPO training with Stable-Baselines3 SubprocVecEnv (Section 5.2)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from rlportfolio.env.market_replay import DEFAULT_LOOKBACK, MarketReplayEnv
from rlportfolio.env.panel import EnvArrays

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

PPO_DEFAULTS: dict[str, Any] = {
    "n_steps": 756,
    "batch_size": 1260,
    "n_epochs": 16,
    "gamma": 0.9,
    "gae_lambda": 0.9,
    "clip_range": 0.25,
    "total_timesteps": 7_500_000,
    "n_envs": 10,
    "learning_rate_start": 3e-4,
    "learning_rate_end": 1e-5,
    "net_arch": [64, 64],
    "log_std_init": -1.0,
}


def linear_schedule(start: float, end: float) -> Callable[[float], float]:
    """
    SB3 schedule: ``progress_remaining`` goes from 1 → 0 during training.

    Maps to learning rate ``start`` → ``end``.
    """

    def schedule(progress_remaining: float) -> float:
        return float(end + (start - end) * progress_remaining)

    return schedule


def _env_kwargs_from_arrays(arrays: EnvArrays, lookback: int) -> dict[str, Any]:
    return {
        "prices": np.array(arrays.prices, copy=True),
        "log_returns": np.array(arrays.log_returns, copy=True),
        "vol_20": np.array(arrays.vol_20, copy=True),
        "vol_ratio": np.array(arrays.vol_ratio, copy=True),
        "vix": np.array(arrays.vix, copy=True),
        "lookback": lookback,
        "asset_names": list(arrays.asset_names),
    }


def make_env_fn(arrays: EnvArrays, lookback: int = DEFAULT_LOOKBACK) -> Callable[[], MarketReplayEnv]:
    """Pickle-friendly env constructor for vectorized workers."""
    kwargs = _env_kwargs_from_arrays(arrays, lookback)

    def _init() -> MarketReplayEnv:
        return MarketReplayEnv(**kwargs)

    return _init


def make_subproc_vec_env(
    arrays: EnvArrays,
    n_envs: int = 10,
    lookback: int = DEFAULT_LOOKBACK,
    *,
    use_subproc: bool = True,
) -> VecEnv:
    """Build ``SubprocVecEnv`` (or ``DummyVecEnv`` when ``use_subproc=False``)."""
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    fns = [make_env_fn(arrays, lookback=lookback) for _ in range(n_envs)]
    if use_subproc and n_envs > 1:
        return SubprocVecEnv(fns)
    return DummyVecEnv(fns)


def evaluate_mean_reward(
    model: PPO,
    arrays: EnvArrays,
    *,
    n_episodes: int = 1,
    lookback: int = DEFAULT_LOOKBACK,
    deterministic: bool = False,
) -> float:
    """Average episodic return over ``n_episodes`` on a single env."""
    env = MarketReplayEnv(**_env_kwargs_from_arrays(arrays, lookback))
    totals: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            done = bool(terminated or truncated)
        totals.append(ep_reward)
    return float(np.mean(totals))


@dataclass(frozen=True)
class TrainResult:
    best_model_path: Path
    best_seed: int
    best_val_reward: float
    seed_rewards: dict[int, float]


def train_ppo_window(
    arrays_train: EnvArrays,
    arrays_val: EnvArrays,
    *,
    seeds: Sequence[int] = (0,),
    n_envs: int = PPO_DEFAULTS["n_envs"],
    total_timesteps: int = PPO_DEFAULTS["total_timesteps"],
    lookback: int = DEFAULT_LOOKBACK,
    pretrained_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    window_index: int = 0,
    use_subproc: bool = True,
    device: str = "auto",
) -> TrainResult:
    """
    Train PPO for one sliding window across ``seeds``; keep best by val reward.

    If ``pretrained_path`` is set, each seed loads that policy before ``learn``.
    """
    out_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # For very short smoke runs, shrink rollout / batch so learn() can start.
    n_steps = min(PPO_DEFAULTS["n_steps"], max(64, total_timesteps // max(n_envs, 1)))
    batch_size = min(PPO_DEFAULTS["batch_size"], n_steps * n_envs)
    # batch_size must divide n_steps * n_envs for clean updates; shrink if needed.
    buffer_size = n_steps * n_envs
    while batch_size > 1 and buffer_size % batch_size != 0:
        batch_size //= 2
    batch_size = max(1, batch_size)

    lr_schedule = linear_schedule(
        PPO_DEFAULTS["learning_rate_start"],
        PPO_DEFAULTS["learning_rate_end"],
    )

    seed_rewards: dict[int, float] = {}
    best_seed = int(seeds[0])
    best_val = -np.inf
    best_path = out_dir / f"window{window_index}_seed{best_seed}.zip"

    policy_kwargs = {
        "net_arch": list(PPO_DEFAULTS["net_arch"]),
        "activation_fn": nn.Tanh,
        "log_std_init": float(PPO_DEFAULTS["log_std_init"]),
    }

    for seed in seeds:
        seed = int(seed)
        vec_env = make_subproc_vec_env(
            arrays_train,
            n_envs=n_envs,
            lookback=lookback,
            use_subproc=use_subproc,
        )
        try:
            if pretrained_path is not None:
                model = PPO.load(
                    str(pretrained_path),
                    env=vec_env,
                    device=device,
                )
                # Ensure schedule / hyperparams still apply for continued training.
                model.learning_rate = lr_schedule
            else:
                model = PPO(
                    "MlpPolicy",
                    vec_env,
                    learning_rate=lr_schedule,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=PPO_DEFAULTS["n_epochs"],
                    gamma=PPO_DEFAULTS["gamma"],
                    gae_lambda=PPO_DEFAULTS["gae_lambda"],
                    clip_range=PPO_DEFAULTS["clip_range"],
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                    seed=seed,
                    device=device,
                )

            model.learn(total_timesteps=int(total_timesteps), progress_bar=False)

            ckpt = out_dir / f"window{window_index}_seed{seed}.zip"
            model.save(str(ckpt))

            val_reward = evaluate_mean_reward(
                model, arrays_val, n_episodes=1, lookback=lookback, deterministic=False
            )
            seed_rewards[seed] = val_reward
            if val_reward > best_val:
                best_val = val_reward
                best_seed = seed
                best_path = ckpt
        finally:
            vec_env.close()

    return TrainResult(
        best_model_path=best_path,
        best_seed=best_seed,
        best_val_reward=float(best_val),
        seed_rewards=seed_rewards,
    )


# Silence unused import warning if torch is only needed for CUDA detection elsewhere.
_ = torch
