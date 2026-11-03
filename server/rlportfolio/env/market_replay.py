"""Gymnasium market-replay environment (broker + simulator)."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlportfolio.rewards.differential_sharpe import DifferentialSharpe

DEFAULT_LOOKBACK = 60
DEFAULT_INITIAL_CASH = 100_000.0


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


class MarketReplayEnv(gym.Env):
    """
    Market replay Gymnasium environment.

    Softmaxes continuous actions into long-only, fully invested weights, marks
    the portfolio to market, rebalances in whole shares with a cash residual,
    and rewards the Differential Sharpe Ratio of the resulting simple return.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: np.ndarray,
        log_returns: np.ndarray,
        vol_20: np.ndarray,
        vol_ratio: np.ndarray,
        vix: np.ndarray,
        *,
        lookback: int = DEFAULT_LOOKBACK,
        initial_cash: float = DEFAULT_INITIAL_CASH,
        eta: float = 1.0 / 252.0,
        asset_names: list[str] | None = None,
    ) -> None:
        super().__init__()

        prices = np.asarray(prices, dtype=np.float64)
        log_returns = np.asarray(log_returns, dtype=np.float64)
        vol_20 = np.asarray(vol_20, dtype=np.float64)
        vol_ratio = np.asarray(vol_ratio, dtype=np.float64)
        vix = np.asarray(vix, dtype=np.float64)

        if prices.ndim != 2:
            raise ValueError("prices must be shaped (T_days, n_assets)")
        n_days, n_assets = prices.shape
        if log_returns.shape != prices.shape:
            raise ValueError("log_returns must match prices shape")
        for name, arr in (
            ("vol_20", vol_20),
            ("vol_ratio", vol_ratio),
            ("vix", vix),
        ):
            if arr.shape != (n_days,):
                raise ValueError(f"{name} must be shaped (T_days,)")

        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        # Need lookback prior days for the return window and one future day for PnL.
        if n_days < lookback + 2:
            raise ValueError(
                f"need at least lookback+2={lookback + 2} days, got {n_days}"
            )

        self.prices = prices
        self.log_returns = np.nan_to_num(log_returns, nan=0.0)
        self.vol_20 = vol_20
        self.vol_ratio = vol_ratio
        self.vix = vix
        self.lookback = int(lookback)
        self.initial_cash = float(initial_cash)
        self.n_assets = n_assets
        self.n_slots = n_assets + 1  # sectors + cash
        self.asset_names = asset_names or [f"asset_{i}" for i in range(n_assets)]

        self._dsr = DifferentialSharpe(eta=eta)
        self._t = self.lookback
        self._shares = np.zeros(n_assets, dtype=np.float64)
        self._cash = self.initial_cash
        self._weights = np.zeros(self.n_slots, dtype=np.float64)
        self._weights[-1] = 1.0
        self._port_val = self.initial_cash

        # Finite bounds required by Stable-Baselines3; softmax still maps to the simplex.
        self.action_space = spaces.Box(
            low=-20.0,
            high=20.0,
            shape=(self.n_slots,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_slots * self.lookback,),
            dtype=np.float32,
        )

    @property
    def portfolio_value(self) -> float:
        return float(self._port_val)

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def shares(self) -> np.ndarray:
        return self._shares.copy()

    @property
    def cash(self) -> float:
        return float(self._cash)

    @property
    def current_index(self) -> int:
        return int(self._t)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._dsr.reset()
        self._t = self.lookback
        self._shares = np.zeros(self.n_assets, dtype=np.float64)
        self._cash = self.initial_cash
        self._weights = np.zeros(self.n_slots, dtype=np.float64)
        self._weights[-1] = 1.0
        self._port_val = self._mark_to_market(self._t)
        obs = self._build_observation()
        info = {
            "portfolio_value": self._port_val,
            "weights": self._weights.copy(),
            "t": self._t,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        target = softmax(np.asarray(action, dtype=np.float64).reshape(-1))
        if target.shape != (self.n_slots,):
            raise ValueError(
                f"action must have shape ({self.n_slots},), got {np.shape(action)}"
            )
        return self._step_with_target_weights(target)

    def step_weights(
        self, weights: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Rebalance using an explicit simplex weight vector (no softmax).

        ``weights`` must have length ``n_assets + 1`` (sectors + cash), non-negative,
        and sum to 1 (within a small tolerance; renormalized if needed).
        """
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape != (self.n_slots,):
            raise ValueError(
                f"weights must have shape ({self.n_slots},), got {w.shape}"
            )
        w = np.clip(w, 0.0, None)
        total = float(w.sum())
        if total <= 1e-12:
            raise ValueError("weights must sum to a positive value")
        w = w / total
        return self._step_with_target_weights(w)

    def _step_with_target_weights(
        self, target: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._t >= len(self.prices) - 1:
            raise RuntimeError("episode already terminated; call reset()")

        port_val_t = self._mark_to_market(self._t)
        self._rebalance(target, port_val_t)

        # Realize next-day mark-to-market return.
        self._t += 1
        port_val_next = self._mark_to_market(self._t)
        simple_return = port_val_next / port_val_t - 1.0
        self._port_val = port_val_next

        reward = float(self._dsr.update(simple_return))
        terminated = self._t >= len(self.prices) - 1
        truncated = False
        obs = self._build_observation()
        info = {
            "portfolio_value": self._port_val,
            "weights": self._weights.copy(),
            "simple_return": float(simple_return),
            "target_weights": target.copy(),
            "t": self._t,
        }
        return obs, reward, terminated, truncated, info

    def _mark_to_market(self, t: int) -> float:
        px = self.prices[t]
        return float(np.dot(self._shares, px) + self._cash)

    def _rebalance(self, weights: np.ndarray, port_val: float) -> None:
        """Allocate ``port_val`` with whole shares; residual stays in cash."""
        px = self.prices[self._t]
        shares = np.zeros(self.n_assets, dtype=np.float64)
        for i in range(self.n_assets):
            if px[i] <= 0.0:
                continue
            notional = weights[i] * port_val
            shares[i] = np.floor(notional / px[i])

        invested = float(np.dot(shares, px))
        cash = port_val - invested
        if cash < -1e-6:
            # Numerical guard: drop shares until non-negative cash.
            order = np.argsort(-px)  # sell expensive first
            for i in order:
                while shares[i] > 0 and cash < 0:
                    shares[i] -= 1.0
                    cash += px[i]

        self._shares = shares
        self._cash = max(cash, 0.0)
        holdings_value = float(np.dot(self._shares, px) + self._cash)
        if holdings_value <= 0.0:
            self._weights = np.zeros(self.n_slots, dtype=np.float64)
            self._weights[-1] = 1.0
            return

        w = np.zeros(self.n_slots, dtype=np.float64)
        w[: self.n_assets] = (self._shares * px) / holdings_value
        w[-1] = self._cash / holdings_value
        # Renormalize tiny float drift.
        w = np.clip(w, 0.0, None)
        w_sum = w.sum()
        self._weights = w / w_sum if w_sum > 0 else w

    def _expanding_z(self, series: np.ndarray, t: int) -> float:
        hist = series[: t + 1]
        hist = hist[np.isfinite(hist)]
        if hist.size < 2:
            return 0.0
        mu = float(hist.mean())
        sigma = float(hist.std(ddof=1))
        if sigma <= 1e-12:
            return 0.0
        return float((series[t] - mu) / sigma)

    def _build_observation(self) -> np.ndarray:
        t = self._t
        obs = np.zeros((self.n_slots, self.lookback), dtype=np.float32)

        # Column 0: current portfolio weights.
        obs[:, 0] = self._weights.astype(np.float32)

        # Asset rows: past lookback-1 log returns ending at t-1.
        for i in range(self.n_assets):
            for lag in range(1, self.lookback):
                # lag=1 -> r_{t-1}, ..., lag=lookback-1 -> r_{t-(lookback-1)}
                obs[i, lag] = self.log_returns[t - lag, i]

        # Cash / market-regime row.
        obs[-1, 0] = np.float32(self._weights[-1])
        obs[-1, 1] = np.float32(self._expanding_z(self.vol_20, t))
        obs[-1, 2] = np.float32(self._expanding_z(self.vol_ratio, t))
        obs[-1, 3] = np.float32(self._expanding_z(self.vix, t))
        # Remaining entries stay 0.

        return obs.reshape(-1)
