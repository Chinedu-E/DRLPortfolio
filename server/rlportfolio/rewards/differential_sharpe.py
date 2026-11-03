"""Differential Sharpe Ratio (Moody et al., 1998) as a per-step RL reward."""

from __future__ import annotations

import math


class DifferentialSharpe:
    """
    Online Differential Sharpe Ratio with exponential moving moments.

    Uses the paper's recursion with timescale ``eta`` (default ``1/252``) and
    ``A_0 = B_0 = 0``. Returns ``0.0`` when the denominator is numerically zero.
    """

    def __init__(self, eta: float = 1.0 / 252.0, eps: float = 1e-12) -> None:
        if eta <= 0.0:
            raise ValueError("eta must be positive")
        self.eta = float(eta)
        self.eps = float(eps)
        self.A = 0.0
        self.B = 0.0

    def reset(self) -> None:
        self.A = 0.0
        self.B = 0.0

    def update(self, portfolio_return: float) -> float:
        """Update moments with simple return ``R_t`` and return ``D_t``."""
        r = float(portfolio_return)
        delta_a = r - self.A
        delta_b = r * r - self.B
        denom = self.B - self.A * self.A
        if denom <= self.eps:
            d_t = 0.0
        else:
            d_t = (self.B * delta_a - 0.5 * self.A * delta_b) / math.pow(denom, 1.5)

        self.A = self.A + self.eta * delta_a
        self.B = self.B + self.eta * delta_b
        if not math.isfinite(d_t):
            return 0.0
        return float(d_t)
