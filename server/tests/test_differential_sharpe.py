"""Tests for Differential Sharpe Ratio reward."""

from __future__ import annotations

import math

import pytest

from rlportfolio.rewards.differential_sharpe import DifferentialSharpe


def _independent_dsr(returns: list[float], eta: float = 1.0 / 252.0) -> list[float]:
    a = 0.0
    b = 0.0
    out: list[float] = []
    for r in returns:
        delta_a = r - a
        delta_b = r * r - b
        denom = b - a * a
        if denom <= 1e-12:
            d_t = 0.0
        else:
            d_t = (b * delta_a - 0.5 * a * delta_b) / math.pow(denom, 1.5)
        a = a + eta * delta_a
        b = b + eta * delta_b
        out.append(0.0 if not math.isfinite(d_t) else d_t)
    return out


def test_first_steps_are_stable_zero() -> None:
    dsr = DifferentialSharpe()
    assert dsr.update(0.01) == 0.0  # A=B=0 => denom 0
    # After one update, moments are non-zero; subsequent steps should be finite.
    second = dsr.update(-0.005)
    assert math.isfinite(second)


def test_matches_independent_implementation() -> None:
    returns = [0.01, -0.005, 0.002, 0.004, -0.001, 0.003]
    dsr = DifferentialSharpe(eta=1.0 / 252.0)
    got = [dsr.update(r) for r in returns]
    expected = _independent_dsr(returns)
    for g, e in zip(got, expected, strict=True):
        assert g == pytest.approx(e)


def test_reset_clears_state() -> None:
    dsr = DifferentialSharpe()
    dsr.update(0.02)
    dsr.update(-0.01)
    dsr.reset()
    assert dsr.A == 0.0
    assert dsr.B == 0.0
    assert dsr.update(0.02) == 0.0
