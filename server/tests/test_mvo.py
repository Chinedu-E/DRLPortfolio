"""Tests for Ledoit–Wolf MVO baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlportfolio.baselines.mvo import MVOStrategy, equal_weights, psd_repair


def test_psd_repair_zeros_negative_eigenvalues() -> None:
    # Symmetric matrix with a negative eigenvalue.
    cov = np.array([[1.0, 0.9], [0.9, 0.5]], dtype=np.float64)
    eigvals_before = np.linalg.eigvalsh(cov)
    assert eigvals_before.min() < 0

    repaired = psd_repair(cov)
    eigvals = np.linalg.eigvalsh(repaired)
    assert eigvals.min() >= -1e-10
    assert repaired.shape == cov.shape
    assert np.allclose(repaired, repaired.T)


def test_weights_on_day_in_simplex() -> None:
    rng = np.random.default_rng(0)
    n_days, n_assets = 120, 2
    # Asset 0 drifts up; asset 1 drifts down → expect tilt toward asset 0.
    shocks = rng.normal(0.0, 0.01, size=(n_days, n_assets))
    shocks[:, 0] += 0.002
    shocks[:, 1] -= 0.001
    log_returns = shocks

    strategy = MVOStrategy(lookback=60)
    w = strategy.weights_on_day(log_returns, t_index=80)
    assert w.shape == (2,)
    assert np.all(w >= -1e-9)
    assert w.sum() == pytest.approx(1.0)


def test_early_day_equal_weight() -> None:
    panel = np.zeros((10, 3))
    strategy = MVOStrategy(lookback=60)
    w = strategy.weights_on_day(panel, t_index=5)
    assert w == pytest.approx(equal_weights(3))


def test_optimize_failure_returns_equal_weight(monkeypatch) -> None:
    strategy = MVOStrategy(lookback=5)
    window = np.ones((5, 2)) * 0.01

    def boom(*args, **kwargs):
        raise RuntimeError("solver failed")

    monkeypatch.setattr("rlportfolio.baselines.mvo.EfficientFrontier", boom)
    w = strategy._optimize(window)
    assert w == pytest.approx(equal_weights(2))


def test_generate_weight_path() -> None:
    idx = pd.date_range("2012-01-01", periods=80, freq="B")
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(80, 2)),
        index=idx,
        columns=["A", "B"],
    )
    path = MVOStrategy(lookback=60).generate_weight_path(rets)
    assert path.shape == (80, 2)
    assert path.iloc[-1].sum() == pytest.approx(1.0)
