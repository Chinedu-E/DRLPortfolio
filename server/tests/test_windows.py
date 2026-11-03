"""Tests for sliding window calendar."""

from __future__ import annotations

import pandas as pd

from rlportfolio.windows import get_window, sliding_windows


def test_sliding_windows_count_and_bounds() -> None:
    windows = sliding_windows()
    assert len(windows) == 10

    w0 = windows[0]
    assert w0.train_start == pd.Timestamp("2006-01-01")
    assert w0.train_end == pd.Timestamp("2011-01-01")
    assert w0.val_start == pd.Timestamp("2011-01-01")
    assert w0.val_end == pd.Timestamp("2012-01-01")
    assert w0.test_start == pd.Timestamp("2012-01-01")
    assert w0.test_end == pd.Timestamp("2013-01-01")

    w9 = windows[9]
    assert w9.train_start == pd.Timestamp("2015-01-01")
    assert w9.train_end == pd.Timestamp("2020-01-01")
    assert w9.val_start == pd.Timestamp("2020-01-01")
    assert w9.test_start == pd.Timestamp("2021-01-01")
    assert w9.test_end == pd.Timestamp("2022-01-01")


def test_get_window() -> None:
    assert get_window(3).index == 3
