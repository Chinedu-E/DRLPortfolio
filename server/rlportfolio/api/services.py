"""Dataset cache, LRU, and backtest orchestration for the API."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any

from rlportfolio.backtest.range_backtest import (
    RangeBacktestPayload,
    dataset_meta,
    run_range_backtest,
)
from rlportfolio.backtest.runner import MODELS_DIR
from rlportfolio.data_engine import MarketDataset, build_dataset

_MAX_CACHE = 32


class DashboardService:
    """Process-wide dataset holder with a small backtest response LRU."""

    def __init__(self, models_dir: Path | str | None = None) -> None:
        self.models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
        self._dataset: MarketDataset | None = None
        self._lock = Lock()
        self._lru: OrderedDict[tuple[str, str, float], dict[str, Any]] = OrderedDict()

    def load(self) -> None:
        with self._lock:
            if self._dataset is None:
                self._dataset = build_dataset(refresh=False)

    @property
    def dataset(self) -> MarketDataset:
        if self._dataset is None:
            self.load()
        assert self._dataset is not None
        return self._dataset

    def meta(self) -> dict[str, Any]:
        return dataset_meta(self.dataset, models_dir=self.models_dir)

    def _models_mtime(self) -> float:
        root = self.models_dir
        if not root.exists():
            return 0.0
        mtimes = [p.stat().st_mtime for p in root.glob("window*_seed*.zip")]
        return max(mtimes) if mtimes else 0.0

    def backtest(self, start: str, end: str) -> dict[str, Any]:
        key = (start, end, self._models_mtime())
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)
                return self._lru[key]

        payload: RangeBacktestPayload = run_range_backtest(
            self.dataset,
            start,
            end,
            models_dir=self.models_dir,
        )
        data = payload.to_dict()

        with self._lock:
            self._lru[key] = data
            self._lru.move_to_end(key)
            while len(self._lru) > _MAX_CACHE:
                self._lru.popitem(last=False)
        return data


service = DashboardService()
