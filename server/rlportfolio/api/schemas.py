"""Pydantic models for the dashboard API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
import pandas as pd


MAX_RANGE_DAYS = 365 * 5 + 10  # ~5 calendar years


class BacktestRequest(BaseModel):
    start: str = Field(..., description="ISO date YYYY-MM-DD")
    end: str = Field(..., description="ISO date YYYY-MM-DD")

    @field_validator("start", "end")
    @classmethod
    def _valid_date(cls, value: str) -> str:
        try:
            pd.Timestamp(value)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid date: {value}") from exc
        return value

    @model_validator(mode="after")
    def _check_order_and_span(self) -> BacktestRequest:
        start = pd.Timestamp(self.start)
        end = pd.Timestamp(self.end)
        if end < start:
            raise ValueError("end must be on or after start")
        if (end - start).days > MAX_RANGE_DAYS:
            raise ValueError(
                f"Date range exceeds maximum of ~5 years ({MAX_RANGE_DAYS} days)"
            )
        return self


class MetaResponse(BaseModel):
    min_date: str
    max_date: str
    sectors: list[str]
    windows: list[dict[str, Any]]
    models: list[str]
    presets: dict[str, dict[str, str]]


class HealthResponse(BaseModel):
    status: str
