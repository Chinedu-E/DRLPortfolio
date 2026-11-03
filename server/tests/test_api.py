"""API tests for the research dashboard FastAPI app."""

from __future__ import annotations

from fastapi.testclient import TestClient

from rlportfolio.api.app import app


def test_health() -> None:
    with TestClient(app) as client:
        res = client.get("/api/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"


def test_meta() -> None:
    with TestClient(app) as client:
        res = client.get("/api/meta")
        assert res.status_code == 200
        data = res.json()
        assert "min_date" in data and "max_date" in data
        assert "presets" in data
        assert "2012" in data["presets"]


def test_backtest_2012_window() -> None:
    with TestClient(app) as client:
        res = client.post(
            "/api/backtest",
            json={"start": "2012-01-01", "end": "2012-06-30"},
        )
        assert res.status_code == 200, res.text
        data = res.json()
        assert len(data["dates"]) > 20
        assert len(data["equity"]["mvo"]) == len(data["dates"])
        assert len(data["equity"]["spx"]) == len(data["dates"])
        assert "ppo" in data["equity"]
        assert data["weight_columns"]
        assert len(data["weights"]) == len(data["dates"])
        assert "mvo" in data["metrics"]


def test_backtest_invalid_range() -> None:
    with TestClient(app) as client:
        res = client.post(
            "/api/backtest",
            json={"start": "2020-01-01", "end": "2019-01-01"},
        )
        assert res.status_code == 422
