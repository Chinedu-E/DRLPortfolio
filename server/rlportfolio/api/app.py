"""FastAPI application for the RLPortfolio research dashboard."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from rlportfolio.api.schemas import BacktestRequest, HealthResponse, MetaResponse
from rlportfolio.api.services import service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    service.load()
    yield


app = FastAPI(
    title="RLPortfolio Research API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/api/meta", response_model=MetaResponse)
def meta() -> MetaResponse:
    data = service.meta()
    return MetaResponse(**data)


@app.post("/api/backtest")
def backtest(body: BacktestRequest) -> dict[str, Any]:
    try:
        return service.backtest(body.start, body.end)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
