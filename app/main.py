"""Training API service."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import monitor, store
from .config import Settings
from .db import init_db, reader
from .features import InputError

settings = Settings.from_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate()
    init_db(settings.db_path)
    yield


app = FastAPI(
    title="Fan Shutdown Early-Warning Training API",
    description=(
        "Submit 8-64 fan vibration/current records and train a logistic "
        "regression with batch size 8, learning rate 0.1 and momentum 0.9.\n\n"
        "All training is executed by workers. Every batch commits weights, "
        "momentum, loss and progress in one atomic checkpoint; after a failure "
        "the next worker resumes from the newest committed checkpoint."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class TrainingRequest(BaseModel):
    features: list[list[float]] = Field(
        ...,
        description=(
            "8-64 records; each record is a fan vibration/current feature vector. "
            "Columns are standardised automatically."
        ),
        examples=[[[0.12, 3.4], [0.15, 3.5], [0.62, 5.1], [0.71, 5.4]]],
    )
    labels: list[int] = Field(
        ...,
        description="One shutdown label per record: 0 = healthy, 1 = shutdown warning.",
        examples=[[0, 0, 1, 1]],
    )
    epochs: int = Field(2, ge=2, le=5, description="Training epochs (2-5).")


class BaselineRequest(BaseModel):
    job_id: str = Field(
        ...,
        description=(
            "Job to use as the drift-monitoring baseline. Must be completed and "
            "carry raw-sample column statistics (jobs created before drift "
            "monitoring existed are refused)."
        ),
    )


class ReadingsRequest(BaseModel):
    report_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description=(
            "Client-chosen idempotency key. A retry with the same key returns "
            "the first response and is never counted twice."
        ),
        examples=["shift-2026-11-17-report-0001"],
    )
    readings: list[list[float]] = Field(
        ...,
        min_length=1,
        max_length=4096,
        description=(
            "Raw vibration/current measurements, one vector per reading, in "
            "arrival order. Every 16 readings form one non-overlapping window."
        ),
        examples=[[[0.12, 3.4], [0.15, 3.5]]],
    )


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    with reader(settings.db_path) as conn:
        conn.execute("SELECT 1")
    return {"status": "ok", "service": "api", "database": str(settings.db_path)}


@app.post("/jobs", status_code=201, tags=["training"])
def submit_job(request: TrainingRequest) -> dict[str, Any]:
    """Validate and enqueue a training job; it is picked up by a worker."""
    try:
        return store.create_job(
            settings.db_path, request.features, request.labels, request.epochs
        )
    except InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/jobs", tags=["training"])
def get_jobs(limit: int = Query(50, ge=1, le=200)) -> list[dict[str, Any]]:
    return store.list_jobs(settings.db_path, limit)


@app.get("/jobs/{job_id}", tags=["training"])
def get_job(job_id: str) -> dict[str, Any]:
    """Confirmed progress, loss, and the latest weights/momentum."""
    try:
        job = store.get_job(settings.db_path, job_id)
    except store.StateConflict as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/jobs/{job_id}/checkpoints", tags=["training"])
def get_checkpoints(
    job_id: str,
    after_step: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=1000),
) -> list[dict[str, Any]]:
    """Batch checkpoints in step order, each with weights, momentum and loss."""
    try:
        return store.list_checkpoints(settings.db_path, job_id, after_step, limit)
    except store.JobNotFound as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc


@app.get("/jobs/{job_id}/events", tags=["training"])
def get_events(job_id: str) -> list[dict[str, Any]]:
    """Audit trail: creation, takeover/resume points and completion."""
    try:
        return store.list_events(settings.db_path, job_id)
    except store.JobNotFound as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc


@app.get("/workers", tags=["system"])
def get_workers() -> dict[str, Any]:
    """Worker liveness and current lease state, for the demo."""
    now = time.time()
    with reader(settings.db_path) as conn:
        workers = conn.execute(
            "SELECT worker_id, status, started_at, heartbeat_at FROM workers ORDER BY worker_id"
        ).fetchall()
        leases = conn.execute(
            "SELECT job_id, worker_id, epoch, heartbeat_at FROM leases ORDER BY job_id"
        ).fetchall()
    return {
        "lease_seconds": settings.lease_seconds,
        "workers": [
            {
                "worker_id": row["worker_id"],
                "status": row["status"],
                "started_at": row["started_at"],
                "heartbeat_at": row["heartbeat_at"],
                "seconds_since_heartbeat": round(max(0.0, now - row["heartbeat_at"]), 3),
                "alive": now - row["heartbeat_at"] < settings.lease_seconds,
            }
            for row in workers
        ],
        "leases": [
            {
                "job_id": row["job_id"],
                "worker_id": row["worker_id"],
                "lease_epoch": int(row["epoch"]),
                "expired": now - float(row["heartbeat_at"]) >= settings.lease_seconds,
                "seconds_since_heartbeat": round(max(0.0, now - row["heartbeat_at"]), 3),
            }
            for row in leases
        ],
    }


@app.post("/monitor/baseline", tags=["monitor"])
def set_monitor_baseline(request: BaselineRequest) -> dict[str, Any]:
    """Set the drift-monitoring baseline to a completed job.

    The job must have finished training and carry the raw-sample column
    statistics saved at submission.  Switching to a different baseline
    discards the partially filled window; already evaluated windows keep
    pointing at the baseline that formed them.
    """
    try:
        return monitor.set_baseline(settings.db_path, request.job_id)
    except store.JobNotFound as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    except monitor.BaselineNotEligible as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/monitor/status", tags=["monitor"])
def monitor_status() -> dict[str, Any]:
    """Current baseline and how full the not-yet-evaluated window is."""
    return monitor.get_status(settings.db_path)


@app.post("/monitor/readings", tags=["monitor"])
def submit_monitor_readings(request: ReadingsRequest) -> dict[str, Any]:
    """Report raw measurements, one vector per reading.

    Every 16 readings form one non-overlapping window.  When a window fills,
    its per-column means are compared with the baseline: a column alarms when
    the mean moved by more than two baseline standard deviations, and a
    baseline-constant column alarms on any differing reading.  The response
    lists, per alarmed window, the alarmed columns, their offsets and the
    baseline job id.  Retrying with the same ``report_id`` never counts twice.
    """
    try:
        return monitor.submit_readings(
            settings.db_path, request.report_id, request.readings
        )
    except monitor.NoBaseline as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except monitor.BadReading as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/monitor/windows", tags=["monitor"])
def monitor_windows(limit: int = Query(100, ge=1, le=1000)) -> list[dict[str, Any]]:
    """Evaluated windows, newest first; each keeps the baseline that formed it."""
    return monitor.list_windows(settings.db_path, limit)
