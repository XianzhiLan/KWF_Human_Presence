"""BirdNET bridge — FastAPI service.

Endpoint shape deliberately mirrors the sidecar REST API in Daan's AI Backend
design doc (2026-05-27), so that client code written against this interim bridge
carries over to the pack framework unchanged:

    POST   /jobs                  submit an analysis job
    GET    /jobs/{id}             status + results
    GET    /jobs/{id}/stream      live progress (SSE)
    DELETE /jobs/{id}             cancel
    GET    /health                readiness
    GET    /models                loaded variants  (sidecar's /packs, scoped to one pack)

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    open http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
from contextlib import asynccontextmanager
from pathlib import Path

import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse

from .jobs import JobManager
from .registry import Registry, RegistryError
from .schemas import (
    HealthResponse,
    JobRequest,
    JobState,
    ModelInfo,
    sse,
)

logging.basicConfig(
    level=os.getenv("BRIDGE_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("bridge")

CONFIG_PATH = Path(os.getenv("BRIDGE_MODELS_CONFIG", "models.json"))
# Determinism is the default, not an opt-in: a bridge that returns different
# numbers run-to-run on identical input is a worse failure than a slow one.
# Raise either for a throughput benchmark, deliberately, via the env var.
INTRA_OP_THREADS = int(os.getenv("BRIDGE_INTRA_OP_THREADS", "1"))
INTER_OP_THREADS = int(os.getenv("BRIDGE_INTER_OP_THREADS", "1"))

registry = Registry()
manager: JobManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load every variant once, at startup, behind the gates in registry.py.

    A gate failure raises here and the service does not start. That is the point:
    a mispaired model and label file must be impossible to serve, not merely
    unlikely.
    """
    global manager
    registry.load_from_config(CONFIG_PATH, INTRA_OP_THREADS, INTER_OP_THREADS)
    manager = JobManager(registry)
    log.info("bridge ready with variants: %s", sorted(registry.models))
    yield
    registry.clear()


app = FastAPI(
    title="BirdNET Bridge",
    description=(
        "Benchmarking and inference service for the pruned BirdNET ONNX models. "
        "Endpoint shape follows the DeepData Portal sidecar contract."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _mgr() -> JobManager:
    if manager is None:
        raise HTTPException(503, "Service still starting.")
    return manager


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """The first thing anyone does with a local server is open the root."""
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        ready=bool(registry.models),
        variants_loaded=sorted(registry.models),
        onnxruntime_version=ort.__version__,
        providers=ort.get_available_providers(),
    )


@app.get("/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    return [
        ModelInfo(
            variant=m.variant,
            model_path=str(m.model_path),
            labels_path=str(m.labels_path),
            num_classes=m.num_classes,
            file_mb=m.file_mb,
            loaded=True,
            allow_list_size=m.allow_list_size(),
        )
        for m in registry.models.values()
    ]


@app.post("/jobs", status_code=202)
def submit_job(request: JobRequest) -> dict:
    try:
        job = _mgr().submit(request)
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RegistryError as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"job_id": job.job_id, "status": job.status.value}


@app.get("/jobs/{job_id}", response_model=JobState)
def job_status(job_id: str) -> JobState:
    job = _mgr().get(job_id)
    if job is None:
        raise HTTPException(404, f"No job {job_id}")
    return job.to_state()


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str) -> dict:
    if not _mgr().cancel(job_id):
        raise HTTPException(409, "Job is not cancellable (unknown or already finished).")
    return {"job_id": job_id, "status": "cancelling"}


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str) -> StreamingResponse:
    """Server-Sent Events: progress / result / done / error.

    Event type names match the provisional pack-contract event shapes.
    """
    job = _mgr().get(job_id)
    if job is None:
        raise HTTPException(404, f"No job {job_id}")

    async def gen():
        q = job.subscribe()
        try:
            # Replay current state so a late subscriber is not left guessing.
            yield sse("progress", {"job_id": job.job_id,
                                   "completed": job.completed, "total": job.total})
            while True:
                try:
                    event_type, payload = await asyncio.to_thread(q.get, True, 15.0)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    if job.status.value in ("done", "error", "cancelled"):
                        return
                    continue
                yield sse(event_type, payload)
                if event_type in ("done", "error"):
                    return
        finally:
            job.unsubscribe(q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
