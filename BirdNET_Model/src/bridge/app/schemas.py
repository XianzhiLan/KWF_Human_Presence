"""Request/response schemas for the BirdNET bridge.

Event type names (progress / result / done / error) are snake_case to match the
provisional pack-contract event shapes, so this service's stream can be consumed
by the same client code that will later consume Daan's sidecar.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"
    cancelled = "cancelled"


class Detection(BaseModel):
    """One species detection above threshold for one 3-second window.

    Field names and types mirror the output_schema column vocabulary in Daan's
    AI Backend doc (TEXT / REAL) so this maps 1:1 onto a pack output table.
    """

    data_id: str                       # source file identifier          (display: hidden)
    species: str                       # label string from the paired labels file (display: label)
    confidence: float                  # post-flat_sigmoid score          (display: percentage)
    start_time: float                  # window start within source file  (display: seconds)
    end_time: float                    # window end within source file    (display: seconds)
    in_allow_list: bool = True         # False when the GeoModel allow-list is loaded and
                                       # this species falls outside it


class ClipTiming(BaseModel):
    """Per-window instrumentation. This is the point of the service."""

    data_id: str
    window_index: int
    preprocess_ms: float
    inference_ms: float


class JobMetrics(BaseModel):
    """Aggregate benchmark result for one job on one model variant."""

    variant: str
    clips: int
    windows: int
    wall_seconds: float
    windows_per_second: float

    inference_ms_mean: float
    inference_ms_p50: float
    inference_ms_p95: float
    inference_ms_max: float
    preprocess_ms_mean: float

    process_cpu_seconds: float
    peak_rss_mb: float | None = None      # None when psutil is unavailable
    model_file_mb: float
    intra_op_num_threads: int
    inter_op_num_threads: int


class JobRequest(BaseModel):
    paths: list[str] = Field(
        ...,
        description="Absolute paths to WAV files, or to directories which will be "
                    "scanned non-recursively for *.wav.",
        min_length=1,
    )
    variant: str = Field(
        "fp16_pruned493",
        description="Key from models.json. Load two variants and submit two jobs "
                    "to produce a matched A/B benchmark.",
    )
    min_confidence: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Detection threshold applied after the internal floor.",
    )
    apply_allow_list: bool = Field(
        True,
        description="Mark detections outside the 218-species GeoModel allow-list. "
                    "Never removes classes from the model; filtering is post-inference, "
                    "exactly as production does it.",
    )
    limit: int | None = Field(
        None,
        description="Cap the number of clips processed. Use for gate runs.",
    )
    collect_detections: bool = Field(
        True,
        description="Set False for pure latency benchmarking to keep memory flat "
                    "on large runs.",
    )


class JobState(BaseModel):
    job_id: str
    status: JobStatus
    variant: str
    total: int
    completed: int
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    metrics: JobMetrics | None = None
    detections: list[Detection] = []


class ModelInfo(BaseModel):
    variant: str
    model_path: str
    labels_path: str
    num_classes: int
    file_mb: float
    loaded: bool
    allow_list_size: int | None = None


class HealthResponse(BaseModel):
    ready: bool
    variants_loaded: list[str]
    onnxruntime_version: str
    providers: list[str]


# --- stream event envelopes -------------------------------------------------

class ProgressEvent(BaseModel):
    type: Literal["progress"] = "progress"
    job_id: str
    completed: int
    total: int


class ResultEvent(BaseModel):
    type: Literal["result"] = "result"
    job_id: str
    detections: list[Detection]


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    job_id: str
    metrics: JobMetrics


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    job_id: str
    message: str


StreamEvent = ProgressEvent | ResultEvent | DoneEvent | ErrorEvent


def sse(event_type: str, payload: dict[str, Any]) -> str:
    """Format one Server-Sent Event frame."""
    import json

    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
