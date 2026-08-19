"""Job queue, background execution, cancellation, and metric aggregation.

Long runs are not request/response. A submit returns a job id immediately, work
happens on a worker thread, and the caller polls /jobs/{id} or subscribes to
/jobs/{id}/stream. This mirrors the sidecar job model in Daan's design doc, so
the client code written against this service carries over.

ONNX Runtime releases the GIL inside session.run, so a thread (rather than a
process) is sufficient here and keeps model weights shared.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import AudioError, collect_paths, read_windows
from .inference import score_window, to_detections, warmup
from .registry import Registry
from .schemas import Detection, JobMetrics, JobRequest, JobState, JobStatus

log = logging.getLogger("bridge.jobs")

try:
    import psutil
    _PROC = psutil.Process()
except Exception:                                    # pragma: no cover
    psutil = None
    _PROC = None


@dataclass
class Job:
    job_id: str
    request: JobRequest
    status: JobStatus = JobStatus.queued
    total: int = 0
    completed: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    metrics: JobMetrics | None = None
    detections: list[Detection] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    cancel: threading.Event = field(default_factory=threading.Event)
    subscribers: list[queue.Queue] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def publish(self, event_type: str, payload: dict) -> None:
        with self.lock:
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait((event_type, payload))
            except queue.Full:                       # slow consumer: drop, don't block
                pass

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def to_state(self) -> JobState:
        return JobState(
            job_id=self.job_id,
            status=self.status,
            variant=self.request.variant,
            total=self.total,
            completed=self.completed,
            started_at=self.started_at,
            finished_at=self.finished_at,
            error=self.error,
            metrics=self.metrics,
            detections=self.detections if self.request.collect_detections else [],
        )


class JobManager:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, request: JobRequest) -> Job:
        self.registry.get(request.variant)           # fail fast on unknown variant
        job = Job(job_id=str(uuid.uuid4()), request=request)
        with self._lock:
            self.jobs[job.job_id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True,
                         name=f"job-{job.job_id[:8]}").start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.status in (JobStatus.done, JobStatus.error, JobStatus.cancelled):
            return False
        job.cancel.set()
        return True

    # --- worker -------------------------------------------------------------

    def _run(self, job: Job) -> None:
        req = job.request
        model = self.registry.get(req.variant)
        inf_ms: list[float] = []
        pre_ms: list[float] = []
        windows = 0

        try:
            paths = collect_paths(req.paths, req.limit)
            job.total = len(paths)
            job.status = JobStatus.running
            job.started_at = time.time()

            warmup(model)

            if _PROC is not None:
                _PROC.cpu_times()                    # prime the counter
            cpu0 = time.process_time()
            wall0 = time.perf_counter()
            peak_rss = _rss_mb()

            for path in paths:
                if job.cancel.is_set():
                    job.status = JobStatus.cancelled
                    job.finished_at = time.time()
                    job.publish("error", {"job_id": job.job_id, "message": "cancelled"})
                    return

                try:
                    t0 = time.perf_counter()
                    wins = read_windows(path)
                    pre_ms.append((time.perf_counter() - t0) * 1000.0 / max(len(wins), 1))
                except (AudioError, RuntimeError) as exc:
                    job.failures.append(f"{path.name}: {exc}")
                    job.completed += 1
                    continue

                batch: list[Detection] = []
                for w_idx, (window, start_s, end_s) in enumerate(wins):
                    conf, ms = score_window(model, window)
                    inf_ms.append(ms)
                    windows += 1
                    if req.collect_detections:
                        batch.extend(
                            to_detections(model, conf, path.name, start_s, end_s,
                                          req.min_confidence, req.apply_allow_list)
                        )

                if batch:
                    job.detections.extend(batch)
                    job.publish("result", {
                        "job_id": job.job_id,
                        "detections": [d.model_dump() for d in batch],
                    })

                job.completed += 1
                if job.completed % 25 == 0 or job.completed == job.total:
                    peak_rss = max(peak_rss or 0.0, _rss_mb() or 0.0) or None
                    job.publish("progress", {
                        "job_id": job.job_id,
                        "completed": job.completed,
                        "total": job.total,
                    })

            wall = time.perf_counter() - wall0
            cpu = time.process_time() - cpu0

            job.metrics = _aggregate(
                model=model, clips=job.total, windows=windows, wall=wall, cpu=cpu,
                inf_ms=inf_ms, pre_ms=pre_ms, peak_rss=peak_rss,
            )
            job.status = JobStatus.done
            job.finished_at = time.time()
            job.publish("done", {"job_id": job.job_id,
                                 "metrics": job.metrics.model_dump()})
            if job.failures:
                log.warning("job %s: %d files skipped", job.job_id, len(job.failures))

        except Exception as exc:                     # noqa: BLE001 - surfaced to caller
            log.exception("job %s failed", job.job_id)
            job.status = JobStatus.error
            job.error = f"{type(exc).__name__}: {exc}"
            job.finished_at = time.time()
            job.publish("error", {"job_id": job.job_id, "message": job.error})


def _rss_mb() -> float | None:
    if _PROC is None:
        return None
    return round(_PROC.memory_info().rss / (1024 * 1024), 1)


def _aggregate(*, model, clips, windows, wall, cpu, inf_ms, pre_ms, peak_rss) -> JobMetrics:
    a = np.asarray(inf_ms, dtype=np.float64) if inf_ms else np.zeros(1)
    p = np.asarray(pre_ms, dtype=np.float64) if pre_ms else np.zeros(1)
    return JobMetrics(
        variant=model.variant,
        clips=clips,
        windows=windows,
        wall_seconds=round(wall, 3),
        windows_per_second=round(windows / wall, 2) if wall > 0 else 0.0,
        inference_ms_mean=round(float(a.mean()), 3),
        inference_ms_p50=round(float(np.percentile(a, 50)), 3),
        inference_ms_p95=round(float(np.percentile(a, 95)), 3),
        inference_ms_max=round(float(a.max()), 3),
        preprocess_ms_mean=round(float(p.mean()), 3),
        process_cpu_seconds=round(cpu, 3),
        peak_rss_mb=peak_rss,
        model_file_mb=model.file_mb,
        intra_op_num_threads=model.intra_op_num_threads,
        inter_op_num_threads=model.inter_op_num_threads,
    )
