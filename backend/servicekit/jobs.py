"""In-process job store: upload, run the analyzer off the event loop, stream progress.

Design notes that matter to product teams
-----------------------------------------
* The analyzer is **synchronous** and CPU-bound. It runs in a worker thread, so it
  must not touch the event loop. `JobContext.progress()` is the only way back and
  it is thread-safe.
* Progress events go to every current subscriber. Late subscribers get the
  backlog replayed first, so a browser that connects after the job started still
  sees the whole story. That is why `Job.events` keeps history.
* State is per-process. Five products on App Runner with one instance each is
  fine; anything that needs to survive a restart writes to S3 or DynamoDB from
  inside the analyzer.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from visioncore import Evidence, RunRecord, recording

from .errors import ServiceError
from .logs import extra, get_logger

log = get_logger("servicekit.jobs")

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"


@dataclass
class JobContext:
    """What an analyzer is handed. Everything it needs, nothing it does not."""

    job_id: str
    input_path: Path
    filename: str
    params: dict[str, Any]
    record: RunRecord
    _emit: Callable[[dict[str, Any]], None]
    evidence_dir: Path

    def progress(self, percent: float, message: str = "", **extra: Any) -> None:
        """Report progress from the worker thread. Safe to call as often as you like."""
        self._emit(
            {
                "type": "progress",
                "percent": max(0.0, min(100.0, float(percent))),
                "message": message,
                **extra,
            }
        )

    def note(self, message: str, **extra: Any) -> None:
        self._emit({"type": "note", "message": message, **extra})

    def save_evidence(self, name: str, data: bytes) -> str:
        """Write an evidence image and return the URI the UI will fetch it from."""
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(c for c in name if c.isalnum() or c in "._-") or "evidence"
        (self.evidence_dir / safe).write_bytes(data)
        return f"/api/jobs/{self.job_id}/evidence/{safe}"

    def add_evidence(self, label: str, image_bytes: bytes, **kwargs: Any) -> Evidence:
        """Save an image and attach it to the run record in one call."""
        index = len(self.record.evidence)
        uri = self.save_evidence(f"{index:03d}-{label}.jpg", image_bytes)
        return self.record.add_evidence(Evidence(label=label, uri=uri, **kwargs))


class Analyzer(Protocol):
    """The one thing a product must implement."""

    def __call__(self, ctx: JobContext) -> RunRecord: ...


@dataclass
class Job:
    job_id: str
    filename: str
    params: dict[str, Any]
    input_path: Path
    status: str = QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    subscribers: set[asyncio.Queue] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "filename": self.filename,
            "params": self.params,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": (
                round(self.finished_at - self.started_at, 3)
                if self.started_at and self.finished_at
                else None
            ),
            "result": self.result,
            "error": self.error,
        }


class JobStore:
    def __init__(
        self,
        analyzer: Analyzer,
        upload_dir: Path,
        *,
        product: str = "product",
        max_jobs: int = 64,
        max_concurrent: int = 2,
        ttl_seconds: int = 3600,
    ) -> None:
        self.analyzer = analyzer
        self.upload_dir = Path(upload_dir)
        self.product = product
        self.max_jobs = max_jobs
        self.ttl_seconds = ttl_seconds
        self._jobs: dict[str, Job] = {}
        self._tasks: set[asyncio.Task] = set()
        self._semaphore = asyncio.Semaphore(max_concurrent)

    # ---- lifecycle --------------------------------------------------------
    def create(self, filename: str, data: bytes, params: dict[str, Any]) -> Job:
        self._evict()
        job_id = uuid.uuid4().hex[:12]
        suffix = Path(filename).suffix.lower()
        path = self.upload_dir / f"{job_id}{suffix}"
        path.write_bytes(data)
        job = Job(job_id=job_id, filename=filename, params=params, input_path=path)
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise ServiceError("NOT_FOUND", f"no job {job_id}", job_id=job_id)
        return job

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def start(self, job: Job) -> asyncio.Task:
        task = asyncio.create_task(self._run(job))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _run(self, job: Job) -> None:
        loop = asyncio.get_running_loop()

        def emit(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(self._publish, job, event)

        async with self._semaphore:
            job.status = RUNNING
            job.started_at = time.time()
            self._publish(job, {"type": "status", "status": RUNNING})
            record = RunRecord(
                product=self.product,
                input={"filename": job.filename, "job_id": job.job_id},
                params=dict(job.params),
            )
            ctx = JobContext(
                job_id=job.job_id,
                input_path=job.input_path,
                filename=job.filename,
                params=dict(job.params),
                record=record,
                _emit=emit,
                evidence_dir=self.upload_dir / job.job_id,
            )
            try:
                result = await loop.run_in_executor(None, self._invoke, ctx)
                job.result = result.to_dict()
                job.status = DONE
                log.info(
                    "job finished",
                    extra=extra(
                        job_id=job.job_id,
                        refused=result.refused,
                        total_ms=round(result.total_ms, 1),
                    ),
                )
            except ServiceError as exc:
                job.status = FAILED
                job.error = exc.payload(job.job_id)["error"]
                log.warning("job rejected", extra=extra(job_id=job.job_id, code=exc.code))
            except Exception as exc:  # analyzer bug: surface it, do not hide it
                job.status = FAILED
                job.error = {
                    "code": "ANALYSIS_FAILED",
                    "message": str(exc) or exc.__class__.__name__,
                    "request_id": job.job_id,
                    "details": {"type": exc.__class__.__name__},
                }
                log.error(
                    "job failed",
                    extra=extra(job_id=job.job_id, traceback=traceback.format_exc()),
                )
            finally:
                job.finished_at = time.time()
                self._publish(
                    job,
                    {
                        "type": "status",
                        "status": job.status,
                        "result": job.result,
                        "error": job.error,
                    },
                )
                self._publish(job, {"type": "end"})

    def _invoke(self, ctx: JobContext) -> RunRecord:
        """Runs in a worker thread. `recording` binds the record to THIS thread."""
        with recording(ctx.record):
            result = self.analyzer(ctx)
        return result if isinstance(result, RunRecord) else ctx.record

    # ---- events -----------------------------------------------------------
    def _publish(self, job: Job, event: dict[str, Any]) -> None:
        event = {"job_id": job.job_id, "ts": time.time(), **event}
        job.events.append(event)
        for queue in list(job.subscribers):
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(event)

    def subscribe(self, job: Job) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        for event in job.events:  # replay, so a late browser sees the whole run
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(event)
        job.subscribers.add(queue)
        return queue

    def unsubscribe(self, job: Job, queue: asyncio.Queue) -> None:
        job.subscribers.discard(queue)

    # ---- housekeeping -----------------------------------------------------
    def _evict(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        stale = [j for j in self._jobs.values() if j.created_at < cutoff and j.status in
                 (DONE, FAILED)]
        for job in stale:
            self._drop(job)
        while len(self._jobs) >= self.max_jobs:
            oldest = min(self._jobs.values(), key=lambda j: j.created_at)
            self._drop(oldest)

    def _drop(self, job: Job) -> None:
        self._jobs.pop(job.job_id, None)
        with contextlib.suppress(OSError):
            job.input_path.unlink(missing_ok=True)
        evidence = self.upload_dir / job.job_id
        if evidence.is_dir():
            for item in evidence.iterdir():
                with contextlib.suppress(OSError):
                    item.unlink()
            with contextlib.suppress(OSError):
                evidence.rmdir()
