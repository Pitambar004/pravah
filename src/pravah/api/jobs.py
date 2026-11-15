"""Single-worker prediction queue.

One GPU, one model, one worker thread: jobs run strictly one at a time so a
GB10 is never oversubscribed and timings are not contaminated by concurrency.
Job state is mirrored to ``<outputs>/<run_id>/job.json`` so the dashboard and a
restarted API can see what happened.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TERMINAL = ("succeeded", "failed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    job_id: str
    run_id: str
    scene_id: str
    threshold: float
    status: str = "queued"  # queued | running | succeeded | failed
    submitted_at_utc: str = field(default_factory=_now)
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    tiles_done: int = 0
    tiles_total: int | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    # Location jobs: {"lat", "lon", "place", ...}; scene_id is filled once the scene is fetched.
    location: dict[str, Any] | None = None
    stage: str | None = None  # fetching | predicting (location jobs)
    stage_detail: str | None = None
    stage_progress: float | None = None  # 0..1 within the current stage

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Runner = Callable[[Job, Callable[[int, int], None]], dict[str, Any]]


class JobQueue:
    def __init__(self, runner: Runner, outputs_dir: Path, max_queued: int = 8):
        self._runner = runner
        self._outputs = outputs_dir
        self._max = max_queued
        self._jobs: dict[str, Job] = {}
        self._q: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="pravah-inference", daemon=True)
        self._thread.start()

    # ----------------------------------------------------------------- public
    def submit(self, run_id: str, scene_id: str, threshold: float,
               location: dict[str, Any] | None = None) -> Job:
        with self._lock:
            pending = sum(j.status in ("queued", "running") for j in self._jobs.values())
            if pending >= self._max:
                raise OverflowError(f"Queue full ({pending} pending jobs)")
            if any(j.run_id == run_id for j in self._jobs.values()) or (self._outputs / run_id).exists():
                raise FileExistsError(f"Run {run_id} already exists")
            job = Job(job_id=uuid.uuid4().hex[:12], run_id=run_id, scene_id=scene_id, threshold=threshold,
                      location=location)
            self._jobs[job.job_id] = job
        self._persist(job)
        self._q.put(job.job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.submitted_at_utc, reverse=True)

    def pending(self) -> int:
        with self._lock:
            return sum(j.status in ("queued", "running") for j in self._jobs.values())

    def wait(self, job_id: str, timeout: float = 60.0) -> Job:
        """Block until a job is terminal (tests and CLI helpers)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.get(job_id)
            if job and job.status in TERMINAL:
                return job
            time.sleep(0.05)
        raise TimeoutError(job_id)

    def shutdown(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=5)

    # ---------------------------------------------------------------- private
    def _persist(self, job: Job) -> None:
        try:
            d = self._outputs / job.run_id
            d.mkdir(parents=True, exist_ok=True)
            (d / "job.json").write_text(json.dumps(job.to_dict(), indent=2, default=str) + "\n")
        except OSError:
            pass  # status is still served from memory; storage errors surface via /status

    def _loop(self) -> None:
        while True:
            job_id = self._q.get()
            if job_id is None:
                return
            job = self.get(job_id)
            if job is None:
                continue
            job.status, job.started_at_utc = "running", _now()
            self._persist(job)

            def progress(done: int, total: int, _job: Job = job) -> None:
                _job.tiles_done, _job.tiles_total = done, total

            try:
                job.result = self._runner(job, progress)
                job.status = "succeeded"
            except Exception as exc:  # noqa: BLE001 - report every failure to the client, never crash the worker
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            job.finished_at_utc = _now()
            self._persist(job)
