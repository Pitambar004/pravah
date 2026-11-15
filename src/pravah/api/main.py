"""PRAVAH inference API (FastAPI).

Run locally::

    PRAVAH_DATA_ROOT=./data uvicorn pravah.api.main:app --port 8000

On Fuzzball the service sets ``PRAVAH_REQUIRE_CUDA=1`` so it refuses to start
inference on CPU. If no checkpoint is staged, the API still starts (so status is
inspectable) but prediction endpoints return 503 — weights are never invented.

Routes (all JSON):

* ``GET  /health``                         liveness
* ``GET  /api/v1/status``                  device, GPU, model hash, storage, queue
* ``GET  /api/v1/scenes``                  GeoTIFFs available under the scenes dir
* ``POST /api/v1/predictions``             queue a prediction  (token if configured)
* ``GET  /api/v1/predictions[/{job_id}]``  job status
* ``GET  /api/v1/runs``                    finished runs in the outputs dir
* ``GET  /api/v1/runs/{run_id}``           run metadata
* ``GET  /api/v1/runs/{run_id}/files/{name}`` download one output artifact
* ``POST /api/v1/model/reload``            re-load the checkpoint (token if configured)
"""
from __future__ import annotations

import hmac
import json
import os
import re
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from pravah import __version__
from pravah.api.jobs import Job, JobQueue
from pravah.api.settings import Settings
from pravah.configuration import configured_url
from pravah.inference.engine import CheckpointError, InferenceEngine, describe_raster

RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
RASTER_SUFFIXES = (".tif", ".tiff")
DOWNLOADABLE = {
    "observed_water.tif": "image/tiff",
    "invalid_mask.tif": "image/tiff",
    "water_probability.tif": "image/tiff",
    "observed_metadata.json": "application/json",
    "observed_water.geojson": "application/geo+json",
    "job.json": "application/json",
}

EngineFactory = Callable[[Settings], InferenceEngine]


def _default_engine_factory(s: Settings) -> InferenceEngine:
    return InferenceEngine(s.checkpoint, device=s.device, require_cuda=s.require_cuda, batch_size=s.batch_size)


class PredictionRequest(BaseModel):
    scene_id: str = Field(..., description="Path of the GeoTIFF relative to the scenes directory")
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    run_id: str | None = Field(None, description="Optional; letters, digits, '_' and '-' only")


class ServiceState:
    """Holds the loaded engine; swapped atomically on reload."""

    def __init__(self, settings: Settings, factory: EngineFactory):
        self.settings = settings
        self.factory = factory
        self.engine: InferenceEngine | None = None
        self.load_error: str | None = None
        self.lock = threading.Lock()

    def load(self) -> None:
        try:
            engine = self.factory(self.settings)
        except (CheckpointError, RuntimeError, ValueError, OSError) as exc:
            with self.lock:
                self.engine, self.load_error = None, str(exc)
            return
        with self.lock:
            self.engine, self.load_error = engine, None


def _safe_scene(settings: Settings, scene_id: str) -> Path:
    root = settings.scenes_dir.resolve()
    candidate = (root / scene_id).resolve()
    if root not in candidate.parents or candidate.suffix.lower() not in RASTER_SUFFIXES:
        raise HTTPException(400, "scene_id must be a .tif/.tiff path inside the scenes directory")
    if not candidate.is_file():
        raise HTTPException(404, f"Scene not found: {scene_id}")
    return candidate


def _run_dir(settings: Settings, run_id: str) -> Path:
    if not RUN_ID_RE.match(run_id):
        raise HTTPException(400, "Invalid run_id")
    return settings.outputs_dir / run_id


def _dir_state(p: Path) -> dict[str, Any]:
    return {"path": str(p), "exists": p.is_dir(), "writable": p.is_dir() and os.access(p, os.W_OK)}


class StripPrefix:
    """ASGI middleware: accept requests with or without the endpoint prefix.

    Works whether or not the Fuzzball proxy strips ``/endpoints/<id>`` (to be
    confirmed on v4.3.1), so the same image serves both cases.
    """

    def __init__(self, app, prefix: str):
        self.app, self.prefix = app, prefix.rstrip("/")

    async def __call__(self, scope, receive, send):
        if self.prefix and scope["type"] in (configured_url("PRAVAH_LOCAL_SCHEME"), "websocket"):
            path = scope.get("path", "")
            if path == self.prefix or path.startswith(self.prefix + "/"):
                scope = {**scope, "path": path[len(self.prefix):] or "/", "root_path": self.prefix}
        await self.app(scope, receive, send)


def create_app(settings: Settings | None = None, engine_factory: EngineFactory | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    state = ServiceState(settings, engine_factory or _default_engine_factory)

    def run_job(job: Job, progress: Callable[[int, int], None]) -> dict[str, Any]:
        with state.lock:
            engine = state.engine
        if engine is None:
            raise RuntimeError(f"Model not loaded: {state.load_error}")
        scene = _safe_scene(settings, job.scene_id)
        result = engine.predict(
            scene, settings.outputs_dir / job.run_id, run_id=job.run_id, threshold=job.threshold,
            progress=progress, extra_metadata={"scene_id": job.scene_id, "job_id": job.job_id},
        )
        md = result.metadata
        return {"files": result.files, "timings_seconds": md["timings_seconds"], "runtime": md["runtime"],
                "water_pixel_count": md["water_pixel_count"], "water_area_m2": md["water_area_m2"],
                "valid_pixel_count": md["valid_pixel_count"], "tile_count": md["tile_count"]}

    holder: dict[str, JobQueue] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # reported as not writable by /api/v1/status
        state.load()
        holder["q"] = JobQueue(run_job, settings.outputs_dir, settings.max_queued_jobs)
        yield
        holder["q"].shutdown()

    app = FastAPI(title="PRAVAH inference API", version=__version__, lifespan=lifespan,
                  description="Retrospective satellite water evidence. Not a flood forecast.")
    if settings.root_path:
        app.add_middleware(StripPrefix, prefix=settings.root_path)

    def require_token(x_pravah_token: str | None = Header(default=None)) -> None:
        if settings.api_token and not (x_pravah_token and hmac.compare_digest(x_pravah_token, settings.api_token)):
            raise HTTPException(401, "Missing or invalid X-Pravah-Token")

    def jobs() -> JobQueue:
        return holder["q"]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/status")
    def status() -> dict[str, Any]:
        with state.lock:
            engine, err = state.engine, state.load_error
        return {
            "service": "pravah-api",
            "version": __version__,
            "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_loaded": engine is not None,
            "model": engine.info.to_dict() if engine else None,
            "model_error": err,
            "runtime": engine.runtime() if engine else None,
            "require_cuda": settings.require_cuda,
            "storage": {
                "data_root": _dir_state(settings.data_root),
                "scenes_dir": _dir_state(settings.scenes_dir),
                "outputs_dir": _dir_state(settings.outputs_dir),
                "checkpoint": {"path": str(settings.checkpoint), "exists": settings.checkpoint.is_file()},
            },
            "queue": {"pending": jobs().pending(), "max": settings.max_queued_jobs},
            "auth_required_for_writes": bool(settings.api_token),
        }

    @app.get("/api/v1/scenes")
    def scenes() -> dict[str, Any]:
        root = settings.scenes_dir
        items: list[dict[str, Any]] = []
        if root.is_dir():
            for p in sorted(root.rglob("*")):
                if p.suffix.lower() not in RASTER_SUFFIXES or not p.is_file():
                    continue
                entry: dict[str, Any] = {"scene_id": str(p.relative_to(root)), "size_bytes": p.stat().st_size}
                try:
                    entry.update(describe_raster(p))
                except Exception as exc:  # noqa: BLE001 - unreadable files are listed, not hidden
                    entry.update({"model_compatible": False, "error": str(exc)})
                items.append(entry)
        return {"scenes_dir": str(root), "scenes": items}

    @app.post("/api/v1/predictions", status_code=202, dependencies=[Depends(require_token)])
    def create_prediction(req: PredictionRequest) -> dict[str, Any]:
        with state.lock:
            loaded = state.engine is not None
        if not loaded:
            raise HTTPException(503, f"Model not loaded: {state.load_error}")
        scene = _safe_scene(settings, req.scene_id)
        info = describe_raster(scene)
        if not info["model_compatible"]:
            raise HTTPException(422, f"Scene not model-compatible (bands={info['band_count']}, crs={info['crs']})")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = req.run_id or re.sub(r"[^A-Za-z0-9_-]+", "_", scene.stem)[:60] + "_" + stamp
        _run_dir(settings, run_id)
        try:
            job = jobs().submit(run_id, req.scene_id, req.threshold)
        except OverflowError as exc:
            raise HTTPException(429, str(exc)) from exc
        except FileExistsError as exc:
            raise HTTPException(409, str(exc)) from exc
        return job.to_dict()

    @app.get("/api/v1/predictions")
    def list_predictions() -> dict[str, Any]:
        return {"jobs": [j.to_dict() for j in jobs().list()]}

    @app.get("/api/v1/predictions/{job_id}")
    def get_prediction(job_id: str) -> dict[str, Any]:
        job = jobs().get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        return job.to_dict()

    @app.get("/api/v1/runs")
    def runs() -> dict[str, Any]:
        out: list[dict[str, Any]] = []
        root = settings.outputs_dir
        if root.is_dir():
            for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
                meta_p = d / "observed_metadata.json"
                row: dict[str, Any] = {"run_id": d.name, "complete": meta_p.is_file()}
                if meta_p.is_file():
                    try:
                        md = json.loads(meta_p.read_text())
                        row.update({k: md.get(k) for k in ("created_at_utc", "scene_id", "threshold",
                                                           "water_area_m2", "tile_count")})
                        row["device"] = (md.get("runtime") or {}).get("device")
                    except (OSError, ValueError):
                        row["error"] = "unreadable observed_metadata.json"
                out.append(row)
        return {"outputs_dir": str(root), "runs": out}

    @app.get("/api/v1/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, Any]:
        meta_p = _run_dir(settings, run_id) / "observed_metadata.json"
        if not meta_p.is_file():
            raise HTTPException(404, "Run has no metadata (missing or still running)")
        return json.loads(meta_p.read_text())

    @app.get("/api/v1/runs/{run_id}/files/{name}")
    def run_file(run_id: str, name: str) -> FileResponse:
        if name not in DOWNLOADABLE:
            raise HTTPException(404, "Unknown artifact")
        p = _run_dir(settings, run_id) / name
        if not p.is_file():
            raise HTTPException(404, "Artifact not found")
        return FileResponse(p, media_type=DOWNLOADABLE[name], filename=f"{run_id}_{name}")

    @app.post("/api/v1/model/reload", dependencies=[Depends(require_token)])
    def reload_model() -> dict[str, Any]:
        if jobs().pending():
            raise HTTPException(409, "Cannot reload while jobs are pending")
        state.load()
        with state.lock:
            return {"model_loaded": state.engine is not None, "model_error": state.load_error,
                    "model": state.engine.info.to_dict() if state.engine else None}

    return app


def __getattr__(name: str):  # lazy module-level ``app`` for ``uvicorn pravah.api.main:app``
    if name == "app":
        return create_app()
    raise AttributeError(name)
