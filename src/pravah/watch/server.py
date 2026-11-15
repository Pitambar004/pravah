"""Live Watch API factory (GB10 service on Fuzzball).

  python -m uvicorn --factory pravah.watch.server:create_app --host 0.0.0.0 --port 8000

Read routes are open to anyone who can reach the endpoint (CORS GET for LAN browsers).
Routes that change shared state need ``X-Pravah-Token`` when ``PRAVAH_API_TOKEN`` is set.
The existing water-segmentation inference routes stay mounted unchanged.
"""
from __future__ import annotations

import hmac
import json
import os
import threading
import time
from pathlib import Path

from fastapi import Depends, Header, HTTPException, Query

from pravah.watch.config import watch_root


def _load_components(root: Path):
    comp = {"river": None, "emulator": None, "observer": None, "errors": {}}
    try:
        from pravah.watch.river import RiverModel
        comp["river"] = RiverModel(root / "river_model")
    except Exception as exc:  # noqa: BLE001 — a 24/7 service must outlive any provider failure
        comp["errors"]["river_model"] = f"{type(exc).__name__}: {exc}"
    try:
        from pravah.watch.emulator import Emulator
        comp["emulator"] = Emulator(root / "emulator")
    except Exception as exc:  # noqa: BLE001 — a 24/7 service must outlive any provider failure
        comp["errors"]["emulator"] = f"{type(exc).__name__}: {exc}"
    try:
        from pravah.watch import layout
        from pravah.watch.observe import Observer
        comp["observer"] = Observer(layout.load(root / "river_model/windows.json"), root / "obs_live", threads=8)
    except Exception as exc:  # noqa: BLE001 — a 24/7 service must outlive any provider failure
        comp["errors"]["observer"] = f"{type(exc).__name__}: {exc}"
    return comp


def attach_watch_routes(app, root: Path, token: str | None, start_engine: bool = True):
    from pravah.watch.engine import WatchEngine
    comp = _load_components(root)
    engine = WatchEngine(root, comp["river"], comp["emulator"], comp["observer"])
    replay = {"status": "pending", "done": 0, "total": None, "data": None, "error": None}

    def require_token(x_pravah_token: str | None = Header(default=None)):
        if token and not (x_pravah_token and hmac.compare_digest(x_pravah_token, token)):
            raise HTTPException(401, "Missing or invalid X-Pravah-Token")

    def build_replay():
        if comp["river"] is None or not (root / "replay/pack.json").exists():
            replay.update(status="unavailable", error="River model or replay pack missing (run the GB10 build)")
            return
        from pravah.watch.replay import load_or_build
        replay["status"] = "building"
        try:
            replay["data"] = load_or_build(root, comp["river"], comp["emulator"],
                                           progress=lambda d, t: replay.update(done=d, total=t))
            replay["status"] = "ready"
        except Exception as exc:  # noqa: BLE001 — a 24/7 service must outlive any provider failure
            replay.update(status="error", error=f"{type(exc).__name__}: {exc}")

    # The inference app owns a lifespan handler, so background work starts here (one uvicorn worker).
    if start_engine:
        if os.environ.get("PRAVAH_WATCH_ENABLED", "1") == "1":
            engine.start()
        threading.Thread(target=build_replay, name="pravah-replay", daemon=True).start()

    @app.get("/api/v1/watch/state")
    def state():
        s = engine.public_state()
        if not s:
            raise HTTPException(503, "First live tick not finished yet")
        return s

    @app.get("/api/v1/watch/status")
    def status():
        import torch
        return {"components_loaded": {k: comp[k] is not None for k in ("river", "emulator", "observer")},
                "component_errors": comp["errors"], "source_errors": engine.errors,
                "cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "replay": {k: replay[k] for k in ("status", "done", "total", "error")},
                "emulator_members": len(comp["emulator"].members) if comp["emulator"] else 0,
                "river_model": (comp["river"].meta.get("main_stem"), comp["river"].meta.get("sections")) if comp["river"] else None,
                "cadence_s": __import__("pravah.watch.engine", fromlist=["CADENCE"]).CADENCE}

    @app.get("/api/v1/watch/timeline")
    def timeline(limit: int = Query(288, ge=1, le=5000)):
        p = root / "state/timeline.jsonl"
        if not p.exists():
            return {"rows": []}
        lines = p.read_text().splitlines()[-limit:]
        return {"rows": [json.loads(x) for x in lines]}

    @app.get("/api/v1/watch/geography")
    def geography():
        if comp["river"] is None:
            raise HTTPException(503, "River model not built")
        m = comp["river"].meta
        return {"main_stem": comp["river"].main_stem, "stations": m["stations"], "reaches": m["reaches"],
                "assets_in_corridor": m["asset_counts_in_corridor"], "method": m["method"]}

    @app.get("/api/v1/watch/model")
    def model_card():
        out = {}
        for name, p in (("emulator", root / "emulator/ensemble_summary.json"), ("width_fits", root / "river_model/width_fits.json"),
                        ("lake_baselines", root / "baselines/lakes.json"), ("glacier_baselines", root / "baselines/glacier.json")):
            doc = json.loads(p.read_text()) if p.exists() else None
            if name == "width_fits" and doc:
                doc = {"years": doc["years"], "reaches": doc["reaches"]}
            out[name] = doc
        out["build_steps"] = {p.stem.replace(".done", ""): json.loads(p.read_text()) for p in sorted((root / "steps").glob("*.done.json"))} \
            if (root / "steps").exists() else {}
        return out

    @app.post("/api/v1/watch/refresh", dependencies=[Depends(require_token)])
    def refresh():
        t0 = time.perf_counter()
        engine.tick(force=True)
        return {"refreshed": True, "seconds": round(time.perf_counter() - t0, 2)}

    @app.get("/api/v1/watch/replay")
    def replay_meta():
        d = replay["data"]
        if replay["status"] != "ready" or not d:
            return {k: replay[k] for k in ("status", "done", "total", "error")}
        return {"status": "ready", "event": d["event"], "start_utc": d["start_utc"], "end_utc": d["end_utc"],
                "frame_count": d["frame_count"], "compute_seconds_measured": d["compute_seconds_measured"],
                "device": d["device"], "availability": d["availability"], "gauges_note": d["gauges_note"],
                "levels": [f["assessment"]["level"] for f in d["frames"]], "method": d["method"]}

    @app.get("/api/v1/watch/replay/frames/{index}")
    def replay_frame(index: int):
        d = replay["data"]
        if not d:
            raise HTTPException(503, f"Replay {replay['status']}")
        if not 0 <= index < d["frame_count"]:
            raise HTTPException(404, "Frame out of range")
        return d["frames"][index]

    app.state.watch_engine = engine
    return engine


def create_app():
    from fastapi.middleware.cors import CORSMiddleware

    from pravah.api.main import create_app as create_inference_app
    app = create_inference_app()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])
    attach_watch_routes(app, watch_root(), os.environ.get("PRAVAH_API_TOKEN") or None)
    return app
