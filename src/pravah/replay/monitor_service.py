"""Live monitoring and exploratory scenarios attached to the existing API."""

import asyncio
import hmac
import json
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path

import geopandas as gpd
from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel, Field
from shapely.geometry import shape

from pravah.configuration import configured_url
from pravah.replay.image_pairs import load_image_pairs
from pravah.replay.lakes import load_lakes, surface_change
from pravah.replay.live import LiveMonitor
from pravah.replay.scenario import explore, load_geography, load_national_rivers
from pravah.replay.water_watch import eligible, identity


class ScenarioRequest(BaseModel):
    assumed_half_width_m: float = Field(default=25, ge=5, le=100)
    extra_margin_m: float = Field(default=100, ge=0, le=250)


class LakeChangeRequest(BaseModel):
    lake_id: str = Field(max_length=100)
    pre_run_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,96}$")
    post_run_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,96}$")


def attach_monitor_routes(app, root: Path, geography: Path, token=None, enabled=True, runs=None, lakes=None, national=None, imagery=None):
    monitor = LiveMonitor(root, runs=runs, preview=enabled)
    app.state.nepal_monitor = monitor
    original = app.router.lifespan_context

    async def poll():
        while True:
            await asyncio.to_thread(monitor.refresh)
            await asyncio.sleep(300)

    @asynccontextmanager
    async def lifespan(application):
        async with original(application):
            task = asyncio.create_task(poll()) if enabled else None
            try:
                yield
            finally:
                if task:
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task

    app.router.lifespan_context = lifespan

    def require_token(x_pravah_token: str | None = Header(default=None)):
        if token and not (x_pravah_token and hmac.compare_digest(token, x_pravah_token)):
            raise HTTPException(401, "Missing or invalid X-Pravah-Token")

    @app.get("/api/v1/nepal/live")
    def live():
        return {**monitor.snapshot(), "automatic_monitoring_enabled": enabled}

    @app.post("/api/v1/nepal/live/refresh", dependencies=[Depends(require_token)])
    async def refresh():
        # Await a bounded refresh; it does not block inference/status requests.
        updated = await asyncio.to_thread(monitor.refresh)
        return {"updated": updated, **monitor.snapshot()}

    @app.get("/api/v1/nepal/geography")
    def context():
        try:
            metadata, layers = load_geography(geography)
            return {"metadata": metadata, "layers": layers}
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(503, "Prepared Nepal river geometry is unavailable or failed integrity verification") from exc

    @app.get("/api/v1/nepal/imagery")
    def image_context():
        try:
            return load_image_pairs(Path(imagery))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(503, "Verified satellite image pairs are unavailable") from exc

    @app.post("/api/v1/nepal/scenarios", dependencies=[Depends(require_token)])
    def scenario(request: ScenarioRequest):
        try:
            metadata, layers = load_geography(geography)
            return {**explore(layers, request.assumed_half_width_m, request.extra_margin_m),
                    "geography_snapshot": metadata.get("retrieved_at_utc"), "geography_source": metadata.get("source"),
                    "attribution": "OpenStreetMap contributors · ODbL 1.0 · " + configured_url("PRAVAH_OSM_ATTRIBUTION_URL")}
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(503, "Scenario geography is unavailable or failed validation") from exc

    @app.get("/api/v1/nepal/lakes")
    def lake_context():
        try:
            return load_lakes(lakes)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(503, "Prepared glacier-lake inventory unavailable") from exc

    @app.get("/api/v1/nepal/rivers")
    def national_rivers():
        try:
            return load_national_rivers(national)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(503, "National river geometry unavailable") from exc

    @app.post("/api/v1/nepal/lakes/change", dependencies=[Depends(require_token)])
    def lake_change(request: LakeChangeRequest):
        try:
            context = load_lakes(lakes)
            feature = next(f for f in context["layers"]["s2"]["features"] if f["properties"]["GLO_ID"] == request.lake_id)
            pre, post = Path(runs) / request.pre_run_id, Path(runs) / request.post_run_id
            if any(p.is_symlink() or Path(runs).resolve() not in p.resolve().parents for p in [pre, post]):
                raise ValueError("Invalid run path")
            a, b = [json.loads((p / "observed_metadata.json").read_text()) for p in [pre, post]]
            if not eligible(a) or not eligible(b) or identity(a) != identity(b):
                raise ValueError("Need two compatible cloud-valid Nepal optical water runs")
            dates = [datetime.fromisoformat(m["acquired_at_utc"].replace("Z", "+00:00")) for m in [a, b]]
            if any(d.tzinfo is None for d in dates) or dates[0] >= dates[1]:
                raise ValueError("Acquisition dates must be timezone-aware and ordered")
            # Use a bounded 300 m expansion around the inventory baseline so
            # new water can be detected beyond the historic maximum outline.
            roi = gpd.GeoSeries([shape(feature["geometry"])], crs="EPSG:4326").to_crs("EPSG:32645").buffer(300).to_crs("EPSG:4326").iloc[0]
            return {**surface_change(pre / "observed_water.tif", post / "observed_water.tif", roi.__geo_interface__),
                    "lake_id": request.lake_id, "search_margin_m": 300, "pre_acquired_at_utc": a["acquired_at_utc"], "post_acquired_at_utc": b["acquired_at_utc"]}
        except StopIteration as exc:
            raise HTTPException(404, "Inventory lake not found") from exc
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    return app
