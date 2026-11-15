"""Synthetic operator fixtures remain in memory or pytest temporary storage."""

import copy
import inspect
import io
import json
from datetime import datetime, timedelta, timezone

import httpx
from PIL import Image

from app.sections.nepal_monitor import operations_layers
from app.sections.operations_data import npt_time, watch_reasons
from pravah.replay.imagery import latest_preview, retain_preview
from pravah.replay.live import LiveMonitor


def test_rerun_replaces_cached_map_component_with_current_bounds_support(monkeypatch):
    import app.components.map_view as component
    from app import hot_reload

    monkeypatch.setattr(component, "render_map_workspace", lambda layers, height=680: None)
    hot_reload._VERSIONS.clear()
    hot_reload.synchronize_modules()
    assert "initial_bounds" in inspect.signature(component.render_map_workspace).parameters
    calls = []
    monkeypatch.setattr(component.st.components.v1, "html", lambda html, height: calls.append(html))
    component.render_map_workspace([], initial_bounds=[[26, 80], [30, 88]])
    assert '"initialBounds":[[26,80],[30,88]]' in calls[0]


def test_nepal_monitor_endpoint_keeps_hosted_and_explicit_configuration(monkeypatch):
    from app.nepal_data import monitor_api_url

    monkeypatch.setenv("PRAVAH_API_URL", "http://api.service:8000")
    monkeypatch.delenv("PRAVAH_NEPAL_API_URL", raising=False)
    assert monitor_api_url() == "http://api.service:8000"
    monkeypatch.setenv("PRAVAH_NEPAL_API_URL", "http://monitor.service:8000/")
    assert monitor_api_url() == "http://monitor.service:8000"


def watch_state():
    now = datetime.now(timezone.utc).isoformat()
    return {"sources": {"gauges": {"status": "ok", "retrieved_at_utc": now,
            "records": [{"id": "4913", "name": "Synthetic station", "observed_at_utc": now,
                         "quality": "current", "level_m": 8, "warning_level_m": 6, "danger_level_m": 7}]}}}


def test_cached_stale_or_failed_gauges_never_activate_infrastructure_watch():
    state = watch_state()
    assert "danger" in watch_reasons(state)[0]
    layers, reasons = operations_layers(None, None, None, state)
    assert reasons and all(layer.visible for layer in layers if layer.group == "Infrastructure")
    for update in [{"status": "error"}, {"cache_stale": True},
                   {"retrieved_at_utc": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()}]:
        changed = copy.deepcopy(state)
        changed["sources"]["gauges"].update(update)
        layers, reasons = operations_layers(None, None, None, changed)
        assert reasons == [] and not any(layer.group == "Infrastructure" for layer in layers)
    changed = copy.deepcopy(state)
    changed["sources"]["gauges"]["records"][0]["observed_at_utc"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    assert watch_reasons(changed) == []


def test_scenario_overlay_remains_separate_and_disappears_when_cleared():
    state = {"sources": {}}
    original = copy.deepcopy(state)
    empty = {"type": "FeatureCollection", "features": []}
    scenario = {"kind": "exploratory_scenario", "baseline": empty, "overflow": empty,
                "intersections": {"buildings": empty, "roads": empty}, "limitation": "Synthetic scenario fixture"}
    layers, reasons = operations_layers(None, None, None, state, scenario)
    assert not reasons and state == original
    assert all(layer.kind == "scenario" for layer in layers if layer.id.startswith("scenario_"))
    assert next(layer for layer in layers if layer.id == "assumed_overflow").color == "#FB7185"
    cleared, _ = operations_layers(None, None, None, state)
    assert not any(layer.group in {"Infrastructure", "Scenarios"} for layer in cleared)


def test_missing_reading_requires_a_recent_observed_transition_for_watch():
    state = watch_state()
    row = state["sources"]["gauges"]["records"][0]
    row.update(quality="missing", level_m=None)
    assert watch_reasons(state) == []
    state["alerts"] = [{"source": "gauges", "record_id": row["id"],
                        "message": "Current gauge reading unavailable or stale; check provider and station",
                        "created_at_utc": datetime.now(timezone.utc).isoformat()}]
    assert "verify the station" in watch_reasons(state)[0]
    state["alerts"][0]["created_at_utc"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    assert watch_reasons(state) == []


def test_history_keeps_real_received_times_without_duplicates_or_failed_reappend(tmp_path):
    observation = datetime.now(timezone.utc)
    failed = False
    def fetch(name, now):
        if failed:
            raise httpx.ReadTimeout("Synthetic failure")
        rows = [{"id": "4913", "name": "Synthetic station", "quality": "current", "level_m": 3,
                 "observed_at_utc": observation.isoformat()}] if name == "gauges" else []
        doc = {"records": rows, "retrieved_at_utc": now.isoformat()}
        return doc, json.dumps(doc).encode()
    monitor = LiveMonitor(tmp_path, fetch)
    monitor.refresh(force=True)
    monitor.refresh(force=True)
    assert len(monitor.snapshot()["gauge_history"]["4913"]) == 1
    observation += timedelta(seconds=1)
    monitor.refresh(force=True)
    assert len(monitor.snapshot()["gauge_history"]["4913"]) == 2
    failed = True
    monitor.state["gauge_history"]["4913"] = []
    monitor.refresh(force=True)
    assert monitor.snapshot()["gauge_history"]["4913"] == []


def test_preview_uses_actual_catalog_timestamp_and_retains_older_image(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    acquired = (now - timedelta(days=3)).isoformat()
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="JPEG")
    thumbnail = "https://datahub.creodias.eu/odata/v1/Assets(test)/$value"
    def respond(request):
        if request.method == "POST":
            return httpx.Response(200, json={"features": [{"id": "SYNTHETIC_SCENE", "collection": "sentinel-2-l1c",
                                  "properties": {"datetime": acquired}, "assets": {"thumbnail": {"href": thumbnail}}}]})
        return httpx.Response(200, content=buffer.getvalue())
    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    preview = latest_preview(tmp_path, now)
    assert preview["acquired_at_utc"] == acquired and preview["model_input_available"] is False
    assert preview["preview_data_url"].startswith("data:image/jpeg;base64,")
    newer = {"scene_id": "NEW_SYNTHETIC_SCENE", "acquired_at_utc": now.isoformat(),
             "retrieved_at_utc": now.isoformat(), "reason": "Synthetic unavailable preview"}
    retained = retain_preview(preview, newer)
    assert retained["acquired_at_utc"] == acquired and retained["preview_sha256"] == preview["preview_sha256"]
    assert retained["latest_catalog_scene_id"] == newer["scene_id"]
    assert npt_time("2026-11-15T00:00:00") == "Time zone unavailable"
