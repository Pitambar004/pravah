"""Live Prediction — the newest prediction map, refreshed automatically.

Watches two places and shows whichever has the newest finished run:
* the outputs directory (shared Fuzzball volume or local disk), and
* the inference API's job queue (progress of running jobs + finished runs).

The map is re-built only when a newer run appears, so pan/zoom is kept between
refreshes. Nothing is shown that is not on disk or returned by the API.
"""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

import streamlit as st

from app import api_client
from app.components.context_layers import with_downloaded_context
from app.components.layout import card, html, render_badge, status_list
from app.components.location_bar import (
    box_bounds,
    job_for,
    job_progress,
    local_run_for,
    render_location_bar,
    same_point,
    short_name,
)
from app.components.map_view import MapLayer, build_layers, render_map_workspace
from app.context import AppContext
from app.data_loader import load_run_artifacts
from app.location_search import scenes_covering
from app.remote_runs import list_remote_runs, load_remote_run
from app.runs import outputs_dir, run_title
from app.theme import TOKENS

LIVE_FILES = ("predicted_flood.geojson", "observed_water.geojson", "prediction_metadata.json",
              "observed_metadata.json")


def _newest_local(root: Path) -> tuple[str, float] | None:
    best = None
    if not root.is_dir():
        return None
    for d in root.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        stamps = [(d / f).stat().st_mtime for f in LIVE_FILES if (d / f).is_file()]
        if stamps and (best is None or max(stamps) > best[1]):
            best = (d.name, max(stamps))
    return best


def _cached_local(run_id: str, mtime: float):
    key = ("pv_live_run", run_id, mtime)
    if st.session_state.get("pv_live_key") != key:
        st.session_state["pv_live_key"] = key
        st.session_state["pv_live_art"] = load_run_artifacts(outputs_dir() / run_id)
    return st.session_state["pv_live_art"]


def _ts(iso: str | None) -> float:
    try:
        return datetime.fromisoformat((iso or "").replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _jobs_strip(skip_place: dict | None = None) -> tuple[str, float] | None:
    """Show running/failed jobs; return (run_id, finished timestamp) of the newest finished job.

    Jobs for the searched place are left to the location bar's progress bar."""
    data, err = api_client.jobs()
    if err:
        return None
    rows = data.get("jobs", [])
    active = [j for j in rows if j["status"] in ("queued", "running")
              and not (skip_place and same_point(j.get("location"), skip_place))]
    for j in active[:3]:
        name = j.get("scene_id") or ((j.get("location") or {}).get("place") or "location").split(",")[0]
        if j["status"] == "queued":
            st.caption(f"Queued: {name}")
        else:
            frac, text = job_progress(j) if j.get("location") else (
                (j["tiles_done"] / j["tiles_total"], f"Running model · {j['tiles_done']}/{j['tiles_total']} tiles")
                if j.get("tiles_total") else (0.0, "Starting"))
            st.progress(min(frac, 1.0), text=f"{name}: {text}")
    failed = [j for j in rows if j["status"] == "failed"]
    if failed and (not rows or rows[0]["status"] == "failed") and not (
            skip_place and same_point(failed[0].get("location"), skip_place)):
        st.caption(f"⚠ Last run failed: {failed[0].get('error')}")
    done = [j for j in rows if j["status"] == "succeeded"]
    if not done:
        return None
    newest = max(done, key=lambda j: _ts(j.get("finished_at_utc")))
    return newest["run_id"], _ts(newest.get("finished_at_utc"))


def _run_form(place: dict | None) -> None:
    """Model status, plus running on scenes already on disk. Searched places are fetched by the location bar."""
    st_json, err = api_client.status()
    loaded = bool(st_json and st_json.get("model_loaded"))
    if not loaded:
        model_detail = ("Start the model service first" if err else
                        f"{st_json.get('model_error') or 'No checkpoint'} — restart the service with `make model`")
        rows = [
            ("1 · Model service running", "Run `make model` in a terminal" if err else api_client.api_url(),
             "wait" if err else "ok", render_badge("Missing" if err else "OK", "neutral" if err else "success")),
            ("2 · Your model loaded", model_detail, "wait", render_badge("Missing", "neutral")),
            ("3 · Imagery", "Searched places download Sentinel-2 automatically", "ok", render_badge("Auto", "success")),
        ]
        html(card("Run your model — setup", status_list(rows)))
        return
    model = st_json.get("model") or {}
    rt = st_json.get("runtime") or {}
    model_line = (f"Model {model.get('architecture', '—')} · epoch {model.get('epoch', '—')} · "
                  f"device {rt.get('device', '—')} · input units {st_json.get('input_units', 'dn')}")
    if place is None:
        st.caption("Search a location to run your model there — the newest cloud-free Sentinel-2 image is "
                   f"downloaded automatically · {model_line}")
        return
    sc, _ = api_client.scenes()
    compatible = [x for x in (sc or {}).get("scenes", []) if x.get("model_compatible")]
    hits = [h for h in scenes_covering(compatible, place["lat"], place["lon"]) if not h.startswith("auto/")]
    if hits:
        c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
        with c1:
            scene = st.selectbox(f"Or run on a scene already on disk that covers {short_name(place)}", hits,
                                 key="pv_live_scene")
        with c2:
            if st.button("Run model", icon=":material/play_arrow:", width="stretch"):
                data, e = api_client.submit_prediction(scene, 0.5)
                st.toast(e or f"Started {data['run_id']}", icon=":material/error:" if e else ":material/schedule:")
    st.caption(model_line)


def _load_run(rid: str):
    folder = outputs_dir() / rid
    if folder.is_dir():
        mtime = max((folder / f).stat().st_mtime for f in LIVE_FILES if (folder / f).is_file()) \
            if any((folder / f).is_file() for f in LIVE_FILES) else 0.0
        return _cached_local(rid, mtime), f"{outputs_dir()}/{rid}"
    return load_remote_run(rid), f"inference API · {rid}"


def _place_run(place: dict) -> str | None:
    job = job_for(place)
    if job and job["status"] == "succeeded":
        return job["run_id"]
    return local_run_for(place)


def _live_body(height: int, place: dict | None = None) -> None:
    api_newest = _jobs_strip(skip_place=place)
    now = datetime.now().strftime("%H:%M:%S")
    a, source = None, ""
    if place is not None:
        rid = _place_run(place)
        if rid is None:  # still downloading/predicting: frame the place instead of showing an unrelated run
            html(f'<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:6px">'
                 f'{render_badge("Waiting", "neutral", dot=True)}<b>{escape(short_name(place))}</b>'
                 f'<span style="color:var(--pv-text-muted);font-size:12px">The model result appears here when the '
                 f'run finishes · box shows the 10 km area being fetched · checked {now}</span></div>')
            (s_, w_), (n_, e_) = box_bounds(place)
            area = MapLayer(id="requested_area", name=f"Area being fetched · {short_name(place)}", group="Request",
                            kind="context", color=TOKENS["accent"], opacity=0.1, dash="6 4",
                            data={"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                                  "geometry": {"type": "Polygon", "coordinates": [[[w_, s_], [e_, s_], [e_, n_],
                                                                                   [w_, n_], [w_, s_]]]}}]},
                            source="Search result (OpenStreetMap Nominatim)",
                            description="10 km box sent to the imagery download; not an observation.")
            render_map_workspace(with_downloaded_context([area]), height=height, mode="full",
                                 initial_bounds=box_bounds(place))
            return
        a, source = _load_run(rid)
    else:
        local = _newest_local(outputs_dir())
        if api_newest and (not local or api_newest[1] >= local[1]):
            a, source = _load_run(api_newest[0])
        elif local:
            a, source = _cached_local(*local), f"{outputs_dir()}/{local[0]}"
        else:
            remote, _ = list_remote_runs(api_client.api_url())
            if remote:
                a, source = load_remote_run(remote[0]), f"inference API · {remote[0]}"
    if a is None:
        st.caption(f"Waiting for the first model output… checked {now}")
        return
    meta = a.prediction_metadata or a.observed_metadata or {}
    model = meta.get("model_id") or meta.get("model") or "—"
    html(f'<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:6px">'
         f'{render_badge("Live", "success", dot=True)}<b>{escape(run_title(a))}</b>'
         f'<span style="color:var(--pv-text-muted);font-size:12px">{escape(str(model))} · '
         f'{escape(source)} · checked {now}</span></div>')
    render_map_workspace(with_downloaded_context(build_layers(a, show_retrospective=False)), height=height, mode="full")


def render_live_section(ctx: AppContext) -> None:
    place, _ = render_location_bar()
    _run_form(place)
    live = st.toggle("Auto-refresh", value=True, key="pv_live_on")
    st.fragment(run_every=5 if live else None)(_live_body)(660, place)
