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
from app.components.layout import card, html, render_badge, status_list
from app.components.map_view import build_layers, render_map_workspace
from app.context import AppContext, render_page_frame
from app.data_loader import load_run_artifacts
from app.remote_runs import list_remote_runs, load_remote_run
from app.runs import outputs_dir, run_title

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


def _jobs_strip() -> tuple[str, float] | None:
    """Show running/failed jobs; return (run_id, finished timestamp) of the newest finished job."""
    data, err = api_client.jobs()
    if err:
        return None
    rows = data.get("jobs", [])
    for j in [j for j in rows if j["status"] in ("queued", "running")][:3]:
        if j["status"] == "running" and j.get("tiles_total"):
            st.progress(j["tiles_done"] / j["tiles_total"],
                        text=f"Running model on {j['scene_id']} · {j['tiles_done']}/{j['tiles_total']} tiles")
        else:
            st.caption(f"Queued: {j['scene_id']}")
    failed = [j for j in rows if j["status"] == "failed"]
    if failed and (not rows or rows[0]["status"] == "failed"):
        st.caption(f"⚠ Last run failed: {failed[0].get('error')}")
    done = [j for j in rows if j["status"] == "succeeded"]
    if not done:
        return None
    newest = max(done, key=lambda j: _ts(j.get("finished_at_utc")))
    return newest["run_id"], _ts(newest.get("finished_at_utc"))


def _run_form() -> None:
    """Pick a scene and run the loaded model. Shows a setup checklist until everything is ready."""
    st_json, err = api_client.status()
    loaded = bool(st_json and st_json.get("model_loaded"))
    sc, _ = api_client.scenes() if st_json else (None, None)
    scenes = [x["scene_id"] for x in (sc or {}).get("scenes", []) if x.get("model_compatible")]
    if not (loaded and scenes):
        scenes_dir = (sc or {}).get("scenes_dir") or "data/raw/sentinel2"
        model_detail = ("Start the model service first" if err else
                        f"{st_json.get('model_error') or 'No checkpoint'} — restart the service with `make model`")
        rows = [
            ("1 · Model service running", "Run `make model` in a terminal" if err else api_client.api_url(),
             "wait" if err else "ok", render_badge("Missing" if err else "OK", "neutral" if err else "success")),
            ("2 · Your model loaded", model_detail if not loaded else
             f"{(st_json.get('model') or {}).get('checkpoint_path', '')}",
             "ok" if loaded else "wait", render_badge("OK" if loaded else "Missing", "success" if loaded else "neutral")),
            ("3 · Scenes to run on", f"{len(scenes)} usable in {scenes_dir}" if scenes else
             f"Copy Sentinel-2 GeoTIFFs (6 or 13 bands) into {scenes_dir}",
             "ok" if scenes else "wait", render_badge("OK" if scenes else "Missing", "success" if scenes else "neutral")),
        ]
        html(card("Run your model — setup", status_list(rows)))
        return
    model = st_json.get("model") or {}
    rt = st_json.get("runtime") or {}
    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    with c1:
        scene = st.selectbox("Scene", scenes, key="pv_live_scene")
    with c2:
        if st.button("Run model", type="primary", icon=":material/play_arrow:", width="stretch"):
            data, e = api_client.submit_prediction(scene, 0.5)
            st.toast(e or f"Started {data['run_id']}", icon=":material/error:" if e else ":material/schedule:")
    st.caption(f"Model {model.get('architecture', '—')} · epoch {model.get('epoch', '—')} · "
               f"device {rt.get('device', '—')} · input units {st_json.get('input_units', 'dn')}")


def _live_body(height: int) -> None:
    api_newest = _jobs_strip()
    local = _newest_local(outputs_dir())
    a, source = None, ""
    if api_newest and (not local or api_newest[1] >= local[1]):
        rid = api_newest[0]
        if (outputs_dir() / rid).is_dir():
            a, source = _cached_local(rid, api_newest[1]), f"{outputs_dir()}/{rid}"
        else:
            a, source = load_remote_run(rid), f"inference API · {rid}"
    elif local:
        a, source = _cached_local(*local), f"{outputs_dir()}/{local[0]}"
    else:
        remote, _ = list_remote_runs(api_client.api_url())
        if remote:
            a, source = load_remote_run(remote[0]), f"inference API · {remote[0]}"
    now = datetime.now().strftime("%H:%M:%S")
    if a is None:
        st.caption(f"Waiting for the first model output… checked {now}")
        return
    meta = a.prediction_metadata or a.observed_metadata or {}
    model = meta.get("model_id") or meta.get("model") or "—"
    html(f'<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:6px">'
         f'{render_badge("Live", "success", dot=True)}<b>{escape(run_title(a))}</b>'
         f'<span style="color:var(--pv-text-muted);font-size:12px">{escape(str(model))} · '
         f'{escape(source)} · checked {now}</span></div>')
    render_map_workspace(build_layers(a, show_retrospective=False), height=height, mode="full")


def render_live_section(ctx: AppContext) -> None:
    render_page_frame(ctx, "Live Prediction", "Live Prediction",
                      "Run your model on a scene and watch the result appear. Refreshes every 5 s.")
    _run_form()
    live = st.toggle("Auto-refresh", value=True, key="pv_live_on")
    st.fragment(run_every=5 if live else None)(_live_body)(660)
