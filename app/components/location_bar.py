"""Shared place search: pick a location, get imagery + a model run there, show it on every tab.

Flow for a selected place:
* a finished run for it already exists (API job list or a ``loc_*`` run folder) → it becomes the
  active run (``pv_run``) on all run-based tabs;
* a job is queued/running → a progress bar (download ≈ 60 %, model tiles ≈ 40 %) refreshes every 3 s;
* nothing yet → ``POST /api/v1/locations/predictions`` is sent once (newest usable Sentinel-2 L1C).

The selected place and query live in session state, so the search carries across pages.
"""
from __future__ import annotations

import json
import math
from typing import Any

import streamlit as st

from app import api_client
from app.location_search import geocode
from app.runs import outputs_dir

PLACE_KEY = "pv_place"            # selected place dict, shared by all pages
QUERY_KEY = "pv_loc_query"        # last search text, shared by all pages
PENDING_RUN_KEY = "pv_pending_run"  # consumed by Home.py before the run selector renders
APPLIED_KEY = "pv_loc_applied"    # runs already made active once (user may switch away afterwards)
HALF_KM = 5.0                     # matches the API default box


@st.cache_data(ttl=86400, show_spinner="Searching…")
def _geocode(query: str):
    return geocode(query)


def place_key(place: dict[str, Any]) -> str:
    return f"{place['lat']:.4f},{place['lon']:.4f}"


def short_name(place: dict[str, Any]) -> str:
    return place["name"].split(",")[0]


def box_bounds(place: dict[str, Any], half_km: float = HALF_KM) -> list[list[float]]:
    """[[south, west], [north, east]] of the square box the API fetches."""
    dlat = half_km / 111.32
    dlon = half_km / (111.32 * max(math.cos(math.radians(place["lat"])), 0.01))
    return [[place["lat"] - dlat, place["lon"] - dlon], [place["lat"] + dlat, place["lon"] + dlon]]


def _keep(widget_key: str, shared_key: str) -> None:
    st.session_state[shared_key] = st.session_state[widget_key]


def search_place() -> dict[str, Any] | None:
    """Search bar + result picker; searches only on Enter (geocoder policy forbids type-ahead).

    Streamlit drops widget values on page switch, so both widgets are re-seeded every run from
    shared keys that their on_change callbacks keep current."""
    wq, ws = "pv_loc_q_w", "pv_loc_sel_w"
    st.session_state[wq] = st.session_state.get(QUERY_KEY, "")
    q = st.text_input("Search location", key=wq, icon=":material/search:", label_visibility="collapsed",
                      placeholder="Search a town, river or district and press Enter",
                      on_change=_keep, args=(wq, QUERY_KEY)).strip()
    if not q:
        st.session_state.pop(PLACE_KEY, None)
        return None
    places, err = _geocode(q)
    if err:
        st.warning(err)
        return None
    if not places:
        st.caption(f"No places found for “{q}”.")
        return None
    names = [p["name"] for p in places]
    saved = st.session_state.get(PLACE_KEY)
    st.session_state[ws] = names.index(saved["name"]) if saved and saved["name"] in names else 0
    i = st.selectbox("Location", range(len(places)), format_func=lambda k: names[k], key=ws,
                     on_change=lambda: st.session_state.__setitem__(PLACE_KEY, places[st.session_state[ws]]))
    st.caption("Search © OpenStreetMap contributors")
    st.session_state[PLACE_KEY] = places[i]
    return places[i]


def same_point(loc: dict[str, Any] | None, place: dict[str, Any]) -> bool:
    return bool(loc) and round(loc["lat"], 4) == round(place["lat"], 4) and round(loc["lon"], 4) == round(place["lon"], 4)


def job_for(place: dict[str, Any]) -> dict[str, Any] | None:
    """Newest API job for this point (jobs are listed newest first)."""
    data, err = api_client.jobs()
    if err:
        return None
    return next((j for j in data.get("jobs", []) if same_point(j.get("location"), place)), None)


def local_run_for(place: dict[str, Any]) -> str | None:
    """Newest finished ``loc_*`` run folder for this point (survives API restarts)."""
    root = outputs_dir()
    if not root.is_dir():
        return None
    best = None
    for d in root.glob("loc_*"):
        md = d / "observed_metadata.json"
        if not md.is_file():
            continue
        try:
            study = json.loads(md.read_text()).get("case_study") or {}
        except (OSError, ValueError):
            continue
        if same_point(study, place) and (best is None or md.stat().st_mtime > best[1]):
            best = (d.name, md.stat().st_mtime)
    return best[0] if best else None


def job_progress(job: dict[str, Any]) -> tuple[float, str]:
    """Overall fraction (download ≈ 60 %, model ≈ 40 %) and a status line."""
    if job["status"] == "queued":
        return 0.0, "Queued behind other model runs"
    if job.get("stage") == "fetching":
        return 0.6 * (job.get("stage_progress") or 0.0), job.get("stage_detail") or "Starting download"
    if job.get("tiles_total"):
        done, total = job["tiles_done"], job["tiles_total"]
        return 0.6 + 0.4 * done / total, f"Running model · {done}/{total} tiles"
    return 0.6, "Starting model"


def _activate(run_id: str) -> None:
    """Make the place's run the active run on every tab, once (the user can switch away afterwards)."""
    applied = st.session_state.setdefault(APPLIED_KEY, [])
    if run_id in applied:
        return
    applied.append(run_id)
    if st.session_state.get("pv_run") != run_id:
        st.session_state[PENDING_RUN_KEY] = run_id
        st.rerun()


def _submit(place: dict[str, Any]) -> str | None:
    return api_client.submit_location_prediction(place["lat"], place["lon"], place["name"])[1]


def place_state(place: dict[str, Any]) -> dict[str, Any]:
    """{"status": done|pending|failed|error, "run_id", "job", "error"} — submits a job if nothing exists."""
    job = job_for(place)
    if job and job["status"] in ("queued", "running"):
        return {"status": "pending", "job": job, "run_id": job["run_id"]}
    if job and job["status"] == "succeeded":
        return {"status": "done", "job": job, "run_id": job["run_id"]}
    local = local_run_for(place)
    if local:
        return {"status": "done", "job": job, "run_id": local}
    if job and job["status"] == "failed":
        return {"status": "failed", "job": job, "run_id": job["run_id"], "error": job.get("error")}
    err = _submit(place)
    if err:
        return {"status": "error", "error": err}
    return {"status": "pending", "job": job_for(place), "run_id": None}


def _status_body(place: dict[str, Any], activate: bool) -> None:
    state = place_state(place)
    name = short_name(place)
    if state["status"] == "pending":
        job = state.get("job")
        frac, text = job_progress(job) if job else (0.0, "Submitting")
        st.progress(min(frac, 1.0), text=f"{name}: {text}")
    elif state["status"] == "done":
        st.caption(f"✓ {name}: model run `{state['run_id']}` is shown on all map and evidence tabs.")
        if activate:
            _activate(state["run_id"])
    elif state["status"] == "failed":
        st.warning(f"{name}: {state.get('error')}")
        if st.button("Try again", key=f"pv_loc_retry_{place_key(place)}", icon=":material/refresh:"):
            err = _submit(place)
            st.toast(err or "Started again", icon=":material/error:" if err else ":material/schedule:")
            st.rerun()
    else:
        st.warning(f"Could not start imagery download for {name}: {state.get('error')}")


def render_location_bar(activate: bool = True) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Search bar + status for the selected place. Returns (place, state at render time)."""
    place = search_place()
    if not place:
        return None, None
    state = place_state(place)
    pending = state["status"] == "pending"
    st.fragment(run_every=3 if pending else None)(_status_body)(place, activate)
    return place, state
