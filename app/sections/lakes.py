"""Glacier/lake baseline watch, keeping area, elevation and depth distinct."""

import os
from pathlib import Path

import streamlit as st

from app.components.layout import render_badge, render_page_header, render_topbar
from app.components.map_view import MapLayer, render_map_workspace
from app.nepal_data import monitor_api_url, request_monitor
from app.theme import TOKENS
from pravah.configuration import configured_url
from pravah.replay.lakes import load_lakes

LAKE_CHANGE_KEY = "pv_lake_change"


def surface_change_tile(result):
    """(value, help) for the surface-change metric from the latest successful /lakes/change result."""
    if not result:
        return "Awaiting imagery", "Run a lake comparison below with two cloud-valid Nepal optical water runs."
    note = (f"{result.get('lake_id')} · {str(result.get('pre_acquired_at_utc'))[:10]} → "
            f"{str(result.get('post_acquired_at_utc'))[:10]} · model-derived area on commonly valid pixels; "
            "not water level, depth or storage.")
    change = result.get("area_change_m2_common_valid")
    if change is None:
        return "No common valid pixels", note
    frac = result.get("comparable_fraction_of_search")
    if frac is not None:
        note += f" {frac:.0%} of the search area was comparable."
    return f"{change / 1e6:+.4f} km²", note


@st.cache_data(ttl=300, show_spinner=False)
def lake_context(root, version, endpoint):
    try:
        return load_lakes(Path(root)), None
    except (OSError, ValueError, KeyError):
        return request_monitor("GET", "/api/v1/nepal/lakes", timeout=5)


def get_lakes():
    root = os.environ.get("PRAVAH_LAKE_ROOT", "data/processed/nepal_lakes")
    manifest = Path(root) / "manifest.json"
    return lake_context(root, manifest.stat().st_mtime if manifest.exists() else 0, monitor_api_url())


def render_lakes(ctx):
    render_topbar(["PRAVAH", "Glacier & Lake Watch"], [render_badge("Rasuwa and transboundary mountains", "outline"), render_badge("Inventory + experimental change", "warning")])
    render_page_header("Glacier & Lake Watch", "Track mapped lake surfaces and separate optical area change from water level, depth and mountain failure.")
    st.page_link("views/simulate.py", label="Open 15-day Rasuwa replay", icon=":material/history:")
    context, error = get_lakes()
    if error or not context:
        st.info("Lake inventory unavailable. " + str(error or "Prepare the GLO source bundle."))
        return
    metadata = context["metadata"]
    st.caption("Glacial Lake Observatory · 2017–2024 merged maximum outlines · CC BY 4.0 · " + metadata["doi"])
    sensor = st.radio("Inventory sensor", ["Sentinel-2 optical", "Sentinel-1 SAR"], horizontal=True)
    key = "s2" if sensor == "Sentinel-2 optical" else "s1"
    doc = context["layers"][key]
    layers = [MapLayer(id="glacial_lakes", name=sensor + " lake inventory (2017–2024)", group="Geographic context", kind="context",
                       color=TOKENS["map_water"], data=doc, source="GLO / Rawlins et al. (2025), CC BY 4.0",
                       timestamp="Annual mosaics 2017–2024; merged maximum extent", description="Real inventory polygons and reported glacier connectivity.",
                       limitation=metadata["limits"])]
    render_map_workspace(layers, height=540, initial_bounds=[[28.1, 85.1], [28.7, 85.8]])
    cols = st.columns(4)
    cols[0].metric("Mapped lake baselines", len(doc["features"]))
    change = st.session_state.get(LAKE_CHANGE_KEY)
    cols[1].metric("Lake surface change", *surface_change_tile(change))
    cols[2].metric("Current water level", "Unmeasured")
    cols[3].metric("Water depth", "Unmeasured")
    st.info("Optical water masks can measure surface area and shoreline changes in cloud-valid images. Surface elevation needs altimetry or calibrated terrain/shoreline measurements; depth and storage need bathymetry. This inventory’s DSM elevations are not current water levels.")
    rows = [{"Lake ID": f["properties"].get("GLO_ID"), "Connectivity": f["properties"].get("CONNECTIVITY"),
             "Country": f["properties"].get("COUNTRY"), "Merged area (km²)": f["properties"].get("AREA_DISSOLVED"),
             "2017–2024 change (km²/year)": f["properties"].get("EXPANSION_RATE"),
             "Trend significant": f["properties"].get("EXPANSION_RATE_SIG"),
             "DSM median elevation (m)": f["properties"].get("ELEVATION_MEDIAN")} for f in doc["features"]]
    st.dataframe(rows, hide_index=True)
    st.caption("Annual source trends are not a 15-day 2026 filling curve. Individual inventory lakes are not established as the source of the Rasuwa disaster.")
    st.link_button("Inventory and source methods", metadata["source_uri"])
    st.link_button("Satellite water elevation methods", configured_url("PRAVAH_NASA_ELEVATION_URL"))
    with st.expander("Compare completed Nepal optical water runs"):
        st.caption("Requires matching model/bands/threshold, projected metric grids and verified cloud-validity. Benchmark runs are rejected.")
        ids = [f["properties"]["GLO_ID"] for f in context["layers"]["s2"]["features"]]
        if not ids:
            return
        lake_id = st.selectbox("Lake search area", ids)
        pre = st.text_input("Earlier Nepal run ID")
        post = st.text_input("Later Nepal run ID")
        if st.button("Compare lake surface", disabled=not pre or not post):
            result, error = request_monitor("POST", "/api/v1/nepal/lakes/change", json={"lake_id": lake_id, "pre_run_id": pre, "post_run_id": post}, timeout=30)
            if error:
                st.warning(error)
            else:
                st.session_state[LAKE_CHANGE_KEY] = result
                st.rerun()
        if change:
            st.json(change)
