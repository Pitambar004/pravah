"""Alerts — who would be warned, from a prediction (or, retrospectively, an observation).

Alert preview only: groups exposed buildings/roads/bridges by nearest OSM settlement
and drafts a message for a human operator. No message is sent. Tiers are illustrative
thresholds, not an official warning standard.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from app.components.empty_state import render_empty_state
from app.components.layout import render_badge
from app.components.map_view import (
    MapLayer,
    build_layers,
    render_map_workspace,
)
from app.components.metric_card import render_metric_card
from app.context import AppContext, render_page_frame
from app.runs import all_runs, model_tag, run_role, run_title
from app.theme import TOKENS
from pravah.gis.alerts import (
    DEFAULT_TIERS,
    UNASSIGNED,
    Settlement,
    draft_message,
    group_by_settlement,
)

TIER_VARIANT = {"Warning": "error", "Watch": "warning", "Advisory": "accent", "None": "neutral"}


def _settlements_path() -> Path:
    return Path(os.environ.get("PRAVAH_SETTLEMENTS", "data/processed/osm/thematic/settlements.gpkg"))


@st.cache_data(show_spinner=False)
def _load_settlements(path: str) -> tuple[list[Settlement], str | None]:
    p = Path(path)
    if not p.is_file():
        return [], f"{path} not found"
    try:
        import geopandas as gpd

        df = gpd.read_file(p)
        if df.crs is not None and df.crs.to_epsg() != 4326:
            df = df.to_crs(4326)
        pts = df[df.geometry.geom_type == "Point"]
        return [Settlement(str(r["name"]), r.geometry.x, r.geometry.y, str(r.get("osm_value", "")))
                for _, r in pts.iterrows()], None
    except Exception as exc:  # noqa: BLE001 - shown to the user, never hidden
        return [], f"Could not read settlements: {exc}"


def render_alerts_section(ctx: AppContext) -> None:
    render_page_frame(ctx, "Alerts", "Alerts",
                      "Settlements that would be alerted from a predicted flood map, with a draft message.")
    st.caption("⚠ Simulated: nothing is sent and these are not official warnings (issued in Nepal by DHM and "
               "local authorities). Alert levels are illustrative.")
    if ctx.is_synthetic:
        return
    runs = [r for r in all_runs() if run_role(r) in ("prediction", "post_event", "unknown")]
    if not runs:
        render_empty_state("Nothing to alert on yet", "Needs a prediction (or a post-event run as a drill) with "
                           "exposure layers.", icon_name="campaign", compact=True)
        return
    runs.sort(key=lambda r: (run_role(r) != "prediction", model_tag(r) != "Your model"))
    by_id = {r.run_id: r for r in runs}
    rid = st.selectbox("Based on", list(by_id), format_func=lambda i: run_title(by_id[i]), key="pv_alert_run")
    a = by_id[rid]
    kind = "prediction" if a.is_prediction else "observed"
    if kind == "observed":
        st.caption("Drill: this map is from after the event — these are the alerts a correct forecast should have "
                   "sent beforehand.")
    if not a.exposure_features:
        render_empty_state(
            "No exposure layers for this run", "Run the exposure analysis on its flood mask with --features-dir.",
            action_command=f"python scripts/exposure.py --mask outputs/{rid}/<mask>.tif --buildings "
                           "data/processed/osm/thematic/buildings.gpkg --roads data/processed/osm/thematic/roads.gpkg "
                           "--bridges data/processed/osm/thematic/bridges.gpkg --osm-date 2026-01-01 "
                           f"--out outputs/{rid}/exposure_summary.json --features-dir outputs/{rid}",
            icon_name="apartment", compact=True)
        return

    settlements, s_err = _load_settlements(str(_settlements_path()))
    if s_err:
        st.caption(f"⚠ {s_err}; set PRAVAH_SETTLEMENTS to an OSM place layer.")
    areas = group_by_settlement(a.exposure_features, settlements, tiers=DEFAULT_TIERS)
    alerted = [x for x in areas if x.tier != "None" and x.settlement != UNASSIGNED]
    exp = a.exposure_summary or {}
    cols = st.columns(3)
    specs = [
        {"label": "Settlements alerted", "value": len(alerted), "icon_name": "campaign"},
        {"label": "Buildings in flood extent", "value": exp.get("building_footprints_intersecting"), "icon_name": "home"},
        {"label": "Road in flood extent", "value": None if exp.get("road_length_m_intersecting") is None
         else exp["road_length_m_intersecting"] / 1000, "unit": "km", "icon_name": "road"},
    ]
    for col, spec in zip(cols, specs, strict=True):
        with col:
            render_metric_card(**spec)

    layers = build_layers(a, show_retrospective=False)
    pts = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"label": f"{x.tier}: {x.settlement} ({x.buildings} buildings)"},
         "geometry": {"type": "Point", "coordinates": [x.lon, x.lat]}} for x in alerted if x.lon is not None]}
    layers.insert(0, MapLayer(id="alerts", name="Alerted settlements", group="Infrastructure", kind="derived",
                              color=TOKENS["danger"], data=pts, opacity=0.9, source="OSM places · nearest-feature",
                              description="Settlements with exposed buildings at or above the Advisory level.",
                              limitation="Illustrative levels; not an official warning.",
                              unavailable_reason="No settlement reaches an alert level"))
    render_map_workspace(layers, height=560, mode="full")

    if alerted:
        left, right = st.columns([3, 2])
        with left:
            st.dataframe([{"Level": x.tier, "Settlement": x.settlement, "Buildings": x.buildings,
                           "Road (m)": round(x.road_m), "Bridges": x.bridges} for x in alerted],
                         hide_index=True, width="stretch", height=300)
        with right:
            names = [x.settlement for x in alerted]
            area = alerted[names.index(st.selectbox("Draft message for", names, key="pv_alert_pick"))]
            issued = (a.prediction_metadata or {}).get("issued_at_utc") or a.acquired_at or "—"
            st.markdown(render_badge(area.tier, TIER_VARIANT[area.tier], dot=True), unsafe_allow_html=True)
            st.code(draft_message(area, kind, f"{issued}"), language=None, wrap_lines=True)
