"""Infrastructure Exposure — mapped intersections between water extent and infrastructure."""
from __future__ import annotations

import streamlit as st

from app.components.empty_state import render_empty_state
from app.components.layout import (
    card,
    html,
    kv_list,
    render_callout,
    render_section_header,
)
from app.components.map_view import build_layers, render_map_workspace
from app.components.metric_card import render_metric_card
from app.context import AppContext, render_page_frame


def exposure_rows(exp: dict) -> list[dict]:
    """Category table rows. Missing values stay None (rendered as blank), never zero."""
    unavailable = set(exp.get("unavailable_fields") or [])

    def status(key: str, value) -> str:
        if key in unavailable or value is None:
            return "Not evaluated"
        return "Evaluated"

    water = exp.get("water_area_m2")
    roads = exp.get("road_length_m_intersecting")
    pop = exp.get("modelled_population_estimate")
    return [
        {"Category": "Mapped water extent", "Measure": "Area", "Value": None if water is None else round(water / 10_000, 2),
         "Unit": "ha", "Status": status("water_area_m2", water), "Source": "Model-derived mask"},
        {"Category": "Buildings", "Measure": "Footprints intersecting", "Value": exp.get("building_footprints_intersecting"),
         "Unit": "footprints", "Status": status("buildings", exp.get("building_footprints_intersecting")), "Source": "OpenStreetMap"},
        {"Category": "Roads", "Measure": "Length intersecting", "Value": None if roads is None else round(roads / 1000, 2),
         "Unit": "km", "Status": status("roads", roads), "Source": "OpenStreetMap"},
        {"Category": "Bridges", "Measure": "Tagged features intersecting", "Value": exp.get("bridge_tagged_features_intersecting"),
         "Unit": "features", "Status": status("bridges", exp.get("bridge_tagged_features_intersecting")), "Source": "OpenStreetMap"},
        {"Category": "Resident population", "Measure": "Estimated within overlay", "Value": pop,
         "Unit": "people (modelled)", "Status": status("population", pop), "Source": "WorldPop (if ingested)"},
    ]


def render_exposure_section(ctx: AppContext) -> None:
    render_page_frame(
        ctx,
        "Infrastructure Exposure",
        "Infrastructure Exposure",
        "Mapped intersections between the model-derived water extent and OpenStreetMap infrastructure, "
        "computed in a metric projection (EPSG:32645).",
    )
    a = ctx.artifacts
    exp = a.exposure_summary
    if not exp:
        render_empty_state(
            "No exposure summary for this run",
            "Exposure metrics need a stitched water mask and OSM infrastructure layers (buildings, roads, bridges).",
            action_command="python scripts/exposure.py --mask outputs/<run_id>/observed_water.tif ...",
            expected_files=["exposure_summary.json", "exposure.geojson"],
            icon_name="apartment",
        )
        return

    render_callout(
        "Intersection with mapped water indicates <strong>potential exposure</strong> only. It is not confirmed damage, "
        "collapse or casualties.",
        title="Exposure is not damage",
        variant="warning",
    )

    roads = exp.get("road_length_m_intersecting")
    src = ctx.source_note("exposure_summary.json")
    cols = st.columns(4)
    specs = [
        {"label": "Buildings in mapped extent", "icon_name": "apartment", "value": exp.get("building_footprints_intersecting"),
             "unit": "footprints", "subtext": src},
        {"label": "Road length in mapped extent", "icon_name": "road", "value": None if roads is None else roads / 1000,
             "unit": "km", "subtext": src},
        {"label": "Bridges in mapped extent", "icon_name": "bridge", "value": exp.get("bridge_tagged_features_intersecting"),
             "unit": "features", "subtext": src},
        {"label": "Estimated resident population", "icon_name": "groups", "value": exp.get("modelled_population_estimate"),
             "unit": "modelled", "empty_label": "Not computed", "subtext": "Modelled estimate, not a headcount"},
    ]
    for col, spec in zip(cols, specs, strict=True):
        with col:
            render_metric_card(**spec)

    render_pre_event_baseline(a, exp)

    col_map, col_side = st.columns([3, 2])
    with col_map:
        render_section_header("Exposure map", "Water extent with study boundary. Infrastructure feature layers appear when exported.")
        render_map_workspace(build_layers(a, show_retrospective=False), height=520, mode="full", synthetic=a.is_synthetic)
    with col_side:
        render_section_header("Source coverage", "Inputs and invariants used by scripts/exposure.py")
        unavailable = exp.get("unavailable_fields") or []
        html(card(None, kv_list([
            ("Mask artifact", exp.get("mask_artifact", "—")),
            ("Mask class", exp.get("source_type", "—")),
            ("Valid pixels", f"{exp['valid_pixel_count']:,}" if isinstance(exp.get("valid_pixel_count"), int) else "—"),
            ("OSM snapshot", exp.get("osm_snapshot_timestamp", "Not recorded")),
            ("Metric CRS", "EPSG:32645 (UTM 45N)"),
            ("Not evaluated", ", ".join(unavailable) if unavailable else "None reported"),
        ], mono_keys=("Mask artifact", "Mask class"))))

    render_section_header("Category breakdown", "Values are mapped intersections. Blank means not evaluated — not zero.")
    st.dataframe(
        exposure_rows(exp),
        hide_index=True,
        width="stretch",
    )
    with st.expander("Raw exposure_summary.json"):
        st.json(exp)


def _pre_event_exposure(a) -> tuple[dict | None, str | None]:
    """Exposure summary of the pre-event run this post-event run was compared against."""
    pre_id = (getattr(a, "change_summary", None) or {}).get("pre_run_id")
    folder = getattr(a, "folder", None)
    if not pre_id or folder is None or getattr(a, "remote_source", None):
        return None, pre_id
    from app.data_loader import safe_read_json

    data, _ = safe_read_json(folder.parent / pre_id / "exposure_summary.json")
    return data, pre_id


def render_pre_event_baseline(a, exp: dict) -> None:
    """Same metrics on the pre-event image. Intersections that already exist before the event
    (rivers, shadow, mixed pixels) are a baseline, not event impact."""
    pre, pre_id = _pre_event_exposure(a)
    if not pre:
        return
    cs = a.change_summary or {}

    def row(label, key, scale=1.0, digits=0):
        def v(d):
            x = d.get(key)
            return None if x is None else round(x / scale, digits) if digits else int(x / scale)
        return {"Metric": label, f"Pre-event {(cs.get('pre_acquired_at_utc') or '')[:10]}": v(pre),
                f"Post-event {(cs.get('post_acquired_at_utc') or '')[:10]}": v(exp)}

    render_section_header(
        "Pre-event baseline",
        f"Same OSM snapshot and method on the pre-event image ({pre_id}). Whole valid area of each date; "
        "cloud cover differs between dates.")
    st.dataframe([
        row("Valid (cloud-free) area, km²", "valid_coverage_area_m2", 1e6, 1),
        row("Model water area, km²", "water_area_m2", 1e6, 2),
        row("Buildings intersecting water", "building_footprints_intersecting"),
        row("Road length intersecting water, km", "road_length_m_intersecting", 1e3, 1),
        row("Bridge-tagged features intersecting", "bridge_tagged_features_intersecting"),
    ], hide_index=True, width="stretch")
    pb, qb = pre.get("building_footprints_intersecting"), exp.get("building_footprints_intersecting")
    if pb is not None and qb is not None and pb >= qb:
        render_callout(
            f"The pre-event image already shows {pb:,} buildings intersecting model water (post-event: {qb:,}). "
            "Intersections at this level before the event point to over-detection (cloud/terrain shadow, river-edge "
            "mixed pixels), so these post-event counts are <strong>not evidence of flood impact</strong>.",
            title="No exposure increase over the pre-event baseline", variant="warning")
