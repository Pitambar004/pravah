"""Overview — what PRAVAH shows and which evidence exists for the case study."""
from __future__ import annotations

import streamlit as st

from app.components.empty_state import render_empty_state
from app.components.layout import (
    card,
    html,
    render_badge,
    render_section_header,
    status_list,
)
from app.components.map_view import build_layers, render_map_workspace
from app.components.metric_card import render_metric_card
from app.components.timeline import build_timeline_svg, timeline_legend
from app.context import AppContext, render_page_frame
from pravah.replay.eligibility import classify_asset

OVERVIEW_CUTOFF = "2026-08-26T11:00:00+05:45"


def _is_fixture(obj) -> bool:
    return bool(isinstance(obj, dict) and obj.get("is_synthetic"))


def artifact_status_rows(ctx: AppContext) -> list[tuple[str, str | None, str, str]]:
    """Processing status derived only from files actually present (or fixtures, labelled)."""
    a = ctx.artifacts

    def row(name, detail, present, fixture=False):
        if fixture:
            return (name, detail, "warn", render_badge("Fixture", "warning"))
        if present:
            return (name, detail, "ok", render_badge("Present", "success"))
        return (name, detail, "wait", render_badge("Missing", "neutral"))

    return [
        row("Replay source manifest",
            ctx.catalog_source if ctx.catalog_source and not ctx.catalog_is_synthetic else "data/manifests/assets.json",
            bool(ctx.catalog), ctx.catalog_is_synthetic),
        row("Study area (AOI)", "config/aois/*.geojson", bool(a.aoi_geojson), _is_fixture(a.aoi_geojson)),
        row("Inference metadata", "observed_metadata.json", bool(a.observed_metadata), _is_fixture(a.observed_metadata)),
        row("Stitched water mask", "observed_water.tif", bool(a.observed_water_tif)),
        row("Water polygons", "exposure.geojson / polygons.geojson", bool(a.water_geojson), _is_fixture(a.water_geojson)),
        row("Exposure summary", "exposure_summary.json", bool(a.exposure_summary), _is_fixture(a.exposure_summary)),
        row("Compute benchmark", "benchmark.json", bool(a.benchmark), _is_fixture(a.benchmark)),
    ]


def render_overview_section(ctx: AppContext) -> None:
    render_page_frame(
        ctx,
        "Overview",
        "Flood Intelligence Overview",
        "Real satellite water segmentation, mapped infrastructure intersections and source provenance.",
    )

    bar_l, bar_r = st.columns([3, 2], vertical_alignment="bottom")
    with bar_l:
        st.selectbox(
            "Case study",
            [f"{ctx.study['region']} · {ctx.study['corridor']} · {ctx.study['event_date']}"],
            help="Only one case study is configured in this prototype.",
            key="pv_case_study",
        )
    with bar_r:
        st.page_link("views/flood_map.py", label="Open map workspace", icon=":material/arrow_forward:")

    a = ctx.artifacts
    exp = a.exposure_summary or {}
    sats = [c for c in ctx.catalog if "sentinel" in str(c.get("platform", "")).lower()]
    water = exp.get("water_area_m2")
    roads = exp.get("road_length_m_intersecting")

    cols = st.columns(4)
    specs = [
        {"label": "Satellite catalog records", "icon_name": "satellite_alt", "value": len(sats) if ctx.catalog else None,
             "unit": "records", "empty_label": "No manifest",
             "subtext": ("Synthetic fixture catalog" if ctx.catalog_is_synthetic else (ctx.catalog_source or "Awaiting data/manifests/assets.json"))},
        {"label": "Mapped water extent", "icon_name": "water", "value": (water / 10_000) if water is not None else None,
             "unit": "ha", "empty_label": "Awaiting processing", "subtext": ctx.source_note("exposure_summary.json") if water is not None else "Requires stitched mask"},
        {"label": "Buildings in mapped extent", "icon_name": "apartment", "value": exp.get("building_footprints_intersecting"),
             "unit": "footprints", "empty_label": "Awaiting processing",
             "subtext": "Potential exposure · OSM" if exp else "Requires exposure run",
             "help_text": "OSM building footprints intersecting the mapped water extent. Not damage."},
        {"label": "Road length in mapped extent", "icon_name": "road", "value": (roads / 1000) if roads is not None else None,
             "unit": "km", "empty_label": "Awaiting processing",
             "subtext": "Potential exposure · OSM" if roads is not None else "Requires exposure run"},
    ]
    for col, spec in zip(cols, specs, strict=True):
        with col:
            render_metric_card(**spec)

    col_map, col_side = st.columns([2, 1])
    with col_map:
        render_section_header("Image coverage", f"{ctx.study['region']} · {ctx.study['corridor']}")
        render_map_workspace(build_layers(a), height=460, mode="preview", synthetic=a.is_synthetic)
    with col_side:
        render_section_header("Processing status", "Artifacts present for the selected run")
        html(card(None, status_list(artifact_status_rows(ctx))))

    render_section_header(
        "Evidence availability",
        "Nepal catalog records include metadata-only entries. Use Historical Replay to inspect download and availability status.",
    )
    if ctx.catalog:
        assets = [{**c, "_status": classify_asset(c, OVERVIEW_CUTOFF)} for c in ctx.catalog]
        html(
            '<div class="pv-timeline"><div class="pv-timeline-scroll">'
            + build_timeline_svg(assets[:20], OVERVIEW_CUTOFF, strict=False)
            + "</div>" + timeline_legend(strict=False) + "</div>"
        )
        st.caption(f"Previewing {min(20, len(assets))} of {len(assets)} catalog records. Catalog entries are not downloaded images.")
        st.page_link("views/replay.py", label="Open historical replay", icon=":material/arrow_forward:")
    else:
        render_empty_state(
            "No source manifest",
            "Observation timestamps appear here once the data role publishes an audited asset manifest with "
            "acquisition and verified availability times.",
            expected_files=["data/manifests/assets.json"],
            icon_name="event_busy",
            compact=True,
        )
