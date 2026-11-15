"""Flood Intelligence Map — the primary map-centric workspace."""
from __future__ import annotations

import json

import streamlit as st

from app.components.layout import render_section_header
from app.components.map_view import KIND_LABELS, build_layers, render_map_workspace
from app.context import AppContext, render_page_frame


def render_flood_map_section(ctx: AppContext) -> None:
    render_page_frame(
        ctx,
        "Flood Intelligence Map",
        "Flood Intelligence Map",
        "Model-derived water extent, study boundary and retrospective reference layers. "
        "Click a feature to inspect its attributes, source and limitations.",
    )
    a = ctx.artifacts
    layers = build_layers(a)
    available = [layer for layer in layers if layer.available]

    t_l, t_r = st.columns([4, 1], vertical_alignment="center")
    with t_l:
        n_avail = len(available)
        st.caption(
            f"{n_avail} of {len(layers)} layers have data in this run · display CRS EPSG:4326 · "
            "metric calculations in EPSG:32645"
        )
    with t_r:
        export = {
            "type": "FeatureCollection",
            "pravah_export": {
                "run_id": a.run_id,
                "is_synthetic": a.is_synthetic,
                "note": "Model-derived water polygons. Not ground truth; not damage.",
            },
            "features": (a.water_geojson or {}).get("features", []),
        }
        st.download_button(
            "Export water GeoJSON",
            data=json.dumps(export, indent=2),
            file_name=f"pravah_water_{a.run_id}.geojson",
            mime="application/geo+json",
            icon=":material/download:",
            disabled=not a.water_geojson,
            width="stretch",
            help="Downloads the model-derived water polygons for the selected run.",
        )

    render_map_workspace(layers, height=720, mode="full", synthetic=a.is_synthetic)

    tab_layers, tab_features, tab_change = st.tabs(["Layer catalogue", "Water feature attributes", "Pre/post change"])
    with tab_layers:
        st.dataframe(
            [
                {
                    "Layer": layer.name,
                    "Group": layer.group,
                    "Type": KIND_LABELS.get(layer.kind, layer.kind),
                    "Status": "Available" if layer.available else "Unavailable",
                    "Features": len(layer.data["features"]) if layer.available else None,
                    "Source / reason": layer.source if layer.available else layer.unavailable_reason,
                }
                for layer in layers
            ],
            hide_index=True,
            width="stretch",
            column_config={"Features": st.column_config.NumberColumn(format="%d")},
        )
    with tab_features:
        feats = (a.water_geojson or {}).get("features", [])
        if feats:
            st.dataframe(
                [
                    {
                        "#": i + 1,
                        "Label": (f.get("properties") or {}).get("label", f"Feature {i + 1}"),
                        "Class": (f.get("properties") or {}).get("class", "—"),
                        "Area (m²)": (f.get("properties") or {}).get("area_m2"),
                        "Geometry": (f.get("geometry") or {}).get("type", "—"),
                    }
                    for i, f in enumerate(feats)
                ],
                hide_index=True,
                width="stretch",
                column_config={"Area (m²)": st.column_config.NumberColumn(format="%.1f")},
            )
        else:
            render_section_header("No water polygons", "Stitch inference tiles and run scripts/exposure.py to produce them.")
    with tab_change:
        render_change_summary(a.change_summary)


def render_change_summary(summary: dict | None) -> None:
    """Tabular pre/post change summary from change_summary.json (counts measured, never estimated)."""
    if not summary:
        render_section_header(
            "No change analysis for this run",
            "Select a post-event run and run scripts/compare_observations.py --pre <run> --post <run>.")
        return
    area = summary.get("area_km2") or {}
    pc = summary.get("pixel_counts") or {}
    render_section_header(
        f"{(summary.get('pre_acquired_at_utc') or '')[:10]} → {(summary.get('post_acquired_at_utc') or '')[:10]}",
        f"{summary.get('pre_run_id')} vs {summary.get('post_run_id')} · same model: {summary.get('same_model')} · "
        f"comparable: {summary.get('comparable_fraction_of_grid', 0) * 100:.1f}% of grid",
    )
    labels = {"stable_no_water": "No water on both dates", "persistent_water": "Water on both dates",
              "new_water": "Water newly detected", "water_no_longer_detected": "Water no longer detected",
              "not_comparable": "Not comparable (cloud / no data on either date)"}
    st.dataframe([{"Class": labels.get(k, k), "Area (km²)": area.get(k), "Pixels": pc.get(k)} for k in labels],
                 hide_index=True, width="stretch",
                 column_config={"Area (km²)": st.column_config.NumberColumn(format="%.3f"),
                                "Pixels": st.column_config.NumberColumn(format="%d")})
    w = summary.get("water_km2_within_comparable") or {}
    st.caption(f"Water inside the comparable area: pre {w.get('pre')} km² → post {w.get('post')} km².")
    for lim in summary.get("interpretation_limits") or []:
        st.caption(f"• {lim}")
