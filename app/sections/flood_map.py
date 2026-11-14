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

    tab_layers, tab_features = st.tabs(["Layer catalogue", "Water feature attributes"])
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
