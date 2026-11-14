"""Flood Evidence Map — Large Geospatial Canvas with Standardized Controls and Attributes."""
import streamlit as st

from app.components.empty_state import render_empty_state
from app.components.layout import (
    render_page_header,
)
from app.components.map_view import render_flood_map
from app.data_loader import RunArtifacts


def render_flood_map_section(artifacts: RunArtifacts):
    """Render the full-width Geospatial Evidence Map."""
    render_page_header(
        title="Flood Evidence Map",
        subtitle="Interactive geospatial intelligence canvas displaying satellite-inferred surface water, core corridor boundaries, and retrospective UNOSAT reference layers.",
        badge="WGS 84 (EPSG:4326)" if artifacts.has_any_artifacts else None,
        badge_variant="neutral",
    )

    if not artifacts.has_any_artifacts:
        render_empty_state(
            title="Awaiting Geographic Map Layers",
            description="Geospatial vectors and satellite-derived water polygons will appear here once tile inference and mask stitching complete.",
            action_command="python scripts/exposure.py --mask outputs/<run_id>/observed_water.tif",
            expected_files=["exposure.geojson", "observed_water.tif", "config/aois/analysis_coverage.geojson"],
        )
        return

    # Compact Layer Toolbar
    col_t1, col_t2, col_t3 = st.columns([1, 1, 2])
    with col_t1:
        show_water = st.toggle("Model Inferred Water", value=True)
    with col_t2:
        show_unosat = st.toggle("UNOSAT Retrospective Extent", value=False)
    with col_t3:
        st.markdown(
            '<div style="text-align: right; font-size: 0.75rem; color: #64748B; padding-top: 6px;">Coordinate Reference: EPSG:4326 Display · EPSG:32645 Metric Calculations</div>',
            unsafe_allow_html=True,
        )

    # Full-Width Map Canvas
    render_flood_map(
        water_geojson=artifacts.water_geojson if show_water else None,
        aoi_geojson=artifacts.aoi_geojson,
        unosat_geojson=artifacts.unosat_geojson if show_unosat else None,
        height=600,
        show_unosat=show_unosat,
    )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # 3-Column Layer Semantics Cards
    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        st.markdown(
            """
            <div class="pv-card" style="margin-bottom: 0.5rem;">
                <div class="pv-card-header">
                    <span class="pv-card-title">Inferred Water Extent</span>
                    <span class="pv-badge pv-badge-accent">Optical Sensor</span>
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.45;">
                    Binary water segmentation generated from Sentinel-2 L1C optical reflectance bands (threshold &ge; 0.5).
                    10m spatial resolution.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_s2:
        st.markdown(
            """
            <div class="pv-card" style="margin-bottom: 0.5rem;">
                <div class="pv-card-header">
                    <span class="pv-card-title">UNOSAT Reference Extent</span>
                    <span class="pv-badge pv-badge-warning">Retrospective</span>
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.45;">
                    Satellite-detected mudflow and rockflow polygon published August 28, 2026.
                    Qualitative post-event reference; physically distinct from clear surface water.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_s3:
        st.markdown(
            """
            <div class="pv-card" style="margin-bottom: 0.5rem;">
                <div class="pv-card-header">
                    <span class="pv-card-title">Core Corridor AOI</span>
                    <span class="pv-badge pv-badge-neutral">Analysis Scope</span>
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.45;">
                    Bounded analysis corridor across the Bhote Koshi and Trishuli valleys.
                    Pixels outside this polygon or occluded by cloud are masked as invalid (<code>255</code>).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Mapped Polygons Attribute Table (if features exist)
    if artifacts.water_geojson and artifacts.water_geojson.get("features"):
        with st.expander("Feature Attributes (Mapped Water Polygons)", expanded=False):
            features_list = []
            for i, f in enumerate(artifacts.water_geojson["features"]):
                props = f.get("properties", {})
                features_list.append({
                    "Index": i + 1,
                    "Feature Name": props.get("label", f"Water Feature #{i + 1}"),
                    "Area (m²)": f"{props.get('area_m2', 0):,.1f}" if props.get("area_m2") else "—",
                    "Geometry Type": f.get("geometry", {}).get("type", "Polygon"),
                    "Classification": props.get("class", "inferred_water"),
                })
            st.dataframe(features_list, use_container_width=True, hide_index=True)
