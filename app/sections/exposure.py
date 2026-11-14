"""Infrastructure Exposure — Spatial Overlays & Vulnerability Analysis."""
import streamlit as st

from app.components.empty_state import render_empty_state
from app.components.layout import (
    render_callout,
    render_page_header,
    render_section_header,
)
from app.components.metric_card import render_metric_card
from app.data_loader import RunArtifacts


def render_exposure_section(artifacts: RunArtifacts):
    """Render the standardized Infrastructure Exposure page."""
    render_page_header(
        title="Infrastructure Exposure",
        subtitle="Calculates spatial intersections between model-derived optical water extents and OpenStreetMap infrastructure footprints (EPSG:32645 metric calculations).",
        badge="Metric Projected: EPSG:32645" if artifacts.exposure_summary else None,
        badge_variant="neutral",
    )

    exp = artifacts.exposure_summary
    if not exp:
        render_empty_state(
            title="Awaiting Infrastructure Exposure Summary",
            description="Exposure metrics require a stitched water mask and OpenStreetMap infrastructure layers (buildings, roads, bridges).",
            action_command="python scripts/exposure.py --mask outputs/<run_id>/observed_water.tif",
            expected_files=["outputs/<run_id>/exposure_summary.json", "outputs/<run_id>/exposure.geojson"],
        )
        return

    # Critical Scientific Callout
    render_callout(
        message="Spatial intersection with OpenStreetMap footprints indicates inundation exposure or proximity, NOT confirmed structural destruction, bridge collapse, or human casualties.",
        title="Methodological Principle (Exposure ≠ Damage)",
        variant="warning",
    )

    # 1. Standardized KPI Metrics Row (4 Cards)
    water = exp.get("water_area_m2")
    buildings = exp.get("building_footprints_intersecting")
    roads_m = exp.get("road_length_m_intersecting")
    bridges = exp.get("bridge_tagged_features_intersecting")
    pop = exp.get("modelled_population_estimate")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_metric_card(
            label="Water Extent",
            value=f"{water / 10000:.1f}" if water else "0.0",
            unit="ha",
            subtext=f"{water:,.0f} m² total" if water else "No water mapped",
            badge="Optical Mask",
            badge_variant="accent",
        )

    with c2:
        render_metric_card(
            label="Building Footprints",
            value=buildings,
            unit="structures" if buildings is not None else None,
            subtext="Intersecting polygons",
            badge="OSM Overlap" if buildings is not None else "Missing",
            badge_variant="warning" if buildings else "neutral",
        )

    with c3:
        render_metric_card(
            label="Road Alignment",
            value=f"{roads_m / 1000:.2f}" if roads_m is not None else None,
            unit="km" if roads_m is not None else None,
            subtext=f"{roads_m:,.0f} m intersecting" if roads_m is not None else "Layer missing",
            badge="Road Network" if roads_m is not None else "Missing",
            badge_variant="warning" if roads_m else "neutral",
        )

    with c4:
        render_metric_card(
            label="Bridge Spans",
            value=bridges,
            unit="features" if bridges is not None else None,
            subtext="Tagged bridge features",
            badge="High Impact" if bridges and bridges > 0 else "None",
            badge_variant="error" if bridges and bridges > 0 else "neutral",
        )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # 2. Detailed Exposure Audit & Geospatial Invariants
    col_l, col_r = st.columns([3, 2])

    with col_l:
        render_section_header("Methodological Breakdown & Coverage Audit", "Inventory of evaluated asset layers.")

        audit_rows = [
            {"Asset Category": "Water Mask Artifact", "Recorded Value": exp.get("mask_artifact", "observed_water.tif"), "Status": "Valid Invariant"},
            {"Asset Category": "Valid Observed Pixels", "Recorded Value": f"{exp.get('valid_pixel_count', 0):,} pixels", "Status": "Valid Invariant"},
            {"Asset Category": "OSM Extract Snapshot", "Recorded Value": exp.get("osm_snapshot_timestamp", "2026-01-01"), "Status": "Pre-Event Baseline"},
            {"Asset Category": "Modelled Resident Population", "Recorded Value": f"~{pop:,} residents" if pop else "Not computed (WorldPop unindexed)", "Status": "Contextual Reference"},
        ]
        st.dataframe(audit_rows, use_container_width=True, hide_index=True)

        unavail = exp.get("unavailable_fields", [])
        if unavail:
            render_callout(
                message=f"The following feature types were omitted or unindexed during this run: {', '.join(unavail)}.",
                title="Layer Availability Notice",
                variant="info",
            )

    with col_r:
        render_section_header("Geospatial Invariants", "Scientific contracts enforced by scripts/exposure.py.")

        st.markdown(
            f"""
            <div class="pv-card">
                <div style="display: flex; flex-direction: column; gap: 0.5rem; font-size: 0.8rem;">
                    <div><strong>Projected Coordinate System:</strong> <code style="color: #38BDF8;">EPSG:32645</code></div>
                    <div><strong>Spatial Units:</strong> Meters & Square Meters</div>
                    <div><strong>Source Mask Type:</strong> <code>{exp.get('source_type', 'model_inferred_water')}</code></div>
                    <div><strong>Deduplication:</strong> Unique OSM ID tracking</div>
                    <div><strong>Zero vs Missing:</strong> Evaluated layers with zero overlap report 0, not null.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Raw Exposure Summary JSON"):
            st.json(exp)
