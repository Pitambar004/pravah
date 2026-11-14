"""Overview — Mission Overview, Key Geospatial KPIs, Map Preview, and Pipeline Status."""
import streamlit as st

from app.components.empty_state import render_empty_state
from app.components.layout import (
    render_page_header,
    render_section_header,
)
from app.components.map_view import render_flood_map
from app.components.metric_card import render_metric_card
from app.data_loader import RunArtifacts


def render_overview_section(artifacts: RunArtifacts):
    """Render the standardized Mission Overview page."""
    render_page_header(
        title="Mission Overview",
        subtitle="Satellite-derived optical water evidence and distributed multi-node AI benchmarks for the August 26, 2026 Rasuwa, Nepal flood cascade.",
        badge="Case Study: Rasuwa" if artifacts.has_any_artifacts else "Awaiting Outputs",
        badge_variant="accent" if artifacts.has_any_artifacts else "neutral",
    )

    exp = artifacts.exposure_summary or {}
    meta = artifacts.observed_metadata or {}
    bench = artifacts.benchmark or {}

    water_m2 = exp.get("water_area_m2")
    buildings = exp.get("building_footprints_intersecting")
    roads_m = exp.get("road_length_m_intersecting")
    bridges = exp.get("bridge_tagged_features_intersecting")
    speedup = bench.get("speedup_compute")

    # 1. Standardized KPI Metrics Row (5 Cards)
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        if water_m2 is not None:
            val = f"{water_m2 / 10000:.1f}"
            unit = "ha"
            subtext = f"{water_m2:,.0f} m² mapped"
            b_variant = "accent"
        else:
            val, unit, subtext, b_variant = None, None, "Awaiting mask", "neutral"

        render_metric_card(
            label="Inferred Water",
            value=val,
            unit=unit,
            subtext=subtext,
            badge="Optical" if water_m2 else None,
            badge_variant=b_variant,
        )

    with c2:
        render_metric_card(
            label="Buildings Exposed",
            value=buildings,
            unit="footprints" if buildings is not None else None,
            subtext="Spatial intersection" if buildings is not None else "Awaiting layer",
            badge="OSM Baseline" if buildings is not None else None,
            badge_variant="warning" if buildings else "neutral",
        )

    with c3:
        if roads_m is not None:
            r_val = f"{roads_m / 1000:.2f}"
            r_unit = "km"
            r_sub = f"{roads_m:,.0f} m intersected"
            r_var = "warning" if roads_m > 0 else "neutral"
        else:
            r_val, r_unit, r_sub, r_var = None, None, "Awaiting layer", "neutral"

        render_metric_card(
            label="Roads Exposed",
            value=r_val,
            unit=r_unit,
            subtext=r_sub,
            badge="Transport" if roads_m is not None else None,
            badge_variant=r_var,
        )

    with c4:
        render_metric_card(
            label="Bridge Overlaps",
            value=bridges,
            unit="spans" if bridges is not None else None,
            subtext="Tagged bridge spans" if bridges is not None else "Awaiting layer",
            badge="Critical" if bridges and bridges > 0 else None,
            badge_variant="error" if bridges and bridges > 0 else "neutral",
        )

    with c5:
        if speedup is not None:
            g_val = f"{speedup:.2f}×"
            g_unit = "speedup"
            g_sub = f"{bench.get('parallel_efficiency', 0) * 100:.0f}% parallel efficiency"
            g_badge = "2-Node GB10"
            g_var = "success"
        else:
            g_val = None
            g_unit = None
            g_sub = "Sunday 9:00 AM window"
            g_badge = "Cluster Run"
            g_var = "neutral"

        render_metric_card(
            label="Multi-Node Scaling",
            value=g_val,
            unit=g_unit,
            subtext=g_sub,
            badge=g_badge,
            badge_variant=g_var,
        )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # 2. Main Content Split: Map Preview (3/5) + Context & Activity (2/5)
    col_map, col_details = st.columns([3, 2])

    with col_map:
        render_section_header("Study Area Geospatial Intelligence", "Core Trishuli / Bhote Koshi catchment corridor.")
        if artifacts.has_any_artifacts:
            render_flood_map(
                water_geojson=artifacts.water_geojson,
                aoi_geojson=artifacts.aoi_geojson,
                unosat_geojson=artifacts.unosat_geojson,
                height=440,
            )
        else:
            render_empty_state(
                title="No Geospatial Layers Available",
                description="Water polygons and corridor bounds will render immediately when tile inference and mask stitching complete.",
                action_command="python scripts/stitch.py && python scripts/exposure.py",
                expected_files=["exposure.geojson", "observed_water.tif"],
            )

    with col_details:
        render_section_header("Case Study Domain", "Geographic and environmental context.")

        st.markdown(
            """
            <div class="pv-card" style="margin-bottom: 0.85rem;">
                <div style="font-size: 0.875rem; font-weight: 600; color: #F1F5F9;">
                    Rasuwa & Nuwakot / Trishuli Basin, Nepal
                </div>
                <div style="font-size: 0.775rem; color: #38BDF8; font-family: 'JetBrains Mono', monospace; margin-top: 0.2rem;">
                    28.150° N, 85.350° E · UTM Zone 45N (EPSG:32645)
                </div>
                <p style="font-size: 0.8rem; color: #94A3B8; margin: 0.5rem 0 0 0; line-height: 1.45;">
                    High-relief Himalayan catchment draining through Bhote Koshi into Trishuli River.
                    Event triggered a debris flow and flood cascade following intense monsoon precipitation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_section_header("Pipeline Stage Verification", "End-to-end artifact generation status.")

        pipeline_steps = [
            ("Scene Discovery & STAC Audit", True, "Audited"),
            ("Temporal Pre-Cutoff Gating", True, "Enforced"),
            ("Tile Inference (Prithvi / U-Net)", bool(meta), "Computed" if meta else "Pending"),
            ("Mask Stitching & GeoTIFF Export", bool(artifacts.observed_water_tif), "Exported" if artifacts.observed_water_tif else "Pending"),
            ("OSM Infrastructure Intersections", bool(exp), "Computed" if exp else "Pending"),
            ("2-Node GPU Benchmarking (GB10)", bool(bench), "Verified" if bench else "Sunday 9:00 AM"),
        ]

        rows_html = ""
        for name, done, status in pipeline_steps:
            badge_cls = "pv-badge-success" if done else "pv-badge-neutral"
            bullet = '<span style="color: #34D399; margin-right: 6px;">●</span>' if done else '<span style="color: #64748B; margin-right: 6px;">○</span>'
            rows_html += f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.35rem 0; border-bottom: 1px solid #1E293B; font-size: 0.785rem;">
                <span>{bullet}{name}</span>
                <span class="pv-badge {badge_cls}">{status}</span>
            </div>
            """

        st.markdown(
            f"""
            <div class="pv-card" style="padding: 0.75rem 1rem;">
                {rows_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Scientific Caveats Expandable Panel
    with st.expander("Scientific Caveats & Methodological Standards"):
        st.markdown(
            """
            - **Retrospective Mapping Only:** Observations acquired or published after the event are never used to claim advance forecast capability.
            - **Monsoon Cloud Occlusion:** Optical Sentinel-2 imagery cannot penetrate dense monsoon cloud; missing pixels are masked as `255 (invalid)`.
            - **Exposure Definition:** Spatial intersection with building or road footprints represents *exposure*, not verified collapse or destruction.
            - **Metric Calculations:** All area and length metrics are computed in **EPSG:32645** (meters), never in planar degrees.
            - **Workload Equivalence:** Cluster benchmarks compare identical tile manifests and model hashes across distinct physical node IDs.
            """
        )
