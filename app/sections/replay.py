"""Historical Replay — Strict Temporal Gating & Disaster Evidence Timeline."""
import streamlit as st

from app.components.layout import (
    render_callout,
    render_page_header,
)
from app.components.metric_card import render_metric_card
from app.data_loader import RunArtifacts, load_replay_catalog
from pravah.replay.eligibility import classify_asset, parse_time

DEFAULT_CUTOFF = "2026-08-26T11:00:00+05:45"

PRESETS = {
    "Disaster Onset Morning (Aug 26 11:00 NPT)": "2026-08-26T11:00:00+05:45",
    "Pre-Event Baseline (Aug 20 12:00 NPT)": "2026-08-20T12:00:00+05:45",
    "Post-Event Evening (Aug 26 18:00 NPT)": "2026-08-26T18:00:00+05:45",
    "Full Retrospective Archive (Aug 29 12:00 NPT)": "2026-08-29T12:00:00+05:45",
    "Custom Timestamp": None,
}


def render_replay_section(artifacts: RunArtifacts):
    """Render the Historical Replay timeline and temporal audit."""
    render_page_header(
        title="Historical Evidence Replay",
        subtitle="Forensic evidence timeline enforcing strict pre-cutoff gating: an observation is eligible only if both its acquisition and verified availability timestamps precede the decision cutoff.",
        badge="Zero-Leakage Replay",
        badge_variant="success",
    )

    # 1. Compact Cutoff Filter Toolbar
    col_f1, col_f2 = st.columns([2, 1])

    with col_f2:
        preset = st.selectbox("Disaster Milestone Presets", list(PRESETS.keys()), index=0)

    with col_f1:
        default_val = PRESETS[preset] if PRESETS[preset] else DEFAULT_CUTOFF
        cutoff_str = st.text_input(
            "Evidence Decision Cutoff (ISO 8601 with Offset)",
            value=default_val,
            help="E.g. 2026-08-26T11:00:00+05:45 (Nepal Time) or 2026-08-26T05:15:00Z (UTC)",
        )

    try:
        parse_time(cutoff_str)
    except (ValueError, TypeError) as e:
        render_callout(f"Invalid timezone-aware ISO timestamp: {e}", title="Timestamp Error", variant="error")
        return

    # 2. Classify Assets
    catalog = load_replay_catalog(allow_synthetic=True)

    eligible_assets, retrospective_assets, excluded_assets = [], [], []
    for item in catalog:
        status = classify_asset(item, cutoff_str)
        enriched = {**item, "_status": status}
        if status == "known_before_cutoff":
            eligible_assets.append(enriched)
        elif status == "retrospective_reveal":
            retrospective_assets.append(enriched)
        else:
            excluded_assets.append(enriched)

    # 3. 3-Card Summary Row
    c1, c2, c3 = st.columns(3)

    with c1:
        render_metric_card(
            label="Eligible at Cutoff",
            value=len(eligible_assets),
            unit="observations",
            subtext="Acquired & available ≤ cutoff",
            badge="Pre-Cutoff Input",
            badge_variant="success",
        )

    with c2:
        render_metric_card(
            label="Retrospective Reveal",
            value=len(retrospective_assets),
            unit="observations",
            subtext="Acquired or available > cutoff",
            badge="Post-Event Validation",
            badge_variant="warning",
        )

    with c3:
        render_metric_card(
            label="Excluded (Unknown)",
            value=len(excluded_assets),
            unit="observations",
            subtext="Missing publication time",
            badge="Integrity Rejection",
            badge_variant="error",
        )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # 4. Tabbed Table Views
    tab_pre, tab_post, tab_excl, tab_all = st.tabs([
        f"Eligible Before Cutoff ({len(eligible_assets)})",
        f"Retrospective Reveal ({len(retrospective_assets)})",
        f"Excluded / Missing Availability ({len(excluded_assets)})",
        "Complete Evidence Manifest",
    ])

    with tab_pre:
        st.markdown(
            '<div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 0.5rem;">These datasets were publicly accessible and acquired prior to the decision point. They qualify as legitimate model inputs.</div>',
            unsafe_allow_html=True,
        )
        if eligible_assets:
            _render_manifest_table(eligible_assets)
        else:
            st.info("No satellite observations qualify under pre-cutoff rules for this timestamp.")

    with tab_post:
        st.markdown(
            '<div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 0.5rem;">Post-cutoff evidence (e.g. Aug 27 Sentinel-2 or Aug 28 UNOSAT). Strictly segregated from pre-cutoff models.</div>',
            unsafe_allow_html=True,
        )
        if retrospective_assets:
            _render_manifest_table(retrospective_assets)
        else:
            st.info("No retrospective assets recorded beyond this timestamp.")

    with tab_excl:
        st.markdown(
            '<div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 0.5rem;">Observations without a verifiable catalog publishing timestamp are excluded to prevent simulated foresight.</div>',
            unsafe_allow_html=True,
        )
        if excluded_assets:
            _render_manifest_table(excluded_assets)
        else:
            st.success("Zero assets with unverified availability timestamps.")

    with tab_all:
        _render_manifest_table(eligible_assets + retrospective_assets + excluded_assets)


def _render_manifest_table(items: list[dict]):
    """Format and display catalog manifest items."""
    table_rows = [
        {
            "Asset ID": a.get("asset_id", "—"),
            "Platform": a.get("platform", "—"),
            "Product": a.get("product_type", "—"),
            "Acquisition Time (UTC)": a.get("acquired_at_utc", "—"),
            "Availability Time (UTC)": a.get("available_at_utc") or "UNKNOWN (Excluded)",
            "Cloud %": f"{a.get('cloud_cover_percent'):.1f}%" if a.get("cloud_cover_percent") is not None else "—",
            "Classification Status": a.get("_status", "—"),
        }
        for a in items
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)
