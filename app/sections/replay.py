"""Historical Replay — what evidence existed at a chosen moment.

Strict mode never renders assets acquired or made available after the cutoff.
Eligibility is delegated to ``pravah.replay.eligibility`` (acquisition AND
verified availability <= cutoff; unknown availability is excluded).
"""
from __future__ import annotations

from datetime import datetime, time

import streamlit as st

from app.components.empty_state import render_empty_state
from app.components.layout import html, render_callout, render_section_header
from app.components.metric_card import render_metric_card
from app.components.timeline import (
    NPT,
    build_timeline_svg,
    evidence_card,
    timeline_legend,
)
from app.context import AppContext, render_page_frame
from pravah.replay.eligibility import classify_asset, parse_time

DEFAULT_CUTOFF = "2026-08-26T11:00:00+05:45"

PRESETS = {
    "Pre-event baseline · 20 Aug 12:00 NPT": "2026-08-20T12:00:00+05:45",
    "Event morning · 26 Aug 11:00 NPT": "2026-08-26T11:00:00+05:45",
    "Event evening · 26 Aug 18:00 NPT": "2026-08-26T18:00:00+05:45",
    "After UNOSAT release · 29 Aug 12:00 NPT": "2026-08-29T12:00:00+05:45",
    "Custom…": None,
}
MODES = ["Evidence at cutoff", "Retrospective reveal"]


def partition(catalog: list[dict], cutoff: str) -> dict[str, list[dict]]:
    """Classify every asset; returns lists keyed by status."""
    out: dict[str, list[dict]] = {"known_before_cutoff": [], "retrospective_reveal": [], "excluded_unknown_availability": []}
    for item in catalog:
        try:
            status = classify_asset(item, cutoff)
        except (ValueError, TypeError):
            status = "excluded_unknown_availability"
        out[status].append({**item, "_status": status})
    return out


def _acquired_before(asset: dict, cutoff: str) -> bool:
    try:
        acq = parse_time(asset.get("acquired_at_utc"))
        return acq is not None and acq <= parse_time(cutoff)
    except (ValueError, TypeError):
        return False


def render_replay_section(ctx: AppContext) -> None:
    render_page_frame(
        ctx,
        "Historical Replay",
        "Historical Replay",
        "Step back to a moment in time and see only the evidence that was actually available then. "
        "Later observations stay hidden unless you switch to retrospective reveal.",
    )

    c_mode, c_preset = st.columns([3, 2], vertical_alignment="bottom")
    with c_mode:
        if hasattr(st, "segmented_control"):
            mode = st.segmented_control("Replay mode", MODES, default=MODES[0], key="pv_replay_mode") or MODES[0]
        else:  # pragma: no cover - older Streamlit
            mode = st.radio("Replay mode", MODES, horizontal=True, key="pv_replay_mode")
    with c_preset:
        preset = st.selectbox("Cutoff", list(PRESETS), index=1, key="pv_replay_preset")

    cutoff = PRESETS[preset]
    if cutoff is None:
        d_col, t_col, _ = st.columns([1, 1, 3])
        with d_col:
            d = st.date_input("Date (NPT)", value=datetime(2026, 8, 26).date(), key="pv_cutoff_date")
        with t_col:
            t = st.time_input("Time (NPT)", value=time(11, 0), step=900, key="pv_cutoff_time")
        cutoff = datetime.combine(d, t, tzinfo=NPT).isoformat()
    strict = mode == MODES[0]

    if not ctx.catalog:
        render_empty_state(
            "No source manifest available",
            "Replay needs an audited asset manifest with acquisition and verified availability timestamps. "
            "Enable synthetic fixtures in the sidebar to preview this page.",
            expected_files=["data/manifests/assets.json"],
            icon_name="history",
        )
        return

    groups = partition(ctx.catalog, cutoff)
    eligible = groups["known_before_cutoff"]
    later = groups["retrospective_reveal"]
    excluded_all = groups["excluded_unknown_availability"]
    excluded = [x for x in excluded_all if _acquired_before(x, cutoff)] if strict else excluded_all

    if strict:
        render_callout(
            "An item is shown only when <strong>both</strong> its acquisition time and its verified availability time "
            "are at or before the cutoff. Items with unknown availability are excluded, never assumed available.",
            title="Strict replay",
            variant="info",
        )
    else:
        render_callout(
            "Later evidence (diamonds) was <strong>not available</strong> at the selected cutoff. It is shown for "
            "retrospective comparison only and must never be treated as an input known at that time.",
            title="Retrospective reveal",
            variant="warning",
        )

    cols = st.columns(3)
    with cols[0]:
        render_metric_card("Available at cutoff", len(eligible), unit="items", icon_name="check_circle",
                           subtext="Acquired and verified available ≤ cutoff")
    with cols[1]:
        render_metric_card("Excluded", len(excluded), unit="items", icon_name="block",
                           subtext="Availability time unknown or unverified")
    with cols[2]:
        render_metric_card("Later evidence", None if strict else len(later), unit="items", icon_name="update",
                           empty_label="Hidden in strict mode",
                           subtext="Switch to retrospective reveal to inspect" if strict else "Acquired or available after cutoff")

    render_section_header("Timeline", "Each line runs from acquisition to verified availability; markers sit at availability.")
    shown = eligible + excluded + ([] if strict else later)
    html(
        '<div class="pv-timeline"><div class="pv-timeline-scroll">'
        + build_timeline_svg(shown, cutoff, strict=strict)
        + "</div>" + timeline_legend(strict) + "</div>"
    )

    def cards(title: str, desc: str, items: list[dict], empty: str) -> None:
        render_section_header(f"{title} · {len(items)}", desc)
        if items:
            html('<div class="pv-evidence-list">' + "".join(evidence_card(i) for i in items) + "</div>")
        else:
            render_empty_state(empty, "", badge=None, icon_name="inbox", compact=True)

    left, right = st.columns(2)
    with left:
        cards("Available at cutoff", "Legitimate inputs at this moment.", eligible,
              "Nothing was available at this cutoff")
    with right:
        cards("Excluded", "Not used: availability could not be verified.", excluded, "No excluded items")
    if not strict:
        cards("Later evidence", "Revealed for comparison only — not available at the cutoff.", later,
              "No later evidence after this cutoff")
        with st.expander("Complete manifest table"):
            st.dataframe(
                [
                    {
                        "Asset ID": x.get("asset_id"),
                        "Platform": x.get("platform"),
                        "Product": x.get("product_type"),
                        "Acquired (UTC)": x.get("acquired_at_utc"),
                        "Available (UTC)": x.get("available_at_utc") or "Unknown",
                        "Cloud %": x.get("cloud_cover_percent"),
                        "Status": x.get("_status"),
                    }
                    for x in eligible + later + excluded_all
                ],
                hide_index=True,
                width="stretch",
                column_config={"Cloud %": st.column_config.NumberColumn(format="%.1f")},
            )
