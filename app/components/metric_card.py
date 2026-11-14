"""Standardized KPI Metric Card Component for PRAVAH Design System."""
from typing import Any

import streamlit as st


def render_metric_card(
    label: str,
    value: Any,
    unit: str | None = None,
    subtext: str | None = None,
    badge: str | None = None,
    badge_variant: str = "neutral",
    as_html: bool = False,
) -> str | None:
    """Render a standardized KPI card matching the PRAVAH design tokens."""
    # Format value cleanly
    if value is None:
        display_val = "—"
    elif isinstance(value, float):
        display_val = f"{value:,.1f}" if value >= 10 else f"{value:,.3f}"
    elif isinstance(value, int):
        display_val = f"{value:,}"
    else:
        display_val = str(value)

    badge_html = ""
    if badge:
        badge_html = f'<span class="pv-badge pv-badge-{badge_variant}">{badge}</span>'

    unit_html = f'<span class="pv-metric-unit">{unit}</span>' if unit else ""
    sub_html = f'<div class="pv-metric-subtext">{subtext}</div>' if subtext else ""

    html = f"""
    <div class="pv-metric-card">
        <div class="pv-metric-label">
            <span>{label}</span>
            {badge_html}
        </div>
        <div class="pv-metric-value-row">
            <span class="pv-metric-value">{display_val}</span>
            {unit_html}
        </div>
        {sub_html}
    </div>
    """

    if as_html:
        return html
    st.markdown(html, unsafe_allow_html=True)
    return None
