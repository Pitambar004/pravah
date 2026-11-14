"""Standardized Empty & Awaiting State Component for PRAVAH Design System."""

import streamlit as st


def render_empty_state(
    title: str = "Awaiting Artifacts",
    description: str = "No computed outputs are present in the selected run folder.",
    action_command: str | None = None,
    expected_files: list[str] | None = None,
    badge: str = "Awaiting Execution",
):
    """Render an intentional, informative empty state."""
    files_html = ""
    if expected_files:
        items = "".join([f"<li style='margin-bottom: 2px;'><code>{f}</code></li>" for f in expected_files])
        files_html = f"""
        <div style="margin-top: 0.75rem; font-size: 0.775rem; color: #94A3B8; text-align: left; max-width: 440px; margin-left: auto; margin-right: auto;">
            <div style="font-weight: 500; text-transform: uppercase; font-size: 0.7rem; color: #64748B; margin-bottom: 0.25rem;">Expected Artifacts:</div>
            <ul style="margin: 0; padding-left: 1.25rem;">{items}</ul>
        </div>
        """

    cmd_html = ""
    if action_command:
        cmd_html = f"""
        <div style="margin-top: 0.75rem;">
            <span class="pv-empty-code">$ {action_command}</span>
        </div>
        """

    html = f"""
    <div class="pv-empty-state">
        <div>
            <span class="pv-badge pv-badge-neutral">{badge}</span>
        </div>
        <div class="pv-empty-title">{title}</div>
        <div class="pv-empty-desc">{description}</div>
        {cmd_html}
        {files_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
