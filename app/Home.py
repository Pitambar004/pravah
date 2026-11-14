# ruff: noqa: N999
"""PRAVAH application shell: page config, theme, sidebar, routing."""
import sys
from html import escape
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from app.components.layout import html, render_callout, render_footer
from app.context import build_context
from app.data_loader import discover_available_runs, load_run_artifacts
from app.theme import TOKENS, apply_theme

APP_VERSION = "0.1.0"

st.set_page_config(
    page_title="PRAVAH · Flood Intelligence",
    page_icon=":material/water:",
    layout="wide",
    initial_sidebar_state="auto",
)
apply_theme()

# Pages (route names kept stable; titles follow the design brief)
PAGES = [
    st.Page("views/overview.py", title="Overview", icon=":material/space_dashboard:", url_path="overview", default=True),
    st.Page("views/flood_map.py", title="Flood Intelligence Map", icon=":material/map:", url_path="map"),
    st.Page("views/replay.py", title="Historical Replay", icon=":material/history:", url_path="replay"),
    st.Page("views/exposure.py", title="Infrastructure Exposure", icon=":material/apartment:", url_path="exposure"),
    st.Page("views/compute.py", title="Compute & Performance", icon=":material/memory:", url_path="compute"),
    st.Page("views/methodology.py", title="Sources & Methodology", icon=":material/menu_book:", url_path="methodology"),
]
current = st.navigation(PAGES, position="hidden")

_MARK = (
    '<svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">'
    f'<rect x="0.5" y="0.5" width="27" height="27" rx="7.5" fill="{TOKENS["surface_raised"]}" stroke="{TOKENS["border"]}"/>'
    f'<path d="M6 10.5c3-2.2 5.2 2.2 8 0s5-2.2 8 0" stroke="{TOKENS["accent"]}" stroke-width="1.7" stroke-linecap="round"/>'
    f'<path d="M6 15c3-2.2 5.2 2.2 8 0s5-2.2 8 0" stroke="{TOKENS["accent"]}" stroke-opacity=".65" stroke-width="1.7" stroke-linecap="round"/>'
    f'<path d="M6 19.5c3-2.2 5.2 2.2 8 0s5-2.2 8 0" stroke="{TOKENS["accent"]}" stroke-opacity=".35" stroke-width="1.7" stroke-linecap="round"/>'
    "</svg>"
)

with st.sidebar:
    html(
        f'<div class="pv-brand">{_MARK}<div><div class="pv-brand-name">PRAVAH'
        '<span class="pv-brand-deva" lang="ne">प्रवाह</span></div>'
        '<div class="pv-brand-sub">Flood Intelligence Platform</div></div></div>'
    )
    html('<div class="pv-nav-label">Workspace</div>')
    for page in PAGES:
        st.page_link(page, label=page.title, icon=page.icon)

    html('<div class="pv-side-divider"></div><div class="pv-nav-label">Data source</div>')
    runs = discover_available_runs("outputs")
    if runs:
        chosen = st.selectbox("Run", runs, index=0, help="Run folders discovered under outputs/")
        run_folder = f"outputs/{chosen}"
    else:
        run_folder = st.text_input("Run folder", "outputs/latest", help="No run folders found under outputs/")
    use_demo = st.toggle(
        "Show synthetic fixtures",
        value=not bool(runs),
        help="Loads clearly labelled synthetic test fixtures to preview layouts when no run artifacts exist. "
        "Fixtures are never real observations.",
    )

artifacts = load_run_artifacts(run_folder, allow_synthetic_fallback=use_demo)
ctx = build_context(artifacts)

with st.sidebar:
    label, variant = ctx.data_status
    html(
        f'<div class="pv-side-meta"><span>Status</span><span class="pv-badge pv-badge-{variant}"><span class="dot"></span>{label}</span></div>'
        f'<div class="pv-side-meta"><span>Run</span><span class="pv-mono" title="{escape(artifacts.run_id)}">{escape(artifacts.run_id)}</span></div>'
        f'<div class="pv-side-footer"><strong>Team 12</strong> · CIQ Hackathon 2026<br>v{APP_VERSION} · retrospective prototype</div>'
    )

if artifacts.errors:
    render_callout(
        "Some artifacts in this run could not be read: " + escape("; ".join(artifacts.errors)),
        title="Artifact read errors",
        variant="error",
    )
if artifacts.is_synthetic:
    render_callout(
        "All geometries, counts, scenes and benchmark numbers on these pages are synthetic test fixtures used to "
        "verify the interface. They are not observations of the Rasuwa event.",
        title="Synthetic fixture mode",
        variant="warning",
    )

current.run()
render_footer()
