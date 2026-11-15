# ruff: noqa: N999
"""PRAVAH application shell: page config, theme, sidebar, routing."""
import hmac
import os
import sys
from html import escape
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from app import api_client
from app.components.layout import html, render_callout, render_footer
from app.context import build_context
from app.data_loader import (
    discover_available_runs,
    load_run_artifacts,
    run_has_artifacts,
)
from app.remote_runs import list_remote_runs, load_remote_run
from app.theme import TOKENS, apply_theme

APP_VERSION = "0.1.0"
# Run folders: ./outputs locally; the Pravah-Dataset predictions dir on Fuzzball.
OUTPUTS_DIR = Path(os.environ.get("PRAVAH_OUTPUTS_DIR", "outputs/live_app/runs" if Path("outputs/live_app/config.json").exists() else "outputs"))

st.set_page_config(
    page_title="PRAVAH · Flood Intelligence",
    page_icon=":material/water:",
    layout="wide",
    initial_sidebar_state="auto",
)
apply_theme()

# Optional shared passcode for venue access (set PRAVAH_DASHBOARD_PASSCODE on the
# Fuzzball service). Fuzzball endpoint scope is the primary access control.
_PASSCODE = os.environ.get("PRAVAH_DASHBOARD_PASSCODE")
if _PASSCODE and not st.session_state.get("pv_authed"):
    st.title("PRAVAH")
    entered = st.text_input("Team passcode", type="password")
    if entered and hmac.compare_digest(entered, _PASSCODE):
        st.session_state["pv_authed"] = True
        st.rerun()
    elif entered:
        st.error("Incorrect passcode")
    st.stop()

# Pages (route names kept stable; titles follow the design brief)
PAGES = [
    st.Page("views/overview.py", title="Overview", icon=":material/space_dashboard:", url_path="overview", default=True),
    st.Page("views/flood_map.py", title="Flood Intelligence Map", icon=":material/map:", url_path="map"),
    st.Page("views/replay.py", title="Historical Replay", icon=":material/history:", url_path="replay"),
    st.Page("views/experiment.py", title="Nepal Experiment", icon=":material/science:", url_path="experiment"),
    st.Page("views/exposure.py", title="Infrastructure Exposure", icon=":material/apartment:", url_path="exposure"),
    st.Page("views/inference.py", title="Run Inference", icon=":material/play_circle:", url_path="inference"),
    st.Page("views/environment.py", title="Nepal Data", icon=":material/landscape:", url_path="environment"),
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
    runs = discover_available_runs(OUTPUTS_DIR)
    local_real = [r for r in runs if run_has_artifacts(OUTPUTS_DIR / r)]
    runs = sorted(local_real, key=lambda run: (OUTPUTS_DIR / run).stat().st_mtime, reverse=True)
    # Run source: local disk (shared volume) or the inference API over HTTP, e.g. a laptop
    # dashboard reading runs produced on Fuzzball. PRAVAH_RUN_SOURCE = auto | local | api.
    source_pref = os.environ.get("PRAVAH_RUN_SOURCE", "auto")
    remote_runs: list[str] = []
    remote_err = None
    if source_pref == "api" or (source_pref == "auto" and not local_real):
        remote_runs, remote_err = list_remote_runs(api_client.api_url())
    use_remote = bool(remote_runs) and (source_pref == "api" or not local_real)
    options = remote_runs if use_remote else runs
    requested_run = st.query_params.get("run")
    if requested_run in options and st.session_state.get("pv_requested_run") != requested_run:
        st.session_state["pv_run"] = requested_run
        st.session_state["pv_requested_run"] = requested_run
    if st.session_state.get("pv_run") not in (None, *options):
        st.session_state.pop("pv_run")  # run disappeared from storage
    if use_remote:
        chosen = st.selectbox("Run · inference API", remote_runs, key="pv_run",
                              help=f"Completed runs served by {api_client.api_url()}")
        run_folder = None
    elif runs:
        chosen = st.selectbox("Run", runs, key="pv_run", help=f"Run folders discovered under {OUTPUTS_DIR}/")
        run_folder = str(OUTPUTS_DIR / chosen)
    else:
        run_folder = st.text_input("Run folder", str(OUTPUTS_DIR / "latest"),
                                   help=f"No run folders found under {OUTPUTS_DIR}/")
    if source_pref == "api" and not remote_runs:
        st.caption(f"No completed runs from the inference API{': ' + remote_err if remote_err else ''}")
    # Test fixtures require an explicit developer flag and manual opt-in.
    st.session_state.setdefault("pv_demo", False)
    if os.environ.get("PRAVAH_ENABLE_TEST_FIXTURES") == "1":
        use_demo = st.toggle("Show synthetic fixtures", key="pv_demo", help="Developer tests only")
    else:
        use_demo = False
        st.session_state["pv_demo"] = False

if use_remote and not use_demo:
    artifacts = load_remote_run(chosen)
else:
    artifacts = load_run_artifacts(run_folder or str(OUTPUTS_DIR / "latest"), allow_synthetic_fallback=use_demo)
ctx = build_context(artifacts)

if not use_demo and any((item or {}).get("is_synthetic") for item in
                       [artifacts.observed_metadata, artifacts.exposure_summary, artifacts.benchmark]):
    st.error("This run is marked synthetic. It cannot be displayed in the real-data application.")
    st.stop()

with st.sidebar:
    label, variant = ctx.data_status
    html(
        f'<div class="pv-side-meta"><span>Status</span><span class="pv-badge pv-badge-{variant}"><span class="dot"></span>{label}</span></div>'
        f'<div class="pv-side-meta"><span>Run</span><span class="pv-mono" title="{escape(artifacts.run_id)}">{escape(artifacts.run_id)}</span></div>'
        + (f'<div class="pv-side-meta"><span>Source</span><span class="pv-mono" title="{escape(artifacts.remote_source)}">'
           f'API</span></div>' if artifacts.remote_source else "")
        + f'<div class="pv-side-footer"><strong>Team 12</strong> · CIQ Hackathon 2026<br>v{APP_VERSION} · retrospective prototype</div>'
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
