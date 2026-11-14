# ruff: noqa: N999
"""PRAVAH — Satellite-Derived Flood Evidence & Distributed Inference Platform."""
import sys
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from app.data_loader import discover_available_runs, load_run_artifacts
from app.theme import apply_theme

# 1. Page Configuration
st.set_page_config(
    page_title="Pravah — Flood Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Unified Design System CSS
apply_theme()

# 3. Persistent Sidebar Navigation
with st.sidebar:
    # Wordmark & Brand Header
    st.markdown(
        """
        <div style="padding-bottom: 0.75rem; border-bottom: 1px solid #263244; margin-bottom: 1rem;">
            <div style="font-size: 1.25rem; font-weight: 700; color: #F1F5F9; letter-spacing: -0.02em; display: flex; align-items: center; gap: 0.5rem;">
                <span>🌊</span> Pravah
            </div>
            <div style="font-size: 0.725rem; color: #94A3B8; font-weight: 500; margin-top: 0.2rem;">
                Scientific Flood Intelligence & Distributed AI
            </div>
            <div style="font-size: 0.685rem; color: #64748B; font-family: 'JetBrains Mono', monospace; margin-top: 0.35rem;">
                Rasuwa, Nepal · Aug 26, 2026
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Clean Navigation
    NAV_PAGES = [
        "Overview",
        "Flood Evidence Map",
        "Historical Replay",
        "Infrastructure Exposure",
        "GPU Performance",
        "Sources & Methodology",
    ]

    page = st.radio(
        "Navigation",
        NAV_PAGES,
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Run Artifact Selector
    st.markdown(
        '<div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; margin-bottom: 0.5rem;">Active Execution Run</div>',
        unsafe_allow_html=True,
    )

    runs = discover_available_runs("outputs")
    if runs:
        chosen = st.selectbox("Select Run", runs, index=0, label_visibility="collapsed")
        run_folder = f"outputs/{chosen}"
    else:
        run_folder = st.text_input("Output Directory", "outputs/latest", label_visibility="collapsed")

    # Demo Fixtures Toggle
    use_demo = st.toggle(
        "UI Demo Fixtures",
        value=not bool(runs),
        help="Loads clearly labeled [SYNTHETIC TEST FIXTURE] datasets to verify UI layouts when outputs are awaiting execution.",
    )

    # Load artifacts
    artifacts = load_run_artifacts(run_folder, allow_synthetic_fallback=use_demo)

    # Status Pill
    if artifacts.is_synthetic:
        status_html = '<span class="pv-badge pv-badge-warning">🧪 Demo Fixtures</span>'
    elif artifacts.has_any_artifacts:
        status_html = '<span class="pv-badge pv-badge-success">● Live Artifacts</span>'
    else:
        status_html = '<span class="pv-badge pv-badge-neutral">○ Awaiting Run</span>'

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem; font-size: 0.75rem; color: #64748B;">
            <span>Status</span>
            {status_html}
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.25rem; font-size: 0.72rem; color: #64748B;">
            <span>Run ID</span>
            <span style="font-family: 'JetBrains Mono', monospace; color: #94A3B8;">{artifacts.run_id}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Footer Metadata
    st.markdown(
        """
        <div style="font-size: 0.72rem; color: #64748B; line-height: 1.45;">
            <strong>Team 12</strong> · CIQ Local AI Hackathon<br>
            2× Lenovo PGX / NVIDIA GB10
        </div>
        """,
        unsafe_allow_html=True,
    )

# 4. Global Scientific Alert Banner (if Demo Fixture is active)
if artifacts.is_synthetic:
    st.markdown(
        """
        <div class="pv-callout pv-callout-warning" style="margin-top: 0.5rem;">
            <strong>Development Fixture Mode Active:</strong> Current views display synthetic test fixtures for UI/UX verification.
            These geometries and benchmark metrics are not authentic disaster observations.
        </div>
        """,
        unsafe_allow_html=True,
    )

# 5. Route to Selected Section
if page == "Overview":
    from app.sections.overview import render_overview_section
    render_overview_section(artifacts)
elif page == "Flood Evidence Map":
    from app.sections.flood_map import render_flood_map_section
    render_flood_map_section(artifacts)
elif page == "Historical Replay":
    from app.sections.replay import render_replay_section
    render_replay_section(artifacts)
elif page == "Infrastructure Exposure":
    from app.sections.exposure import render_exposure_section
    render_exposure_section(artifacts)
elif page == "GPU Performance":
    from app.sections.gpu_benchmarks import render_gpu_section
    render_gpu_section(artifacts)
elif page == "Sources & Methodology":
    from app.sections.methodology import render_methodology_section
    render_methodology_section(artifacts)

# 6. Global Scientific Integrity Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem; color: #64748B; flex-wrap: wrap; gap: 0.5rem;">
        <div>
            <strong>Pravah</strong> · Scientific AI & Multi-Node Inference Prototype · Team 12
        </div>
        <div>
            Retrospective evidence only · Not an operational emergency warning system · EPSG:32645 UTM 45N
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
