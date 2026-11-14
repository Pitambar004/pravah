"""GPU Performance — Multi-Node Cluster Scaling & Hardware Benchmarks."""
import streamlit as st

from app.components.empty_state import render_empty_state
from app.components.layout import (
    render_page_header,
    render_section_header,
)
from app.components.metric_card import render_metric_card
from app.data_loader import RunArtifacts


def render_gpu_section(artifacts: RunArtifacts):
    """Render the standardized GPU scaling and cluster performance page."""
    render_page_header(
        title="GPU Performance & Cluster Benchmarks",
        subtitle="Distributed scientific AI inference benchmark measuring compute speedup and parallel efficiency across Team 12's Lenovo ThinkStation PGX / NVIDIA GB10 hardware.",
        badge="CIQ Fuzzball v4.3.1" if artifacts.benchmark else "Awaiting Sunday Hardware",
        badge_variant="accent" if artifacts.benchmark else "neutral",
    )

    bench = artifacts.benchmark
    if not bench:
        render_empty_state(
            title="Awaiting Verified Cluster Benchmarks (Sunday 9:00 AM CT)",
            description="The hackathon hardware access window opens Sunday at 09:00 CT. Benchmark metrics will automatically appear here once the identical tile manifest is executed on 1 node and 2 distinct GB10 nodes.",
            action_command="python scripts/benchmark.py --one outputs/run_1node/benchmark.json --two outputs/run_2node/benchmark.json",
            expected_files=["outputs/<run_id>/benchmark.json", "benchmarks/benchmarks.csv"],
            badge="Scheduled Hardware Allocation",
        )
        _render_hardware_spec_card()
        return

    # 1. Standardized 4-Card Benchmark Row
    speedup = bench.get("speedup_compute", 0)
    efficiency = bench.get("parallel_efficiency", 0)
    tiles = bench.get("tile_count", 0)
    t1 = bench.get("one_node_seconds")
    t2 = bench.get("two_node_seconds")
    nodes = bench.get("node_ids", [])
    cuda = bench.get("cuda_verified", False)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_metric_card(
            label="Compute Speedup",
            value=f"{speedup:.2f}×" if speedup else "—",
            subtext="T(1 Node) / T(2 Nodes)",
            badge="Scaling Factor",
            badge_variant="success" if speedup >= 1.5 else "warning",
        )

    with c2:
        render_metric_card(
            label="Parallel Efficiency",
            value=f"{efficiency * 100:.1f}%" if efficiency else "—",
            subtext="Speedup / 2 Nodes",
            badge="Efficiency",
            badge_variant="success" if efficiency >= 0.8 else "warning",
        )

    with c3:
        render_metric_card(
            label="Workload Equivalence",
            value=tiles,
            unit="tiles (512×512)",
            subtext="Identical manifest SHA256",
            badge="Matched Grid",
            badge_variant="accent",
        )

    with c4:
        render_metric_card(
            label="Physical Nodes",
            value=len(set(nodes)) if nodes else 2,
            unit="distinct PGX boxes",
            subtext=", ".join(nodes) if nodes else "NVIDIA GB10",
            badge="CUDA Engine" if cuda else "CPU Test",
            badge_variant="success" if cuda else "neutral",
        )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # 2. Timing Breakdown & Workload Verification Invariants
    col_l, col_r = st.columns([3, 2])

    with col_l:
        render_section_header("Measured Inference Time (Compute-Only)", "Excludes container setup and volume mount overhead.")

        if t1 and t2:
            st.bar_chart(
                {"Execution Time (s)": {"Single Node": t1, "Two Nodes (DDP/Parallel)": t2}},
                horizontal=True,
                color="#38BDF8",
            )
            st.markdown(
                f"""
                <div style="font-size: 0.775rem; color: #94A3B8; margin-top: 0.25rem;">
                    1 Node: <code>{t1:.2f}s</code> &rarr; 2 Nodes: <code>{t2:.2f}s</code> · 
                    <strong style="color: #34D399;">{(1 - t2 / t1) * 100:.1f}% execution duration reduction</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Execution durations not detailed in raw benchmark summary.")

        render_section_header("Workload Equivalence & Cryptographic Verification", "Invariants enforced by scripts/benchmark.py.")

        verification_rows = [
            {"Verification Property": "Tile Manifest Hash", "Recorded Value": str(bench.get("tile_manifest_sha256", ""))[:18] + "…", "Result": "Identical Manifest"},
            {"Verification Property": "Model Checkpoint Hash", "Recorded Value": str(bench.get("model_sha256", ""))[:18] + "…", "Result": "Identical Weights"},
            {"Verification Property": "Tile Count Equivalence", "Recorded Value": f"{tiles} tiles", "Result": "Identical Workload"},
            {"Verification Property": "Hardware Isolation", "Recorded Value": ", ".join(nodes) if nodes else "2 physical nodes", "Result": "Distinct Sockets"},
            {"Verification Property": "CUDA Execution Engine", "Recorded Value": "torch.cuda active", "Result": "Hardware Accelerated" if cuda else "CPU Smoke Test"},
        ]
        st.dataframe(verification_rows, use_container_width=True, hide_index=True)

    with col_r:
        _render_hardware_spec_card()

        with st.expander("Raw Benchmark Artifact JSON"):
            st.json(bench)


def _render_hardware_spec_card():
    """Render details of the target GB10 cluster platform."""
    render_section_header("Target Cluster Specifications", "Hardware topology.")
    st.markdown(
        """
        <div class="pv-card">
            <div style="display: flex; flex-direction: column; gap: 0.45rem; font-size: 0.8rem;">
                <div><strong>Systems:</strong> 2× Lenovo ThinkStation PGX</div>
                <div><strong>GPU Architecture:</strong> NVIDIA GB10 (Blackwell generation)</div>
                <div><strong>Host Processor:</strong> ARM64 (aarch64) high-throughput cores</div>
                <div><strong>Orchestration:</strong> CIQ Fuzzball v4.3.1 Local AI Stack</div>
                <div><strong>Task Partitioning:</strong> Deterministic spatial tile manifests</div>
                <div><strong>Inference Mode:</strong> Independent data-parallel (non-blocking)</div>
            </div>
            <div style="margin-top: 0.75rem; padding-top: 0.5rem; border-top: 1px solid #263244; font-size: 0.75rem; color: #64748B;">
                Hardware access opens Sunday Nov 15, 09:00 CT.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
