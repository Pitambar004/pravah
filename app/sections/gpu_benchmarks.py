"""Compute & Performance — verified compute environment and benchmark results only."""
from __future__ import annotations

import altair as alt
import streamlit as st

from app.components.empty_state import render_empty_state
from app.components.layout import (
    card,
    html,
    kv_list,
    render_badge,
    render_section_header,
)
from app.components.metric_card import render_metric_card
from app.context import AppContext, render_page_frame
from app.data_loader import RunArtifacts
from app.theme import TOKENS


def runtime_chart(t1: float, t2: float) -> alt.Chart:
    """Single-series horizontal bar chart of compute-only runtime."""
    data = alt.Data(values=[
        {"Configuration": "1 node", "Seconds": t1},
        {"Configuration": "2 nodes", "Seconds": t2},
    ])
    base = alt.Chart(data).encode(
        y=alt.Y("Configuration:N", sort=None, title=None,
                axis=alt.Axis(labelColor=TOKENS["text_secondary"], labelFontSize=12, ticks=False, domain=False, labelPadding=8)),
        x=alt.X("Seconds:Q", title="Compute time (s)",
                axis=alt.Axis(labelColor=TOKENS["text_muted"], titleColor=TOKENS["text_muted"], titleFontWeight="normal",
                              gridColor=TOKENS["border_subtle"], domain=False, ticks=False, tickCount=5)),
        tooltip=[alt.Tooltip("Configuration:N"), alt.Tooltip("Seconds:Q", format=".2f")],
    )
    bars = base.mark_bar(color=TOKENS["accent"], height=22, cornerRadiusEnd=4)
    labels = base.mark_text(align="left", dx=6, color=TOKENS["text_secondary"], fontSize=12).encode(
        text=alt.Text("Seconds:Q", format=".2f"))
    return (bars + labels).properties(height=120).configure(background="transparent").configure_view(stroke=None)


def render_compute_section(ctx: AppContext) -> None:
    render_page_frame(
        ctx,
        "Compute & Performance",
        "Compute & Performance",
        "Compute environment, processing runs and multi-node inference benchmarks. "
        "Only measured values from benchmark artifacts are shown.",
    )
    a: RunArtifacts = ctx.artifacts
    bench = a.benchmark or {}
    meta = a.observed_metadata or {}

    col_env, col_runs = st.columns([2, 3])
    with col_env:
        render_section_header("Compute environment", "Planned target; verified only by a benchmark artifact")
        nodes = bench.get("node_ids") or []
        cuda = bench.get("cuda_verified")
        env_state = render_badge("Fixture", "warning") if a.is_synthetic else (
            render_badge("Verified run", "success") if bench else render_badge("Not yet verified", "neutral"))
        html(card("Target cluster", kv_list([
            ("Hardware (planned)", "2× Lenovo ThinkStation PGX · NVIDIA GB10"),
            ("Orchestration (planned)", "CIQ Fuzzball 4.3.1"),
            ("Access window", "Team plan: Sun 15 Nov 2026, 09:00 CT"),
            ("Node IDs", ", ".join(nodes) if nodes else "Not recorded"),
            ("CUDA verified", "Yes" if cuda is True else "No" if cuda is False else "Not recorded"),
            ("Workflow status", "Recorded in benchmark.json" if bench else "No workflow run recorded"),
        ], mono_keys=("Node IDs",)), trailing=env_state))
    with col_runs:
        render_section_header("Processing runs", "From observed_metadata.json and benchmark.json")
        if meta or bench:
            st.dataframe(
                [{
                    "Run ID": meta.get("run_id") or a.run_id,
                    "Model": meta.get("model") or meta.get("model_name") or "Not recorded",
                    "Input scene": meta.get("source_raster") or "Not recorded",
                    "Tiles": bench.get("tile_count") or meta.get("tile_count"),
                    "Status": "Synthetic fixture" if a.is_synthetic else "Completed (artifact present)",
                    "2-node compute (s)": bench.get("two_node_seconds"),
                }],
                hide_index=True,
                width="stretch",
            )
        else:
            render_empty_state("No processing runs recorded", "Run metadata appears after tile inference and stitching.",
                               expected_files=["observed_metadata.json"], icon_name="pending_actions", compact=True)

    render_section_header("Multi-node performance", "Compute-only time for an identical tile manifest on 1 vs 2 nodes")
    if not bench:
        render_empty_state(
            "Benchmark results will appear after a verified compute run",
            "Speedup and efficiency are only shown when the same tile manifest and model hash have been executed on "
            "one node and on two distinct nodes with CUDA verified.",
            action_command="python scripts/benchmark.py --one <1-node benchmark.json> --two <2-node benchmark.json>",
            expected_files=["outputs/<run_id>/benchmark.json"],
            icon_name="speed",
        )
        return

    t1, t2 = bench.get("one_node_seconds"), bench.get("two_node_seconds")
    speedup, eff = bench.get("speedup_compute"), bench.get("parallel_efficiency")
    src = ctx.source_note("benchmark.json")
    cols = st.columns(4)
    specs = [
        {"label": "Single-node time", "icon_name": "timer", "value": t1, "unit": "s", "subtext": src},
        {"label": "Two-node time", "icon_name": "timer", "value": t2, "unit": "s", "subtext": src},
        {"label": "Speedup", "icon_name": "speed", "value": None if speedup is None else f"{speedup:.2f}×",
             "subtext": "T(1 node) / T(2 nodes)"},
        {"label": "Parallel efficiency", "icon_name": "percent", "value": None if eff is None else f"{eff * 100:.0f}%",
             "subtext": "Speedup / 2 nodes"},
    ]
    for col, spec in zip(cols, specs, strict=True):
        with col:
            render_metric_card(**spec)

    col_chart, col_ver = st.columns([3, 2])
    with col_chart:
        if t1 and t2:
            st.altair_chart(runtime_chart(t1, t2), width="stretch")
            thr1, thr2 = bench.get("throughput_one_node_tiles_per_sec"), bench.get("throughput_two_node_tiles_per_sec")
            if thr1 and thr2:
                st.caption(f"Throughput: {thr1:.2f} → {thr2:.2f} tiles/s · {bench.get('note', 'compute-only, not wall-clock')}")
        else:
            render_empty_state("Runtimes not recorded", "benchmark.json does not include per-configuration times.",
                               badge=None, icon_name="timer_off", compact=True)
    with col_ver:
        html(card("Workload equivalence", kv_list([
            ("Tile manifest SHA-256", str(bench.get("tile_manifest_sha256", "—"))[:16] + "…"),
            ("Model SHA-256", str(bench.get("model_sha256", "—"))[:16] + "…"),
            ("Tile count", str(bench.get("tile_count", "—"))),
            ("Distinct nodes", str(len(set(bench.get("node_ids") or []))) if bench.get("node_ids") else "Not recorded"),
        ], mono_keys=("Tile manifest SHA-256", "Model SHA-256"))))

    with st.expander("Execution details · raw benchmark.json"):
        st.json(bench)


# Backwards-compatible name
render_gpu_section = render_compute_section
