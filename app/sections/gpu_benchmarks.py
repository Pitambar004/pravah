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
    runtime = meta.get("runtime") or {}

    col_env, col_runs = st.columns([2, 3])
    with col_env:
        render_section_header("Compute environment", "Actual runtime recorded with the selected prediction")
        nodes = bench.get("node_ids") or []
        cuda = runtime.get("cuda_verified", bench.get("cuda_verified"))
        env_state = render_badge("Fixture", "warning") if a.is_synthetic else (
            render_badge("Recorded run", "success") if runtime or bench else render_badge("Not yet recorded", "neutral"))
        html(card("Execution environment", kv_list([
            ("Device", runtime.get("device", "Not recorded")),
            ("GPU", runtime.get("gpu_name") or "No CUDA GPU used"),
            ("Host", runtime.get("hostname", "Not recorded")),
            ("PyTorch", runtime.get("torch_version", "Not recorded")),
            ("Node IDs", ", ".join(nodes) if nodes else "Not recorded"),
            ("CUDA verified", "Yes" if cuda is True else "No" if cuda is False else "Not recorded"),
            ("Execution", "Fuzzball only when recorded in workflow provenance" if cuda else "Local prediction"),
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
                    "Model compute (s)": (meta.get("timings_seconds") or {}).get("model_compute"),
                    "Inference total (s)": (meta.get("timings_seconds") or {}).get("end_to_end"),
                }],
                hide_index=True,
                width="stretch",
            )
        else:
            render_empty_state("No processing runs recorded", "Run metadata appears after tile inference and stitching.",
                               expected_files=["observed_metadata.json"], icon_name="pending_actions", compact=True)

    evaluation = meta.get("training_evaluation")
    if evaluation:
        render_section_header("Completed training evaluation", "Recorded full-dataset run; checkpoint selected on validation only")
        st.dataframe([{"Split": name, **evaluation[key]["metrics"]} for name, key in
                      [("Official test", "official_test"), ("Bolivia holdout", "bolivia_holdout")]], hide_index=True)
        st.caption(f"Selected epoch {evaluation['selected_epoch']} · {evaluation['optimizer_steps']:,} training steps · checkpoint {evaluation['checkpoint_sha256'][:16]}…")
    render_section_header("Multi-node performance", "Requires the identical workload measured on one and two actual nodes")
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
