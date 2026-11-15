"""Model Training — live 1-node vs 2-node GB10 training runs from scripts/train_ddp.py.

Reads each run folder under PRAVAH_TRAINING_DIR (default <outputs>/training):
``run_environment.json`` (written at start), ``train_metrics.csv`` (one row per
optimizer step, appended live) and ``training_summary.json`` (written at the end).
A speedup is shown only when the two runs pass every workload-equivalence check;
otherwise the measured numbers are shown side by side with the failing checks.
"""
from __future__ import annotations

import json

import altair as alt
import streamlit as st

from app.components.empty_state import render_empty_state
from app.components.layout import (
    html,
    render_badge,
    render_section_header,
)
from app.components.metric_card import render_metric_card
from app.context import AppContext, render_page_frame
from app.runs import training_dir, training_runs
from app.theme import TOKENS

# Validated dark-mode categorical slots (dataviz reference palette; checked on #111B2A).
CONFIG_COLORS = {"1 node": "#3987e5", "2 nodes": "#d95926", "Laptop (Apple GPU)": "#199e70",
                 "CPU / other": "#c98500"}


def _config(run: dict) -> str:
    info = run["summary"] or run["environment"] or {}
    if info.get("device") == "mps" or info.get("gpu_backend") == "Metal/MPS":
        return "Laptop (Apple GPU)"
    if not info.get("cuda_verified"):
        return "CPU / other"
    return "2 nodes" if info.get("world_size") == 2 else "1 node" if info.get("world_size") == 1 else "CPU / other"


def _row(run: dict) -> dict:
    s = run["summary"] or {}
    env = run["environment"] or {}
    info = s or env
    gpus = sorted({r.get("gpu_name") or r.get("device") for r in info.get("ranks") or []} - {None})
    last = run["metrics"][-1] if run["metrics"] else {}
    ev = run.get("evaluation") or {}
    vm = (((s.get("final_validation") or {}).get("metrics")) or (ev.get("selected_validation") or {}).get("metrics")
          or (run["validation"] or {}).get("metrics") or {})
    tm = (ev.get("official_test") or {}).get("metrics") or {}
    return {
        "Run": run["label"], "Status": run["status"].capitalize(), "Config": _config(run),
        "World size": info.get("world_size"), "Distinct hosts": info.get("distinct_hostnames"),
        "GPU": ", ".join(gpus) or env.get("gpu_backend") or "—", "CUDA verified": info.get("cuda_verified"),
        "Global batch": info.get("global_batch_size") or (run.get("config") or {}).get("batch_size"),
        "Steps": s.get("optimizer_steps") or ev.get("optimizer_steps")
        or (int(last["optimizer_step"]) if last else None),
        "Train time (s)": s.get("train_seconds") or ev.get("elapsed_seconds") or last.get("train_sec"),
        "Samples/s": s.get("samples_per_second"),
        "Last loss": last.get("loss_rank0"),
        "Val IoU": vm.get("iou"), "Test IoU": tm.get("iou"), "Test F1": tm.get("f1"),
    }


def _quality(runs: list[dict]) -> None:
    """Held-out scores of the newest evaluated model (written once, after model selection)."""
    evaluated = [r for r in runs if (r.get("evaluation") or {}).get("official_test")]
    if not evaluated:
        return
    r = evaluated[0]
    ev = r["evaluation"]
    tm = ev["official_test"]["metrics"]
    hold = (ev.get("bolivia_holdout") or {}).get("metrics") or {}
    render_section_header(f"How good is the model? · {r['label']}",
                          "Scored once on images never used for training or model selection")
    cols = st.columns(4)
    specs = [
        {"label": "Test IoU", "value": f"{tm['iou'] * 100:.1f}%", "subtext": "Sen1Floods11 official test split"},
        {"label": "Test F1", "value": f"{tm['f1'] * 100:.1f}%",
         "subtext": f"precision {tm['precision'] * 100:.0f}% · recall {tm['recall'] * 100:.0f}%"},
        {"label": "Unseen flood event IoU", "value": f"{hold['iou'] * 100:.1f}%" if hold.get("iou") is not None
         else None, "subtext": "Bolivia — whole event held out"},
        {"label": "Trained on", "value": _config(r), "subtext": f"epoch {ev.get('selected_epoch')} selected · "
         f"{(ev.get('elapsed_seconds') or 0) / 60:.1f} min"},
    ]
    for col, spec in zip(cols, specs, strict=True):
        with col:
            render_metric_card(**spec)
    st.caption("These measure water detection in Sentinel-2 images (hand-labelled), not flood forecasting. "
               "No Rasuwa accuracy is claimed: there are no independent hand labels for Rasuwa.")


def _curves(runs: list[dict]) -> None:
    rows = []
    for r in runs:
        for m in r["metrics"]:
            rows.append({"Run": r["label"], "Config": _config(r), "Step": m.get("optimizer_step"),
                         "Loss": m.get("loss_rank0"), "Seconds": m.get("train_sec", m.get("elapsed_sec"))})
    if not rows:
        st.caption("No train_metrics.csv rows yet.")
        return
    color = alt.Color("Config:N", scale=alt.Scale(domain=list(CONFIG_COLORS), range=list(CONFIG_COLORS.values())),
                      legend=alt.Legend(orient="top", title=None, labelColor=TOKENS["text_secondary"]))
    axis = {"labelColor": TOKENS["text_muted"], "titleColor": TOKENS["text_muted"], "titleFontWeight": "normal",
            "gridColor": TOKENS["border_subtle"], "domain": False, "ticks": False}
    base = alt.Chart(alt.Data(values=rows)).encode(
        y=alt.Y("Loss:Q", axis=alt.Axis(**axis)), color=color,
        strokeDash=alt.StrokeDash("Run:N", legend=alt.Legend(orient="top", title=None,
                                                             labelColor=TOKENS["text_secondary"])),
        tooltip=["Run:N", "Config:N", alt.Tooltip("Step:Q", format="d"), alt.Tooltip("Loss:Q", format=".4f"),
                 alt.Tooltip("Seconds:Q", format=".1f")])

    chart = (base.mark_line(strokeWidth=2, interpolate="monotone")
             .encode(x=alt.X("Seconds:Q", title="Training time (s)", axis=alt.Axis(**axis)))
             .properties(height=280).configure(background="transparent").configure_view(stroke=None))
    st.altair_chart(chart, width="stretch")


def equivalence_checks(one: dict, two: dict) -> list[tuple[str, bool]]:
    a, b = one.get("summary") or {}, two.get("summary") or {}
    return [
        ("Both runs completed (training_summary.json present)", bool(a and b)),
        ("Same training data (pairs manifest SHA-256)", bool(a.get("pairs_sha256")) and a.get("pairs_sha256") == b.get("pairs_sha256")),
        ("Same global batch size", a.get("global_batch_size") is not None and a.get("global_batch_size") == b.get("global_batch_size")),
        ("Same number of optimizer steps", a.get("optimizer_steps") is not None and a.get("optimizer_steps") == b.get("optimizer_steps")),
        ("Same seed and learning rate", (a.get("seed"), a.get("lr")) == (b.get("seed"), b.get("lr"))),
        ("Baseline ran with WORLD_SIZE = 1", a.get("world_size") == 1),
        ("Distributed run had WORLD_SIZE = 2", b.get("world_size") == 2),
        ("Two distinct hosts in the distributed run", b.get("distinct_hostnames") == 2),
        ("CUDA verified on every rank of both runs", bool(a.get("cuda_verified") and b.get("cuda_verified"))),
    ]


def _comparison(runs: list[dict]) -> None:
    done = [r for r in runs if r["summary"]]
    ones = [r for r in done if (r["summary"] or {}).get("world_size") == 1]
    twos = [r for r in done if (r["summary"] or {}).get("world_size") == 2]
    if not ones or not twos:
        st.caption("Needs one completed WORLD_SIZE=1 run and one completed WORLD_SIZE=2 run.")
        return
    by_id = {r["id"]: r for r in done}
    c1, c2 = st.columns(2)
    with c1:
        one = by_id[st.selectbox("Single-node run", [r["id"] for r in ones],
                                 format_func=lambda i: by_id[i]["label"], key="pv_tr_one")]
    with c2:
        two = by_id[st.selectbox("Two-node run", [r["id"] for r in twos],
                                 format_func=lambda i: by_id[i]["label"], key="pv_tr_two")]
    checks = equivalence_checks(one, two)
    ok = all(p for _, p in checks)
    a, b = one["summary"], two["summary"]
    t1, t2 = a.get("train_seconds"), b.get("train_seconds")
    speed = (t1 / t2) if ok and t1 and t2 else None
    va = ((a.get("final_validation") or {}).get("metrics") or {})
    vb = ((b.get("final_validation") or {}).get("metrics") or {})
    cols = st.columns(4)
    specs = [
        {"label": "1-node training time", "value": t1, "unit": "s", "icon_name": "timer", "subtext": one["label"]},
        {"label": "2-node training time", "value": t2, "unit": "s", "icon_name": "timer", "subtext": two["label"]},
        {"label": "Speedup", "value": None if speed is None else f"{speed:.2f}×", "icon_name": "speed",
         "empty_label": "Not claimed", "subtext": "T(1 node) / T(2 nodes)" if speed else "Checks failed — see below"},
        {"label": "Parallel efficiency", "value": None if speed is None else f"{speed / 2 * 100:.0f}%",
         "icon_name": "percent", "empty_label": "Not claimed", "subtext": "Speedup / 2"},
    ]
    for col, spec in zip(cols, specs, strict=True):
        with col:
            render_metric_card(**spec)
    failed = [name for name, p in checks if not p]
    if failed:
        st.caption("Speedup not claimed — " + "; ".join(failed).lower() + " failed.")
    else:
        q = [f"val IoU {x['iou']:.3f}" if x.get("iou") is not None else "val IoU —" for x in (va, vb)]
        st.caption(f"Same data, batch, steps and seed · CUDA on all ranks · 2 distinct hosts · "
                   f"1 node {q[0]} vs 2 nodes {q[1]}")
    export = {"schema_version": "pravah-training-benchmark-v1", "equivalent": ok,
              "checks": dict(checks), "one_node": a, "two_node": b,
              "speedup_train": speed, "parallel_efficiency": speed / 2 if speed else None,
              "note": "train_seconds = slowest rank's data+compute time; excludes validation and checkpointing."}
    st.download_button("Download benchmark JSON", json.dumps(export, indent=2),
                       file_name="training_benchmark.json", mime="application/json", icon=":material/download:")


def _empty() -> None:
    render_empty_state(
        "No training runs yet",
        f"Runs appear here live once scripts/train_ddp.py writes to {training_dir()}/<run_id>/. "
        "How to launch both experiments: docs/TRAINING_GB10.md.",
        icon_name="model_training", compact=True)


def _body() -> None:
    runs = training_runs()
    if not runs:
        _empty()
        return
    running = [r for r in runs if r["status"] == "running"]
    if running:
        html(" ".join(render_badge(f"Training now: {r['label']}", "success", dot=True) for r in running))
    _quality(runs)
    _comparison(runs)
    render_section_header("Training loss over time", "Colour = 1 or 2 nodes · line style = run")
    _curves(runs[:6])
    with st.expander(f"All runs ({len(runs)})"):
        st.dataframe([_row(r) for r in runs], hide_index=True, width="stretch", column_config={
            "Train time (s)": st.column_config.NumberColumn(format="%.1f"),
            "Samples/s": st.column_config.NumberColumn(format="%.2f"),
            "Last loss": st.column_config.NumberColumn(format="%.4f"),
            "Val IoU": st.column_config.NumberColumn(format="%.3f"),
            "Test IoU": st.column_config.NumberColumn(format="%.3f"),
            "Test F1": st.column_config.NumberColumn(format="%.3f"),
        })
        for r in runs:
            for e in r["errors"]:
                st.caption(f"⚠ {r['label']}: {e}")


def render_training_section(ctx: AppContext) -> None:
    render_page_frame(ctx, "Training", "Training · 1 vs 2 GB10 nodes",
                      "The same model trained on one GB10 node and on two with PyTorch DDP. Updates live.")
    live = st.toggle("Live refresh", value=True, key="pv_train_live")
    st.fragment(run_every=10 if live else None)(_body)()
