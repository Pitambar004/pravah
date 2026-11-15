"""Prediction vs Reality — scores a flood prediction against what satellites later observed.

Every number is read from ``prediction_validation.json`` written by
``scripts/validate_prediction.py``; nothing is computed or estimated here, and no
score is shown when the prediction failed the temporal-integrity check.
"""
from __future__ import annotations

from html import escape

import streamlit as st

from app.components.compare_map import MapSide, render_compare_map
from app.components.context_layers import with_downloaded_context
from app.components.empty_state import render_empty_state
from app.components.layout import render_callout, render_section_header
from app.components.map_view import (
    build_layers,
    prediction_layers,
    render_map_workspace,
    run_layers,
    short_run_label,
)
from app.components.metric_card import render_metric_card
from app.context import CASE_STUDY, AppContext, render_page_frame
from app.data_loader import RunArtifacts
from app.runs import all_runs, run_role


def _pct(v) -> str | None:
    return None if v is None else f"{v * 100:.1f}%"


def _pair(runs: list[RunArtifacts]) -> tuple[RunArtifacts, RunArtifacts] | None:
    """First post-event run and the pre-event run it was change-analysed against."""
    by_id = {r.run_id: r for r in runs}
    for post in (r for r in runs if run_role(r) == "post_event"):
        pre = by_id.get((post.change_summary or {}).get("pre_run_id") or "")
        if pre:
            return pre, post
    return None


def _no_prediction(runs: list[RunArtifacts]) -> None:
    refs = [r for r in runs if run_role(r) == "post_event"]
    pres = [r for r in runs if run_role(r) == "pre_event"]
    render_empty_state(
        "No flood forecast supplied",
        "No forecast run was found in the configured outputs folder. This page is not running a processing job. "
        "Satellite water maps describe conditions at image acquisition; river and rainfall outlooks do not "
        "supply the predicted flood-extent map needed for this comparison.",
        badge="Forecast unavailable", icon_name="info", compact=True)
    with st.expander("How to add and score a flood forecast"):
        st.markdown(
            "Supply a forecast run containing `prediction_metadata.json` and either `predicted_flood.tif` "
            "or `predicted_flood_probability.tif`, following `docs/PREDICTION_CONTRACT.md`. "
            "Every input must have verified acquisition and availability times at or before the forecast cutoff.\n\n"
            "The command below scores an existing forecast; it does not generate one. Replace `<pred_run>` "
            "with its directory and `<UTC>` with the documented event onset. Pair each post-event reference "
            "with a pre-event baseline on the same grid, in the same order."
        )
        st.code(
            "python scripts/validate_prediction.py --prediction outputs/<pred_run> "
            + " ".join(f"--reference outputs/{r.run_id}" for r in refs[:2])
            + "".join(f" --baseline outputs/{r.run_id}" for r in pres[:2])
            + " --target new_water --event-time <UTC>", language="bash")
    pair = _pair(runs)
    if pair:
        pre, post = pair
        render_section_header("Available satellite evidence", "Water before vs after the event · retrospective comparison, not a forecast")
        render_compare_map(
            MapSide("Before", short_run_label(pre.run_id), with_downloaded_context(run_layers(pre, "pre"), light=True), "Satellite"),
            MapSide("After", short_run_label(post.run_id), with_downloaded_context(run_layers(post, "post"), light=True),
                    "Satellite"), height=520)


def _verdict(v: dict) -> None:
    integ = v.get("integrity") or {}
    if v.get("status") == "rejected_integrity":
        render_callout("The prediction used data that was not available before its cutoff, or the cutoff is "
                       "undocumented, so it is not scored.<br>"
                       + "<br>".join(f"• {escape(x)}" for x in integ.get("violations") or []),
                       title="Rejected: not a genuine prediction", variant="error")
        return
    comb = v.get("combined") or {}
    skill = comb.get("skill_vs_persistence") or {}
    lead = integ.get("lead_time_hours_before_event")
    msg = []
    if skill:
        msg.append("The model <b>beats</b> the 'nothing changes' baseline." if skill.get("beats_baseline")
                   else "The model does <b>not</b> beat the 'nothing changes' baseline.")
    if integ.get("claim_type") == "early_warning" and lead is not None:
        msg.append(f"Inputs closed <b>{lead:.1f} h</b> before the event.")
    elif integ.get("claim_type") == "post_event_nowcast":
        msg.append("Inputs closed after the event began, so this is a nowcast, not an early warning.")
    render_callout(" ".join(msg) or "Scored.", title="Result",
                   variant="success" if skill.get("beats_baseline") else "warning")


def _scores(v: dict) -> None:
    comb = v.get("combined") or {}
    m = comb.get("metrics") or {}
    b = (comb.get("persistence_baseline") or {}).get("metrics") or {}

    def sub(k):
        return f"baseline {_pct(b.get(k)) or '—'}" if b else None

    cols = st.columns(4)
    specs = [
        {"label": "Overlap (IoU)", "value": _pct(m.get("csi")), "subtext": sub("csi"),
         "help_text": "Correctly predicted area ÷ (predicted ∪ observed). 100% = perfect."},
        {"label": "Flooding detected", "value": _pct(m.get("pod")), "subtext": sub("pod"),
         "help_text": "Share of observed flooding that was predicted."},
        {"label": "False alarms", "value": _pct(m.get("far")), "subtext": sub("far"),
         "help_text": "Share of predicted flooding that did not happen. Lower is better."},
        {"label": "F1", "value": _pct(m.get("f1")), "subtext": sub("f1")},
    ]
    for col, spec in zip(cols, specs, strict=True):
        with col:
            render_metric_card(**spec)


def _details(pm: dict, v: dict) -> None:
    integ = v.get("integrity") or {}
    with st.expander("Method, sources & limitations"):
        st.markdown(f"**Model** {pm.get('model_id') or '—'} · **input cutoff** {integ.get('input_cutoff_utc') or '—'}"
                    f" · **event** {integ.get('event_time_utc') or '—'} · **scored against** "
                    f"{', '.join(integ.get('reference_acquired_utc') or []) or '—'}")
        cutoff = integ.get("input_cutoff_utc") or pm.get("input_cutoff_utc")
        st.dataframe([{
            "Input": i.get("name"), "Acquired": i.get("acquired_at_utc") or i.get("valid_to_utc"),
            "Available": i.get("available_at_utc"),
            "Before cutoff": bool(cutoff and i.get("available_at_utc") and i["available_at_utc"] <= cutoff),
        } for i in pm.get("inputs") or []], hide_index=True, width="stretch")
        st.dataframe([{
            "Observation": short_run_label(r.get("reference_run_id", "")),
            "Hit km²": (r.get("area_km2") or {}).get("hit"),
            "False alarm km²": (r.get("area_km2") or {}).get("false_alarm"),
            "Miss km²": (r.get("area_km2") or {}).get("miss"),
            "IoU": (r.get("metrics") or {}).get("iou"),
        } for r in v.get("references") or []], hide_index=True, width="stretch",
            column_config={"IoU": st.column_config.NumberColumn(format="percent")})
        for lim in dict.fromkeys([*(pm.get("limitations") or []), *(v.get("interpretation_limits") or [])]):
            st.caption(f"• {lim}")


def render_validation_section(ctx: AppContext) -> None:
    render_page_frame(ctx, "Prediction vs Reality", "Prediction vs Reality",
                      f"Forecasts and baselines compared with satellite-derived water after the {CASE_STUDY['event_date']} event.")
    if ctx.is_synthetic:
        render_callout("Scores are never shown for synthetic fixtures.", title="Fixture mode", variant="warning")
        return
    runs = all_runs()
    preds = [r for r in runs if r.is_prediction]
    if not preds:
        _no_prediction(runs)
        return
    by_id = {r.run_id: r for r in runs}
    def prediction_label(run_id):
        kind = (by_id[run_id].prediction_metadata or {}).get("prediction_kind")
        prefix = "Persistence baseline" if kind == "persistence_baseline" else "Forecast"
        return f"{prefix} · {short_run_label(run_id)}"

    p = by_id[st.selectbox("Evaluation run", [r.run_id for r in preds], key="pv_pred_run", format_func=prediction_label,
                           label_visibility="collapsed" if len(preds) == 1 else "visible")]
    pm, v = p.prediction_metadata or {}, p.prediction_validation
    if pm.get("prediction_kind") == "persistence_baseline":
        render_callout(
            "Assumes pre-event water stays unchanged, so it predicts no new inundation. "
            "Reconstructed after the event; not a trained forecast or a historical early warning.",
            title="Retrospective persistence baseline", variant="warning")
    if not v:
        render_callout("Run scripts/validate_prediction.py on this prediction.", title="Not scored yet")
        return
    if pm.get("prediction_kind") != "persistence_baseline" or v.get("status") != "scored":
        _verdict(v)
    if v.get("status") != "scored":
        _details(pm, v)
        return
    _scores(v)

    refs = [by_id[r["reference_run_id"]] for r in v.get("references", []) if r.get("reference_run_id") in by_id]
    right = []
    for r in refs:
        for lyr in run_layers(r, short_run_label(r.run_id)):
            if v.get("target") == "new_water":
                lyr.visible = lyr.id.startswith("change_new")
            right.append(lyr)
    comparison, errors = st.tabs(["Map comparison", "Hits & misses"], key="pv_validation_map", on_change="rerun")
    with comparison:
        if comparison.open:
            st.caption("Synchronized views · satellite water is model-derived, not field-verified ground truth.")
            render_compare_map(
                MapSide("Predicted", f"inputs up to {pm.get('input_cutoff_utc') or '—'}", with_downloaded_context(prediction_layers(p)[:1], light=True), "Model"),
                MapSide("Observed after the event", ", ".join(short_run_label(r.run_id) for r in refs),
                        with_downloaded_context(right, light=True), "Satellite"),
                height=520)
    with errors:
        if errors.open:
            st.caption("Green: correctly predicted · amber: false alarm · red: missed")
            context = [lyr for lyr in build_layers(ctx.artifacts, show_retrospective=False)
                       if lyr.group == "Geographic context"]
            render_map_workspace(with_downloaded_context(context + prediction_layers(p)), height=560, mode="full")
    _details(pm, v)
