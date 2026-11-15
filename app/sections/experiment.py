"""Nepal experimental replay: audited inputs, immutable freeze and later reveal."""

import json
import os
import uuid
from datetime import datetime, time
from pathlib import Path

import streamlit as st

from app.components.layout import render_badge, render_page_header, render_topbar
from app.components.timeline import NPT
from app.context import AppContext
from pravah.replay.experiment import audit, freeze, register_evaluation, verify_snapshot


def render_experiment(ctx: AppContext):
    render_topbar(
        ["PRAVAH", "Nepal Experiment"],
        [
            render_badge("Rasuwa, Nepal", "outline"),
            render_badge("Historical case study · 26 Aug 2026", "outline"),
        ],
    )
    render_page_header(
        "Nepal Experimental Replay",
        "Freeze evidence available at a historical cutoff, then inspect later observations.",
    )
    st.warning(
        "Research reconstruction. The working models detect satellite water. Downstream forecasts, arrival times and warning accuracy require a separately trained and evaluated model."
    )
    root = Path(os.environ.get("PRAVAH_REPLAY_ROOT", "outputs/nepal_replay/prepared"))
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except (OSError, ValueError):
        st.info("Prepared Nepal evidence is not available in this workspace.")
        return
    date_col, time_col = st.columns(2)
    with date_col:
        date = st.date_input(
            "Cutoff date (Nepal time)",
            value=datetime(2026, 8, 26).date(),
            key="pv_exp_date",
        )
    with time_col:
        clock = st.time_input(
            "Cutoff time (Nepal time)", value=time(8, 30), key="pv_exp_time"
        )
    cutoff = datetime.combine(date, clock, tzinfo=NPT).isoformat()
    try:
        result = audit(manifest, cutoff, root)
    except (ValueError, KeyError) as exc:
        st.error(f"Evidence could not be audited: {exc}")
        return
    left, right = st.columns(2)
    left.metric("Qualifying inputs", result["eligible_input_count"])
    right.metric("Excluded sources", result["excluded_count"])
    st.caption(
        "Both acquisition and verified availability must precede the cutoff. An archived observation date alone is insufficient."
    )
    for blocker in manifest["blockers"]:
        st.write("• " + blocker)
    if st.button("Freeze eligible inputs", key="pv_exp_freeze"):
        identity = (
            "rasuwa_"
            + datetime.now().strftime("%Y%m%dT%H%M%S")
            + "_"
            + uuid.uuid4().hex[:8]
        )
        try:
            freeze(root, manifest, cutoff, identity)
            st.session_state["pv_experiment_id"] = identity
        except (OSError, ValueError) as exc:
            st.error(f"Could not freeze inputs: {exc}")
    identity = st.session_state.get("pv_experiment_id")
    if identity:
        try:
            snapshot = verify_snapshot(root, identity)
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"Frozen input verification failed: {exc}")
            return
        st.success("Input snapshot preserved: " + identity)
        st.caption(
            "Frozen cutoff: "
            + snapshot["cutoff_at_utc"]
            + ". Changing the controls requires a new snapshot."
        )
        st.download_button(
            "Download frozen inputs",
            json.dumps(snapshot, indent=2),
            "snapshot.json",
            "application/json",
        )
        if snapshot["prediction"] is None:
            st.info(
                "No downstream prediction was produced. Forecast extent, exposure and accuracy are unavailable."
            )
        if st.button("Reveal later evaluation evidence", key="pv_exp_reveal"):
            try:
                register_evaluation(root, manifest, identity)
            except FileExistsError:
                pass
            except (OSError, ValueError, KeyError) as exc:
                st.error(f"Could not reveal references: {exc}")
        evaluation_file = root / "experiments" / identity / "evaluation.json"
        if evaluation_file.exists():
            evaluation = json.loads(evaluation_file.read_text())
            st.warning(
                "Later evidence is retrospective. It was not used as a frozen input."
            )
            for reference in evaluation["references"]:
                st.write(reference["asset_id"])
                st.caption(reference.get("source_uri") or "Source citation unavailable")
            st.info(evaluation["reason"])
            st.download_button(
                "Download evaluation record",
                json.dumps(evaluation, indent=2),
                "evaluation.json",
                "application/json",
            )
    with st.expander("Why sources were excluded"):
        for item in result["excluded"]:
            st.write(item["asset_id"] + ": " + item["reason"].replace("_", " "))
