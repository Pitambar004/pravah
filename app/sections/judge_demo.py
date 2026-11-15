"""Automatic judge presentation with explicit exploratory office alerts."""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from app.components.map_view import MapLayer, render_map_workspace
from app.image_data import get_image_pairs
from app.nepal_data import geography_layers, load_nepal_geography
from app.sections.lakes import get_lakes
from app.sections.nepal_monitor import CORRIDOR_VIEW
from app.theme import TOKENS
from pravah.replay.cascade import acquisition_footprints, daily_history
from pravah.replay.experiment import freeze, register_evaluation, verify_snapshot
from pravah.replay.presentation import DURATION_SECONDS, PHASES, presentation_stage
from pravah.replay.scenario import explore

CUTOFF = "2026-08-26T08:30:00+05:45"


def replay_images(demo):
    """Upgrade retained Streamlit sessions without changing their snapshot."""
    if demo.get("images"):
        return demo["images"], None
    context, error = get_image_pairs()
    if context:
        demo["images"] = context
    return context, error


def prepare_judge_demo(root, manifest):
    image_context, image_error = get_image_pairs()
    if not image_context or not {"pre_event", "post_event"}.issubset(image_context.get("pairs", {})):
        raise ValueError("Real replay images are unavailable. Run scripts/nepal_replay/prepare_images.py. " + (image_error or ""))
    st.session_state.pop("pv_judge_evaluation_result", None)
    geography, error = load_nepal_geography()
    if error or not geography:
        raise ValueError("Mapped Nepal corridor is required for the judge replay")
    identity = "judge_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    snapshot = freeze(root, manifest, CUTOFF, identity)
    scenario = explore(geography["layers"], 25, 100)
    scenario.update(geography_snapshot=geography["metadata"].get("retrieved_at_utc"),
                    assumed_signals={"rapid_lake_expansion": True, "seismic_disturbance": True,
                                     "gauge_unavailable": True, "high_water": True},
                    replay_window={"start_date": "2026-08-11", "cutoff": CUTOFF},
                    attribution="OpenStreetMap contributors · ODbL 1.0")
    folder = root / "experiments" / identity
    (folder / "exploratory_scenario.json").write_text(json.dumps(scenario, indent=2) + "\n")
    (folder / "presentation_plan.json").write_text(json.dumps({
        "kind": "exploratory_warning_workflow", "planned_duration_seconds": DURATION_SECONDS,
        "phases": PHASES, "cutoff": CUTOFF, "snapshot_sha256": snapshot["snapshot_sha256"],
        "alert_semantics": "Assumed workflow signals; not verified historical office alerts",
        "forecast": None, "accuracy": None}, indent=2) + "\n")
    return {"identity": identity, "scenario": scenario, "started_monotonic": time.monotonic(),
            "finished": False, "evaluation_saved": False, "history": daily_history(root)["rows"],
            "snapshot": snapshot, "phase_seen": [], "images": image_context}


def render_judge_demo(root: Path, manifest):
    demo = st.session_state["pv_judge_demo"]
    # Streamlit retains active sessions through source reloads. Upgrade a
    # previously started replay before accessing the new image evidence.
    image_context, image_error = replay_images(demo)
    if not image_context:
        st.error("Replay images are unavailable: " + (image_error or "Prepare the real image pairs first."))
        if st.button("Return to replay controls", key="pv_judge_missing_images"):
            st.session_state.pop("pv_judge_demo", None)
            st.rerun()
        return
    elapsed = time.monotonic() - demo["started_monotonic"]
    stage = presentation_stage(elapsed)
    phase = stage["phase"]
    if phase not in demo["phase_seen"]:
        demo["phase_seen"].append(phase)
    if stage["evaluation_revealed"] and not demo["evaluation_saved"]:
        try:
            register_evaluation(root, manifest, demo["identity"])
        except FileExistsError:
            pass
        demo["evaluation_saved"] = True
    if phase == "complete" and not demo["finished"]:
        demo["finished"] = True
        demo["measured_elapsed_seconds"] = elapsed
        (root / "experiments" / demo["identity"] / "presentation_result.json").write_text(json.dumps({
            "kind": "exploratory_warning_workflow", "planned_duration_seconds": DURATION_SECONDS,
            "measured_elapsed_seconds": elapsed, "phases_seen": demo["phase_seen"],
            "evaluation_revealed": demo["evaluation_saved"], "forecast": None, "accuracy": None}, indent=2) + "\n")
        st.rerun()
    top = st.columns([4, 1])
    top[0].subheader(stage["title"])
    top[1].metric("Time remaining", f"{stage['remaining_seconds']//60}:{stage['remaining_seconds']%60:02d}")
    st.progress(stage["progress"], text="11–26 August 2026 · two-minute warning-workflow presentation")
    st.caption("Exploratory warning workflow · catalog context and scenario alerts are distinct from confirmed later records.")
    cutoff = stage["date"] + "T08:30:00+05:45"
    geography, _ = load_nepal_geography()
    layers = [layer for layer in geography_layers(geography) if layer.group != "Infrastructure"]
    inventory, _ = get_lakes()
    if inventory:
        layers.append(MapLayer(id="judge_lakes", name="Glacier-lake inventory baselines", group="Geographic context", kind="context",
                               color=TOKENS["map_water"], data=inventory["layers"]["s2"], opacity=.65,
                               source="GLO · CC BY 4.0", timestamp="2017–2024 merged annual outlines",
                               limitation="Inventory context; no measured 2026 filling curve or identified source lake."))
    layers.append(MapLayer(id="judge_catalog", name="Catalog acquisitions before selected date", group="Geographic context", kind="context",
                           color=TOKENS["text_muted"], data=acquisition_footprints(root, cutoff), fill=False, visible=False,
                           source="Copernicus L1C catalog", limitation="Catalog footprints; imagery and historical availability remain unverified."))
    image_context = demo["images"]
    if image_context and phase in {"glacier", "lake", "water", "evaluation", "complete"}:
        pair_name = "post_event" if stage["evaluation_revealed"] else "pre_event"
        layers[0].extra["review"] = {
            "pair": image_context["pairs"][pair_name], "auto_open": phase in {"glacier", "lake", "evaluation"},
            "title": "Post-flood satellite images arrived" if stage["evaluation_revealed"] else "Actual pre-flood imagery · inspect visual changes",
            "note": "Real acquisition images. Red highlights appearance differences; clouds, snow and shadows can cause changes. Historical availability is unverified; no glacier displacement or lake level is measured."}
    focus = None
    if phase == "glacier":
        focus = {"coordinates": [28.43,85.55], "title": "Mountain-region imagery review",
                 "description": "Exploratory review area; inspect the real images. No measured glacier movement is available.", "zoom": 11}
    elif phase == "seismic":
        focus = {"coordinates": [28.27,85.38], "title": "Exploratory seismic watch",
                 "description": "Animated regional review focus, not an observed earthquake epicentre or waveform.", "zoom": 11}
    elif phase == "lake" and inventory and inventory["layers"]["s2"]["features"]:
        from shapely.geometry import Point, shape
        lake = min(inventory["layers"]["s2"]["features"], key=lambda f: shape(f["geometry"]).distance(Point(85.55,28.43)))
        point = shape(lake["geometry"]).representative_point()
        focus = {"coordinates": [point.y,point.x], "title": "Mapped glacial-lake baseline",
                 "description": "Inventory location for review; not an identified flood source. Current level and depth remain unmeasured.", "zoom": 12}
    elif phase == "gauge":
        focus = {"coordinates": [28.17065,85.342554], "title": "Syabrubesi gauge · data-gap review",
                 "description": "Exploratory communications alert. Focus locks briefly while the operator checks the station.",
                 "last_reading": "Last verified pre-cutoff raw sample: unavailable in the supplied historical logs.", "zoom": 14, "lock_seconds": 6}
    elif stage["evaluation_revealed"]:
        event = json.loads((root / "sources/usgs_us7000tbwb.json").read_text())
        x,y,*_ = event["geometry"]["coordinates"]
        focus = {"coordinates": [y,x], "title": "Reviewed landslide location · later evidence",
                 "description": "USGS origin 08:37:10 NPT, after the frozen 08:30 cutoff. This is not an earlier earthquake precursor.", "zoom": 10}
    if focus:
        layers[0].extra["focus"] = focus
    scenario = demo["scenario"]
    if stage["show_exploratory_assets"]:
        for key, label, color in [("baseline", "Assumed river channel", TOKENS["map_water"]),
                                  ("overflow", "Exploratory overflow corridor", TOKENS["danger"])]:
            layers.append(MapLayer(id="judge_" + key, name=label, group="Scenarios", kind="scenario", color=color,
                                   data=scenario[key], opacity=.5, source="OSM + explicit 25 m channel / 100 m margin",
                                   limitation="Assumed lateral buffer; no calibrated flow, width or arrival prediction."))
        for category, data in scenario["intersections"].items():
            layers.append(MapLayer(id="judge_" + category, name=category.title() + " for scenario review", group="Scenarios", kind="scenario",
                                   color=TOKENS["map_exposure"], data=data, fill=category not in {"roads", "bridges"},
                                   source="Computed mapped OSM intersections", limitation="Scenario review, not observed damage."))
    left, right = st.columns([2.3, 1])
    with left:
        bounds = CORRIDOR_VIEW if phase in {"assets", "evaluation", "complete"} else [[28.05, 85.1], [28.7, 85.8]]
        render_map_workspace(layers, height=440, initial_bounds=bounds)
        st.caption("Blue: mapped rivers/lake baselines. Red appears with the exploratory office warning. Amber: mapped assets for review.")
    with right, st.container(border=True):
        st.markdown("**Rasuwa office · evidence desk**")
        st.write("Replay date: " + stage["date"])
        st.caption("Frozen forecast cutoff: 26 Aug · 08:30 NPT")
        if phase == "history":
            row = next(r for r in demo["history"] if r["date"] == stage["date"])
            st.metric("Catalog acquisitions that day", row["optical_catalog_records"])
            st.caption("Acquisition metadata, not downloaded images or verified historical availability.")
            st.write("Lake level / depth: unmeasured")
        elif phase == "glacier":
            st.info("Real pre-flood images · visual review")
            st.write("The before/current popup highlights appearance differences in actual satellite previews.")
            st.caption("Clouds can obscure the mountains. Glacier displacement has not been measured.")
        elif phase == "lake":
            st.warning("Exploratory alert · lake surface-change signal")
            st.write("Office action: inspect cloud-valid imagery and request elevation evidence.")
            st.caption("No actual August filling measurements are available in this replay.")
        elif phase == "seismic":
            st.warning("Exploratory alert · regional seismic disturbance")
            st.write("Office action: check the event type and upstream conditions.")
            st.caption("This assumed disturbance does not establish an earthquake-triggered glacier burst.")
        elif phase == "gauge":
            st.warning("Exploratory alert · gauge data unavailable")
            st.write("Office action: verify station communications and consult downstream stations.")
            st.caption("Assumed early data gap; not a verified pre-collapse station outage.")
        elif phase == "water":
            st.warning("Exploratory alert · river level above usual")
            st.write("Office action: check the station datum, timestamp and provider thresholds.")
            st.caption("No historical level is invented for this stage.")
        elif phase in {"office", "assets"}:
            st.error("Exploratory office warning · review downstream assets")
            exposure = scenario["exposure"]
            st.metric("Building footprints for review", exposure["building_footprints_intersecting"])
            st.metric("Mapped roads for review", f"{exposure['road_length_m_intersecting']/1000:.2f} km")
            st.write(f"{exposure['bridge_tagged_features_intersecting']} bridge-tagged features")
            st.caption("Explicit scenario intersections through Chitwan; not confirmed affected assets or a flood arrival forecast.")
        else:
            st.info("Later evaluation reveal · confirmed report chronology")
            st.write("08:37:10 NPT · reviewed USGS event: landslide")
            st.write("08:40 · last Syabrubesi reading reported")
            st.write("08:50 · outage reported after source failure")
            st.caption("Later reports were excluded from frozen inputs. No single trigger is confirmed.")
        st.metric("Verified eligible frozen inputs", demo["snapshot"]["eligible_input_count"])
    if phase == "complete":
        st.success("Two-minute replay complete · warning workflow demonstrated; forecast accuracy remains unavailable.")
        verify_snapshot(root, demo["identity"])
        report = {k: v for k, v in demo.items() if k not in {"started_monotonic", "history", "scenario", "images"}}
        report["imagery_sources"] = {name: {"before_scene": pair["before"]["scene_id"],
            "current_scene": pair["current"]["scene_id"], "before_sha256": pair["before"]["preview_sha256"],
            "current_sha256": pair["current"]["preview_sha256"], "historical_eligibility": False}
            for name, pair in image_context["pairs"].items() if name != "current"}
        st.download_button("Download two-minute replay record", json.dumps(report, indent=2), "judge_replay.json", "application/json")
        if st.button("Open scenario in Rasuwa Operations", key="pv_judge_open_operations"):
            st.session_state["pv_operations_scenario"] = scenario
            st.switch_page("views/nepal_monitor.py")
        st.subheader("Post-event data comparison")
        data_type = st.selectbox("Post-event data type", ["Satellite image comparison", "Reported gauge chronology"])
        if st.button("Run post-event comparison", key="pv_judge_evaluate"):
            if data_type == "Satellite image comparison" and image_context:
                st.session_state["pv_judge_evaluation_result"] = {"data_type": data_type,
                    "pair": image_context["pairs"]["post_event"]["summary"],
                    "forecast_accuracy": None, "status": "actual_visual_comparison",
                    "limitation": "Preview appearance differs; cloud-limited optical imagery cannot score a missing downstream forecast or confirm structural damage."}
            elif data_type == "Reported gauge chronology":
                st.session_state["pv_judge_evaluation_result"] = {"data_type": data_type,
                    "reported_last_reading_npt": "2026-08-26T08:40:00+05:45", "reported_outage_npt": "2026-08-26T08:50:00+05:45",
                    "raw_historical_samples": "unavailable", "forecast_accuracy": None,
                    "limitation": "These report times follow source failure and were not frozen pre-cutoff inputs."}
            (root / "experiments" / demo["identity"] / "visual_evaluation.json").write_text(
                json.dumps(st.session_state["pv_judge_evaluation_result"], indent=2) + "\n")
        if st.session_state.get("pv_judge_evaluation_result"):
            result = st.session_state["pv_judge_evaluation_result"]
            if result["data_type"] == "Satellite image comparison":
                summary = result["pair"]
                metrics = st.columns(3)
                fraction = summary["changed_preview_fraction"]
                metrics[0].metric("Preview appearance changed", f"{fraction:.1%}" if fraction is not None else "Unavailable")
                metrics[1].metric("Comparable preview pixels", f"{summary['common_preview_pixels']:,}")
                metrics[2].metric("Scene cloud cover · maximum", f"{summary['maximum_scene_cloud_percent']:.1f}%")
                st.warning("Analyst review required · these are visible image differences; clouds can obscure the river and mountains.")
            else:
                st.write("Reported chronology: last Syabrubesi reading at 08:40 NPT; outage at 08:50 NPT.")
                st.caption("Raw historical reading values are unavailable in the prepared sources.")
            st.caption(result["limitation"])
            st.download_button("Download comparison result", json.dumps(result, indent=2), "rasuwa_post_event_comparison.json", "application/json")
            with st.expander("Comparison evidence details"):
                st.json(result)
    if st.button("Return to manual replay", key="pv_judge_return"):
        st.session_state.pop("pv_judge_demo", None)
        st.rerun()


render_judge_fragment = st.fragment(run_every="1s")(render_judge_demo)
