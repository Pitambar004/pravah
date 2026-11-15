"""Rasuwa evidence replay and separately recorded exploratory exposure scenarios."""

import json
import os
import uuid
from datetime import date as date_type
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.layout import render_badge, render_page_header, render_topbar
from app.components.map_view import MapLayer, render_map_workspace
from app.components.timeline import NPT
from app.nepal_data import geography_layers, load_nepal_geography
from app.sections.judge_demo import (
    prepare_judge_demo,
    render_judge_demo,
    render_judge_fragment,
)
from app.sections.lakes import get_lakes
from app.sections.nepal_monitor import CORRIDOR_VIEW
from app.theme import TOKENS
from pravah.configuration import configured_url
from pravah.replay.cascade import acquisition_footprints, daily_history
from pravah.replay.experiment import audit, freeze, register_evaluation, verify_snapshot
from pravah.replay.scenario import explore


def render_simulate(ctx):
    render_topbar(["PRAVAH", "Rasuwa Replay"], [render_badge("Rasuwa, Nepal", "outline"), render_badge("Experimental replay", "warning")])
    render_page_header("Simulate Rasuwa flood", "Reconstruct the evidence sequence, explore mapped downstream exposure, then reveal later records.")
    st.caption("Historical case: 26 August 2026 · Bhote Koshi–Trishuli–Narayani. The July 2025 event is separate historical context.")
    root = Path(os.environ.get("PRAVAH_REPLAY_ROOT", "outputs/nepal_replay/prepared"))
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except (OSError, ValueError):
        st.info("Prepared Nepal evidence is unavailable. Run the preparation command in the handoff.")
        return
    demo = st.session_state.get("pv_judge_demo")
    if demo:
        (render_judge_demo if demo["finished"] else render_judge_fragment)(root, manifest)
        return
    if st.button("Start 2-minute judges demo", type="primary", key="pv_start_judge_demo"):
        try:
            with st.spinner("Preparing the frozen replay and mapped scenario…"):
                st.session_state["pv_judge_demo"] = prepare_judge_demo(root, manifest)
            st.rerun()
        except (OSError, ValueError, KeyError) as exc:
            st.error("Judge replay preparation failed: " + str(exc))
    st.caption("One click · 2 minutes · automatic satellite-image popups, animated map alerts, gauge focus and downstream infrastructure review.")
    st.info("The replay opens real before/current images during glacier and lake review, zooms to the gauge during its data-gap alert, then reveals post-event imagery for a separate comparison.")
    with st.expander("Manual evidence replay and scenario controls", expanded=False):
        render_manual_replay(root, manifest)


def render_manual_replay(root, manifest):
    """Detailed evidence controls, separate from the automatic judge sequence."""
    geography, geo_error = load_nepal_geography()
    stages = [(date_type(2026, 8, 26) - timedelta(days=d)).isoformat() for d in range(15, -1, -1)]
    stage = st.select_slider("15-day replay stage · Nepal date", options=stages, value=stages[0], key="pv_sim_stage")
    stage_changed = st.session_state.get("pv_sim_stage_applied") != stage
    if stage_changed:
        st.session_state["pv_sim_date"] = date_type.fromisoformat(stage)
        st.session_state["pv_sim_stage_applied"] = stage
    date_col, clock_col, mode_col = st.columns([1, 1, 2])
    date = date_col.date_input("Cutoff date · Nepal time", value=date_type.fromisoformat(stage), key="pv_sim_date")
    clock = clock_col.time_input("Cutoff time · Nepal time", value=time(8, 30), key="pv_sim_clock")
    scenario_enabled = mode_col.toggle("Include exploratory scenario", value=True, key="pv_sim_scenario_enabled",
                                       help="Assumed uniform river corridor, separate from measured event observations.")
    cutoff = datetime.combine(date, clock, tzinfo=NPT).isoformat()
    try:
        report = audit(manifest, cutoff, root)
    except (ValueError, KeyError) as exc:
        st.error("Evidence audit failed: " + str(exc))
        return
    st.info("Historical inputs require both acquisition and verified availability before the cutoff. Current downloads and later reports cannot establish a pre-event forecast.")
    st.caption("Replay starts 15 days before the event, on 11 August. Move the stage to inspect dated catalog coverage; image footprints do not measure glacier movement or lake filling.")
    controls = None
    half_width, margin = 25, 100
    if scenario_enabled:
        with st.expander("Exploratory scenario assumptions", expanded=False):
            st.caption("These controls describe assumed conditions. They do not change the real evidence record or train a forecast model.")
            widths = st.columns(2)
            half_width = widths[0].slider("Assumed channel half-width (m)", 5, 100, 25, 5, key="pv_sim_width")
            margin = widths[1].slider("Additional lateral margin (m)", 0, 250, 100, 10, key="pv_sim_margin")
            signals = st.columns(3)
            controls = {
                "gauge_unavailable": signals[0].checkbox("Assume gauge unavailable", value=True, key="pv_sim_gauge"),
                "high_water": signals[1].checkbox("Assume unusually high water", value=True, key="pv_sim_water"),
                "seismic_disturbance": signals[2].checkbox("Assume seismic disturbance", value=False, key="pv_sim_seismic"),
            }
            controls["rapid_lake_expansion"] = st.checkbox("Assume rapidly expanding lake", key="pv_sim_lake_expansion")
            controls["icefall_into_temporary_lake"] = st.checkbox("Assume icefall into a temporary lake", key="pv_sim_icefall")
            controls["temporary_lake_breach"] = st.checkbox("Assume temporary-lake breach", key="pv_sim_lake_breach")
            st.caption("Assumed signals prompt an evidence-review watch. They do not determine the width, flow, probability or arrival time; the margin above is selected explicitly.")
    if st.button("Run Rasuwa replay", key="pv_sim_run"):
        identity = "rasuwa_demo_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        try:
            snapshot = freeze(root, manifest, cutoff, identity)
            scenario = None
            if scenario_enabled and geography:
                scenario = explore(geography["layers"], half_width, margin)
                scenario["assumed_signals"] = controls
                scenario["geography_snapshot"] = geography["metadata"].get("retrieved_at_utc")
                scenario["geography_source"] = geography["metadata"].get("source")
                scenario["attribution"] = "OpenStreetMap contributors · ODbL 1.0 · " + configured_url("PRAVAH_OSM_ATTRIBUTION_URL")
                scenario["replay_window"] = {"start_date": stages[0], "cutoff": cutoff,
                                              "measurement_history": "Retrospective records remain behind separate evaluation reveal"}
                # Keep assumptions in their own artifact, never snapshot.prediction.
                (root / "experiments" / identity / "exploratory_scenario.json").write_text(json.dumps(scenario, indent=2) + "\n")
            st.session_state["pv_sim_result"] = {"identity": identity, "scenario": scenario,
                                                 "scenario_requested": scenario_enabled}
            st.session_state["pv_sim_revealed"] = False
        except (OSError, ValueError, KeyError) as exc:
            st.error("Replay could not be prepared: " + str(exc))
    result = st.session_state.get("pv_sim_result")
    scenario = result.get("scenario") if result else None
    layers = geography_layers(geography)
    inventory, _ = get_lakes()
    if inventory:
        layers.append(MapLayer(id="lake_baselines", name="Glacier-lake baselines (2017–2024)", group="Geographic context", kind="context",
                               color=TOKENS["map_water"], data=inventory["layers"]["s2"], visible=True,
                               source="Glacial Lake Observatory, Rawlins et al. (2025), CC BY 4.0", timestamp="Merged annual 2017–2024 maximum outlines",
                               limitation="Baseline inventory, not 2026 lake filling or identified source lake."))
    layers.append(MapLayer(id="catalog_footprints", name="Acquired satellite catalog footprints", group="Geographic context", kind="context",
                           color=TOKENS["text_muted"], fill=False, visible=False, dash="4 4", data=acquisition_footprints(root, cutoff),
                           source="Copernicus Sentinel-2 L1C catalog", timestamp=cutoff,
                           limitation="Acquisition metadata only; not image pixels, slope deformation or verified pre-cutoff availability."))
    if scenario:
        layers.extend([
            MapLayer(id="assumed_channel", name="Assumed channel corridor", group="Scenarios", kind="scenario", color=TOKENS["map_water"],
                     data=scenario["baseline"], opacity=.35, source="OSM centrelines + analyst-selected metric buffer", timestamp=scenario["geography_snapshot"],
                     description="Assumed channel half-width; independent of water observations.", limitation=scenario["limitation"]),
            MapLayer(id="assumed_overflow", name="Exploratory overflow corridor", group="Scenarios", kind="scenario", color=TOKENS["danger"],
                     data=scenario["overflow"], opacity=.55, source="Uniform lateral buffer in EPSG:32645", timestamp=scenario["geography_snapshot"],
                     description="Extra assumed corridor outside the assumed channel.", limitation=scenario["limitation"]),
        ])
        for category, data in scenario["intersections"].items():
            layers.append(MapLayer(id="scenario_" + category, name=category.title() + " in assumed corridor", group="Scenarios", kind="scenario",
                                   color=TOKENS["map_exposure"], data=data, visible=True, fill=category not in {"roads", "bridges"},
                                   source="Computed OSM geometry intersections", limitation="Potential scenario exposure, not observed damage."))
    focus = st.radio("Scenario map extent", ["Entire corridor through Chitwan", "Rasuwa headwaters detail"], horizontal=True)
    bounds = CORRIDOR_VIEW if focus == "Entire corridor through Chitwan" else [[28.10, 85.25], [28.55, 85.65]]
    render_map_workspace(layers, height=650, initial_bounds=bounds)
    if geo_error:
        st.warning("Nepal river geometry unavailable: " + geo_error)
    st.caption("Blue: mapped rivers and assumed channel if enabled. Red: additional assumed overflow corridor. The uniform buffer does not reproduce valley hydraulics or measured inundation.")
    st.page_link("views/lakes.py", label="Inspect glacier-lake inventory and area comparison", icon=":material/landscape:")
    if not result:
        st.caption("Select assumptions and run the replay to freeze inputs and calculate mapped scenario exposure.")
        return
    identity = result["identity"]
    try:
        snapshot = verify_snapshot(root, identity)
    except (OSError, ValueError, KeyError) as exc:
        st.error("Frozen input verification failed: " + str(exc))
        return
    st.success("Frozen replay: " + identity)
    st.caption("Recorded cutoff: " + snapshot["cutoff_at_utc"] + ". Changes to controls take effect in a new replay.")
    metrics = st.columns(4)
    metrics[0].metric("Eligible pre-cutoff inputs", snapshot["eligible_input_count"])
    exposure = (scenario or {}).get("exposure", {})
    metrics[1].metric("Buildings in assumed corridor", exposure.get("building_footprints_intersecting", "—"))
    road_length = exposure.get("road_length_m_intersecting")
    metrics[2].metric("Roads in assumed corridor", f"{road_length/1000:.2f} km" if road_length is not None else "—")
    metrics[3].metric("Forecast accuracy", "Unavailable")
    if result["scenario_requested"] and scenario is None:
        st.warning("Scenario was requested but mapped Nepal geometry is missing; no exposure was calculated.")
    if scenario:
        st.caption(f"Computed against mapped vectors: {exposure.get('bridge_tagged_features_intersecting')} bridge-tagged features, "
                   f"{exposure.get('facilities_intersecting')} facilities, {exposure.get('settlements_intersecting')} settlement points. "
                   "Building footprints do not establish household counts, residents, damage or complete coverage.")
        if any((scenario.get("assumed_signals") or {}).values()):
            st.warning("Exploratory watch: verify upstream conditions and review mapped downstream assets. Assumed signals are not measured historical warnings.")
        if st.button("Open exploratory watch in Operations", key="pv_open_operations_scenario"):
            st.session_state["pv_operations_scenario"] = scenario
            st.switch_page("views/nepal_monitor.py")
        st.download_button("Download exploratory scenario", json.dumps(scenario, indent=2), "exploratory_scenario.json", "application/json")
    st.download_button("Download frozen evidence", json.dumps(snapshot, indent=2), "frozen_evidence.json", "application/json")
    st.info("No calibrated downstream forecast was produced. Post-event accuracy remains unavailable until a genuine forecast and comparable independent observations exist.")
    if st.button("Reveal post-event evidence", key="pv_sim_reveal"):
        try:
            register_evaluation(root, manifest, identity)
            st.session_state["pv_sim_revealed"] = True
        except FileExistsError:
            st.session_state["pv_sim_revealed"] = True
        except (OSError, ValueError, KeyError) as exc:
            st.error("Evaluation reveal failed: " + str(exc))
    if st.session_state.get("pv_sim_revealed"):
        evaluation = json.loads((root / "experiments" / identity / "evaluation.json").read_text())
        st.subheader("Later-reported event chronology")
        st.warning("Retrospective reconstruction. These later records were excluded from the frozen pre-event inputs.")
        st.write("08:37:10 NPT · The currently reviewed USGS event is classified as a landslide. It does not establish an earlier earthquake precursor.")
        st.write("08:40 NPT · Last Syabrubesi reading reported in the technical report. Raw historical station samples are still unavailable here.")
        st.write("08:50 NPT · Reported station outage follows the source failure. It can be considered for downstream warning research, not as an earlier collapse predictor.")
        st.link_button("USGS reviewed event", configured_url("PRAVAH_USGS_EVENT_URL", id="us7000tbwb"))
        st.link_button("Hydrology technical report", configured_url("PRAVAH_DHM_EVENT_REPORT_URL"))
        st.subheader("15-day source reconstruction")
        history = daily_history(root)
        st.dataframe([{k: r[k] for k in ["date", "days_before_event", "rainfall_mm_retrospective_reanalysis", "optical_catalog_records", "imagery_status",
                                         "lake_surface_area_m2", "water_level_m", "daily_slope_displacement_m"]} for r in history["rows"]], hide_index=True)
        st.caption(history["limit"])
        st.write("ICIMOD’s October synthesis reports pre-collapse slope deformation measured with Sentinel-1 InSAR over 7 July–24 August. It is later evidence; we do not have a daily deformation series for these 15 days.")
        st.write("The synthesis identifies a rock/ice collapse rather than a conventional glacial-lake outburst, and no single trigger is confirmed. The seismic record is not proof that an earthquake caused a glacier burst.")
        st.link_button("ICIMOD later synthesis and monitoring findings", configured_url("PRAVAH_ICIMOD_FINDINGS_URL"))
        weather_path = root / "sources/august_era5_rasuwa.json"
        if weather_path.is_file():
            d = json.loads(weather_path.read_text())
            st.subheader("Retrospective rainfall context")
            chart = pd.DataFrame({"Date": d["daily"]["time"], "Rainfall (mm)": d["daily"]["precipitation_sum"]}).set_index("Date")
            st.bar_chart(chart)
            st.caption("ERA5 daily reanalysis from Open-Meteo, retrieved after the event. Not a contemporaneous gauge or pre-cutoff forecast.")
        catalog_path = root / "sources/rasuwa_l1c_catalog.json"
        if catalog_path.is_file():
            catalog = json.loads(catalog_path.read_text())
            with st.expander("Pre/post satellite catalog"):
                st.dataframe([{"Scene": i["id"], "Acquired UTC": i["properties"].get("datetime"),
                               "Cloud %": i["properties"].get("eo:cloud_cover"), "Status": "Catalog only; image not downloaded"}
                              for i in catalog["items"]], hide_index=True)
        for item in evaluation["references"]:
            st.caption("Evaluation reference: " + item["asset_id"])
        st.info("Mapped post-event water width, damage counts and forecast overlap scores cannot yet be evaluated: matched Nepal images and calibrated predictions are missing.")
        st.download_button("Download separate evaluation", json.dumps(evaluation, indent=2), "evaluation.json", "application/json")
    with st.expander("Excluded evidence"):
        for item in report["excluded"]:
            st.caption(item["asset_id"] + " · " + item["reason"].replace("_", " "))
