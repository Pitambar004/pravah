"""Rasuwa operator workspace: actual sources and watch-gated mapped assets."""

from dataclasses import replace

import streamlit as st

from app.components.layout import render_badge, render_page_header, render_topbar
from app.components.map_view import MapLayer, render_map_workspace
from app.image_data import get_image_pairs
from app.nepal_data import (
    geography_layers,
    load_national_context,
    load_nepal_geography,
    request_monitor,
)
from app.sections.lakes import get_lakes
from app.sections.operations_data import (
    RASUWA_IDS,
    npt_time,
    render_data_views,
    render_summary_metrics,
    watch_reasons,
)
from app.theme import TOKENS

NEPAL_VIEW = [[26.3, 80.05], [30.5, 88.25]]
CORRIDOR_VIEW = [[27.30, 83.85], [28.5, 85.7]]
RASUWA_VIEW = [[28.05, 85.10], [28.70, 85.80]]


def gauge_layer(state, regional=True):
    source = state.get("sources", {}).get("gauges", {})
    features = []
    for row in source.get("records", []):
        if regional and row["id"] not in RASUWA_IDS:
            continue
        if row.get("latitude") is None or row.get("longitude") is None:
            continue
        features.append({"type": "Feature", "properties": {k: v for k, v in row.items() if k not in {"latitude", "longitude"}},
                         "geometry": {"type": "Point", "coordinates": [row["longitude"], row["latitude"]]}})
    return MapLayer(id="gauges" if regional else "gauges_national", name="Corridor river stations" if regional else "All Nepal river stations", group="Geographic context", kind="observed", visible=regional,
                    color=TOKENS["success"], data={"type": "FeatureCollection", "features": features} if features else None,
                    extra={"legend_label": "River stations", "radius": 5},
                    source="DHM river-watch", timestamp=source.get("retrieved_at_utc", "—"),
                    description="Public station records; inspect reading time and quality.",
                    limitation="Missing readings do not prove station failure. Values use each station’s own datum.")


def operations_layers(document, national, inventory, state, scenario=None):
    context = geography_layers(document)
    layers = [layer for layer in context if layer.group != "Infrastructure"]
    if national:
        metadata = national["metadata"]
        layers.insert(0, MapLayer(id="national_rivers", name="Major rivers of Nepal", group="Geographic context", kind="context",
                                 color=TOKENS["map_water"], data=national["layer"], fill=False, visible=True,
                                 source="OpenStreetMap contributors · ODbL 1.0", timestamp=metadata["retrieved_at_utc"],
                                 description=metadata["selection"], limitation=metadata["limitation"]))
    if inventory:
        layers.append(MapLayer(id="lake_baselines", name="Upstream glacier-lake baselines", group="Geographic context", kind="context",
                               color=TOKENS["map_water"], data=inventory["layers"]["s2"], visible=True,
                               source="GLO / Rawlins et al. (2025) · CC BY 4.0", timestamp="Merged annual 2017–2024 maximum outlines",
                               limitation="Inventory baseline; not a current water level or an identified source lake."))
    layers.append(gauge_layer(state))
    layers.append(gauge_layer(state, regional=False))
    for layer in layers:
        if layer.color == TOKENS["map_water"]:
            layer.extra["legend_label"] = "Mapped rivers & lake baselines"
        if layer.id == "national_rivers":
            layer.extra["weight"] = 2.5
    reasons = watch_reasons(state)
    water = state.get("sources", {}).get("water_change", {})
    if water.get("new_water"):
        layers.append(MapLayer(id="new_nepal_water", name="Newly detected water", group="Change vs pre-event", kind="model",
                               color=TOKENS["danger"], data=water["new_water"], source="Compared Nepal optical water masks",
                               limitation="Model-derived change in comparable cloud-valid pixels; not water depth or forecast overflow."))
    if reasons:
        # These signals have no calibrated flood footprint; show context for review.
        layers.extend(replace(layer, visible=True, color=TOKENS["map_exposure"],
                              name=layer.name + " · watch review", limitation="Mapped corridor context for review; a station/rainfall watch does not establish which assets are affected.")
                      for layer in context if layer.group == "Infrastructure")
    if scenario and scenario.get("kind") == "exploratory_scenario":
        for key, label, color in [("baseline", "Assumed channel corridor", TOKENS["map_water"]),
                                  ("overflow", "Exploratory overflow corridor", TOKENS["danger"])]:
            layers.append(MapLayer(id="assumed_" + key, name=label, group="Scenarios", kind="scenario", color=color,
                                   data=scenario[key], opacity=.4, source="Explicit analyst-selected metric buffer", limitation=scenario["limitation"]))
        for category, data in scenario["intersections"].items():
            layers.append(MapLayer(id="scenario_" + category, name=category.title() + " in assumed corridor", group="Scenarios", kind="scenario",
                                   color=TOKENS["map_exposure"], data=data, visible=True, fill=category not in {"roads", "bridges"},
                                   source="Computed OSM geometry intersections", limitation="Scenario intersections, not observed damage or a validated forecast."))
    return layers, reasons


@st.fragment(run_every="30s")
def render_summary_panel():
    state, _ = request_monitor("GET", "/api/v1/nepal/live", timeout=3)
    if state:
        render_summary_metrics(state)


@st.fragment(run_every="30s")
def render_live_panel():
    state, error = request_monitor("GET", "/api/v1/nepal/live", timeout=3)
    if error:
        st.warning("Live monitoring API unavailable. " + error)
        st.caption("The mapped context and saved replay remain usable.")
        return
    st.subheader("Monitoring data")
    cols = st.columns([3, 1])
    cols[0].caption("Sources polled every 5 minutes · panel refreshed every 30 seconds · last source check: " + npt_time(state.get("updated_at_utc")))
    if cols[1].button("Refresh live sources", key="pv_refresh_nepal"):
        with st.spinner("Checking public sources…"):
            _, error = request_monitor("POST", "/api/v1/nepal/live/refresh", timeout=55)
        if error:
            st.warning(error)
        else:
            st.rerun()
    render_data_views(state)
    with st.expander("Source health & provenance"):
        for key, source in state.get("sources", {}).items():
            st.write(key.replace("_", " ").title() + " · " + source.get("status", "pending") + (" · stale cache" if source.get("cache_stale") else ""))
            st.caption("Retrieved: " + npt_time(source.get("retrieved_at_utc")))
            st.caption(source.get("scope") or source.get("reason") or source.get("semantics", ""))
            if source.get("error"):
                st.warning(source["error"])
            if source.get("source_uri"):
                st.link_button("Open " + key.replace("_", " ") + " provider", source["source_uri"])
        st.caption("Water comparison: " + state.get("sources", {}).get("water_change", {}).get("reason", "Comparable Nepal water masks required."))


@st.fragment(run_every="300s")
def render_nepal_map():
    document, error = load_nepal_geography()
    national, national_error = load_national_context()
    inventory, _ = get_lakes()
    state, _ = request_monitor("GET", "/api/v1/nepal/live", timeout=3)
    scenario = st.session_state.get("pv_operations_scenario")
    layers, reasons = operations_layers(document, national, inventory, state or {}, scenario)
    image_context, _ = get_image_pairs()
    if image_context and layers:
        layers[0].extra["review"] = {"pair": image_context["pairs"]["current"], "title": "Latest satellite image comparison", "auto_open": False}
    if reasons:
        st.warning("Active Rasuwa source watch · " + " · ".join(reasons))
        st.caption("Highlighted infrastructure is mapped context for operator review. No calibrated affected-area forecast is available.")
    else:
        st.caption("No active Rasuwa threshold watch. Check source health and reading quality below.")
    if scenario:
        st.warning("Exploratory watch from Rasuwa replay · assumed corridor and intersecting assets shown separately from live evidence.")
        if st.button("Clear exploratory watch", key="pv_clear_operations_scenario"):
            st.session_state.pop("pv_operations_scenario", None)
            st.rerun(scope="fragment")
    focus = st.radio("Map view", ["Nepal", "Rasuwa office area", "Downstream through Chitwan"], horizontal=True, key="pv_nepal_view")
    bounds = {"Nepal": NEPAL_VIEW, "Rasuwa office area": RASUWA_VIEW, "Downstream through Chitwan": CORRIDOR_VIEW}[focus]
    render_map_workspace(layers, height=580, initial_bounds=bounds)
    with st.expander("Map sources & interpretation"):
        st.caption("Blue: mapped rivers and lake baselines. Red: model water change, or the explicitly labelled exploratory overflow corridor. Infrastructure appears during a source watch or opened replay scenario.")
        st.caption("Infrastructure uses mapped OpenStreetMap geometry. Complete image-based detection of houses, roads and bridges is not trained in this prototype.")
    if error or national_error:
        st.caption("Some mapped context is unavailable: " + str(error or national_error))


def render_nepal_monitor(ctx):
    render_topbar(["PRAVAH", "Rasuwa Operations"], [render_badge("Rasuwa office view", "outline"), render_badge("Experimental monitoring", "warning")])
    render_page_header("Rasuwa Hydropower & River Watch", "Operator workspace for river gauges, upstream lakes, latest satellite imagery and downstream asset review.")
    render_summary_panel()
    render_nepal_map()
    render_live_panel()
