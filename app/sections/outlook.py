"""River Outlook: DHM observed levels beside GloFAS discharge and rainfall forecasts.

Displays third-party forecasts with provenance (pravah.forecast.outlook). PRAVAH has
no forecast model of its own yet; nothing here is an official warning.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from app.components.context_layers import with_downloaded_context
from app.components.empty_state import render_empty_state
from app.components.layout import render_callout, render_section_header
from app.components.map_view import MapLayer, render_map_workspace
from app.components.metric_card import render_metric_card
from app.context import AppContext, render_page_frame
from app.theme import TOKENS

LABELS = {"very_high": "Very high for the month", "high": "High for the month", "normal": "Normal", "unknown": "Unknown"}
COLORS = {"very_high": TOKENS["danger"], "high": TOKENS["warning"], "normal": TOKENS["map_water"], "unknown": TOKENS["text_muted"]}


@st.cache_data(ttl=3600, show_spinner="Fetching GloFAS and rainfall forecasts…")
def _outlook(cells_text: str, gauges: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    from pravah.forecast.outlook import fetch_outlook

    try:
        return fetch_outlook(json.loads(cells_text), gauges), None
    except Exception as exc:  # noqa: BLE001 - shown to the user, never hidden
        return None, f"{type(exc).__name__}: {exc}"[:300]


@st.cache_data(ttl=600, show_spinner=False)
def _gauges() -> dict[str, dict[str, Any]]:
    from app.nepal_data import request_monitor

    state, _ = request_monitor("GET", "/api/v1/nepal/live", timeout=5)
    rows = ((state or {}).get("sources", {}).get("gauges", {}) or {}).get("records", [])
    return {str(r["id"]): r for r in rows}


def _chart(s: dict[str, Any], today: str) -> alt.LayerChart:
    df = pd.DataFrame(s["series"])
    df["date"] = pd.to_datetime(df["date"])
    base = alt.Chart(df).encode(x=alt.X("date:T", title=None))
    band = base.mark_area(opacity=0.18, color=TOKENS["map_water"]).encode(
        y=alt.Y("river_discharge_min:Q", title="Discharge (m³/s)"), y2="river_discharge_max:Q",
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("river_discharge_min:Q", title="ensemble min", format=".0f"),
                 alt.Tooltip("river_discharge_max:Q", title="ensemble max", format=".0f")])
    line = base.mark_line(color=TOKENS["map_water"], strokeWidth=2).encode(
        y="river_discharge:Q", tooltip=[alt.Tooltip("date:T"), alt.Tooltip("river_discharge:Q", title="GloFAS", format=".0f")])
    layers = [band, line]
    rules = [(s.get("month_p90_m3s"), "Month p90", TOKENS["warning"]), (s.get("month_p99_m3s"), "Month p99", TOKENS["danger"])]
    for value, label, color in rules:
        if value is not None:
            layers.append(alt.Chart(pd.DataFrame({"y": [value], "label": [label]})).mark_rule(strokeDash=[4, 4], color=color)
                          .encode(y="y:Q", tooltip=["label:N", alt.Tooltip("y:Q", format=".0f")]))
    layers.append(alt.Chart(pd.DataFrame({"x": [pd.Timestamp(today)]})).mark_rule(color=TOKENS["text_muted"]).encode(x="x:T"))
    return alt.layer(*layers).properties(height=280)


def render_outlook_section(ctx: AppContext) -> None:
    render_page_frame(ctx, "River Outlook", "River Outlook",
                      "Observed DHM river levels beside GloFAS 7-day discharge and rainfall forecasts.", location=False)
    render_callout(
        "These are <b>third-party model forecasts</b> (Copernicus GloFAS via Open-Meteo), shown next to DHM "
        "observations. PRAVAH does not yet have a validated forecast model. GloFAS covers rainfall- and "
        "snowmelt-driven flow only — glacial-lake outbursts and landslide-dam failures are not forecast. "
        "Not an official warning: DHM issues flood warnings in Nepal.",
        title="Forecast display, not a PRAVAH prediction", variant="warning", as_html=False)
    cells = Path(os.environ.get("PRAVAH_FORECAST_CELLS", "data/processed/forecast/glofas_cells.json"))
    if not cells.exists():
        render_empty_state("Forecast stations not mapped", "Map the corridor gauges to GloFAS cells once.",
                           action_command="python scripts/build_forecast_cells.py", expected_files=[str(cells)])
        return
    outlook, error = _outlook(cells.read_text(), _gauges())
    if error or not outlook:
        render_callout(f"Forecast sources unavailable: {error}", title="Could not fetch forecasts", variant="error")
        return
    stations = outlook["stations"]
    flagged = [s for s in stations if s["outlook"] in {"high", "very_high"}]
    rain = [s["rain_next_72h_mm"] for s in stations if s["rain_next_72h_mm"] is not None]
    gauges_live = [s for s in stations if s["dhm_quality"] == "current"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Main-river stations", len(stations), subtext="Bhote Koshi → Trishuli → Narayani")
    with c2:
        render_metric_card("High for the month (7 days)", len(flagged), subtext="GloFAS peak ≥ month p90, 2016–2025",
                           badge="Watch" if flagged else "None", badge_variant="warning" if flagged else "success")
    with c3:
        render_metric_card("DHM gauges reporting", f"{len(gauges_live)}/{len(stations)}",
                           subtext="Rasuwa gauges offline → forecast only")
    with c4:
        render_metric_card("Max rain next 72 h", max(rain) if rain else None, unit="mm", subtext="Open-Meteo at each gauge")

    render_section_header("Stations", f"Forecast issued for {outlook['issued_for_date_utc']} (UTC) · upstream first")
    table = pd.DataFrame([{
        "Station": s["station_name"],
        "DHM level (m)": s["dhm_level_m"], "DHM warning (m)": s["dhm_warning_level_m"],
        "DHM status": s["dhm_quality"] or "not reporting",
        "GloFAS today (m³/s)": s["discharge_today_m3s"], "7-day peak (m³/s)": s["peak_7d_m3s"],
        "Peak date": s["peak_7d_date"], "Month p90 (m³/s)": round(s["month_p90_m3s"]) if s["month_p90_m3s"] else None,
        "Outlook": LABELS[s["outlook"]], "Ensemble worst case": LABELS[s["outlook_ensemble_max"]],
        "Rain next 72 h (mm)": s["rain_next_72h_mm"],
    } for s in stations])
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("DHM level is river stage on each station's own datum (m); GloFAS is modelled discharge (m³/s). "
               "They are different quantities and are not converted into each other.")

    names = [s["station_name"] for s in stations]
    pick = st.selectbox("Station forecast", names, index=len(names) - 1, key="pv_outlook_station")
    s = stations[names.index(pick)]
    st.altair_chart(_chart(s, outlook["issued_for_date_utc"]), use_container_width=True)
    cell = s["cell"]
    st.caption(f"Line: GloFAS control forecast · band: ensemble min–max · dashed: same-month p90/p99 of this cell, "
               f"2016–2025 · grey: today. GloFAS cell {cell['cell_latitude']:.3f}, {cell['cell_longitude']:.3f} "
               f"({cell['cell_distance_km']} km from the gauge, mean {cell['cell_mean_m3s']:.0f} m³/s).")

    render_section_header("Map", "Stations coloured by 7-day outlook")
    layers = []
    for key in ("very_high", "high", "normal", "unknown"):
        pts = [s for s in stations if s["outlook"] == key]
        layers.append(MapLayer(
            id=f"outlook_{key}", name=f"Outlook: {LABELS[key]}", group="Prediction", kind="model", color=COLORS[key],
            data={"type": "FeatureCollection", "features": [
                {"type": "Feature", "properties": {"label": f"{p['station_name']}: {LABELS[p['outlook']]}"},
                 "geometry": {"type": "Point", "coordinates": [p["gauge_longitude"], p["gauge_latitude"]]}} for p in pts]},
            opacity=0.9, source="GloFAS via Open-Meteo Flood API · DHM station table",
            timestamp=outlook["sources"]["glofas"]["retrieved_at_utc"], extra={"radius": 7},
            description="7-day GloFAS peak against the same calendar month, 2016–2025 (model climatology).",
            limitation="Model climatology, not observed return periods; not an official warning.",
            unavailable_reason="No station in this class"))
    render_map_workspace(with_downloaded_context(layers), height=520, mode="full",
                         initial_bounds=[[27.6, 84.2], [28.4, 85.6]])

    with st.expander("Sources, checksums and limitations"):
        for name, prov in outlook["sources"].items():
            st.markdown(f"**{name}** · `{prov['url']}` · retrieved {prov['retrieved_at_utc']} · "
                        f"sha256 `{prov['sha256'][:16]}…` · {prov['bytes']} bytes")
        for line in outlook["limitations"]:
            st.markdown(f"- {line}")
