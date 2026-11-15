"""Live Watch 24/7 — the Rasuwa compound-hazard watch and the Aug-2026 replay.

Everything is computed by the GB10 API service on Fuzzball; this page only renders what
``/api/v1/watch/*`` returns. Live mode refreshes every 30 s. Replay mode plays the hourly
frames of 11–28 Aug 2026 at a chosen speed. Analyst watch rules, not official warnings.
"""
from __future__ import annotations

from datetime import datetime
from html import escape

import pandas as pd
import streamlit as st

from app import api_client
from app.components.layout import html, render_badge, render_page_header, render_topbar
from app.components.map_view import MapLayer, render_map_workspace
from app.components.timeline import NPT
from app.theme import TOKENS

STATUS_COLOR = {"ok": TOKENS["success"], "watch": TOKENS["warning"], "warning": "#FB923C",
                "severe": TOKENS["danger"], "unknown": TOKENS["map_unavailable"]}
STATUS_VARIANT = {"ok": "success", "watch": "warning", "warning": "warning", "severe": "danger", "unknown": "neutral"}
LEVEL_TEXT = {"ok": "Normal", "watch": "Watch", "warning": "Warning", "severe": "Severe"}
VIEW = [[27.70, 84.35], [28.80, 85.85]]


def _get(path, timeout=8.0):
    return api_client._request("GET", path, timeout=timeout)


def _npt(iso):
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone(NPT).strftime("%d %b %H:%M NPT")
    except (TypeError, ValueError):
        return "—"


def _level_banner(level: str, subtitle: str, families: list[str]):
    color = STATUS_COLOR.get(level, TOKENS["map_unavailable"])
    fam = ", ".join(families) if families else "none"
    html(f'<div style="border:1px solid {color};border-left:6px solid {color};border-radius:12px;padding:14px 18px;'
         f'margin:4px 0 14px;background:rgba(17,27,42,.6)"><div style="font-size:12px;color:{TOKENS.get("text_muted", "#94A3B8")}">'
         f'Combined watch level</div><div style="font-size:28px;font-weight:600;color:{color}">{escape(LEVEL_TEXT.get(level, level))}</div>'
         f'<div style="font-size:13px">{escape(subtitle)} · elevated families: {escape(fam)}</div></div>')


def _indicators(assessment: dict):
    rows = assessment.get("indicators", [])
    fams: dict = {}
    for i in rows:
        fams.setdefault(i["family_label"], []).append(i)
    cols = st.columns(len(fams) or 1)
    for col, (fam, items) in zip(cols, fams.items()):
        with col:
            html(f'<div style="font-size:12px;font-weight:600;margin-bottom:6px">{escape(fam)}</div>')
            for i in items:
                c = STATUS_COLOR[i["status"]]
                val = "—" if i["value"] is None else f'{i["value"]} {i["units"]}'
                html(f'<div style="border:1px solid #23324a;border-left:4px solid {c};border-radius:8px;padding:7px 9px;margin-bottom:6px">'
                     f'<div style="font-size:11.5px">{escape(i["label"])}</div>'
                     f'<div style="font-size:15px;font-weight:600;color:{c}">{escape(i["status"].upper())} · {escape(str(val))}</div>'
                     f'<div style="font-size:10.5px;color:#94A3B8">{escape(i.get("detail") or "")}'
                     f'{" · " + escape(_npt(i["observed_at_utc"])) if i.get("observed_at_utc") else ""}</div></div>')


def _exposure_tiles(down: dict | None):
    if not down:
        st.info("Downstream hydraulics not run yet (needs discharge from GloFAS or the emulator).")
        return
    e = down.get("exposure", {})
    labels = [("buildings", "Buildings in wet extent"), ("roads", "Road length (km)"), ("bridges", "Bridges"),
              ("facilities", "Facilities"), ("settlements", "Settlement points")]
    cols = st.columns(len(labels))
    for col, (k, lab) in zip(cols, labels):
        v = e.get(k)
        if v:
            col.metric(lab, f'{v["median"]:.0f}' if k != "roads" else f'{v["median"]:.1f}',
                       f'max {v["max"]:.0f}' if k != "roads" else f'max {v["max"]:.1f}', delta_color="off")
        else:
            col.metric(lab, "—")
    g = down.get("glof")
    note = (f'GLOF scenario active from lake {g["lake_id"]} (peak {g["peak_m3s_low"]:.0f}–{g["peak_m3s_high"]:.0f} m³/s, '
            f'empirical envelope). ' if g else "")
    st.caption(f'{note}{down["member_count"]} ensemble members · hydraulics {down["hydraulics_seconds_measured"]} s on '
               f'{down["device"]} (measured) · median and max across members. Screening model, not a 2-D simulation.')


def _reach_table(down: dict | None):
    if not down:
        return
    rows = []
    for r in down["reaches"]:
        rows.append({"Reach": f'{r["reach_id"]} {r["label"]}', "km": f'{r["start_km"]:.0f}–{r["end_km"]:.0f}',
                     "Q m³/s (min/med/max)": " / ".join(f"{v:.0f}" for v in r["discharge_m3s"]),
                     "Width hydraulic m": f'{r["hydraulic_width_m"][1]:.0f} ({r["hydraulic_width_m"][0]:.0f}–{r["hydraulic_width_m"][2]:.0f})',
                     "Width empirical m": "—" if r["empirical_width_m"] is None else f'{r["empirical_width_m"]:.0f}',
                     "Methods agree": {True: "yes", False: "NO", None: "—"}[r["width_methods_agree"]],
                     "Depth m": f'{r["depth_m"][1]:.1f} ({r["depth_m"][0]:.1f}–{r["depth_m"][2]:.1f})',
                     "Arrival h": "—" if not r["arrival_h"] else f'{r["arrival_h"][0]:.1f}–{r["arrival_h"][2]:.1f}',
                     "Bldgs max": down["exposure"].get("buildings", {}).get("per_reach_max", {}).get(r["reach_id"], 0)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _map(geo: dict | None, state: dict, height=560):
    layers = []
    if geo:
        layers.append(MapLayer(id="stem", name="Bhote Koshi–Trishuli main stem", group="River model", kind="context",
                               color=TOKENS["map_water"], data=geo["main_stem"], fill=False, source="OSM + Copernicus DEM"))
        layers.append(MapLayer(id="stations", name="Main-stem gauges", group="River model", kind="observed", color=TOKENS["success"],
                               data={"type": "FeatureCollection", "features": [
                                   {"type": "Feature", "properties": {"name": s["station_name"], "chainage_km": round(s["chainage_m"] / 1000, 1)},
                                    "geometry": {"type": "Point", "coordinates": [s["gauge_longitude"], s["gauge_latitude"]]}}
                                   for s in geo["stations"]]}, extra={"radius": 6}, source="DHM"))
    down = state.get("downstream") or {}
    for key, name, color, vis in (("extent_max", "Wet extent — max member", TOKENS["danger"], True),
                                  ("extent_median", "Wet extent — median member", TOKENS["map_scenario"], True)):
        if down.get(key):
            layers.append(MapLayer(id=key, name=name, group="Downstream prediction", kind="prediction", color=color, visible=vis,
                                   data={"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"member": key},
                                                                                     "geometry": down[key]}]},
                                   limitation="Manning normal depth on 30 m DEM sections; screening only"))
    if down.get("exposed_points_max"):
        layers.append(MapLayer(id="exposed", name="Assets in max wet extent", group="Downstream prediction", kind="derived",
                               color=TOKENS["map_exposure"], data=down["exposed_points_max"], extra={"radius": 3}, source="OSM"))
    ev = state.get("evidence") or state.get("assessment", {}).get("evidence") or {}
    quakes = [q for q in (ev.get("seismic") or []) if q.get("magnitude")]
    if quakes:
        layers.append(MapLayer(id="quakes", name="Earthquakes (USGS)", group="Live evidence", kind="observed", color=TOKENS["danger"],
                               data={"type": "FeatureCollection", "features": [
                                   {"type": "Feature", "properties": {"M": q["magnitude"], "place": q.get("name"), "time": q["observed_at_utc"]},
                                    "geometry": {"type": "Point", "coordinates": [q["longitude"], q["latitude"]]}} for q in quakes]},
                               extra={"radius": 7}))
    lakes = (ev.get("lakes") or {}).get("s2") or []
    feats = []
    for r in lakes:
        try:
            _, lon, lat = r["lake_id"].split("_")[:3]
        except ValueError:
            continue
        ch = None if not (r.get("area_m2") and r.get("baseline_m2")) else round(100 * (r["area_m2"] / r["baseline_m2"] - 1), 1)
        feats.append({"type": "Feature", "properties": {"lake": r["lake_id"], "change_%": ch},
                      "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]}})
    if feats:
        layers.append(MapLayer(id="lakes", name="Watched glacial lakes (S2)", group="Live evidence", kind="observed",
                               color=TOKENS["accent"], data={"type": "FeatureCollection", "features": feats}, extra={"radius": 5}))
    render_map_workspace(layers, height=height, mode="full", initial_bounds=VIEW)


def _live_body():
    state, err = _get("/api/v1/watch/state")
    if err or not state:
        st.warning(f"Live state unavailable: {err or 'no data yet'}. The GB10 service fills it within a few minutes of starting.")
        return
    a = state["assessment"]
    _level_banner(a["level"], f'Updated {_npt(state["updated_at_utc"])} · tick {state.get("tick_seconds_measured")} s', a["families_elevated"])
    if state.get("source_errors"):
        st.caption("Source problems (last good evidence kept): " + "; ".join(f"{k}: {v[:80]}" for k, v in state["source_errors"].items()))
    _indicators(a)
    geo, _ = _get("/api/v1/watch/geography")
    _map(geo, state)
    st.subheader("Downstream river response")
    _exposure_tiles(state.get("downstream"))
    _reach_table(state.get("downstream"))
    passes = (state.get("evidence") or {}).get("latest_passes") or []
    if passes:
        st.caption("Latest satellite passes: " + " · ".join(f'{p["collection"]} {_npt(p["acquired_at_utc"])} (cloud {p.get("cloud_cover_percent") or "—"}%)' for p in passes))
    tl, _ = _get("/api/v1/watch/timeline?limit=600")
    if tl and tl.get("rows"):
        df = pd.DataFrame([{"time": r["t"], "level": ["ok", "watch", "warning", "severe"].index(r["level"])} for r in tl["rows"]])
        df["time"] = pd.to_datetime(df["time"])
        st.caption("Watch level history (0 normal · 1 watch · 2 warning · 3 severe)")
        st.line_chart(df.set_index("time"), height=140)


def _replay_body():
    meta, err = _get("/api/v1/watch/replay", timeout=15)
    if err or not meta:
        st.warning(f"Replay unavailable: {err}")
        return
    if meta.get("status") != "ready":
        done, total = meta.get("done") or 0, meta.get("total")
        st.info(f"Replay frames are being computed on the GB10: {meta['status']} {done}/{total or '?'} {meta.get('error') or ''}")
        return
    n = meta["frame_count"]
    ss = st.session_state
    ss.setdefault("pv_watch_frame", 0)
    if ss.get("pv_watch_playing"):
        ss["pv_watch_frame"] = min(n - 1, ss["pv_watch_frame"] + ss.get("pv_watch_step", 3))
        if ss["pv_watch_frame"] >= n - 1:
            ss["pv_watch_playing"] = False
    i = ss["pv_watch_frame"]
    frame, ferr = _get(f"/api/v1/watch/replay/frames/{i}", timeout=15)
    if ferr:
        st.error(ferr)
        return
    levels = meta["levels"]
    first_alarm = next((k for k, lv in enumerate(levels) if lv in ("warning", "severe")), None)
    a = frame["assessment"]
    _level_banner(a["level"], f'Replay clock {_npt(frame["clock_utc"])} · frame {i + 1}/{n}', a["families_elevated"])
    st.progress((i + 1) / n, text=("First warning at frame " + str(first_alarm + 1) + f' ({_npt(meta["start_utc"])} + {first_alarm} h)'
                                    if first_alarm is not None else "No warning-level frame in this replay"))
    if frame.get("new_imagery"):
        st.success("New satellite pass at this hour: " + ", ".join(p["pass_id"] for p in frame["new_imagery"]))
    _indicators(a)
    _map(None, frame, height=500)
    _exposure_tiles(frame.get("downstream"))
    _reach_table(frame.get("downstream"))


def render_watch_section(ctx=None):
    render_topbar(["PRAVAH", "Live Watch"], [render_badge("Rasuwa · Bhote Koshi–Trishuli", "outline"),
                                              render_badge("GB10 · Fuzzball", "accent"), render_badge("Analyst rules, not official warnings", "warning")])
    render_page_header("Live Watch 24/7", "Rain, earthquakes, gauges, glacial lakes, glacier motion and satellite water, "
                                          "combined into one watch level with downstream width, depth, arrival and exposure.")
    mode = st.segmented_control("Mode", ["Live 24/7", "Rasuwa 26 Aug 2026 replay"], default="Live 24/7", key="pv_watch_mode")
    if mode == "Live 24/7":
        st.fragment(run_every=30)(_live_body)()
        return
    c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
    if c1.button("▶ Play", key="pv_watch_play", type="primary"):
        st.session_state["pv_watch_playing"] = True
    if c2.button("⏸ Pause", key="pv_watch_pause"):
        st.session_state["pv_watch_playing"] = False
    st.session_state["pv_watch_step"] = c3.select_slider("Speed (replay hours per tick)", [1, 2, 3, 6, 12], value=3, key="pv_watch_speed")
    if c4.button("⏮ Restart", key="pv_watch_restart"):
        st.session_state["pv_watch_frame"] = 0
    st.caption("Each frame sees only evidence acquired up to its clock. ERA5 rainfall and real satellite acquisition times; "
               "provider availability was not verified, so this is a reconstruction, not a cutoff-eligible forecast.")
    st.fragment(run_every=1.5 if st.session_state.get("pv_watch_playing") else None)(_replay_body)()
