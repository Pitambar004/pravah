"""Actual operator data views: readings, levels, scene age, weather and alerts."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from app.components.notifications import render_notifications
from app.image_data import get_image_pairs
from pravah.configuration import configured_url
from pravah.replay.live import age_seconds

RASUWA_IDS = {"4913", "191", "52", "5705"}
NPT = timezone(timedelta(hours=5, minutes=45))


def age_label(timestamp):
    try:
        date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if date.tzinfo is None:
            return "Time zone unavailable"
        hours = (datetime.now(timezone.utc) - date).total_seconds()/3600
        if hours < 0:
            return "Provider time is ahead of current time"
        return f"{hours*60:.0f} min old" if hours < 1 else f"{hours:.1f} hours old"
    except (AttributeError, ValueError):
        return "Age unavailable"


def npt_time(timestamp):
    try:
        date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if date.tzinfo is None:
            return "Time zone unavailable"
        return date.astimezone(NPT).strftime("%d %b %Y · %H:%M NPT")
    except (AttributeError, ValueError):
        return "Not supplied"


def watch_reasons(state):
    now = datetime.now(timezone.utc)
    reasons = []
    sources = state.get("sources", {})
    def current_source(key):
        source = sources.get(key, {})
        age = age_seconds(source.get("retrieved_at_utc"), now)
        return source.get("status") == "ok" and not source.get("cache_stale") and age is not None and 0 <= age <= 900
    gauges = sources.get("gauges", {}).get("records", []) if current_source("gauges") else []
    for row in gauges:
        age = age_seconds(row.get("observed_at_utc"), now)
        if row["id"] not in RASUWA_IDS or row.get("quality") != "current" or row.get("level_m") is None or age is None or not -300 <= age <= 3600:
            continue
        danger, warning = row.get("danger_level_m"), row.get("warning_level_m")
        if danger is not None and row["level_m"] >= danger:
            reasons.append(row["name"] + " exceeds the provider danger level")
        elif warning is not None and row["level_m"] >= warning:
            reasons.append(row["name"] + " exceeds the provider warning level")
        history = state.get("gauge_history", {}).get(row["id"], [])
        if len(history) >= 2 and history[-1]["observed_at_utc"] == row["observed_at_utc"] and history[-1]["level_m"] - history[-2]["level_m"] >= .5:
            reasons.append(row["name"] + " rose ≥0.5 m between received samples (analyst watch)")
    weather = sources.get("weather", {}).get("records", []) if current_source("weather") else []
    for row in weather:
        rain = row.get("forecast_rain_next_24h_mm")
        if row.get("name") in {"Rasuwa", "Upstream Bhote Koshi"} and rain is not None and rain >= 50:
            reasons.append(row["name"] + " exceeds the analyst rainfall watch")
    regional_rows = {row["id"]: row for row in gauges if row["id"] in RASUWA_IDS}
    for alert in state.get("alerts", []):
        age = age_seconds(alert.get("created_at_utc"), now)
        row = regional_rows.get(alert.get("record_id"))
        if alert.get("source") == "gauges" and row and row.get("quality") != "current" and "unavailable or stale" in alert.get("message", "") and age is not None and 0 <= age <= 900:
            reasons.append(row["name"] + " current reading became unavailable; verify the station")
    water = sources.get("water_change", {})
    if current_source("water_change") and any((r.get("new_water_area_km2") or 0) >= .01 for r in water.get("records", [])):
        reasons.append("Comparable Nepal water masks exceed the analyst new-water watch")
    return reasons


def render_summary_metrics(state):
    sources = state.get("sources", {})
    gauges = sources.get("gauges", {}).get("records", [])
    gauge_source = sources.get("gauges", {})
    regional = [g for g in gauges if g["id"] in RASUWA_IDS]
    image = sources.get("rasuwa_imagery", {})
    cols = st.columns(4)
    cols[0].metric("Corridor stations", len(regional), border=True)
    cols[1].metric("Current readings", sum(g.get("quality") == "current" for g in regional) if gauge_source.get("status") == "ok" and not gauge_source.get("cache_stale") else "Unavailable", border=True)
    cols[2].metric("Latest satellite image", age_label(image.get("acquired_at_utc")), border=True)
    cols[3].metric("Active source watches", len(watch_reasons(state)), border=True)


def render_data_views(state):
    sources = state.get("sources", {})
    gauges = sources.get("gauges", {}).get("records", [])
    gauge_source = sources.get("gauges", {})
    regional = [g for g in gauges if g["id"] in RASUWA_IDS]
    image = sources.get("rasuwa_imagery", {})
    tabs = st.tabs(["Gauge readings", "Water levels", "Satellite images", "Weather & seismic", "Alerts"])
    with tabs[0]:
        if gauge_source.get("status") != "ok" or gauge_source.get("cache_stale"):
            st.warning("Gauge source unavailable or stale. Retained readings below are cached context.")
        scope = st.radio("Station network", ["Rasuwa & downstream", "All Nepal"], horizontal=True, key="pv_station_network")
        rows = regional if scope == "Rasuwa & downstream" else gauges
        st.dataframe([{"Station": r["name"], "Reading (m)": r.get("level_m"), "Warning (m)": r.get("warning_level_m"),
                       "Danger (m)": r.get("danger_level_m"), "Observed": npt_time(r.get("observed_at_utc")),
                       "Age": age_label(r.get("observed_at_utc")), "Quality": r.get("quality"), "Trend": r.get("trend")}
                      for r in rows], hide_index=True, width="stretch", height=210 if scope == "Rasuwa & downstream" else 340,
                     column_config={"Reading (m)": st.column_config.NumberColumn(format="%.2f"),
                                    "Warning (m)": st.column_config.NumberColumn(format="%.2f"),
                                    "Danger (m)": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Metres relative to each provider station’s own datum. Empty readings stay unavailable. Data age and quality require checking before a watch decision.")
        if rows:
            st.link_button("DHM official river-watch", configured_url("PRAVAH_DHM_RIVER_WATCH_URL"))
    with tabs[1]:
        options = {g["id"]: g for g in regional}
        if not options:
            st.info("No Rasuwa station metadata has been received.")
        else:
            selected = st.selectbox("Water-level station", list(options), format_func=lambda key: options[key]["name"])
            row = options[selected]
            values = st.columns(3)
            values[0].metric("Latest provider level", f"{row['level_m']:.2f} m" if row.get("level_m") is not None else "Unavailable")
            values[1].metric("Provider warning level", f"{row['warning_level_m']:.2f} m" if row.get("warning_level_m") is not None else "Not supplied")
            values[2].metric("Provider danger level", f"{row['danger_level_m']:.2f} m" if row.get("danger_level_m") is not None else "Not supplied")
            st.caption(npt_time(row.get("observed_at_utc")) + " · " + age_label(row.get("observed_at_utc")) + " · " + row.get("quality", "unavailable"))
            history = state.get("gauge_history", {}).get(selected, [])
            if len(history) >= 2:
                chart = pd.DataFrame(history)
                chart["Observed"] = pd.to_datetime(chart["observed_at_utc"]).dt.tz_convert("Asia/Kathmandu")
                st.line_chart(chart.set_index("Observed")[["level_m"]], y_label="Provider level (m)")
            else:
                st.info("A trend needs at least two actual current readings. Monitoring history is collecting; missing points are not filled.")
            st.link_button("Open station source", row["source_uri"])
    with tabs[2]:
        st.subheader("Latest available Rasuwa satellite scene")
        st.caption("The last available image remains visible until a new acquisition is listed and its preview is obtainable. Its age is calculated from the actual acquisition time.")
        if image.get("preview_data_url"):
            st.image(image["preview_data_url"], caption="Copernicus Sentinel-2 L1C regional tile overview preview", width=520)
        else:
            st.info(image.get("reason", "Rasuwa scene preview pending. Catalog metadata can be inspected below."))
        st.write(image.get("scene_id", "No Rasuwa scene selected yet"))
        st.caption("Acquired: " + npt_time(image.get("acquired_at_utc")) + " · " + age_label(image.get("acquired_at_utc")))
        cloud = image.get("cloud_cover_percent")
        st.caption("Preview status: " + image.get("status", "pending") + " · Cloud cover: " + (f"{cloud:.1f}%" if cloud is not None else "not supplied"))
        if image.get("latest_catalog_scene_id") and image["latest_catalog_scene_id"] != image.get("scene_id"):
            st.info("A newer scene is listed but its preview is unavailable: " + image["latest_catalog_scene_id"] + " · " + npt_time(image.get("latest_catalog_acquired_at_utc")))
        if image.get("reason"):
            st.caption(image["reason"])
        st.caption("A downsampled provider preview is for inspection. It does not supply the full model bands or establish glacier movement, lake level, depth or flood extent.")
        pairs, _ = get_image_pairs()
        if pairs:
            pair = pairs["pairs"]["current"]
            columns = st.columns(2)
            for column, key, label in zip(columns, ["before", "current"], ["Earlier actual image", "Latest actual image"], strict=True):
                with column:
                    st.markdown("**"+label+"**")
                    st.image(pair[key]["preview_data_url"], width="stretch")
                    st.caption(npt_time(pair[key]["acquired_at_utc"]))
            st.caption(pair["summary"]["measurement_scope"])
        with st.expander("Recent catalog acquisitions across Nepal"):
            st.dataframe([{k: r.get(k) for k in ["id", "observed_at_utc", "cloud_cover_percent", "collection", "downloaded_imagery"]}
                          for r in sources.get("satellite", {}).get("records", [])], hide_index=True)
    with tabs[3]:
        st.subheader("Weather model estimates")
        st.dataframe([{k: r.get(k) for k in ["name", "forecast_rain_next_24h_mm", "current_precipitation_mm", "temperature_c", "observed_at_utc"]}
                      for r in sources.get("weather", {}).get("records", [])], hide_index=True)
        st.subheader("Regional seismic catalog")
        st.dataframe([{k: r.get(k) for k in ["name", "event_type", "magnitude", "depth_km", "observed_at_utc"]}
                      for r in sources.get("seismic", {}).get("records", [])], hide_index=True)
        st.caption("USGS event type is retained. A catalog event alone does not establish a flood trigger or an earthquake-caused ice collapse.")
    with tabs[4]:
        alerts = state.get("alerts", [])
        render_notifications(alerts)
        dismissed = st.session_state.setdefault("pv_alert_dismissed", [])
        active = [a for a in alerts if a["id"] not in dismissed]
        if active and st.button("Dismiss current alerts", key="pv_dismiss_nepal_alerts"):
            st.session_state["pv_alert_dismissed"] = [a["id"] for a in alerts]
            st.rerun(scope="fragment")
        if not active:
            st.info("No new source watch alerts. Missing measurements still need verification.")
        for alert in reversed(active[-10:]):
            st.warning(alert["name"] + " · " + alert["message"])
            st.caption(npt_time(alert.get("created_at_utc")) + " · Experimental source watch")
