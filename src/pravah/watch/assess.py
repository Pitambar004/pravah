"""Warning-sign indicators and the combined watch level — pure functions over one snapshot.

Each indicator: family, status (ok | watch | warning | severe | unknown), value, threshold,
evidence time and source. ``unknown`` means missing evidence and is never read as "safe".
Combined level = worst indicator status, raised one step when ≥ 3 *independent families*
are at watch or above simultaneously (compound-hazard rule, as in Rasuwa 2025/2026 where
rain, upstream cryosphere and gauge behaviour coincided). Analyst rules, not validated.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from pravah.watch.config import FAMILIES, RULES, UPSTREAM_GAUGES

ORDER = ["ok", "watch", "warning", "severe"]
RASUWA = (28.17, 85.33)


def _t(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")) if s else None


def _km(a, b):
    p = math.pi / 180
    h = math.sin((b[0] - a[0]) * p / 2) ** 2 + math.cos(a[0] * p) * math.cos(b[0] * p) * math.sin((b[1] - a[1]) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def _band(value, rule):
    if value is None:
        return "unknown"
    for level in ("severe", "warning", "watch"):
        if level in rule and value >= rule[level]:
            return level
    return "ok"


def _ind(id_, family, label, status, value=None, units="", threshold=None, observed_at=None, source="", detail=""):
    return {"id": id_, "family": family, "family_label": FAMILIES[family], "label": label, "status": status,
            "value": value, "units": units, "threshold": threshold, "observed_at_utc": observed_at,
            "source": source, "detail": detail}


def rain(snapshot):
    pts = snapshot.get("rain", {}).get("points") or []
    out = []
    v72 = [p.get("last72_mm") for p in pts if p.get("last72_mm") is not None]
    mean72 = sum(v72) / len(v72) if v72 else None
    out.append(_ind("rain_72h", "rain", "Catchment rain, last 72 h", _band(mean72, RULES["rain_72h_mm"]),
                    None if mean72 is None else round(mean72, 1), "mm (mean of points)", RULES["rain_72h_mm"],
                    snapshot.get("rain", {}).get("observed_at_utc"), snapshot.get("rain", {}).get("source", "")))
    n24 = [(p.get("next24_mm"), p["name"]) for p in pts if p.get("next24_mm") is not None]
    top = max(n24) if n24 else (None, None)
    out.append(_ind("rain_24h_forecast", "rain", "Heaviest forecast rain, next 24 h", _band(top[0], RULES["rain_24h_max_mm"]),
                    None if top[0] is None else round(top[0], 1), "mm", RULES["rain_24h_max_mm"],
                    snapshot.get("rain", {}).get("observed_at_utc"), snapshot.get("rain", {}).get("source", ""),
                    detail=f"at {top[1]}" if top[1] else "No forecast available (replay: no archived forecast)"))
    return out


def seismic(snapshot):
    now = _t(snapshot["now"])
    events = snapshot.get("seismic")
    if events is None:
        return [_ind("seismic", "seismic", "Earthquakes near Rasuwa", "unknown", source="USGS")]
    near = []
    for e in events:
        t = _t(e.get("observed_at_utc"))
        if t is None or e.get("magnitude") is None or t > now:
            continue
        if (now - t).total_seconds() <= RULES["seismic_window_h"] * 3600 and \
                _km(RASUWA, (e["latitude"], e["longitude"])) <= RULES["seismic_radius_km"]:
            near.append(e)
    top = max(near, key=lambda e: e["magnitude"], default=None)
    return [_ind("seismic", "seismic", f"Strongest quake ≤{RULES['seismic_radius_km']:.0f} km, last 72 h",
                 _band(top["magnitude"] if top else 0.0, RULES["seismic_mag"]), top["magnitude"] if top else None, "M",
                 RULES["seismic_mag"], top.get("observed_at_utc") if top else None, "USGS",
                 detail=(top.get("name") or "") + " — can destabilise moraine dams and slopes" if top else f"{len(events)} events in region, none near")]


def gauges(snapshot):
    now = _t(snapshot["now"])
    rows = snapshot.get("gauges")
    if rows is None:
        return [_ind("gauge_level", "gauge", "Main-stem gauge level", "unknown", source="DHM",
                     detail="No provider readings for this time (DHM serves ~2.5 weeks of history)")]
    out, worst, worst_row = [], "ok", None
    rises, silent = [], []
    for r in rows:
        level, warn, danger = r.get("level_m"), r.get("warning_level_m"), r.get("danger_level_m")
        if r.get("quality") == "current" and level is not None:
            s = "severe" if danger is not None and level >= danger else "warning" if warn is not None and level >= warn else "ok"
            if ORDER.index(s) > ORDER.index(worst):
                worst, worst_row = s, r
        hist = [(_t(h["observed_at_utc"]), h["level_m"]) for h in r.get("history", []) if h.get("level_m") is not None]
        hist = [(t, v) for t, v in hist if t and t <= now]
        if len(hist) >= 2:
            t_last, v_last = hist[-1]
            prior = [v for t, v in hist if 0 < (t_last - t).total_seconds() <= 3600]
            if prior and v_last - min(prior) >= RULES["gauge_rise_m_per_h"]["warning"]:
                rises.append((v_last - min(prior), r))
        last = _t(r.get("observed_at_utc"))
        if r["id"] in UPSTREAM_GAUGES and last and now - last <= __import__("datetime").timedelta(seconds=RULES["gauge_recent_s"]) \
                and (now - last).total_seconds() > RULES["gauge_silent_after_s"]:
            silent.append(r)
    out.append(_ind("gauge_level", "gauge", "Main-stem gauge vs DHM warning/danger level", worst,
                    worst_row.get("level_m") if worst_row else None, "m (station datum)", "provider levels",
                    worst_row.get("observed_at_utc") if worst_row else None, "DHM",
                    detail=worst_row.get("name", "") if worst_row else f"{len(rows)} stations reporting"))
    top = max(rises, key=lambda x: x[0], default=None)
    out.append(_ind("gauge_rise", "gauge", "Fastest rise in the last hour", "warning" if top else "ok",
                    round(top[0], 2) if top else None, "m / h", RULES["gauge_rise_m_per_h"],
                    top[1].get("observed_at_utc") if top else None, "DHM", detail=top[1].get("name", "") if top else ""))
    out.append(_ind("gauge_silent", "gauge", "Upstream gauge stopped reporting", "warning" if silent else "ok",
                    len(silent), "stations", f"silent > {RULES['gauge_silent_after_s'] // 60} min after reporting",
                    None, "DHM", detail=", ".join(s.get("name", s["id"]) for s in silent) +
                    (" — outage and flood damage cannot be distinguished remotely" if silent else "")))
    return out


def lakes(snapshot):
    out = []
    for sensor, label in (("s2", "optical"), ("s1", "SAR")):
        rows = (snapshot.get("lakes") or {}).get(sensor)
        if not rows:
            out.append(_ind(f"lake_change_{sensor}", "cryosphere", f"Glacial-lake area change ({label})", "unknown",
                            source="Sentinel-2" if sensor == "s2" else "Sentinel-1",
                            detail="No valid recent pass or baseline"))
            continue
        worst, top = "ok", None
        for r in rows:
            a, b = r.get("area_m2"), r.get("baseline_m2")
            if a is None or not b or b < RULES["lake_min_baseline_km2"] * 1e6:
                continue
            frac, change_km2 = (a - b) / b, abs(a - b) / 1e6
            s = "ok"
            if change_km2 >= RULES["lake_min_change_km2"]:
                s = _band(frac, RULES["lake_growth_fraction"])
                if frac <= -RULES["lake_drain_fraction"]:
                    s = "warning"   # sudden drainage: possible outburst already under way
            if ORDER.index(s) > ORDER.index(worst) or (s == worst and top and abs(frac) > abs(top[1])):
                worst, top = s, (r, frac)
        r, frac = top if top else (None, None)
        out.append(_ind(f"lake_change_{sensor}", "cryosphere", f"Glacial-lake area change ({label})", worst,
                        None if frac is None else round(100 * frac, 1), "% vs baseline", RULES["lake_growth_fraction"],
                        r.get("acquired_at_utc") if r else None, "Sentinel-2 L2A" if sensor == "s2" else "Sentinel-1 RTC",
                        detail=(f"lake {r['lake_id']}: {r['area_m2'] / 1e6:.3f} km² vs {r['baseline_m2'] / 1e6:.3f} km²") if r else ""))
    return out


def glacier(snapshot):
    rows = [r for r in snapshot.get("glacier") or [] if r.get("median_speed_m_per_day") is not None and r.get("baseline_m_per_day")]
    if not rows:
        return [_ind("glacier_speed", "cryosphere", "Glacier surface speed-up", "unknown", source="Sentinel-2 feature tracking",
                     detail="Needs two clear passes 5–40 days apart and a baseline")]
    top = max(rows, key=lambda r: r["median_speed_m_per_day"] / r["baseline_m_per_day"])
    ratio = top["median_speed_m_per_day"] / top["baseline_m_per_day"]
    return [_ind("glacier_speed", "cryosphere", "Glacier surface speed-up", _band(ratio, RULES["glacier_speedup_ratio"]),
                 round(ratio, 2), "× baseline", RULES["glacier_speedup_ratio"], top.get("acquired_at_utc"),
                 "Sentinel-2 B08 feature tracking (GPU)", detail=f"zone {top['zone_id']}: {top['median_speed_m_per_day']:.2f} m/day")]


def river(snapshot):
    out = []
    rows = [r for r in snapshot.get("widths") or [] if r.get("width_m") and r.get("expected_m")]
    if rows:
        top = max(rows, key=lambda r: r["width_m"] / r["expected_m"])
        ratio = top["width_m"] / top["expected_m"]
        out.append(_ind("width_anomaly", "river", "Observed river width vs expected", _band(ratio, RULES["width_anomaly_ratio"]),
                        round(ratio, 2), "× expected", RULES["width_anomaly_ratio"], top.get("acquired_at_utc"), "Sentinel-2 L2A",
                        detail=f"reach {top['reach_id']}: {top['width_m']:.0f} m vs {top['expected_m']:.0f} m"))
    else:
        out.append(_ind("width_anomaly", "river", "Observed river width vs expected", "unknown", source="Sentinel-2 L2A",
                        detail="No clear recent pass or no fitted width model"))
    q = snapshot.get("discharge") or []
    worst, top = "unknown" if not q else "ok", None
    for r in q:
        v, p90, p99 = r.get("q_m3s"), r.get("p90_m3s"), r.get("p99_m3s")
        if v is None or p90 is None:
            continue
        s = "warning" if p99 is not None and v >= p99 else "watch" if v >= p90 else "ok"
        if ORDER.index(s) > ORDER.index(worst if worst != "unknown" else "ok") or top is None:
            worst, top = s, r
    out.append(_ind("discharge", "river", "Forecast discharge vs monthly climate", worst,
                    round(top["q_m3s"], 1) if top else None, "m³/s", "monthly p90 / p99 (GloFAS 2016–2025)",
                    top.get("day") if top else None, top.get("source", "") if top else "",
                    detail=f"{top['station_name']} ({top['source']})" if top else "No discharge estimate"))
    c = snapshot.get("sar_change")
    out.append(_ind("sar_new_water", "river", "New SAR water in corridor",
                    _band(c["new_water_km2"], RULES["sar_new_water_km2"]) if c else "unknown",
                    round(c["new_water_km2"], 3) if c else None, "km²", RULES["sar_new_water_km2"],
                    c.get("acquired_at_utc") if c else None, "Sentinel-1 RTC",
                    detail="Same-orbit pass pair" if c else "Needs two passes of the same orbit"))
    return out


def assess(snapshot: dict) -> dict:
    indicators = rain(snapshot) + seismic(snapshot) + gauges(snapshot) + lakes(snapshot) + glacier(snapshot) + river(snapshot)
    known = [i for i in indicators if i["status"] != "unknown"]
    worst = max((ORDER.index(i["status"]) for i in known), default=0)
    fams = sorted({i["family"] for i in known if ORDER.index(i["status"]) >= 1})
    level = min(3, worst + 1) if len(fams) >= 3 else worst
    return {"level": ORDER[level], "level_index": level, "families_elevated": fams,
            "indicators": indicators, "unknown": [i["id"] for i in indicators if i["status"] == "unknown"],
            "assessed_at_utc": snapshot["now"], "rule": __doc__.strip().split("\n\n")[1]}


def now_iso():
    return datetime.now(timezone.utc).isoformat()
