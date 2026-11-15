"""Build one assessment snapshot from observations available at a clock time.

Shared by the live engine (clock = now) and the replay (clock = simulated time). Every
selector filters on ``acquired/observed ≤ clock`` so the replay cannot see its future.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np

from pravah.watch import assess, hydro


def _t(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def rain_from_hourly(hourly: dict, clock: datetime, forecast: bool) -> dict:
    times = [_t(t + "+00:00" if len(t) <= 16 else t) for t in hourly["times"]]
    P = np.array([[np.nan if v is None else v for v in r] for r in hourly["precip"]], float)
    past = np.array([(clock - timedelta(hours=72)) < t <= clock for t in times])
    nxt = np.array([clock < t <= clock + timedelta(hours=24) for t in times])
    pts = []
    for j, name in enumerate(hourly["points"]):
        last72 = float(np.nansum(P[past, j])) if past.any() else None
        n24 = float(np.nansum(P[nxt, j])) if forecast and nxt.sum() >= 20 else None
        pts.append({"name": name, "last72_mm": last72, "next24_mm": n24})
    return {"points": pts, "observed_at_utc": clock.isoformat(), "source": hourly.get("source", "")}


def latest_lakes(passes: list[dict], baselines: dict, clock: datetime, sensor: str) -> list[dict]:
    latest: dict = {}
    for p in passes:
        t = _t(p["pass"]["acquired_at_utc"])
        if t > clock:
            continue
        for x in p.get("lakes", []):
            if x.get("area_m2") is not None:
                latest[x["lake_id"]] = {**x, "acquired_at_utc": p["pass"]["acquired_at_utc"]}
    base = baselines.get(sensor, {})
    return [{**v, "baseline_m2": base.get(k, {}).get("median_m2")} for k, v in latest.items()]


def latest_glacier(pairs: list[dict], baselines: dict, clock: datetime) -> list[dict]:
    seen = [p for p in pairs if _t(p["acquired_at_utc"]) <= clock]
    if not seen:
        return []
    last = seen[-1]
    return [{**z, "acquired_at_utc": last["acquired_at_utc"],
             "baseline_m_per_day": (baselines.get(z["zone_id"]) or {}).get("baseline_m_per_day")} for z in last["zones"]]


def latest_widths(passes: list[dict], fits: dict, reach_q: dict, clock: datetime) -> list[dict]:
    seen = [p for p in passes if _t(p["pass"]["acquired_at_utc"]) <= clock and p.get("reaches")]
    if not seen:
        return []
    last = seen[-1]
    out = []
    for w in last["reaches"]:
        fit = fits.get(w["reach_id"])
        q = reach_q.get(w["reach_id"])
        exp = float(hydro.empirical_width(fit, q)) if fit and fit.get("a") and q else None
        out.append({**w, "expected_m": exp, "acquired_at_utc": last["pass"]["acquired_at_utc"]})
    return out


def latest_before(rows: list[dict], clock: datetime, key="acquired_at_utc"):
    seen = [r for r in rows if _t(r[key]) <= clock]
    return seen[-1] if seen else None


def discharge_rows(station_q: dict, stations: list[dict], day: date, source: str) -> list[dict]:
    rows = []
    for s in stations:
        q = station_q.get(s["station_id"])
        clim = (s.get("monthly_climatology_m3s") or {}).get(f"{day.month:02d}", {})
        if q is None:
            continue
        rows.append({"station": s["station_id"], "station_name": s.get("station_name"), "q_m3s": float(q),
                     "p90_m3s": clim.get("p90"), "p99_m3s": clim.get("p99"), "day": day.isoformat(), "source": source})
    return rows


def lake_centroid(lake_id: str):
    """GLO ids encode the centroid: GLO_<lon>_<lat>."""
    try:
        _, lon, lat = lake_id.split("_")[:3]
        return float(lon), float(lat)
    except ValueError:
        return None


def glof_trigger(assessment: dict, lakes_rows: list[dict], river_model) -> dict | None:
    """When a lake indicator is at watch or above, size an outburst from the most-changed lake."""
    fired = [i for i in assessment["indicators"] if i["id"].startswith("lake_change") and i["status"] in ("watch", "warning", "severe")]
    if not fired:
        return None
    cands = [r for r in lakes_rows if r.get("area_m2") and r.get("baseline_m2")]
    if not cands:
        return None
    r = max(cands, key=lambda x: abs(x["area_m2"] - x["baseline_m2"]) / x["baseline_m2"])
    c = lake_centroid(r["lake_id"])
    g = hydro.glof_peak(max(r["area_m2"], r["baseline_m2"]))
    g.update({"lake_id": r["lake_id"], "origin_chainage_m": river_model.chainage_of(*c) if c else 0.0,
              "trigger": fired[0]["id"]})
    return g


def assess_snapshot(snap: dict) -> dict:
    return assess.assess(snap)
