"""River outlook: DHM observed levels beside third-party GloFAS and rainfall forecasts.

This module does not run a PRAVAH forecast model. It assembles, with provenance:

* DHM (Department of Hydrology and Meteorology, Nepal) observed water level and
  the provider's warning/danger levels, per station (provider datum, metres).
* GloFAS river-discharge forecasts (Copernicus Emergency Management Service,
  served by the Open-Meteo Flood API): daily m³/s, 7 days back and 7 ahead,
  with the ensemble min/p25/median/p75/max.
* Open-Meteo precipitation forecasts at each gauge (gridded model estimate, mm).

GloFAS is a ~5 km gridded model, so a gauge's nearest cell is often a small side
stream. For main-stem stations we search a 3×3 neighbourhood (±0.05°) and keep
the cell with the highest 2024 mean discharge, recorded as an approximation.
GloFAS discharge (m³/s) and DHM stage (m) are different quantities on different
datums and are never compared numerically. "Unusual" compares a forecast with the
same calendar month of the chosen cell's 2016–2025 GloFAS history (model
climatology, not observed return periods). None of this is an official warning.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from pravah.configuration import configured_url

# Main-stem Bhote Koshi → Trishuli → Narayani stations (DHM station ids). Tributary
# gauges are excluded because neighbourhood snapping would jump to the main river.
CORRIDOR_MAIN_STEM = ["4913", "191", "52", "5705", "4781", "265"]  # 4657 Dhunche is a tributary
SEARCH_OFFSETS = (-0.05, 0.0, 0.05)
CLIMATOLOGY = ("2016-01-01", "2025-12-31")
SELECTION_YEAR = ("2024-01-01", "2024-12-31")  # one year is enough to find the main channel
DISCHARGE_STATS = ["river_discharge", "river_discharge_min", "river_discharge_p25", "river_discharge_median",
                   "river_discharge_p75", "river_discharge_max"]
LIMITATIONS = [
    "GloFAS is a ~5 km global model; small, steep Himalayan catchments are poorly resolved.",
    "Chosen GloFAS cell is the highest-mean cell within ±0.05° of the gauge — an approximation of the main channel.",
    "Rainfall- and snowmelt-driven flow only: glacial-lake outbursts and landslide-dam failures are not forecast.",
    "Discharge (m³/s) and DHM stage (m) are different quantities; they are shown side by side, never converted.",
    "Not an official warning. DHM issues flood warnings in Nepal.",
]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _get_json(client: httpx.Client, url: str, params: dict[str, Any], retry_wait_s: float = 65.0) -> tuple[Any, dict[str, Any]]:
    r = client.get(url, params=params)
    if r.status_code == 429 and retry_wait_s:  # free-tier rate limit: wait once, then fail loudly
        time.sleep(retry_wait_s)
        r = client.get(url, params=params)
    r.raise_for_status()
    raw = r.content
    data = json.loads(raw)
    if isinstance(data, dict) and data.get("error"):
        raise ValueError(f"{url}: {data.get('reason')}")
    return data, {"url": url, "params": params, "sha256": _sha(raw), "bytes": len(raw),
                  "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def _percentile(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    k = (len(vals) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return vals[lo] if lo == hi else vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def map_station(client: httpx.Client, station: dict[str, Any]) -> dict[str, Any]:
    """Pick the main-channel GloFAS cell near a gauge and derive its monthly climatology."""
    lat, lon = station["latitude"], station["longitude"]
    pts = [(round(lat + a, 4), round(lon + b, 4)) for a in SEARCH_OFFSETS for b in SEARCH_OFFSETS]
    data, prov = _get_json(client, configured_url("PRAVAH_OPEN_METEO_FLOOD_URL"), {
        "latitude": ",".join(str(p[0]) for p in pts), "longitude": ",".join(str(p[1]) for p in pts),
        "daily": "river_discharge", "start_date": SELECTION_YEAR[0], "end_date": SELECTION_YEAR[1]})
    best = None
    for cell in data if isinstance(data, list) else [data]:
        vals = [v for v in cell["daily"]["river_discharge"] if v is not None]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        if best is None or mean > best[0]:
            best = (mean, cell)
    if best is None:
        raise ValueError(f"No GloFAS discharge near station {station['id']}")
    _, chosen = best
    cell, clim_prov = _get_json(client, configured_url("PRAVAH_OPEN_METEO_FLOOD_URL"), {
        "latitude": chosen["latitude"], "longitude": chosen["longitude"],
        "daily": "river_discharge", "start_date": CLIMATOLOGY[0], "end_date": CLIMATOLOGY[1]})
    clim_vals = [v for v in cell["daily"]["river_discharge"] if v is not None]
    mean = sum(clim_vals) / len(clim_vals)
    monthly: dict[str, dict[str, float | None]] = {}
    for m in range(1, 13):
        vals = [v for t, v in zip(cell["daily"]["time"], cell["daily"]["river_discharge"]) if v is not None and int(t[5:7]) == m]
        monthly[f"{m:02d}"] = {"p50": _percentile(vals, 0.5), "p90": _percentile(vals, 0.9), "p99": _percentile(vals, 0.99),
                               "max": max(vals) if vals else None, "days": len(vals)}
    return {"station_id": station["id"], "station_name": station.get("name"),
            "gauge_latitude": lat, "gauge_longitude": lon,
            "cell_latitude": cell["latitude"], "cell_longitude": cell["longitude"],
            "cell_distance_km": round(_km(lat, lon, cell["latitude"], cell["longitude"]), 2),
            "cell_mean_m3s": round(mean, 2), "candidates_checked": len(pts),
            "method": "max 2024 mean discharge among 3×3 cells at ±0.05°; climatology from that cell 2016–2025",
            "climatology_period": list(CLIMATOLOGY), "monthly_climatology_m3s": monthly,
            "provenance": {"cell_selection": prov, "climatology": clim_prov}}


def build_station_cells(stations: list[dict[str, Any]], out: Path, pause_s: float = 3.0) -> dict[str, Any]:
    """Map every station with coordinates and write the mapping file (one request per station)."""
    rows, failures = [], []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for st in stations:
            if st.get("latitude") is None or st.get("longitude") is None:
                failures.append({"station_id": st["id"], "reason": "no coordinates"})
                continue
            try:
                rows.append(map_station(client, st))
                time.sleep(pause_s)
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                failures.append({"station_id": st["id"], "reason": str(exc)[:200]})
    doc = {"schema_version": "pravah-glofas-cells-v1", "source": "GloFAS via Open-Meteo Flood API",
           "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "stations": rows, "failures": failures, "is_synthetic": False, "limitations": LIMITATIONS}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def classify(peak: float | None, clim: dict[str, float | None]) -> str:
    """Model-climatology flag for a forecast peak against the same calendar month."""
    if peak is None or clim.get("p90") is None:
        return "unknown"
    if clim.get("p99") is not None and peak >= clim["p99"]:
        return "very_high"
    if peak >= clim["p90"]:
        return "high"
    return "normal"


def fetch_outlook(cells: dict[str, Any], gauges: dict[str, dict[str, Any]] | None = None,
                  today: date | None = None, client: httpx.Client | None = None) -> dict[str, Any]:
    """Current GloFAS ensemble + rain forecast for every mapped station, with DHM observations attached."""
    stations = cells["stations"]
    if not stations:
        return {"stations": [], "sources": {}, "limitations": LIMITATIONS}
    today = today or datetime.now(timezone.utc).date()
    own = client is None
    client = client or httpx.Client(timeout=60, follow_redirects=True)
    try:
        flood, flood_prov = _get_json(client, configured_url("PRAVAH_OPEN_METEO_FLOOD_URL"), {
            "latitude": ",".join(str(s["cell_latitude"]) for s in stations),
            "longitude": ",".join(str(s["cell_longitude"]) for s in stations),
            "daily": ",".join(DISCHARGE_STATS), "past_days": 7, "forecast_days": 7})
        rain, rain_prov = _get_json(client, configured_url("PRAVAH_OPEN_METEO_URL"), {
            "latitude": ",".join(str(s["gauge_latitude"]) for s in stations),
            "longitude": ",".join(str(s["gauge_longitude"]) for s in stations),
            "daily": "precipitation_sum", "forecast_days": 7, "timezone": "UTC"})
    finally:
        if own:
            client.close()
    flood = flood if isinstance(flood, list) else [flood]
    rain = rain if isinstance(rain, list) else [rain]
    if len(flood) != len(stations) or len(rain) != len(stations):
        raise ValueError("Forecast response count does not match stations")
    if flood[0]["daily_units"]["river_discharge"] != "m³/s" or rain[0]["daily_units"]["precipitation_sum"] != "mm":
        raise ValueError("Unexpected forecast units")

    out = []
    for st, f, r in zip(stations, flood, rain):
        d = f["daily"]
        series = [{"date": t, **{k: d[k][i] for k in DISCHARGE_STATS}} for i, t in enumerate(d["time"])]
        future = [row for row in series if row["date"] > today.isoformat()]
        now_row = next((row for row in series if row["date"] == today.isoformat()), None)
        peak_row = max(future, key=lambda row: row["river_discharge"] or -1) if future else None
        month = (date.fromisoformat(peak_row["date"]) if peak_row else today).strftime("%m")
        clim = st["monthly_climatology_m3s"].get(month, {})
        rain_days = dict(zip(r["daily"]["time"], r["daily"]["precipitation_sum"]))
        upcoming = [rain_days.get((today + timedelta(days=i)).isoformat()) for i in range(3)]
        g = (gauges or {}).get(st["station_id"], {})
        out.append({
            "station_id": st["station_id"], "station_name": g.get("name") or st.get("station_name"),
            "dhm_level_m": g.get("level_m"), "dhm_warning_level_m": g.get("warning_level_m"),
            "dhm_danger_level_m": g.get("danger_level_m"), "dhm_quality": g.get("quality"),
            "dhm_observed_at_utc": g.get("observed_at_utc"), "dhm_trend": g.get("trend"),
            "discharge_today_m3s": now_row["river_discharge"] if now_row else None,
            "peak_7d_m3s": peak_row["river_discharge"] if peak_row else None,
            "peak_7d_date": peak_row["date"] if peak_row else None,
            "peak_7d_ensemble_max_m3s": peak_row["river_discharge_max"] if peak_row else None,
            "month_p90_m3s": clim.get("p90"), "month_p99_m3s": clim.get("p99"),
            "outlook": classify(peak_row["river_discharge"] if peak_row else None, clim),
            "outlook_ensemble_max": classify(peak_row["river_discharge_max"] if peak_row else None, clim),
            "rain_today_mm": upcoming[0], "rain_next_72h_mm": (sum(x for x in upcoming if x is not None)
                                                              if any(x is not None for x in upcoming) else None),
            "series": series, "cell": {k: st[k] for k in ("cell_latitude", "cell_longitude", "cell_distance_km", "cell_mean_m3s")},
            "gauge_latitude": st["gauge_latitude"], "gauge_longitude": st["gauge_longitude"],
        })
    return {"schema_version": "pravah-river-outlook-v1", "issued_for_date_utc": today.isoformat(),
            "stations": out, "sources": {"glofas": flood_prov, "rain": rain_prov},
            "kind": "third_party_forecast_display", "pravah_forecast_model": None,
            "is_synthetic": False, "limitations": LIMITATIONS}
