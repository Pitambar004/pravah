"""Public time-series sources for the watch: ERA5 / forecast rainfall, GloFAS discharge, USGS, DEM.

Each fetch returns data plus provenance (URL, params, sha256, retrieval time). Open-Meteo's
free tier is rate limited, so long histories are requested in multi-year chunks with backoff.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

from pravah.configuration import configured_url
from pravah.watch.config import CATCHMENT_POINTS


def _get(client, url, params, tries=5):
    for k in range(tries):
        r = client.get(url, params=params)
        if r.status_code in (429, 502, 503, 504) and k < tries - 1:
            time.sleep(min(90, 10 * 2 ** k))
            continue
        r.raise_for_status()
        raw = r.content
        doc = json.loads(raw)
        if isinstance(doc, dict) and doc.get("error"):
            raise ValueError(doc.get("reason"))
        return doc, {"url": url, "params": params, "sha256": hashlib.sha256(raw).hexdigest(),
                     "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    raise RuntimeError("unreachable")


def _many(doc):
    return doc if isinstance(doc, list) else [doc]


def era5_daily(start: date, end: date, points=CATCHMENT_POINTS, chunk_years=8) -> dict:
    """Daily precipitation_sum (mm) and temperature_2m_mean (°C) per point from the ERA5 archive."""
    url = configured_url("PRAVAH_OPEN_METEO_ARCHIVE_URL")
    days, P, T, prov = [], [], [], []
    with httpx.Client(timeout=180) as c:
        s = start
        while s <= end:
            e = min(end, date(s.year + chunk_years - 1, 12, 31))
            doc, p = _get(c, url, {"latitude": ",".join(str(x[1]) for x in points),
                                   "longitude": ",".join(str(x[2]) for x in points),
                                   "start_date": s.isoformat(), "end_date": e.isoformat(),
                                   "daily": "precipitation_sum,temperature_2m_mean", "timezone": "UTC", "models": "era5"})
            cells = _many(doc)
            days += cells[0]["daily"]["time"]
            P += list(zip(*[cl["daily"]["precipitation_sum"] for cl in cells]))
            T += list(zip(*[cl["daily"]["temperature_2m_mean"] for cl in cells]))
            prov.append(p)
            s = e + timedelta(days=1)
            time.sleep(2)
    return {"days": days, "precip": [list(r) for r in P], "temp": [list(r) for r in T],
            "points": [p[0] for p in points], "units": {"precip": "mm/day", "temp": "°C"},
            "source": "ERA5 via Open-Meteo archive", "provenance": prov}


def era5_hourly(start: date, end: date, points=CATCHMENT_POINTS) -> dict:
    url = configured_url("PRAVAH_OPEN_METEO_ARCHIVE_URL")
    with httpx.Client(timeout=120) as c:
        doc, prov = _get(c, url, {"latitude": ",".join(str(x[1]) for x in points),
                                  "longitude": ",".join(str(x[2]) for x in points),
                                  "start_date": start.isoformat(), "end_date": end.isoformat(),
                                  "hourly": "precipitation,temperature_2m", "timezone": "UTC", "models": "era5"})
    cells = _many(doc)
    return {"times": cells[0]["hourly"]["time"], "points": [p[0] for p in points],
            "precip": [list(r) for r in zip(*[cl["hourly"]["precipitation"] for cl in cells])],
            "temp": [list(r) for r in zip(*[cl["hourly"]["temperature_2m"] for cl in cells])],
            "units": {"precip": "mm/h", "temp": "°C"}, "source": "ERA5 hourly via Open-Meteo archive", "provenance": prov}


def glofas_daily(cells: list[dict], start: date, end: date, ensemble=False) -> dict:
    """Daily river_discharge (m³/s) at fixed GloFAS cells (lat/lon from glofas_cells.json)."""
    url = configured_url("PRAVAH_OPEN_METEO_FLOOD_URL")
    params = {"latitude": ",".join(str(c["cell_latitude"]) for c in cells),
              "longitude": ",".join(str(c["cell_longitude"]) for c in cells),
              "start_date": start.isoformat(), "end_date": end.isoformat()}
    params["daily"] = ("river_discharge,river_discharge_median,river_discharge_max,river_discharge_min"
                       if ensemble else "river_discharge")
    with httpx.Client(timeout=180) as c:
        doc, prov = _get(c, url, params)
    out = _many(doc)
    return {"days": out[0]["daily"]["time"], "stations": [c["station_id"] for c in cells],
            "discharge": [list(r) for r in zip(*[o["daily"]["river_discharge"] for o in out])],
            **({k: [list(r) for r in zip(*[o["daily"][k] for o in out])]
                for k in ("river_discharge_median", "river_discharge_max", "river_discharge_min")} if ensemble else {}),
            "units": "m³/s", "source": "GloFAS via Open-Meteo Flood API", "provenance": prov}


def forecast_rain(past_days=92, forecast_days=7, points=CATCHMENT_POINTS) -> dict:
    """Live: hourly + daily precipitation/temperature, recent past (model analysis) and forecast."""
    url = configured_url("PRAVAH_OPEN_METEO_URL")
    with httpx.Client(timeout=60) as c:
        doc, prov = _get(c, url, {"latitude": ",".join(str(x[1]) for x in points),
                                  "longitude": ",".join(str(x[2]) for x in points),
                                  "hourly": "precipitation", "daily": "precipitation_sum,temperature_2m_mean",
                                  "past_days": past_days, "forecast_days": forecast_days, "timezone": "UTC"})
    cells = _many(doc)
    return {"hourly_times": cells[0]["hourly"]["time"], "days": cells[0]["daily"]["time"],
            "hourly_precip": [list(r) for r in zip(*[cl["hourly"]["precipitation"] for cl in cells])],
            "precip": [list(r) for r in zip(*[cl["daily"]["precipitation_sum"] for cl in cells])],
            "temp": [list(r) for r in zip(*[cl["daily"]["temperature_2m_mean"] for cl in cells])],
            "points": [p[0] for p in points], "source": "Open-Meteo forecast (gridded model estimate)", "provenance": prov}


def usgs_events(start: datetime, end: datetime, bbox=(83.5, 26.8, 87.5, 30.0), min_mag=2.5) -> list[dict]:
    url = configured_url("PRAVAH_USGS_QUERY_URL")
    with httpx.Client(timeout=60) as c:
        doc, _ = _get(c, url, {"format": "geojson", "starttime": start.isoformat(), "endtime": end.isoformat(),
                               "minlatitude": bbox[1], "maxlatitude": bbox[3], "minlongitude": bbox[0],
                               "maxlongitude": bbox[2], "minmagnitude": min_mag, "orderby": "time-asc", "limit": 2000})
    return [{"id": f["id"], "name": f["properties"].get("place"), "magnitude": f["properties"].get("mag"),
             "event_type": f["properties"].get("type"), "longitude": f["geometry"]["coordinates"][0],
             "latitude": f["geometry"]["coordinates"][1], "depth_km": f["geometry"]["coordinates"][2],
             "observed_at_utc": datetime.fromtimestamp(f["properties"]["time"] / 1000, timezone.utc).isoformat()}
            for f in doc["features"]]


def dem_tiles(bbox) -> list[str]:
    names = []
    for lat in range(math.floor(bbox[1]), math.floor(bbox[3]) + 1):
        for lon in range(math.floor(bbox[0]), math.floor(bbox[2]) + 1):
            names.append(f"Copernicus_DSM_COG_10_{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_"
                         f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM")
    return names


def download_dem(bbox, dest: Path) -> list[dict]:
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    with httpx.Client(timeout=300, follow_redirects=True) as c:
        for name in dem_tiles(bbox):
            path = dest / f"{name}.tif"
            if not path.exists():
                r = c.get(configured_url("PRAVAH_COP_DEM_TILE_URL", name=name))
                if r.status_code == 404:       # ocean / missing tile
                    continue
                r.raise_for_status()
                tmp = path.with_suffix(".part")
                tmp.write_bytes(r.content)
                tmp.replace(path)
            out.append({"tile": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    return out
