"""15-day source chronology without invented daily lake/slope measurements."""

import json
from datetime import date, datetime, timedelta
from pathlib import Path


def daily_history(root: Path, event_day=date(2026, 8, 26)):
    rainfall, scenes = {}, []
    weather = root / "sources/august_era5_rasuwa.json"
    if weather.exists():
        doc = json.loads(weather.read_text())
        rainfall = dict(zip(doc["daily"]["time"], doc["daily"]["precipitation_sum"]))
    catalog = root / "sources/rasuwa_l1c_catalog.json"
    if catalog.exists():
        scenes = json.loads(catalog.read_text())["items"]
    rows = []
    for offset in range(15, -1, -1):
        day = event_day - timedelta(days=offset)
        matches = [s for s in scenes if (s["properties"].get("datetime") or s["properties"].get("start_datetime") or "")[:10] == day.isoformat()]
        rows.append({"date": day.isoformat(), "days_before_event": offset,
                     "rainfall_mm_retrospective_reanalysis": rainfall.get(day.isoformat()),
                     "optical_catalog_records": len(matches), "scene_ids": [s["id"] for s in matches],
                     "imagery_status": "catalog_only" if matches else "no_catalog_pass_in_saved_search",
                     "lake_surface_area_m2": None, "water_level_m": None, "depth_m": None,
                     "daily_slope_displacement_m": None, "historical_availability_verified": False})
    return {"start_date": rows[0]["date"], "event_date": event_day.isoformat(), "rows": rows,
            "role": "retrospective_source_chronology", "forecast": None,
            "limit": "Metadata/reanalysis context; missing lake, slope and seismic time series are not filled or interpolated. Government data not supplied is unavailable."}


def acquisition_footprints(root: Path, cutoff):
    path = root / "sources/rasuwa_l1c_catalog.json"
    if not path.exists():
        return {"type": "FeatureCollection", "features": []}
    limit = datetime.fromisoformat(cutoff)
    features = []
    for item in json.loads(path.read_text())["items"]:
        timestamp = item["properties"].get("datetime") or item["properties"].get("start_datetime")
        if timestamp and datetime.fromisoformat(timestamp.replace("Z", "+00:00")) <= limit:
            features.append({"type": "Feature", "geometry": item["geometry"],
                             "properties": {"scene_id": item["id"], "acquired_at_utc": timestamp,
                                            "status": "Catalog only; no image pixels or verified historical availability",
                                            "kind": "catalog_footprint", "cloud_cover_percent": item["properties"].get("eo:cloud_cover")}})
    return {"type": "FeatureCollection", "features": features}
