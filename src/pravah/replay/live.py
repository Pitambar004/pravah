"""Bounded public-source monitoring, separate from time-eligible event inputs.

All retrievals establish availability *now*. Catalog discovery does not download
imagery, and provider failures or empty samples never establish a gauge outage.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from pravah.configuration import configured_url
from pravah.replay.imagery import latest_preview, retain_preview
from pravah.replay.water_watch import watch_water

NEPAL_BBOX = [80.05, 26.3, 88.25, 30.5]
WEATHER_POINTS = [
    ("Rasuwa", 28.15, 85.35), ("Upstream Bhote Koshi", 28.43, 85.55),
    ("Kathmandu", 27.7172, 85.324), ("Pokhara", 28.2096, 83.9856),
    ("Chitwan", 27.6833, 84.4333), ("Nepalgunj", 28.05, 81.6167),
    ("Biratnagar", 26.4525, 87.2718),
]
URLS = {
    "gauges": configured_url("PRAVAH_DHM_RIVER_WATCH_URL"),
    "weather": configured_url("PRAVAH_OPEN_METEO_URL"),
    "seismic": configured_url("PRAVAH_USGS_QUERY_URL"),
    "satellite": configured_url("PRAVAH_CDSE_SEARCH_URL"),
}


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def numeric(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def age_seconds(timestamp, now):
    try:
        date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if date.tzinfo is None:
            return None
        return (now - date).total_seconds()
    except (AttributeError, ValueError, TypeError):
        return None


def parse_gauges(page: str, now: datetime) -> dict:
    marker = re.search(r"var\s+coordinates\s*=\s*", page)
    if not marker:
        raise ValueError("DHM public station schema changed")
    stations, _ = json.JSONDecoder().raw_decode(page[marker.end():])
    if not isinstance(stations, list):
        raise TypeError("DHM station list missing")
    rows = []
    for s in stations:
        # HS and MHS are the provider's hydrological-station tags. RF-only
        # stations do not have a river stage and are excluded from this adapter.
        if not set(s.get("tags") or []).intersection({"HS", "MHS", "RLS", "Bubbler"}):
            continue
        level = s.get("waterLevel") or {}
        value = numeric(level.get("value"))
        age = age_seconds(level.get("datetime"), now)
        quality = "missing" if value is None else "current"
        if value is not None and not -1000 <= value <= 10000:
            quality = "implausible_provider_value"
            value = None
        elif age is None and value is not None:
            quality = "timestamp_unknown"
        elif age is not None and age < -300:
            quality = "future_timestamp"
        elif age is not None and age > 3600:
            quality = "stale"
        rows.append({
            "id": str(s["id"]), "name": s.get("name", "Unnamed station").strip(),
            "latitude": numeric(s.get("latitude")), "longitude": numeric(s.get("longitude")),
            "level_m": value, "observed_at_utc": level.get("datetime"),
            "warning_level_m": numeric(s.get("warning_level")),
            "danger_level_m": numeric(s.get("danger_level")),
            "provider_status": s.get("status"), "trend": s.get("steady"),
            "quality": quality, "datum": "Provider station datum; compare within the same station only",
            "source_uri": configured_url("PRAVAH_DHM_STATION_URL", id=s["id"]),
        })
    return {"records": rows, "scope": "DHM public hydrological stations across Nepal",
            "units": "m (provider water-level table)", "empty_is_outage": False}


def read_response(client, method, url, **kwargs):
    with client.stream(method, url, **kwargs) as r:
        r.raise_for_status()
        blocks, count = [], 0
        for block in r.iter_bytes():
            count += len(block)
            if count > 8_000_000:
                raise ValueError("Response exceeds the 8 MB monitoring limit")
            blocks.append(block)
    return b"".join(blocks)


def fetch_source(name, now):
    url = URLS[name]
    with httpx.Client(timeout=15, follow_redirects=True) as client:
        if name == "gauges":
            raw = read_response(client, "GET", url)
            doc = parse_gauges(raw.decode(), now)
        elif name == "weather":
            params = {
                "latitude": ",".join(str(p[1]) for p in WEATHER_POINTS),
                "longitude": ",".join(str(p[2]) for p in WEATHER_POINTS),
                "current": "precipitation,temperature_2m",
                "hourly": "precipitation", "forecast_days": 2, "timezone": "UTC",
            }
            raw = read_response(client, "GET", url, params=params)
            data = json.loads(raw)
            if len(data) != len(WEATHER_POINTS):
                raise ValueError("Weather location count differs from request")
            rows = []
            for point, d in zip(WEATHER_POINTS, data):
                if d["hourly_units"]["precipitation"] != "mm":
                    raise ValueError("Unexpected rainfall units")
                pairs = list(zip(d["hourly"]["time"], d["hourly"]["precipitation"]))
                future = [v for t, v in pairs if 0 <= (datetime.fromisoformat(t).replace(tzinfo=timezone.utc) - now).total_seconds() < 86400]
                values = [numeric(v) for v in future]
                total = sum(values) if len(values) == 24 and all(v is not None for v in values) else None
                rows.append({"id": point[0], "name": point[0], "latitude": point[1], "longitude": point[2],
                             "forecast_rain_next_24h_mm": total,
                             "current_precipitation_mm": numeric(d["current"].get("precipitation")),
                             "temperature_c": numeric(d["current"].get("temperature_2m")),
                             "observed_at_utc": d["current"]["time"] + "+00:00",
                             "kind": "weather_model_estimate", "hourly": dict(pairs)})
            doc = {"records": rows, "scope": "Seven selected Nepal/upstream forecast points; gridded model estimates",
                   "units": "mm, °C", "rain_watch_threshold_mm": 50,
                   "threshold_semantics": "Analyst-selected 24h rain watch; not a DHM flood threshold"}
        elif name == "seismic":
            params = {"format": "geojson", "starttime": (now - timedelta(days=7)).isoformat(),
                      "endtime": now.isoformat(), "minlatitude": NEPAL_BBOX[1],
                      "maxlatitude": NEPAL_BBOX[3], "minlongitude": NEPAL_BBOX[0],
                      "maxlongitude": NEPAL_BBOX[2], "limit": 200, "orderby": "time"}
            raw = read_response(client, "GET", url, params=params)
            data = json.loads(raw)
            rows = []
            for f in data["features"]:
                p = f["properties"]
                xyz = f["geometry"]["coordinates"]
                rows.append({"id": f["id"], "name": p.get("place"), "magnitude": p.get("mag"),
                             "event_type": p.get("type"), "longitude": xyz[0], "latitude": xyz[1],
                             "depth_km": xyz[2], "observed_at_utc": datetime.fromtimestamp(p["time"]/1000, timezone.utc).isoformat(),
                             "source_uri": configured_url("PRAVAH_USGS_EVENT_URL", id=f["id"])})
            doc = {"records": rows, "scope": "Nepal and surrounding rectangular region, last 7 days",
                   "bbox": NEPAL_BBOX, "truncated": len(rows) >= 200,
                   "semantics": "Event type is preserved; events do not establish a flood trigger"}
        else:
            query = {"collections": ["sentinel-2-l1c"], "bbox": NEPAL_BBOX,
                     "datetime": (now - timedelta(days=14)).isoformat() + "/" + now.isoformat(),
                     "limit": 100, "sortby": [{"field": "datetime", "direction": "desc"}]}
            raw = read_response(client, "POST", url, json=query)
            data = json.loads(raw)
            rows = []
            for f in data["features"]:
                if f["collection"] != "sentinel-2-l1c":
                    raise ValueError("Satellite collection mismatch")
                p = f["properties"]
                rows.append({"id": f["id"], "collection": f["collection"],
                             "observed_at_utc": p.get("datetime") or p.get("start_datetime"),
                             "cloud_cover_percent": p.get("eo:cloud_cover"),
                             "geometry": f["geometry"], "downloaded_imagery": False})
            # Asset URLs/links can contain signed tokens; persist only sanitized metadata.
            doc = {"records": rows, "scope": "Sentinel-2 L1C catalog in Nepal bounding rectangle, last 14 days",
                   "truncated": any(x.get("rel") == "next" for x in data.get("links", [])),
                   "query": query, "imagery_change_detection": "Requires downloaded compatible imagery and valid comparable masks"}
            raw = json.dumps(doc, sort_keys=True).encode()
    retrieved = now_utc()
    return {**doc, "source_uri": url, "retrieved_at_utc": retrieved,
            "availability_verified_at_utc": retrieved, "historical_eligibility": False}, raw


def changes(name, before, after, first=False):
    """Only evidence changes and current threshold crossings become alerts."""
    old = {r["id"]: r for r in (before or {}).get("records", [])}
    result = []
    for r in after.get("records", []):
        prev = old.get(r["id"])
        reason, severity = None, "info"
        if name == "gauges":
            if r["quality"] == "current":
                def band(row):
                    value = row.get("level_m")
                    if value is None or row.get("quality") != "current":
                        return 0
                    danger, warning = row.get("danger_level_m"), row.get("warning_level_m")
                    return 2 if danger is not None and value >= danger else 1 if warning is not None and value >= warning else 0
                status = band(r)
                if status and (prev is None or band(prev) < status):
                    reason = "Current reading exceeds provider " + ("danger" if status == 2 else "warning") + " level"
                    severity = "warning"
                elif prev and prev.get("quality") == "current" and prev.get("level_m") is not None and r["observed_at_utc"] != prev.get("observed_at_utc") and r["level_m"] - prev["level_m"] >= .5:
                    reason, severity = "River stage rose ≥0.5 m between received samples (analyst watch rule)", "warning"
            if prev and prev.get("quality") == "current" and r["quality"] != "current":
                reason, severity = "Current gauge reading unavailable or stale; check provider and station", "warning"
        elif name == "weather":
            v = r.get("forecast_rain_next_24h_mm")
            previous = (prev or {}).get("forecast_rain_next_24h_mm")
            if v is not None and v >= 50 and (previous is None or previous < 50):
                reason, severity = "Forecast rainfall exceeds analyst 50 mm / 24 h watch", "warning"
        elif not first and prev is None:
            reason = "New regional " + (str(r.get("event_type") or "seismic event") if name == "seismic" else "Sentinel-2 L1C catalog item")
        if reason:
            identity = hashlib.sha256(json.dumps([name, r["id"], r.get("observed_at_utc"), reason]).encode()).hexdigest()[:24]
            result.append({"id": identity, "source": name, "record_id": r["id"], "name": r.get("name") or r["id"],
                           "message": reason, "severity": severity, "observed_at_utc": r.get("observed_at_utc"),
                           "created_at_utc": after["retrieved_at_utc"], "kind": "source_watch",
                           "limitation": "Evidence watch; not a validated flood forecast or station-failure diagnosis"})
    return result


class LiveMonitor:
    def __init__(self, root: Path, fetcher=fetch_source, runs: Path | None = None, preview=False):
        self.root, self.fetcher = root, fetcher
        self.runs = runs
        self.preview_enabled = preview
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.state = {"schema_version": 1, "sources": {}, "alerts": [], "updated_at_utc": None}
        try:
            self.state = json.loads((root / "latest.json").read_text())
        except (OSError, ValueError):
            pass

    def snapshot(self):
        with self.state_lock:
            state = copy.deepcopy(self.state)
        now = datetime.now(timezone.utc)
        for source in state["sources"].values():
            age = age_seconds(source.get("retrieved_at_utc"), now)
            source["cache_stale"] = age is None or age > 900
        for row in state["sources"].get("gauges", {}).get("records", []):
            age = age_seconds(row.get("observed_at_utc"), now)
            if row.get("quality") == "current" and (age is None or age > 3600):
                row["quality"] = "stale" if age is not None else "timestamp_unknown"
        return {**state, "refresh_in_progress": self.lock.locked(), "poll_interval_seconds": 300,
                "forecast_model_loaded": False, "forecast": None}

    def refresh(self, force=False):
        if not self.lock.acquire(blocking=False):
            return False
        try:
            now = datetime.now(timezone.utc)
            with self.state_lock:
                state = copy.deepcopy(self.state)
            age = age_seconds(state.get("updated_at_utc"), now)
            if not force and age is not None and age < 60:
                return False
            self.root.mkdir(parents=True, exist_ok=True)
            stamp = now.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
            directory = self.root / "responses" / stamp
            directory.mkdir(parents=True, exist_ok=False)
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {name: pool.submit(self.fetcher, name, now) for name in URLS}
                for name, future in futures.items():
                    previous = state["sources"].get(name)
                    try:
                        doc, raw = future.result()
                        path = directory / (name + (".html" if name == "gauges" else ".json"))
                        path.write_bytes(raw)
                        # DHM page may contain transient CSRF/session tokens: do not
                        # export it. Retain only normalized public station records.
                        if name == "gauges":
                            path.unlink()
                            path = directory / "gauges.json"
                            path.write_text(json.dumps(doc))
                            raw = path.read_bytes()
                        fresh = changes(name, previous, doc, first=previous is None)
                        known = {a["id"] for a in state["alerts"]}
                        state["alerts"].extend(a for a in fresh if a["id"] not in known)
                        state["sources"][name] = {**doc, "status": "empty" if not doc["records"] else "ok",
                                                 "response_sha256": hashlib.sha256(raw).hexdigest(),
                                                 "response_file": str(path.relative_to(self.root)), "last_attempt_at_utc": now.isoformat()}
                    except (httpx.HTTPError, ValueError, KeyError, TypeError, OSError) as exc:
                        was_ok = previous and previous.get("status") in {"ok", "empty"}
                        if was_ok:
                            state["alerts"].append({"id": stamp + "_" + name, "source": name, "name": name,
                                                    "message": "Public source refresh failed; cached evidence retained",
                                                    "severity": "warning", "kind": "source_watch", "created_at_utc": now.isoformat()})
                        state["sources"][name] = {**(previous or {"records": []}), "status": "error",
                                                 "last_attempt_at_utc": now.isoformat(), "error": type(exc).__name__ + ": source unavailable or schema changed"}
            histories = state.setdefault("gauge_history", {})
            gauge_source = state["sources"].get("gauges", {})
            gauge_rows = gauge_source.get("records", []) if gauge_source.get("status") == "ok" else []
            for row in gauge_rows:
                age = age_seconds(row.get("observed_at_utc"), now)
                if row.get("quality") != "current" or row.get("level_m") is None:
                    continue
                if age is None or not -300 <= age <= 3600:
                    continue
                values = histories.setdefault(row["id"], [])
                if not any(v["observed_at_utc"] == row["observed_at_utc"] for v in values):
                    values.append({"observed_at_utc": row["observed_at_utc"], "level_m": row["level_m"]})
                values.sort(key=lambda v: datetime.fromisoformat(v["observed_at_utc"].replace("Z", "+00:00")))
                histories[row["id"]] = values[-288:]
            if self.preview_enabled:
                try:
                    state["sources"]["rasuwa_imagery"] = retain_preview(state["sources"].get("rasuwa_imagery"), latest_preview(self.root, now))
                except (httpx.HTTPError, OSError, ValueError, KeyError):
                    state["sources"]["rasuwa_imagery"] = {**state["sources"].get("rasuwa_imagery", {"records": []}),
                                                           "status": "error", "reason": "Rasuwa catalog refresh failed; prior preview retained"}
            if self.runs is not None:
                water = watch_water(self.runs, self.root)
                water["retrieved_at_utc"] = now_utc()
                state["sources"]["water_change"] = water
                for record in water.get("records", []):
                    alert_id = "water_" + record["id"]
                    if record.get("new_water_area_km2") is not None and record["new_water_area_km2"] >= .01 and not any(a["id"] == alert_id for a in state["alerts"]):
                        state["alerts"].append({"id": alert_id, "source": "water_change", "name": "Nepal optical water change",
                                               "message": "Newly classed water exceeds analyst 0.01 km² watch in comparable pixels",
                                               "severity": "warning", "kind": "source_watch", "created_at_utc": now_utc()})
            state["alerts"] = state["alerts"][-100:]
            state["updated_at_utc"] = now.isoformat()
            temporary = self.root / "latest.json.part"
            temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
            temporary.replace(self.root / "latest.json")
            with self.state_lock:
                self.state = state
            return True
        finally:
            self.lock.release()
