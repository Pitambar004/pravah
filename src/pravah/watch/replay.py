"""Rasuwa 26-Aug-2026 replay: the live engine's rules and hydraulics stepped through history.

Hourly frames from 11 Aug to 28 Aug 2026. At each clock time the frame sees only
observations with acquisition/observation time ≤ clock: ERA5 hourly rainfall up to the clock
(no archived forecast → forecast indicators are "unknown"), USGS events, Sentinel-2/1
passes and glacier pairs, and emulator discharge driven by rainfall ≤ clock. GloFAS
reanalysis is shown as after-the-fact context only and never drives a trigger. Imagery
provider availability was not verified, so this is a reconstruction, not a cutoff-eligible
forecast. Frames are computed on the API's GPU at startup and cached.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from pravah.watch import snapshot as S


def _t(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def emulator_station_q(emulator, rain_daily: dict, rain_hourly: dict, clock: datetime) -> dict | None:
    if emulator is None:
        return None
    days = [date.fromisoformat(d) for d in rain_daily["days"]]
    P = np.array([[0 if v is None else v for v in r] for r in rain_daily["precip"]], float)
    T = np.array([[0 if v is None else v for v in r] for r in rain_daily["temp"]], float)
    end = clock.date()
    keep = [i for i, d in enumerate(days) if d <= end]
    P, T, days = P[keep], T[keep], [days[i] for i in keep]
    # clock day: only hours up to the clock count
    hours = [_t(t + "+00:00" if len(t) <= 16 else t) for t in rain_hourly["times"]]
    sel = [i for i, h in enumerate(hours) if h.date() == end and h <= clock]
    if days and days[-1] == end:
        P[-1] = np.nansum(np.array([[0 if v is None else v for v in rain_hourly["precip"][i]] for i in sel], float), 0) if sel else 0
    if len(days) < 90:
        return None
    pred = emulator.predict(P, T, days)
    last = -1
    return {"median": dict(zip(pred["stations"], np.asarray(pred["discharge_m3s"])[last])),
            "max": dict(zip(pred["stations"], np.asarray(pred["member_max_m3s"])[last])) if "member_max_m3s" in pred else None}


def build_frames(pack: dict, river_model, emulator, baselines: dict, glacier_base: dict, step_hours: int = 1,
                 progress=None) -> dict:
    start, end = _t(pack["start_utc"]), _t(pack["end_utc"])
    frames, clock, last_poly, last_level = [], start, None, None
    t0 = time.perf_counter()
    stations = river_model.meta["stations"]
    reach_station = {r["reach_id"]: min(stations, key=lambda s: abs(s["chainage_m"] - (r["start_km"] + r["end_km"]) * 500))["station_id"]
                     for r in river_model.meta["reaches"]} if stations else {}
    glo_days = dict(zip(pack["glofas_reanalysis"]["days"], pack["glofas_reanalysis"]["discharge"]))
    emu_cache: dict = {}
    total = int((end - start).total_seconds() // (3600 * step_hours)) + 1
    while clock <= end:
        key = clock.strftime("%Y-%m-%dT%H")
        emu = emu_cache.get(key) or emulator_station_q(emulator, pack["rain_daily"], pack["rain_hourly"], clock)
        emu_cache[key] = emu
        reach_q = {rid: (emu or {}).get("median", {}).get(sid) for rid, sid in reach_station.items()}
        lakes_s2 = S.latest_lakes(pack["s2"], baselines, clock, "s2")
        lakes_s1 = S.latest_lakes(pack["s1"], baselines, clock, "s1")
        sar = S.latest_before(pack["s1_changes"], clock)
        snap = {"now": clock.isoformat(),
                "rain": S.rain_from_hourly(pack["rain_hourly"], clock, forecast=False),
                "seismic": [e for e in pack["seismic"] if _t(e["observed_at_utc"]) <= clock],
                "gauges": pack.get("gauges"),
                "lakes": {"s2": lakes_s2, "s1": lakes_s1},
                "glacier": S.latest_glacier(pack["glacier_pairs"], glacier_base, clock),
                "widths": S.latest_widths(pack["s2"], river_model.fits, reach_q, clock),
                "discharge": S.discharge_rows((emu or {}).get("median", {}), stations, clock.date(), "PRAVAH GRU emulator (rain ≤ clock)"),
                "sar_change": sar}
        a = S.assess_snapshot(snap)
        sources = {"emulator": (emu or {}).get("median")}
        if emu and emu.get("max"):
            sources["emulator_max"] = emu["max"]
        glof = S.glof_trigger(a, lakes_s2 + lakes_s1, river_model)
        down = None
        if any(sources.values()):
            need_poly = last_poly is None or a["level"] != last_level or clock.hour % 6 == 0 or glof is not None
            down = river_model.run({k: v for k, v in sources.items() if v}, glof=glof, polygons=need_poly, month=clock.month)
            if need_poly:
                last_poly = {k: down.get(k) for k in ("extent_median", "extent_max", "exposed_points_max")}
            else:
                down.update(last_poly)
        frames.append({"clock_utc": clock.isoformat(), "assessment": a, "downstream": down,
                       "glofas_reanalysis_context": dict(zip(pack["glofas_reanalysis"]["stations"], glo_days.get(clock.date().isoformat()) or [])),
                       "new_imagery": [p["pass"] for p in pack["s2"] + pack["s1"]
                                       if clock - timedelta(hours=step_hours) < _t(p["pass"]["acquired_at_utc"]) <= clock]})
        last_level = a["level"]
        if progress:
            progress(len(frames), total)
        clock += timedelta(hours=step_hours)
    return {"event": pack["event"], "start_utc": pack["start_utc"], "end_utc": pack["end_utc"], "frames": frames,
            "frame_count": len(frames), "compute_seconds_measured": round(time.perf_counter() - t0, 2),
            "device": frames[-1]["downstream"]["device"] if frames and frames[-1]["downstream"] else None,
            "availability": pack["availability"], "gauges_note": pack.get("gauges_note"), "method": __doc__}


def load_or_build(root: Path, river_model, emulator, progress=None) -> dict:
    cache = root / "replay/frames.json"
    pack_path = root / "replay/pack.json"
    if cache.exists() and pack_path.exists() and cache.stat().st_mtime >= pack_path.stat().st_mtime:
        return json.loads(cache.read_text())
    pack = json.loads(pack_path.read_text())
    lakes = json.loads((root / "baselines/lakes.json").read_text()) if (root / "baselines/lakes.json").exists() else {}
    glac = json.loads((root / "baselines/glacier.json").read_text()) if (root / "baselines/glacier.json").exists() else {}
    out = build_frames(pack, river_model, emulator, lakes, glac, progress=progress)
    tmp = cache.with_suffix(".part")
    tmp.write_text(json.dumps(out))
    tmp.replace(cache)
    return out
