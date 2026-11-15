"""GB10 build pipeline for Live Watch. Runs on Fuzzball, never on a laptop.

  python -m pravah.watch.build --out /data/pravah_watch --steps history,emulator          # GB10 node A
  python -m pravah.watch.build --out /data/pravah_watch --steps river,satellite,replay    # GB10 node B

Steps are resumable: a finished step leaves ``<out>/steps/<name>.done.json`` (with measured
timings) and is skipped unless ``--force``. Leakage rules: width fits, lake and glacier
baselines use imagery ≤ 2024 only; the emulator tests on 2025–2026; the Aug-2026 replay
never reads data after its own clock (see replay.py).
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from pravah.watch import layout, sources, stac
from pravah.watch.config import (
    CORRIDOR_BBOX,
    S1_COLLECTION,
    S2_COLLECTION,
    UPSTREAM_BBOX,
    data_root,
)

HISTORY_START = date(1984, 1, 1)
SAT_YEARS = (2018, 2024)
REPLAY_START = datetime(2026, 8, 11, tzinfo=timezone.utc)
REPLAY_END = datetime(2026, 8, 28, tzinfo=timezone.utc)
WATCH_BBOX = (min(CORRIDOR_BBOX[0], UPSTREAM_BBOX[0]), min(CORRIDOR_BBOX[1], UPSTREAM_BBOX[1]),
              max(CORRIDOR_BBOX[2], UPSTREAM_BBOX[2]), max(CORRIDOR_BBOX[3], UPSTREAM_BBOX[3]))


def log(msg):
    print(f"[watch-build {datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def env_record():
    rec = {"host": socket.gethostname(), "machine": platform.machine(), "python": platform.python_version()}
    try:
        import torch
        rec.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        pass
    return rec


# ----------------------------------------------------------------------------------- steps
def step_history(out: Path, args) -> dict:
    end = date.today() - timedelta(days=7)        # ERA5 lags ~5 days
    era = sources.era5_daily(HISTORY_START, end)
    cells = json.loads((data_root() / "forecast/glofas_cells.json").read_text())["stations"]
    glo = sources.glofas_daily(cells, HISTORY_START, end)
    qmap = dict(zip(glo["days"], glo["discharge"]))
    days, P, T, Q = [], [], [], []
    for d, p, t in zip(era["days"], era["precip"], era["temp"]):
        if d in qmap:
            days.append(d)
            P.append(p)
            T.append(t)
            Q.append([np.nan if v is None else v for v in qmap[d]])
    P = np.array(P, float)
    T = np.array(T, float)
    table = {"days": days, "precip": np.where(np.isnan(P), 0, P).tolist(), "temp": np.where(np.isnan(T), 0, T).tolist(),
             "discharge": [[None if np.isnan(v) else v for v in r] for r in Q], "stations": glo["stations"],
             "points": era["points"], "provenance": {"era5": era["provenance"], "glofas": glo["provenance"]}}
    (out / "history").mkdir(parents=True, exist_ok=True)
    (out / "history/table.json").write_text(json.dumps(table, allow_nan=True).replace("NaN", "null"))
    return {"days": len(days), "first": days[0], "last": days[-1], "stations": glo["stations"]}


def step_emulator(out: Path, args) -> dict:
    from pravah.watch.emulator import train_ensemble
    path = out / "history/table.json"
    doc = json.loads(path.read_text())
    doc["discharge"] = [[np.nan if v is None else v for v in r] for r in doc["discharge"]]
    tmp = out / "history/table_np.json"
    tmp.write_text(json.dumps(doc, allow_nan=True))
    return train_ensemble(tmp, out / "emulator", members=args.members, workers=args.workers, epochs=args.epochs)


def step_river(out: Path, args) -> dict:
    from pravah.watch.river import build_river_model
    dem_dir = out / "dem"
    tiles = sources.download_dem(WATCH_BBOX, dem_dir)
    meta = build_river_model(data_root() / "nepal_monitor", sorted(dem_dir.glob("*.tif")),
                             data_root() / "forecast/glofas_cells.json", out / "river_model")
    windows = layout.build_windows(out / "river_model", data_root() / "nepal_lakes")
    layout.save(windows, out / "river_model/windows.json")
    return {"dem_tiles": tiles, "sections": meta["sections"], "main_stem": meta["main_stem"],
            "stations": [s["station_id"] for s in meta["stations"]], "assets": meta["asset_counts_in_corridor"],
            "windows": {k: sum(w.kind == k for w in windows) for k in ("reach", "lakes", "glacier")},
            "build_seconds": meta["build_seconds_measured"]}


def _passes(collection, start, end, max_cloud=None, per_month=None):
    items = []
    s = start
    while s < end:                                 # month-by-month keeps each search small
        e = min(end, (s.replace(day=1) + timedelta(days=32)).replace(day=1))
        items += stac.search(collection, WATCH_BBOX, s, e, limit=500, max_cloud=max_cloud)
        s = e
    groups = stac.group_by_pass(items)
    if per_month:
        by: dict = {}
        for g in groups:
            by.setdefault(stac.item_time(g[0]).strftime("%Y-%m"), []).append(g)
        groups = []
        for month, gs in sorted(by.items()):
            gs.sort(key=lambda g: np.mean([i["properties"].get("eo:cloud_cover") or 0 for i in g]))
            groups += gs[:per_month]
    return sorted(groups, key=lambda g: stac.item_time(g[0]))


def _process(observer, groups, kind, workers):
    fn = observer.s2_pass if kind == "s2" else observer.s1_pass
    done, results = 0, []

    def one(g):
        try:
            return fn(g)
        except Exception as exc:  # noqa: BLE001 — one bad pass must not stop a long history run
            return {"error": f"{type(exc).__name__}: {exc}", "pass": stac.pass_record(g, S2_COLLECTION if kind == "s2" else S1_COLLECTION)}
    with ThreadPoolExecutor(workers) as pool:
        for r in pool.map(one, groups):
            results.append(r)
            done += 1
            if done % 10 == 0 or done == len(groups):
                log(f"{kind} passes {done}/{len(groups)}")
    return results


def step_satellite(out: Path, args) -> dict:
    """Historical S2/S1 (≤2024): width–discharge fits, lake baselines, glacier baseline speeds."""
    from pravah.watch.hydro import fit_hydraulic_geometry
    from pravah.watch.observe import Observer
    windows = layout.load(out / "river_model/windows.json")
    obs = Observer(windows, out / "obs", threads=8)
    start = datetime(SAT_YEARS[0], 1, 1, tzinfo=timezone.utc)
    end = datetime(SAT_YEARS[1] + 1, 1, 1, tzinfo=timezone.utc)
    s2_groups = _passes(S2_COLLECTION, start, end, max_cloud=40, per_month=args.s2_per_month)
    log(f"S2 history: {len(s2_groups)} passes")
    s2 = [r for r in _process(obs, s2_groups, "s2", args.workers) if "error" not in r]
    s1_groups = _passes(S1_COLLECTION, datetime(2023, 9, 1, tzinfo=timezone.utc), end, per_month=4)
    log(f"S1 baseline: {len(s1_groups)} passes")
    s1 = [r for r in _process(obs, s1_groups, "s1", args.workers) if "error" not in r]
    # width–discharge fits per reach (GloFAS at the reach's nearest station on that date)
    model = json.loads((out / "river_model/river_model.json").read_text())
    cells = [s for s in model["stations"]]
    glo = sources.glofas_daily([{"station_id": s["station_id"], "cell_latitude": s["cell_latitude"],
                                 "cell_longitude": s["cell_longitude"]} for s in cells], start.date(), end.date())
    qday = dict(zip(glo["days"], glo["discharge"]))
    reach_station = {}
    for r in model["reaches"]:
        mid = (r["start_km"] + r["end_km"]) * 500
        reach_station[r["reach_id"]] = min(range(len(cells)), key=lambda i: abs(cells[i]["chainage_m"] - mid))
    pairs: dict = {}
    for r in s2:
        day = r["pass"]["acquired_at_utc"][:10]
        for w in r["reaches"]:
            q = (qday.get(day) or [None] * len(cells))[reach_station[w["reach_id"]]]
            if w["width_m"] and q:
                pairs.setdefault(w["reach_id"], []).append((w["width_m"], q, day))
    fits = {rid: dict(fit_hydraulic_geometry([p[0] for p in v], [p[1] for p in v]) or {"n": len(v), "fit": None},
                      station=cells[reach_station[rid]]["station_id"]) for rid, v in pairs.items()}
    (out / "river_model/width_fits.json").write_text(json.dumps({"years": list(SAT_YEARS), "reaches": fits,
                                                                 "pairs": pairs}, indent=1))
    # lake baselines: median valid area in post-monsoon (Sep–Nov) of the last two baseline years
    def baseline(rows, sensor):
        by: dict = {}
        for r in rows:
            t = r["pass"]["acquired_at_utc"]
            if int(t[:4]) >= SAT_YEARS[1] - 1 and int(t[5:7]) in (9, 10, 11):
                for x in r["lakes"]:
                    if x["area_m2"] is not None:
                        by.setdefault(x["lake_id"], []).append(x["area_m2"])
        return {k: {"median_m2": float(np.median(v)), "n": len(v), "sensor": sensor} for k, v in by.items() if len(v) >= 2}
    lakes = {"s2": baseline(s2, "sentinel-2"), "s1": baseline(s1, "sentinel-1"),
             "semantics": "Median valid water area Sep–Nov of baseline years; sensor-specific, never mixed"}
    (out / "baselines").mkdir(exist_ok=True)
    (out / "baselines/lakes.json").write_text(json.dumps(lakes, indent=1))
    # glacier baselines: Oct–Dec pairs 5–40 days apart, GPU feature tracking
    late = [r for r in s2 if r.get("glacier_windows") and int(r["pass"]["acquired_at_utc"][5:7]) in (10, 11, 12)
            and int(r["pass"]["acquired_at_utc"][:4]) >= SAT_YEARS[1] - 1]
    speeds: dict = {}
    t0 = time.perf_counter()
    for a, b in itertools.pairwise(late):
        dt = (datetime.fromisoformat(b["pass"]["acquired_at_utc"]) - datetime.fromisoformat(a["pass"]["acquired_at_utc"])).days
        if 5 <= dt <= 40:
            for z in obs.glacier_pair(a, b):
                if z["median_speed_m_per_day"] is not None:
                    speeds.setdefault(z["zone_id"], []).append(z["median_speed_m_per_day"])
    track_s = time.perf_counter() - t0
    glac = {k: {"baseline_m_per_day": float(np.median(v)), "pairs": len(v)} for k, v in speeds.items()}
    (out / "baselines/glacier.json").write_text(json.dumps(glac, indent=1))
    return {"s2_passes": len(s2), "s1_passes": len(s1), "width_fits": {k: v.get("r2") for k, v in fits.items()},
            "lake_baselines": {k: len(v) for k, v in lakes.items() if isinstance(v, dict)},
            "glacier_zones": len(glac), "glacier_tracking_seconds": round(track_s, 2)}


def step_replay(out: Path, args) -> dict:
    """Assemble the Aug 2026 Rasuwa replay pack (observations stamped with acquisition time)."""
    from pravah.watch.observe import Observer
    windows = layout.load(out / "river_model/windows.json")
    obs = Observer(windows, out / "obs", threads=8)
    pre = REPLAY_START - timedelta(days=45)
    s2_groups = _passes(S2_COLLECTION, pre, REPLAY_END, max_cloud=80)
    s1_groups = _passes(S1_COLLECTION, pre, REPLAY_END)
    s2 = [r for r in _process(obs, s2_groups, "s2", args.workers) if "error" not in r]
    s1 = [r for r in _process(obs, s1_groups, "s1", args.workers) if "error" not in r]
    glacier = []
    clear = [r for r in s2 if r.get("glacier_windows")]
    for a, b in itertools.pairwise(clear):
        dt = (datetime.fromisoformat(b["pass"]["acquired_at_utc"]) - datetime.fromisoformat(a["pass"]["acquired_at_utc"])).days
        if 5 <= dt <= 40:
            glacier.append({"acquired_at_utc": b["pass"]["acquired_at_utc"], "zones": obs.glacier_pair(a, b)})
    s1_changes = []
    for i, b in enumerate(s1):
        prev = [a for a in s1[:i] if a["pass"].get("relative_orbit") == b["pass"].get("relative_orbit")]
        if prev:
            s1_changes.append({"acquired_at_utc": b["pass"]["acquired_at_utc"], **obs.s1_change(prev[-1], b)})
    rain_h = sources.era5_hourly((REPLAY_START - timedelta(days=4)).date(), REPLAY_END.date())
    rain_d = sources.era5_daily(date(2026, 4, 1), REPLAY_END.date())
    quakes = sources.usgs_events(REPLAY_START - timedelta(days=4), REPLAY_END)
    model = json.loads((out / "river_model/river_model.json").read_text())
    glo = sources.glofas_daily([{"station_id": s["station_id"], "cell_latitude": s["cell_latitude"],
                                 "cell_longitude": s["cell_longitude"]} for s in model["stations"]],
                               REPLAY_START.date() - timedelta(days=3), REPLAY_END.date())
    pack = {"schema": "pravah-watch-replay-v1", "event": "Rasuwa, Bhote Koshi–Trishuli, 26 Aug 2026",
            "start_utc": REPLAY_START.isoformat(), "end_utc": REPLAY_END.isoformat(),
            "s2": s2, "s1": s1, "glacier_pairs": glacier, "s1_changes": s1_changes,
            "rain_hourly": rain_h, "rain_daily": rain_d, "seismic": quakes, "glofas_reanalysis": glo,
            "gauges": None, "gauges_note": "DHM serves ~2.5 weeks of history; no archived Aug-2026 gauge readings were available",
            "availability": "Imagery acquisition times are real; provider availability times were not verified. "
                            "Reconstruction, not a cutoff-eligible forecast.",
            "built_at_utc": datetime.now(timezone.utc).isoformat()}
    (out / "replay").mkdir(exist_ok=True)
    (out / "replay/pack.json").write_text(json.dumps(pack))
    return {"s2_passes": len(s2), "s1_passes": len(s1), "glacier_pairs": len(glacier), "s1_changes": len(s1_changes),
            "quakes": len(quakes)}


STEPS = {"history": step_history, "emulator": step_emulator, "river": step_river,
         "satellite": step_satellite, "replay": step_replay}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", default="history,emulator,river,satellite,replay")
    p.add_argument("--workers", type=int, default=5, help="concurrent workers (passes / ensemble members)")
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--s2-per-month", type=int, default=4)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "steps").mkdir(exist_ok=True)
    for name in args.steps.split(","):
        marker = args.out / "steps" / f"{name}.done.json"
        if marker.exists() and not args.force:
            log(f"{name}: already done ({marker})")
            continue
        log(f"{name}: start")
        t0 = time.perf_counter()
        result = STEPS[name](args.out, args)
        rec = {"step": name, "seconds_measured": round(time.perf_counter() - t0, 2), "finished_at_utc": datetime.now(timezone.utc).isoformat(),
               "environment": env_record(), "workers": args.workers, "result": result}
        marker.write_text(json.dumps(rec, indent=2, default=str))
        log(f"{name}: done in {rec['seconds_measured']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
