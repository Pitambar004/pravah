"""24/7 Live Watch engine (runs inside the GB10 API service on Fuzzball).

Cadence (seconds, env-overridable):
  fast   300   DHM gauges, Open-Meteo rain (past + 7-day forecast), USGS quakes, then assess
  flow   1800  GloFAS ensemble forecast + GRU emulator ensemble, downstream hydraulics (GPU)
  sat    1800  Planetary Computer search for new S2 L2A / S1 RTC passes; process each new pass
Each tick writes ``state/latest.json`` atomically and appends a compact line to
``state/timeline.jsonl``. Provider failures keep the last good evidence and mark it stale;
they never become "all clear". In live mode the time a pass is first discovered is its
verified availability time.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from pravah.watch import snapshot as S
from pravah.watch import sources, stac
from pravah.watch.config import MAIN_STEM_GAUGES, S1_COLLECTION, S2_COLLECTION

CADENCE = {"fast": int(os.environ.get("PRAVAH_WATCH_FAST_S", "300")),
           "flow": int(os.environ.get("PRAVAH_WATCH_FLOW_S", "1800")),
           "sat": int(os.environ.get("PRAVAH_WATCH_SAT_S", "1800"))}


def utcnow():
    return datetime.now(timezone.utc)


class WatchEngine:
    def __init__(self, root: Path, river_model=None, emulator=None, observer=None):
        self.root = root
        self.state_dir = root / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.river, self.emulator, self.observer = river_model, emulator, observer
        self.lock = threading.Lock()
        self.state: dict = {}
        self.mem = self._load_mem()
        self.last = {"fast": 0.0, "flow": 0.0, "sat": 0.0}
        self.errors: dict = {}
        self._stop = threading.Event()
        self.baselines = self._json(root / "baselines/lakes.json") or {}
        self.glacier_base = self._json(root / "baselines/glacier.json") or {}
        try:
            self.state = json.loads((self.state_dir / "latest.json").read_text())
        except (OSError, ValueError):
            pass

    @staticmethod
    def _json(p):
        try:
            return json.loads(Path(p).read_text())
        except (OSError, ValueError):
            return None

    def _load_mem(self):
        return self._json(self.state_dir / "memory.json") or {"gauge_history": {}, "s2": [], "s1": [], "glacier_pairs": [],
                                                              "s1_changes": [], "seen_passes": []}

    # ------------------------------------------------------------------------- fetchers
    def fetch_fast(self, now):
        from pravah.replay.live import fetch_source
        out = {}
        try:
            doc, _ = fetch_source("gauges", now)
            hist = self.mem["gauge_history"]
            rows = []
            for r in doc["records"]:
                if r["id"] not in MAIN_STEM_GAUGES:
                    continue
                if r.get("level_m") is not None and r.get("observed_at_utc"):
                    h = hist.setdefault(r["id"], [])
                    if not h or h[-1]["observed_at_utc"] != r["observed_at_utc"]:
                        h.append({"observed_at_utc": r["observed_at_utc"], "level_m": r["level_m"]})
                    hist[r["id"]] = h[-288:]
                rows.append({**r, "history": hist.get(r["id"], [])})
            out["gauges"] = rows
            self.errors.pop("gauges", None)
        except Exception as exc:  # noqa: BLE001 — a 24/7 service must outlive any provider failure  # keep running; record the failure
            self.errors["gauges"] = f"{type(exc).__name__}: {exc}"[:300]
        try:
            fc = sources.forecast_rain()
            self.mem["forecast"] = fc
            out["rain"] = S.rain_from_hourly({"times": fc["hourly_times"], "precip": fc["hourly_precip"],
                                              "points": fc["points"], "source": fc["source"]}, now, forecast=True)
            self.errors.pop("rain", None)
        except Exception as exc:  # noqa: BLE001 — a 24/7 service must outlive any provider failure
            self.errors["rain"] = f"{type(exc).__name__}: {exc}"[:300]
        try:
            out["seismic"] = sources.usgs_events(now - timedelta(hours=96), now)
            self.errors.pop("seismic", None)
        except Exception as exc:  # noqa: BLE001 — a 24/7 service must outlive any provider failure
            self.errors["seismic"] = f"{type(exc).__name__}: {exc}"[:300]
        return out

    def fetch_flow(self, now):
        stations = self.river.meta["stations"] if self.river else []
        out = {}
        if stations:
            try:
                g = sources.glofas_daily([{"station_id": s["station_id"], "cell_latitude": s["cell_latitude"],
                                           "cell_longitude": s["cell_longitude"]} for s in stations],
                                         now.date() - timedelta(days=1), now.date() + timedelta(days=6), ensemble=True)
                i = g["days"].index(now.date().isoformat()) if now.date().isoformat() in g["days"] else 0
                window = slice(i, min(i + 3, len(g["days"])))   # today .. +2 days: take the peak
                med = np.nanmax(np.array(g["river_discharge_median"][window], float), 0)
                mx = np.nanmax(np.array(g["river_discharge_max"][window], float), 0)
                out["glofas_median"] = dict(zip(g["stations"], med.tolist()))
                out["glofas_max"] = dict(zip(g["stations"], mx.tolist()))
                self.errors.pop("glofas", None)
            except Exception as exc:  # noqa: BLE001 — a 24/7 service must outlive any provider failure
                self.errors["glofas"] = f"{type(exc).__name__}: {exc}"[:300]
        fc = self.mem.get("forecast")
        if self.emulator is not None and fc:
            try:
                days = [date.fromisoformat(d) for d in fc["days"]]
                end = now.date() + timedelta(days=2)
                keep = [k for k, d in enumerate(days) if d <= end]
                P = np.array([[0 if v is None else v for v in fc["precip"][k]] for k in keep], float)
                T = np.array([[0 if v is None else v for v in fc["temp"][k]] for k in keep], float)
                pred = self.emulator.predict(P, T, [days[k] for k in keep])
                peak = np.max(np.asarray(pred["discharge_m3s"]), 0)
                out["emulator"] = dict(zip(pred["stations"], peak.tolist()))
                if "member_max_m3s" in pred:
                    out["emulator_max"] = dict(zip(pred["stations"], np.max(np.asarray(pred["member_max_m3s"]), 0).tolist()))
                self.errors.pop("emulator", None)
            except Exception as exc:  # noqa: BLE001 — a 24/7 service must outlive any provider failure
                self.errors["emulator"] = f"{type(exc).__name__}: {exc}"[:300]
        return out

    def fetch_sat(self, now):
        if self.observer is None:
            return
        from pravah.watch.build import WATCH_BBOX
        for coll, key in ((S2_COLLECTION, "s2"), (S1_COLLECTION, "s1")):
            try:
                items = stac.search(coll, WATCH_BBOX, now - timedelta(days=4), now, limit=200)
                for g in stac.group_by_pass(items):
                    pid = f"{coll}:{g[0]['id']}"
                    if pid in self.mem["seen_passes"]:
                        continue
                    rec = self.observer.s2_pass(g, discovered_at=now) if key == "s2" else self.observer.s1_pass(g, discovered_at=now)
                    self.mem["seen_passes"].append(pid)
                    prev = self.mem[key][-1] if self.mem[key] else None
                    self.mem[key] = (self.mem[key] + [rec])[-20:]
                    if key == "s2" and prev and rec.get("glacier_windows") and prev.get("glacier_windows"):
                        dt = (stac.item_time(g[0]) - datetime.fromisoformat(prev["pass"]["acquired_at_utc"])).days
                        if 5 <= dt <= 40:
                            self.mem["glacier_pairs"].append({"acquired_at_utc": rec["pass"]["acquired_at_utc"],
                                                              "zones": self.observer.glacier_pair(prev, rec)})
                    if key == "s1":
                        same = [p for p in self.mem["s1"][:-1] if p["pass"].get("relative_orbit") == rec["pass"].get("relative_orbit")]
                        if same:
                            self.mem["s1_changes"].append({"acquired_at_utc": rec["pass"]["acquired_at_utc"],
                                                           **self.observer.s1_change(same[-1], rec)})
                self.mem["seen_passes"] = self.mem["seen_passes"][-500:]
                self.errors.pop(key, None)
            except Exception as exc:  # noqa: BLE001 — a 24/7 service must outlive any provider failure
                self.errors[key] = f"{type(exc).__name__}: {exc}"[:300]

    # ---------------------------------------------------------------------------- tick
    def tick(self, force: bool = False) -> dict:
        now = utcnow()
        mono = time.monotonic()
        t0 = time.perf_counter()
        due = {k: force or mono - self.last[k] >= CADENCE[k] for k in CADENCE}
        prev = self.state
        fast = self.fetch_fast(now) if due["fast"] else {}
        if due["fast"]:
            self.last["fast"] = mono
        if due["sat"]:
            self.fetch_sat(now)
            self.last["sat"] = mono
        flow = prev.get("_flow") or {}
        if due["flow"]:
            flow = self.fetch_flow(now) or flow
            self.last["flow"] = mono
        ev = {**(prev.get("_evidence") or {}), **fast}
        stations = self.river.meta["stations"] if self.river else []
        reach_station = {r["reach_id"]: min(stations, key=lambda s: abs(s["chainage_m"] - (r["start_km"] + r["end_km"]) * 500))["station_id"]
                         for r in self.river.meta["reaches"]} if stations else {}
        qsrc = flow.get("emulator") or flow.get("glofas_median") or {}
        reach_q = {rid: qsrc.get(sid) for rid, sid in reach_station.items()}
        lakes_s2 = S.latest_lakes(self.mem["s2"], self.baselines, now, "s2")
        lakes_s1 = S.latest_lakes(self.mem["s1"], self.baselines, now, "s1")
        discharge = (S.discharge_rows(flow.get("glofas_median", {}), stations, now.date(), "GloFAS ensemble median (Open-Meteo)")
                     + S.discharge_rows(flow.get("emulator", {}), stations, now.date(), "PRAVAH GRU emulator ensemble"))
        snap = {"now": now.isoformat(), "rain": ev.get("rain", {}), "seismic": ev.get("seismic"), "gauges": ev.get("gauges"),
                "lakes": {"s2": lakes_s2, "s1": lakes_s1},
                "glacier": S.latest_glacier(self.mem["glacier_pairs"], self.glacier_base, now),
                "widths": S.latest_widths(self.mem["s2"], self.river.fits if self.river else {}, reach_q, now),
                "discharge": discharge, "sar_change": S.latest_before(self.mem["s1_changes"], now)}
        a = S.assess_snapshot(snap)
        down = prev.get("downstream")
        glof = S.glof_trigger(a, lakes_s2 + lakes_s1, self.river) if self.river else None
        if self.river and (due["flow"] or due["fast"] or force) and any(flow.values()):
            down = self.river.run({k: v for k, v in flow.items() if v}, glof=glof)
        state = {"mode": "live", "updated_at_utc": now.isoformat(), "assessment": a, "downstream": down,
                 "evidence": {"gauges": ev.get("gauges"), "rain": ev.get("rain"),
                              "seismic": (ev.get("seismic") or [])[-30:], "lakes": snap["lakes"], "glacier": snap["glacier"],
                              "widths": snap["widths"], "discharge": discharge, "sar_change": snap["sar_change"],
                              "latest_passes": [p["pass"] for p in (self.mem["s2"][-3:] + self.mem["s1"][-3:])]},
                 "source_errors": dict(self.errors), "cadence_s": CADENCE,
                 "tick_seconds_measured": round(time.perf_counter() - t0, 3),
                 "_evidence": ev, "_flow": flow}
        with self.lock:
            self.state = state
        self._persist(state, a)
        return state

    def _persist(self, state, a):
        tmp = self.state_dir / "latest.json.part"
        tmp.write_text(json.dumps(state, default=str))
        tmp.replace(self.state_dir / "latest.json")
        tmp = self.state_dir / "memory.json.part"
        tmp.write_text(json.dumps(self.mem, default=str))
        tmp.replace(self.state_dir / "memory.json")
        with (self.state_dir / "timeline.jsonl").open("a") as f:
            f.write(json.dumps({"t": state["updated_at_utc"], "level": a["level"],
                                "elevated": a["families_elevated"],
                                "indicators": {i["id"]: i["status"] for i in a["indicators"]}}) + "\n")

    def public_state(self) -> dict:
        with self.lock:
            return {k: v for k, v in self.state.items() if not k.startswith("_")}

    def run_forever(self, period_s: int = 60):
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 — a 24/7 service must outlive any provider failure
                self.errors["engine"] = traceback.format_exc()[-800:]
            self._stop.wait(period_s)

    def start(self):
        t = threading.Thread(target=self.run_forever, name="pravah-watch", daemon=True)
        t.start()
        return t

    def stop(self):
        self._stop.set()
