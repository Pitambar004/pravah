#!/usr/bin/env python3
"""Archive DHM 10-minute river-level history so a forecast training set can accumulate.

DHM's public station view only serves roughly the last 2½ weeks (probed 2026-11-15:
data back to 2026-10-29..31, nothing on 2026-10-28). Run this at least weekly; each
run backfills what the provider still serves and merges it, deduplicated, into

    data/raw/dhm_history/<station_id>.csv     datetime_utc,level_m
    data/raw/dhm_history/manifest.json         per-request provenance (sha256, time)

Levels are river stage on each station's own provider datum (m), not discharge.
Values are archived exactly as served; provider error codes (e.g. -9999985) are kept and must
be filtered before analysis (see manifest "known_provider_error_values").
Uses the provider's public station form; requests are spaced politely. Session
cookies and CSRF tokens are never stored.

    python scripts/collect_dhm_history.py [--stations 265,4781] [--days 21]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from pravah.configuration import configured_url

# Bhote Koshi → Trishuli → Narayani main stem, plus Tadi and Marsyangdi tributaries
DEFAULT_STATIONS = ["4913", "191", "52", "5705", "4781", "265", "66", "4659", "186", "172"]
PERIOD_7_DAYS = "4"  # provider option "7 Days Point": 10-minute samples for the 7 days ending on `date`


def parse_river(chart_html: str) -> list[dict]:
    m = re.search(r"var river = '([^']*)'", chart_html or "")
    if not m:
        raise ValueError("Provider chart format changed")
    rows = json.loads(m.group(1).encode().decode("unicode_escape"))
    return [{"datetime_utc": r["datetime"], "level_m": float(r["value"])} for r in rows
            if r.get("datetime") and r.get("value") is not None]


def merge(path: Path, rows: list[dict]) -> tuple[int, int]:
    existing: dict[str, float] = {}
    if path.exists():
        with path.open() as f:
            existing = {r["datetime_utc"]: float(r["level_m"]) for r in csv.DictReader(f)}
    before = len(existing)
    for r in rows:
        existing[r["datetime_utc"]] = r["level_m"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["datetime_utc", "level_m"])
        for k in sorted(existing):
            w.writerow([k, existing[k]])
    return len(existing) - before, len(existing)


def collect(stations: list[str], days: int, out: Path, pause_s: float = 1.0) -> dict:
    station_url = configured_url("PRAVAH_DHM_STATION_URL")
    history_url = configured_url("PRAVAH_DHM_HISTORY_URL")
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema_version": "pravah-dhm-history-v1", "units": "m (provider stage, station datum)",
        "source": "DHM public station view", "is_synthetic": False, "requests": []}
    today = datetime.now(timezone.utc).date()
    ends = [today - timedelta(days=d) for d in range(0, days, 7)]
    summary = {}
    with httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": "pravah-research"}) as client:
        for sid in stations:
            url = station_url.format(id=sid)
            page = client.get(url)
            page.raise_for_status()
            csrf = re.search(r'name="csrf_test_name" value="([^"]+)"', page.text)
            series = re.search(r'name="seriesid"[^>]*value="([0-9]+)"', page.text)
            if not csrf or not series:
                summary[sid] = {"error": "station form changed or missing"}
                continue
            got = []
            for end in ends:
                time.sleep(pause_s)
                r = client.post(history_url, data={"csrf_test_name": csrf.group(1), "seriesid": series.group(1),
                                                   "date": end.isoformat(), "period": PERIOD_7_DAYS},
                                headers={"Referer": url, "X-Requested-With": "XMLHttpRequest"})
                r.raise_for_status()
                rows = parse_river((r.json().get("data") or {}).get("chart"))
                got += rows
                manifest["requests"].append({
                    "station_id": sid, "series_id": series.group(1), "window_end_date": end.isoformat(),
                    "period": "7 Days Point", "samples": len(rows), "response_sha256": hashlib.sha256(r.content).hexdigest(),
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            added, total = merge(out / f"{sid}.csv", got)
            first = min((x["datetime_utc"] for x in got), default=None)
            summary[sid] = {"fetched": len(got), "added": added, "total": total, "earliest_fetched": first}
    manifest["known_provider_error_values"] = "levels <= -9999 are provider error codes, not measurements"
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stations", default=",".join(DEFAULT_STATIONS))
    p.add_argument("--days", type=int, default=21, help="How far back to request (provider serves ~17 days)")
    p.add_argument("--out", type=Path, default=Path("data/raw/dhm_history"))
    args = p.parse_args(argv)
    summary = collect([s.strip() for s in args.stations.split(",") if s.strip()], args.days, args.out)
    for sid, s in summary.items():
        print(sid, s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
