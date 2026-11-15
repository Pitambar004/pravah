#!/usr/bin/env python3
"""Map DHM main-stem corridor gauges to GloFAS cells (one-time; re-run to refresh).

    python scripts/build_forecast_cells.py [--out data/processed/forecast/glofas_cells.json]

Reads the live DHM station table for names/coordinates, then for each station
queries a 3×3 GloFAS neighbourhood (2016–2025 daily) via the Open-Meteo Flood API.
See pravah.forecast.outlook for the method and limitations.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from pravah.forecast.outlook import CORRIDOR_MAIN_STEM, build_station_cells
from pravah.replay.live import fetch_source


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=Path("data/processed/forecast/glofas_cells.json"))
    p.add_argument("--stations", default=",".join(CORRIDOR_MAIN_STEM), help="Comma-separated DHM station ids")
    args = p.parse_args(argv)
    doc, _ = fetch_source("gauges", datetime.now(timezone.utc))
    by_id = {str(g["id"]): g for g in doc["records"]}
    wanted = [s.strip() for s in args.stations.split(",") if s.strip()]
    missing = [s for s in wanted if s not in by_id]
    stations = [by_id[s] for s in wanted if s in by_id]
    out = build_station_cells(stations, args.out)
    out["failures"] += [{"station_id": s, "reason": "not in DHM station table"} for s in missing]
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    for s in out["stations"]:
        print(f"{s['station_id']:>5} {s['station_name'][:34]:34s} cell {s['cell_latitude']:.3f},{s['cell_longitude']:.3f} "
              f"({s['cell_distance_km']} km) mean {s['cell_mean_m3s']} m3/s")
    for f in out["failures"]:
        print("FAILED", f)
    return 0 if out["stations"] else 1


if __name__ == "__main__":
    sys.exit(main())
