#!/usr/bin/env python3
"""Portable equivalent of scripts/p3_extract_osm.sh when ``ogr2ogr`` is not installed.

Reads an OSM .pbf with GDAL's OSM driver (via pyogrio) and writes
``osm_{points,lines,multipolygons}.gpkg`` for features *intersecting* the bbox,
exactly the inputs ``scripts/p3_build_infrastructure.py`` expects. Like
``ogr2ogr -spat`` this selects, it does not clip.

    python scripts/extract_osm_bbox.py --pbf data/raw/osm/nepal-260101.osm.pbf --out data/processed/osm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyogrio

BBOX = (85.10, 27.85, 85.70, 28.35)  # same provisional bbox as p3_extract_osm.sh (not an official boundary)
LAYERS = ("points", "lines", "multipolygons")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pbf", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--bbox", type=float, nargs=4, default=BBOX, metavar=("W", "S", "E", "N"))
    a = p.parse_args(argv)
    if not a.pbf.is_file():
        raise SystemExit(f"Missing OSM source: {a.pbf}")
    a.out.mkdir(parents=True, exist_ok=True)
    report = {"source": str(a.pbf), "bbox_epsg4326": list(a.bbox), "selection": "intersects (not clipped)", "layers": {}}
    for layer in LAYERS:
        frame = pyogrio.read_dataframe(a.pbf, layer=layer, bbox=tuple(a.bbox))
        path = a.out / f"osm_{layer}.gpkg"
        if path.exists():
            path.unlink()
        frame.to_file(path, layer=layer, driver="GPKG")
        report["layers"][layer] = len(frame)
        print(f"{layer}: {len(frame)} features -> {path}")
    (a.out / "osm_extract_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
