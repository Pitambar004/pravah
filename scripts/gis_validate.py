#!/usr/bin/env python3
"""Validate P3 GeoPackage thematic assets and terrain/rainfall files (no flood claims).

Nonzero exit for missing mandatory GIS assets, null/invalid geometries, or bad CRS.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import rasterio

THEMATIC = (
    "roads", "bridges", "waterways", "buildings", "settlements",
    "schools", "healthcare", "emergency_facilities", "waterbodies", "landuse",
)
RASTERS = {
    "dem": ("processed/terrain/rasuwa_area_dem_utm.tif", 32645),
    "slope": ("processed/terrain/rasuwa_area_slope_degrees.tif", 32645),
    "rainfall_accum": ("processed/rainfall/rainfall_accumulation_aug01_26.tif", 4326),
}


def validate(root):
    report = {
        "study_area": "provisional bbox, not verified district",
        "data_root": str(root),
        "source_osm_snapshot": "2026-01-01",
        "vectors": {}, "rasters": {}, "errors": [],
    }
    for name in THEMATIC:
        path = root / "processed/osm/thematic" / f"{name}.gpkg"
        if not path.exists():
            report["errors"].append(f"missing vector {name}: {path}")
            continue
        try:
            layer = gpd.read_file(path)
            null = int(layer.geometry.isna().sum())
            invalid = int((~layer.geometry.is_valid).sum())
            crs = str(layer.crs) if layer.crs else None
            report["vectors"][name] = {
                "count": len(layer), "crs": crs, "null_geometry": null,
                "invalid_geometry": invalid,
            }
            if crs != "EPSG:4326" or null or invalid:
                report["errors"].append(f"{name}: unexpected CRS/null/invalid geometry")
        except Exception as exc:
            report["errors"].append(f"{name}: {exc}")
    for name, (relative, epsg) in RASTERS.items():
        path = root / relative
        if not path.exists():
            report["errors"].append(f"missing raster {name}: {path}")
            continue
        try:
            with rasterio.open(path) as ds:
                item = {
                    "crs": str(ds.crs), "width": ds.width, "height": ds.height,
                    "bands": ds.count, "nodata_declared": ds.nodata,
                }
                report["rasters"][name] = item
                if ds.crs is None or ds.crs.to_epsg() != epsg or ds.count != 1:
                    report["errors"].append(f"{name}: unexpected CRS/bands")
        except Exception as exc:
            report["errors"].append(f"{name}: {exc}")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(os.environ.get("PRAVAH_DATA_ROOT", "/mnt/c/PRAVAH_DATA")))
    p.add_argument("--out", type=Path, help="Optional validation JSON path")
    args = p.parse_args()
    report = validate(args.root)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
