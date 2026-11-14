#!/usr/bin/env python3
"""Polygonize a real model water mask and compute potential mapped exposure."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely import make_valid

# Keep the CLI usable directly from a clean checkout, where ``src`` may not
# have been installed as an editable package yet.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pravah.gis.exposure import summarize_exposure, validate_model_mask


def _polygonize(mask: np.ndarray, transform: Any, crs: Any) -> gpd.GeoDataFrame:
    geometries = [
        make_valid(shape(geometry))
        for geometry, value in shapes(mask.astype("uint8"), mask=mask.astype(bool), transform=transform)
        if value == 1
    ]
    return gpd.GeoDataFrame({"geometry": geometries}, geometry="geometry", crs=crs)


def _read_vector(path: str | None) -> gpd.GeoDataFrame | None:
    return gpd.read_file(path) if path else None


def _write_geojson(frame: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        path.write_text(json.dumps({"type": "FeatureCollection", "features": []}) + "\n", encoding="utf-8")
    else:
        frame.to_file(path, driver="GeoJSON")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", required=True, help="GeoTIFF: 0=no water, 1=inferred water, 255=invalid")
    parser.add_argument("--buildings")
    parser.add_argument("--roads")
    parser.add_argument("--bridges")
    parser.add_argument("--settlements")
    parser.add_argument("--schools")
    parser.add_argument("--healthcare")
    parser.add_argument("--emergency-facilities")
    parser.add_argument("--aoi", help="Authoritative or explicitly provisional AOI vector")
    parser.add_argument("--osm-date", default="unknown", help="OSM snapshot date")
    parser.add_argument("--out", required=True, help="exposure_summary.json output")
    parser.add_argument("--polygons-out", help="water GeoJSON; defaults to exposure.geojson beside --out")
    parser.add_argument("--metadata-out", help="metadata JSON; defaults to exposure_metadata.json beside --out")
    parser.add_argument("--projected-crs", default="EPSG:32645")
    args = parser.parse_args()

    with rasterio.open(args.mask) as source:
        if source.count != 1 or source.crs is None:
            raise ValueError("model mask must be a one-band georeferenced raster")
        values = source.read(1)
        counts = validate_model_mask(values)
        water = _polygonize(values == 1, source.transform, source.crs)
        coverage = _polygonize(np.isin(values, (0, 1)), source.transform, source.crs)
        mask_crs = source.crs
        mask_transform = source.transform
        mask_shape = [source.height, source.width]
        mask_bounds = [float(source.bounds.left), float(source.bounds.bottom), float(source.bounds.right), float(source.bounds.top)]
    if counts["valid_pixels"] == 0:
        raise ValueError("model mask contains no valid observation pixels")

    vectors = {name: _read_vector(path) for name, path in {
        "buildings": args.buildings, "roads": args.roads, "bridges": args.bridges,
        "settlements": args.settlements, "schools": args.schools,
        "healthcare": args.healthcare, "emergency_facilities": args.emergency_facilities,
    }.items()}
    aoi = _read_vector(args.aoi)
    summary = summarize_exposure(
        water, vectors["buildings"], vectors["roads"], vectors["bridges"],
        projected_crs=args.projected_crs, coverage=coverage, aoi=aoi,
        settlements=vectors["settlements"], schools=vectors["schools"],
        healthcare=vectors["healthcare"], emergency_facilities=vectors["emergency_facilities"],
    )
    output = Path(args.out)
    polygons = Path(args.polygons_out) if args.polygons_out else output.parent / "exposure.geojson"
    metadata_path = Path(args.metadata_out) if args.metadata_out else output.parent / "exposure_metadata.json"
    metadata = {
        "schema_version": "p3-exposure-metadata-v2",
        "mask_artifact": str(Path(args.mask)),
        "mask_classes": {"0": "inferred_non_water", "1": "inferred_water", "255": "invalid_no_observation"},
        "mask_crs": str(mask_crs),
        "mask_transform": [float(value) for value in mask_transform[:6]],
        "mask_shape": mask_shape,
        "mask_bounds": mask_bounds,
        "mask_counts": counts,
        "metric_crs": args.projected_crs,
        "osm_snapshot_timestamp": args.osm_date,
        "aoi_artifact": args.aoi,
        "analysis_mode": "retrospective_model_output",
        "inference_is_not_damage": True,
        "scientific_caveats": [
            "A model-derived water intersection is potential mapped exposure, not confirmed damage.",
            "No exposure is reported outside valid (0/1) mask coverage.",
            "A real P2 model run and its acquisition/inference provenance are required for event claims.",
        ],
    }
    summary.update({
        "source_type": "model_inferred_water",
        "mask_artifact": str(Path(args.mask)),
        "osm_snapshot_timestamp": args.osm_date,
        "valid_pixel_count": counts["valid_pixels"],
        "water_pixel_count": counts["water_pixels"],
        "invalid_pixel_count": counts["invalid_pixels"],
        "inference_is_not_damage": True,
        "aoi_artifact": args.aoi,
        "metadata_artifact": str(metadata_path),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_geojson(water, polygons)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
