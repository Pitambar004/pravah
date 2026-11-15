#!/usr/bin/env python3
"""Turn a P3 observation bundle into a dashboard run folder.

A bundle (written by ``pravah.gis.observation.write_observation_bundle`` on
Fuzzball) holds ``observed_water.tif`` (0/1/255), ``invalid_mask.tif``,
``water_probability.tif`` and ``observed_metadata.json``. This copies them
unchanged (checksums preserved), adds ``observed_water.geojson`` for the map and
a short ``model`` label, and never alters the class raster.

    python scripts/import_observation_bundle.py BUNDLE_DIR --outputs outputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import rasterio

from pravah.inference.engine import water_polygons

REQUIRED = ("observed_water.tif", "observed_metadata.json")
OPTIONAL = ("invalid_mask.tif", "water_probability.tif", "observation_validation.json")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def import_bundle(bundle: Path, outputs: Path, max_polygons: int = 5000) -> Path:
    for f in REQUIRED:
        if not (bundle / f).is_file():
            raise FileNotFoundError(f"{bundle} is missing {f}")
    md = json.loads((bundle / "observed_metadata.json").read_text())
    if md.get("is_synthetic"):
        raise ValueError("refusing to import a synthetic bundle as a run")
    run_id = md.get("run_id") or bundle.name
    dest = outputs / run_id
    dest.mkdir(parents=True, exist_ok=True)
    for f in REQUIRED + OPTIONAL:
        if (bundle / f).is_file():
            shutil.copy2(bundle / f, dest / f)
    recorded = (md.get("output_sha256") or {}).get("observed_water.tif")
    if recorded and recorded != sha256(dest / "observed_water.tif"):
        raise ValueError("observed_water.tif checksum differs from the bundle's recorded output_sha256")
    with rasterio.open(dest / "observed_water.tif") as src:
        cls, crs, tr = src.read(1), src.crs, src.transform
    gj, total = water_polygons(cls, tr, crs, max_polygons=max_polygons)
    (dest / "observed_water.geojson").write_text(json.dumps(gj) + "\n")
    md.setdefault("model", "Prithvi-EO-2.0-300M-TL-Sen1Floods11 (IBM/NASA pretrained)"
                  if "Prithvi" in str(md.get("model_id")) else md.get("model_id"))
    md["dashboard_import"] = {"source_bundle": str(bundle), "water_polygon_count": total,
                              "water_polygons_written": len(gj["features"]),
                              "observed_water_sha256_verified": bool(recorded)}
    (dest / "observed_metadata.json").write_text(json.dumps(md, indent=2) + "\n")
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundles", nargs="+", type=Path)
    p.add_argument("--outputs", type=Path, default=Path("outputs"))
    p.add_argument("--max-polygons", type=int, default=5000)
    a = p.parse_args(argv)
    for b in a.bundles:
        try:
            d = import_bundle(b, a.outputs, a.max_polygons)
        except (OSError, ValueError) as exc:
            print(f"ERROR {b}: {exc}", file=sys.stderr)
            return 2
        print(f"imported {b.name} -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
