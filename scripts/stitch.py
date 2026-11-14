#!/usr/bin/env python3
"""Stitch valid probability tiles from one/two workers to GIS-valid water mask."""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--worker-dir", action="append", required=True, help="Repeat for separate worker dirs")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold out of range")
    doc = json.loads(Path(args.manifest).read_text())
    shape = (doc["raster_height"], doc["raster_width"])
    weighted = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.float64)
    observed = []
    for tile in doc["tiles"]:
        matches = [Path(folder) / f'{tile["tile_id"]}.npz' for folder in args.worker_dir]
        matches = [p for p in matches if p.exists()]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one result for {tile["tile_id"]}, found {len(matches)}')
        with np.load(matches[0], allow_pickle=False) as npz:
            prob, valid = npz["probability"], npz["valid"].astype(bool)
        if prob.shape != (tile["height"], tile["width"]) or valid.shape != prob.shape:
            raise ValueError(f'Bad tile shape {tile["tile_id"]}')
        if not np.isfinite(prob[valid]).all() or not np.logical_and(prob[valid] >= 0, prob[valid] <= 1).all():
            raise ValueError("Invalid probabilities")
        row, col = tile["row_off"], tile["col_off"]
        ys = slice(row, row + tile["height"])
        xs = slice(col, col + tile["width"])
        weighted[ys, xs] += np.where(valid, prob, 0)
        counts[ys, xs] += valid.astype(np.float64)
        observed.append(tile["tile_id"])
    if len(observed) != len(set(observed)):
        raise ValueError("Duplicate tile IDs")
    valid = counts > 0
    pwater = np.full(shape, np.nan, dtype=np.float32)
    pwater[valid] = (weighted[valid] / counts[valid]).astype(np.float32)
    cls = np.full(shape, 255, dtype=np.uint8)
    cls[valid] = (pwater[valid] >= args.threshold).astype(np.uint8)
    invalid = (~valid).astype(np.uint8)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with rasterio.open(doc["raster"]) as src:
        base = src.profile.copy()
        # A strip-organized source often carries blockysize=rows, which is
        # invalid for tiled GeoTIFFs unless it is a multiple of 16.
        base.pop("blockxsize", None)
        base.pop("blockysize", None)
        base.update(count=1, compress="deflate", tiled=True,
                    blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
        with rasterio.open(output / "observed_water.tif", "w", **{**base, "dtype": "uint8", "nodata": 255}) as dst:
            dst.write(cls, 1)
        with rasterio.open(output / "invalid_mask.tif", "w", **{**base, "dtype": "uint8", "nodata": None}) as dst:
            dst.write(invalid, 1)
        with rasterio.open(output / "water_probability.tif", "w", **{**base, "dtype": "float32", "nodata": -9999.0}) as dst:
            dst.write(np.where(valid, pwater, -9999), 1)
    metadata = {"run_id": doc["run_id"], "threshold": args.threshold,
                "source_raster": doc["raster"], "tile_manifest_sha256": doc["tile_manifest_sha256"],
                "valid_pixel_count": int(valid.sum()), "invalid_pixel_count": int(invalid.sum()),
                "class_codes": {"0": "no_water", "1": "model_inferred_water", "255": "invalid"},
                "science_note": "Model-derived water, not calibrated flood forecast or structural damage"}
    (output / "observed_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
