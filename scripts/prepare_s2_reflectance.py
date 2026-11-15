#!/usr/bin/env python3
"""Rebuild a 6-band Sentinel-2 L1C TOA-reflectance GeoTIFF on a reference run's grid.

Reproduces the recipe recorded in a P3 Prithvi run's ``observed_metadata.json``
(schema ``p3-s2-l1c-input-v1``) so another model can be run on the *same* scene
and grid locally:

* Source: original L1C ``.jp2`` band files. Every band's SHA-256 must equal the
  reference run's ``expected_band_sha256``; otherwise the script refuses.
* Radiometry: reflectance = (DN + RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE, both
  read from ``MTD_MSIL1C.xml`` and checked against the reference metadata.
* Grid: nearest resampling of 10 m / 20 m bands onto the reference
  ``observed_water.tif`` grid (CRS, transform, shape).
* Invalid (nodata -9999 in all bands): DN == 0, or flagged opaque cloud, cirrus
  or snow/ice in ``MSK_CLASSI_B00.jp2`` (same classes the reference excluded).

Output band order is [B2, B3, B4, B8A, B11, B12], units dimensionless TOA
reflectance (use ``--input-units reflectance`` for inference). This is optical
Sentinel-2 only; it is not a Sentinel-1 SAR product. The output is a local
re-preparation following the recorded recipe, not the byte-identical Fuzzball file.

    python scripts/prepare_s2_reflectance.py --safe-dir data/raw/sentinel2/<PRODUCT> \
        --reference outputs/p3_prithvi_nepal_20260827_T45RUL --output data/raw/sentinel2/rasuwa_20260827_T45RUL.tif
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

BANDS = ["B2", "B3", "B4", "B8A", "B11", "B12"]
FILE_KEYS = {"B2": "B02", "B3": "B03", "B4": "B04", "B8A": "B8A", "B11": "B11", "B12": "B12"}
# Sentinel-2 band_id order in MTD_MSIL1C.xml
BAND_IDS = {"B2": 1, "B3": 2, "B4": 3, "B8A": 8, "B11": 11, "B12": 12}
NODATA = -9999.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_radiometry(mtd: Path) -> tuple[float, dict[str, float], str]:
    text = mtd.read_text()
    q = re.search(r"<QUANTIFICATION_VALUE[^>]*>([^<]+)<", text)
    if not q:
        raise ValueError("QUANTIFICATION_VALUE missing from MTD_MSIL1C.xml")
    offsets = {int(i): float(v) for i, v in re.findall(r'<RADIO_ADD_OFFSET band_id="(\d+)">([^<]+)<', text)}
    baseline = re.search(r"<PROCESSING_BASELINE>([^<]+)<", text)
    return float(q.group(1)), {b: offsets.get(BAND_IDS[b], 0.0) for b in BANDS}, (baseline.group(1) if baseline else "")


def prepare(safe_dir: Path, reference: Path, output: Path) -> dict:
    meta = json.loads((reference / "observed_metadata.json").read_text())
    if meta.get("is_synthetic"):
        raise ValueError("Reference run is marked synthetic; refusing")
    expected = meta.get("expected_band_sha256") or {}
    band_files: dict[str, Path] = {}
    checks = {}
    for band in BANDS:
        key = FILE_KEYS[band]
        matches = sorted(safe_dir.glob(f"*_{key}.jp2"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one *_{key}.jp2 in {safe_dir}")
        got = sha256(matches[0])
        if expected.get(key) and got != expected[key]:
            raise ValueError(f"{key} checksum {got} != reference {expected[key]}")
        checks[key] = {"file": matches[0].name, "sha256": got, "matches_reference": got == expected.get(key)}
        band_files[band] = matches[0]
    if not all(c["matches_reference"] for c in checks.values()):
        raise ValueError("Reference run does not record checksums for every band; provenance unverifiable")

    quant, offsets, baseline = read_radiometry(safe_dir / "MTD_MSIL1C.xml")
    recorded = meta.get("radiometry") or {}
    if recorded:
        if float(recorded.get("quantification_value", quant)) != quant:
            raise ValueError("QUANTIFICATION_VALUE differs from the reference run")
        for band, off in (recorded.get("radio_add_offset") or {}).items():
            if band in offsets and float(off) != offsets[band]:
                raise ValueError(f"RADIO_ADD_OFFSET for {band} differs from the reference run")

    with rasterio.open(reference / "observed_water.tif") as ref:
        crs, transform, (h, w) = ref.crs, ref.transform, ref.shape

    stack = np.empty((len(BANDS), h, w), dtype=np.float32)
    invalid = np.zeros((h, w), dtype=bool)
    for i, band in enumerate(BANDS):
        dn = np.zeros((h, w), dtype=np.uint16)
        with rasterio.open(band_files[band]) as src:
            reproject(rasterio.band(src, 1), dn, dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.nearest, src_nodata=None, dst_nodata=None)
        invalid |= dn == 0
        stack[i] = (dn.astype(np.float32) + offsets[band]) / quant

    mask_path = safe_dir / "MSK_CLASSI_B00.jp2"
    if mask_path.exists():
        with rasterio.open(mask_path) as src:
            for b in range(1, src.count + 1):  # opaque cloud, cirrus, snow/ice
                flag = np.zeros((h, w), dtype=np.uint8)
                reproject(rasterio.band(src, b), flag, dst_transform=transform, dst_crs=crs,
                          resampling=Resampling.nearest, src_nodata=None, dst_nodata=None)
                invalid |= flag > 0
        mask_used = "MSK_CLASSI_B00 opaque_cloud|cirrus_cloud|snow_ice"
    else:
        mask_used = "none (MSK_CLASSI_B00.jp2 not found)"

    stack[:, invalid] = NODATA
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "height": h, "width": w, "count": len(BANDS), "dtype": "float32",
               "crs": crs, "transform": transform, "nodata": NODATA, "compress": "deflate",
               "tiled": True, "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER"}
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(stack)
        for i, band in enumerate(BANDS, start=1):
            dst.set_band_description(i, band)

    report = {
        "schema_version": "pravah-s2-l1c-reflectance-local-v1",
        "scene_id": meta.get("scene_id"),
        "reference_run": meta.get("run_id"),
        "acquired_at_utc": meta.get("acquired_at_utc"),
        "available_at_utc": meta.get("available_at_utc"),
        "sensor": "S2", "processing_level": "L1C", "processing_baseline": baseline,
        "bands": BANDS, "units": "dimensionless_TOA_reflectance",
        "scaling": "(DN + RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE",
        "quantification_value": quant, "radio_add_offset": offsets,
        "resampling": "nearest onto reference observed_water.tif grid",
        "crs": str(crs), "transform": list(transform)[:6], "shape": [h, w],
        "nodata": NODATA, "invalid_rule": "DN == 0 or " + mask_used,
        "invalid_pixels": int(invalid.sum()), "valid_pixels": int((~invalid).sum()),
        "source_band_checks": checks, "output_sha256": sha256(output),
        "is_synthetic": False,
        "note": "Local re-preparation from checksum-verified source bands following the reference run's "
                "recorded recipe; not the byte-identical Fuzzball-prepared raster.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.with_suffix(".prep.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--safe-dir", type=Path, required=True, help="Folder with the L1C *_B??.jp2, MTD_MSIL1C.xml, MSK_CLASSI_B00.jp2")
    p.add_argument("--reference", type=Path, required=True, help="Reference run folder (observed_metadata.json + observed_water.tif)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    report = prepare(args.safe_dir, args.reference, args.output)
    print(json.dumps({k: report[k] for k in ("scene_id", "shape", "valid_pixels", "invalid_pixels", "output_sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
