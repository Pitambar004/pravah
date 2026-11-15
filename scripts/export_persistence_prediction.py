#!/usr/bin/env python3
"""Export a retrospective persistence baseline using only audited pre-cutoff imagery.

This is not a trained inundation forecast or a historically issued warning.
No reference/post-event raster is accepted by this exporter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Timestamps must include a UTC offset")
    return dt.astimezone(timezone.utc)


def export(pre_run: Path, output: Path, cutoff: str) -> dict:
    limit = utc(cutoff)
    metadata_path = pre_run / "observed_metadata.json"
    m = json.loads(metadata_path.read_text())
    if m.get("is_synthetic") is not False:
        raise ValueError("Source must explicitly be non-synthetic; fixtures cannot be exported as evidence")
    if m.get("source_temporal_role") != "pre_event":
        raise ValueError("Source must be audited as pre_event")
    for key in ("acquired_at_utc", "available_at_utc"):
        if not m.get(key) or utc(m[key]) > limit:
            raise ValueError(f"{key} is unknown or after cutoff")
    if utc(m["available_at_utc"]) < utc(m["acquired_at_utc"]):
        raise ValueError("Availability precedes acquisition")
    if not all(m.get(k) for k in ("availability_basis", "catalog_uri", "scene_id")):
        raise ValueError("Verified availability basis, catalog URI and scene ID are required")
    source = pre_run / "observed_water.tif"
    digest = sha256(source)
    if digest != (m.get("output_sha256") or {}).get(source.name):
        raise ValueError("Source raster does not match its audited output checksum")
    with rasterio.open(source) as src:
        if src.count != 1 or src.crs is None or src.nodata != 255:
            raise ValueError("Source needs one band, a CRS and nodata=255")
        arr, profile = src.read(1), src.profile.copy()
    if not np.isin(arr, (0, 1, 255)).all() or not np.any(arr != 255):
        raise ValueError("Source must contain valid water classes 0/1 and optional nodata 255")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = {
        "run_id": output.name,
        "model_id": "Water persistence baseline v1 (no trained forecast model)",
        "prediction_kind": "persistence_baseline",
        "evaluation_mode": "retrospective_reconstruction",
        "description": "Assume the pre-event water extent stays unchanged; predicts no new inundation.",
        "input_cutoff_utc": cutoff,
        "issued_at_utc": now,
        "created_at_utc": now,
        "is_synthetic": False,
        "inputs": [{
            "name": m["scene_id"], "type": "Sentinel-2 optical imagery",
            "scene_id": m["scene_id"], "source": m["catalog_uri"],
            "acquired_at_utc": m["acquired_at_utc"], "available_at_utc": m["available_at_utc"],
            "availability_basis": m["availability_basis"],
        }],
        "derived_from": {"run_id": pre_run.name, "raster_sha256": digest,
                         "metadata_sha256": sha256(metadata_path),
                         "segmentation_model": m.get("model_id"),
                         "segmentation_checkpoint_sha256": m.get("model_sha256")},
        "exporter_sha256": sha256(Path(__file__)),
        "crs": str(profile["crs"]), "transform": list(profile["transform"])[:6],
        "shape": list(arr.shape), "nodata": 255,
        "classes": {"0": "predicted no water", "1": "persisted pre-event water", "255": "no prediction"},
        "limitations": [
            "Retrospective reconstruction generated now, not a forecast issued before the event.",
            "Cutoff eligibility covers source imagery; the derived segmentation and this exporter were created later.",
            "Persistence predicts no new inundation and is not evidence of trained forecasting skill.",
            "Source domain is provisional; no independent pre-event AOI provenance is established.",
            "Cloud/nodata remain unknown. Optical water is not a debris-flow or damage label.",
        ],
    }
    # Never replace a previous run or leave metadata pointing at an unwritten raster.
    output.mkdir(parents=True, exist_ok=False)
    with rasterio.open(output / "predicted_flood.tif", "w", **profile) as dst:
        dst.write(arr, 1)
    metadata["output_sha256"] = {"predicted_flood.tif": sha256(output / "predicted_flood.tif")}
    (output / "prediction_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cutoff", required=True)
    args = parser.parse_args()
    try:
        result = export(args.pre_run, args.output, args.cutoff)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps({"run_id": result["run_id"], "kind": result["prediction_kind"],
                      "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
