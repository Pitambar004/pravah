#!/usr/bin/env python3
"""Agreement between two models' water maps of the SAME scene (e.g. MiniUNet vs Prithvi).

    python scripts/compare_models.py --run outputs/unet_rasuwa_20260827_T45RUL \
        --other outputs/p3_prithvi_nepal_20260827_T45RUL

Both folders need ``observed_water.tif`` (0/1/255) on the identical grid. Writes
``model_agreement.json`` into ``--run``. This is inter-model AGREEMENT, not accuracy:
neither map is ground truth, and there are no independent hand labels for Rasuwa.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import rasterio

from pravah.analysis.prediction_validation import score_against_reference


def compare(run: Path, other: Path) -> dict:
    with rasterio.open(run / "observed_water.tif") as a, rasterio.open(other / "observed_water.tif") as b:
        if a.crs != b.crs or a.transform != b.transform or a.shape != b.shape:
            raise ValueError("Grids differ (CRS/transform/shape); run both models on the same prepared raster")
        res, _ = score_against_reference(a.read(1), b.read(1))
    c, m = res["counts"], res["metrics"]
    meta = json.loads((run / "observed_metadata.json").read_text()) if (run / "observed_metadata.json").exists() else {}
    out = {
        "analysis": "inter_model_agreement",
        "run_id": run.name, "other_run_id": other.name, "scene_id": meta.get("scene_id"),
        "pixels_compared": c["valid"],
        "both_water": c["tp"], "only_this_model": c["fp"], "only_other_model": c["fn"], "both_dry": c["tn"],
        "agreement_iou": m["iou"],
        "share_of_other_water_also_found": m["recall"],
        "share_of_this_water_also_in_other": m["precision"],
        "note": "Agreement between two models on cloud-free pixels valid in both; not accuracy or ground truth.",
    }
    (run / "model_agreement.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--other", required=True, type=Path)
    a = p.parse_args(argv)
    try:
        print(json.dumps(compare(a.run, a.other), indent=2))
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
