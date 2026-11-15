#!/usr/bin/env python3
"""Stage verified benchmark imagery and real context for the live application."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare():
    base = ROOT / "outputs/local_demo"
    state = json.loads((base / "pipeline_status.json").read_text())
    if state["phase"] != "complete":
        raise RuntimeError("A completed training run is required")
    model_dir = Path(state["checkpoint_dir"])
    evaluation = json.loads((model_dir / "final_evaluation.json").read_text())
    checkpoint = model_dir / "checkpoint_best.pt"
    if sha(checkpoint) != evaluation["checkpoint_sha256"]:
        raise ValueError("Trained checkpoint integrity failed")
    selection = json.loads((base / "unet_demo/selection.json").read_text())
    records = json.loads((base / "data/pairs.json").read_text())
    # Same first alphabetical official-test examples used by the completed run.
    selected = [
        next(
            r
            for r in sorted(records, key=lambda x: x["scene_id"])
            if r["event_id"] == event and r["split"] == "test"
        )
        for event in ["India", "Spain", "USA"]
    ]
    metadata_path = ROOT / "data/raw/sen1floods11/v1.1/Sen1Floods11_Metadata.geojson"
    dates = {
        f["properties"]["location"]: f["properties"]["s2_date"]
        for f in json.loads(metadata_path.read_text())["features"]
    }
    target = ROOT / "data/processed/live_app"
    scenes, references = target / "scenes", target / "references"
    scenes.mkdir(parents=True, exist_ok=True)
    references.mkdir(parents=True, exist_ok=True)
    catalog = {}
    for row in selected:
        if sha(row["source_image_path"]) != row["source_image_sha256"]:
            raise ValueError("Source imagery integrity failed")
        if sha(row["source_label_path"]) != row["source_label_sha256"]:
            raise ValueError("Source label integrity failed")
        image = scenes / (row["scene_id"] + ".tif")
        label = references / (row["scene_id"] + ".tif")
        with rasterio.open(row["image_path"]) as src:
            values = src.read()
            valid = (src.read_masks() != 0).all(0) & (values != 0).any(0)
            if src.count != 6 or not np.isfinite(values).all():
                raise ValueError("Invalid six-band input")
            profile = {**src.profile, "driver": "GTiff", "compress": "deflate"}
            with rasterio.open(image, "w", **profile) as dst:
                dst.write(values)
                dst.write_mask(valid.astype("uint8") * 255)
                for i, band in enumerate(row["bands"], 1):
                    dst.set_band_description(i, band)
            bounds = list(src.bounds)
        shutil.copy2(row["label_path"], label)
        catalog[image.name] = {
            "scene_id": row["scene_id"],
            "country": row["event_id"],
            "dataset": "Sen1Floods11",
            "split": "official test split",
            "product": "Sentinel-2 L1C",
            "bands": row["bands"],
            "source_image_sha256": row["source_image_sha256"],
            "source_label_sha256": row["source_label_sha256"],
            "image_sha256": sha(image),
            "label_sha256": sha(label),
            "label_path": str(label),
            "bounds_wgs84": bounds,
            "acquired_date": dates[row["event_id"]].replace("/", "-"),
            "acquisition_precision": "day, from provider event metadata",
            "verified_available_at_utc": None,
            "analysis_mode": "retrospective benchmark",
            "source_url": "https://github.com/cloudtostreet/Sen1Floods11",
            "selection_record": selection,
        }
    save(target / "scene_catalog.json", catalog)
    context_path = ROOT / "data/processed/user_context/context_catalog.json"
    replay = {}
    if context_path.exists():
        context = json.loads(context_path.read_text())
        for item in context["source_catalogs"]:
            if "/sentinel/" not in item["path"]:
                continue
            source = Path(item["path"])
            if sha(source) != item["sha256"]:
                raise ValueError("Context catalog integrity failed")
            for feature in json.loads(source.read_text()).get("features", []):
                props = feature["properties"]
                replay[feature["id"]] = {
                    "asset_id": feature["id"],
                    "platform": props.get("platform"),
                    "product_type": feature.get("collection"),
                    "acquired_at_utc": props.get("datetime"),
                    "available_at_utc": None,
                    "cloud_cover_percent": props.get("eo:cloud_cover"),
                    "provider": "Copernicus STAC; user-supplied archived response",
                    "description": "Catalog metadata only; imagery not downloaded; historical availability unverified",
                    "download_status": "catalog_only",
                    "source_sha256": item["sha256"],
                }
    save(target / "replay_assets.json", list(replay.values()))
    report = {
        "status": "prepared",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "evaluation": str(model_dir / "final_evaluation.json"),
        "scenes_dir": str(scenes),
        "catalog": str(target / "scene_catalog.json"),
        "outputs_dir": str(ROOT / "outputs/live_app/runs"),
        "replay_manifest": str(target / "replay_assets.json"),
        "scene_count": len(catalog),
        "replay_catalog_entries": len(replay),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save(ROOT / "outputs/live_app/config.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
