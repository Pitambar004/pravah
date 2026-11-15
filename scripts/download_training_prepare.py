#!/usr/bin/env python3
"""Audit downloaded flood datasets and prepare optical U-Net pairs on Fuzzball.

WorldFloods original rasters remain immutable. Six-band 512-pixel VRTs reference
their original pixels, while small GeoTIFF masks encode -1/0/1. No resampling or
per-image normalization occurs. Weak-label training is opt-in via its own manifest.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window, bounds, transform

S2_BANDS = ("B2", "B3", "B4", "B8A", "B11", "B12")
S2_INDEXES = (2, 3, 4, 9, 12, 13)  # rasterio is one-based
GDAL_TYPES = {
    "uint8": "Byte",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "float32": "Float32",
    "float64": "Float64",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def overlapping(box, protected, epsilon=1e-9):
    """Positive-area bbox intersection; shared tile edges do not count."""
    if not len(protected):
        return False
    other = np.asarray(protected)
    width = np.minimum(box[2], other[:, 2]) - np.maximum(box[0], other[:, 0])
    height = np.minimum(box[3], other[:, 3]) - np.maximum(box[1], other[:, 1])
    return bool(np.any((width > epsilon) & (height > epsilon)))


def eligible_combined_pair(
    pair, protected, test_boxes, world_heldout_events, world_test_events
):
    """Retain source splits while removing geography and known event leakage."""
    if pair["dataset"] == "worldfloods_v2":
        if pair["split"] == "train" and pair["event_id"] in world_heldout_events:
            return False
        if pair["split"] == "val" and pair["event_id"] in world_test_events:
            return False
    exclusions = (
        protected
        if pair["split"] == "train"
        else test_boxes
        if pair["split"] == "val"
        else []
    )
    return not overlapping(pair["bounds_wgs84"], exclusions)


def worldfloods_label(raw):
    if raw.shape[0] != 2 or not np.isin(raw, (0, 1, 2)).all():
        raise ValueError("Expected WorldFloods v2 clear/cloud and land/water bands")
    result = np.full(raw.shape[1:], -1, dtype=np.int16)
    clear = raw[0] == 1
    result[clear & (raw[1] == 1)] = 0
    result[clear & (raw[1] == 2)] = 1
    return result


def check_grid(image, label, dataset):
    expected_counts = (13, 15) if dataset == "worldfloods_v2" else (13,)
    if image.count not in expected_counts:
        raise ValueError(f"Unexpected optical band count: {image.count}")
    if image.crs is None or label.crs is None:
        raise ValueError("Missing CRS")
    if image.crs != label.crs:
        raise ValueError("Image/label grid mismatch")
    if (image.width, image.height) != (label.width, label.height):
        raise ValueError("Image/label dimensions differ")
    if grid_displacement_pixels(image, label) > 1e-6:
        raise ValueError("Image/label grid mismatch")
    expected_label_count = 2 if dataset == "worldfloods_v2" else 1
    if label.count != expected_label_count:
        raise ValueError("Unexpected label band count")
    for index, name in zip(S2_INDEXES, S2_BANDS, strict=True):
        if image.descriptions[index - 1] not in (None, name):
            raise ValueError("Input band descriptions disagree with provider order")


def grid_displacement_pixels(image, label):
    """Measure all four grid corners in image pixels, permitting numeric roundoff."""
    inverse = ~image.transform
    source = label.transform
    differences = []
    for x, y in (
        (0, 0),
        (image.width, 0),
        (0, image.height),
        (image.width, image.height),
    ):
        # Explicit affine arithmetic also supports affine 2.x in older CPU envs.
        world_x = source.a * x + source.b * y + source.c
        world_y = source.d * x + source.e * y + source.f
        mapped_x = inverse.a * world_x + inverse.b * world_y + inverse.c
        mapped_y = inverse.d * world_x + inverse.e * world_y + inverse.f
        differences.extend((abs(mapped_x - x), abs(mapped_y - y)))
    return max(differences)


def write_vrt(path, image, window, tile_size):
    """Use source pixel windows verbatim; zero-pad edge tiles without resampling."""
    root = ET.Element(
        "VRTDataset", rasterXSize=str(tile_size), rasterYSize=str(tile_size)
    )
    ET.SubElement(root, "SRS").text = image.crs.to_wkt()
    affine = transform(window, image.transform)
    ET.SubElement(root, "GeoTransform").text = ",".join(map(str, affine.to_gdal()))
    for output_index, source_index in enumerate(S2_INDEXES, 1):
        band = ET.SubElement(
            root,
            "VRTRasterBand",
            dataType=GDAL_TYPES[image.dtypes[source_index - 1]],
            band=str(output_index),
        )
        ET.SubElement(band, "Description").text = S2_BANDS[output_index - 1]
        source = ET.SubElement(band, "SimpleSource")
        ET.SubElement(source, "SourceFilename", relativeToVRT="0").text = str(
            Path(image.name).resolve()
        )
        ET.SubElement(source, "SourceBand").text = str(source_index)
        ET.SubElement(
            source,
            "SrcRect",
            xOff=str(int(window.col_off)),
            yOff=str(int(window.row_off)),
            xSize=str(int(window.width)),
            ySize=str(int(window.height)),
        )
        ET.SubElement(
            source,
            "DstRect",
            xOff="0",
            yOff="0",
            xSize=str(int(window.width)),
            ySize=str(int(window.height)),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))


def geographic_bounds(image, window=None):
    box = image.bounds if window is None else bounds(window, image.transform)
    result = transform_bounds(image.crs, "EPSG:4326", *box, densify_pts=21)
    if (
        not all(math.isfinite(v) for v in result)
        or result[0] >= result[2]
        or result[1] >= result[3]
    ):
        raise ValueError("Invalid/antimeridian bounds require separate handling")
    return result


def hand_sources(sen_root):
    base = sen_root / "data/flood_events/HandLabeled"
    mapping = {
        "train": "flood_train_data.csv",
        "val": "flood_valid_data.csv",
        "test": "flood_test_data.csv",
        "bolivia": "flood_bolivia_data.csv",
    }
    records = []
    seen = set()
    for split, filename in mapping.items():
        with (sen_root / "splits/flood_handlabeled" / filename).open() as stream:
            for row in csv.reader(stream):
                if not row:
                    continue
                source = Path(row[0]).name
                if not source.endswith("_S1Hand.tif"):
                    raise ValueError("Unrecognized official split row")
                chip = source.removesuffix("_S1Hand.tif")
                if chip in seen:
                    raise ValueError("Chip occurs in multiple official splits")
                seen.add(chip)
                records.append(
                    {
                        "image": base / "S2Hand" / (chip + "_S2Hand.tif"),
                        "label": base / "LabelHand" / (chip + "_LabelHand.tif"),
                        "split": split,
                        "scene_id": chip,
                        "event_id": chip.rsplit("_", 1)[0],
                        "dataset": "sen1floods11_hand",
                        "label_quality": "hand_labeled",
                    }
                )
    return records


def wf_sources(wf_root):
    records = []
    expected_encoding = [
        {"0": "invalid", "1": "clear", "2": "cloud"},
        {"0": "invalid", "1": "land", "2": "water"},
    ]
    for split in ("train", "val", "test"):
        for image in sorted((wf_root / split / "S2").glob("*.tif")):
            label = wf_root / split / "gt" / image.name
            meta_path = wf_root / split / "meta" / (image.stem + ".json")
            meta = json.loads(meta_path.read_text())
            if meta.get("encoding_values") != expected_encoding:
                raise ValueError("WorldFloods class metadata differs: " + image.stem)
            event = meta.get("ems_code")
            if event in (None, "NaN", "None", ""):
                event = meta.get("event id") or image.stem
            records.append(
                {
                    "image": image,
                    "label": label,
                    "split": split,
                    "scene_id": image.stem,
                    "event_id": str(event),
                    "acquired_at_utc": meta.get("s2_date"),
                    "dataset": "worldfloods_v2",
                    "label_quality": "provider_supervised_curated_and_derived",
                    "country": None
                    if meta.get("country") in (None, "NaN", "None")
                    else meta["country"],
                }
            )
    return records


def process_scene(
    record, output, tile_size, min_valid_fraction, protected, transfer_index
):
    image_path, label_path = record["image"], record["label"]
    tiles = []
    stats = {
        "dataset": record["dataset"],
        "scene_id": record["scene_id"],
        "split": record["split"],
        "kept": 0,
        "excluded_overlap": 0,
        "excluded_low_validity": 0,
    }
    with rasterio.open(image_path) as image, rasterio.open(label_path) as label:
        check_grid(image, label, record["dataset"])
        stats.update(
            {
                "crs": str(image.crs),
                "source_nodata": image.nodata,
                "label_nodata": label.nodata,
                "source_band_count": image.count,
                "source_dtype": image.dtypes[0],
                "source_shape": [image.height, image.width],
                "label_grid_max_displacement_pixels": grid_displacement_pixels(
                    image, label
                ),
                "bounds_wgs84": geographic_bounds(image),
            }
        )
        for y in range(0, image.height, tile_size):
            for x in range(0, image.width, tile_size):
                window = Window(
                    x,
                    y,
                    min(tile_size, image.width - x),
                    min(tile_size, image.height - y),
                )
                box = geographic_bounds(image, window)
                if record["split"] == "train" and overlapping(box, protected):
                    stats["excluded_overlap"] += 1
                    continue
                raw_label = label.read(window=window)
                if record["dataset"] == "worldfloods_v2":
                    mapped = worldfloods_label(raw_label)
                else:
                    if not np.isin(raw_label, (-1, 0, 1)).all():
                        raise ValueError("Unexpected Sen1Floods11 label values")
                    mapped = raw_label[0].astype(np.int16)
                pixels = image.read(S2_INDEXES, window=window)
                # Ignoring labels alone does not prevent NaNs spreading through
                # convolutions in the unchanged baseline loader/model.
                if not np.isfinite(pixels).all():
                    raise ValueError(
                        "Nonfinite image pixels are incompatible with the baseline loader"
                    )
                valid = (pixels != 0).any(axis=0)
                valid &= (image.read_masks(S2_INDEXES, window=window) != 0).all(axis=0)
                mapped[~valid] = -1
                valid_count = int((mapped != -1).sum())
                threshold = max(
                    1, math.ceil(min_valid_fraction * tile_size * tile_size)
                )
                if valid_count < threshold:
                    stats["excluded_low_validity"] += 1
                    continue
                padded = np.full((tile_size, tile_size), -1, dtype=np.int16)
                padded[: int(window.height), : int(window.width)] = mapped
                stem = f"{record['scene_id']}_y{y:06d}_x{x:06d}"
                directory = output / record["dataset"] / record["split"]
                image_out = directory / "images" / (stem + ".vrt")
                label_out = directory / "labels" / (stem + ".tif")
                label_out.parent.mkdir(parents=True, exist_ok=True)
                affine = transform(window, image.transform)
                with rasterio.open(
                    label_out,
                    "w",
                    driver="GTiff",
                    width=tile_size,
                    height=tile_size,
                    count=1,
                    dtype="int16",
                    crs=image.crs,
                    transform=affine,
                    nodata=-1,
                    compress="deflate",
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                ) as dest:
                    dest.write(padded, 1)
                    dest.set_band_description(
                        1, "-1 invalid/cloud; 0 no_water; 1 water"
                    )
                write_vrt(image_out, image, window, tile_size)
                # Verify VRT layout, band selection, padding and mask geometry.
                with (
                    rasterio.open(image_out) as reopened,
                    rasterio.open(label_out) as mask,
                ):
                    if (
                        reopened.count != 6
                        or reopened.transform != mask.transform
                        or reopened.crs != mask.crs
                    ):
                        raise ValueError("Prepared tile contract failure")
                    if stats["kept"] == 0:
                        actual = reopened.read(
                            window=Window(0, 0, window.width, window.height)
                        )
                        if not np.array_equal(actual, pixels, equal_nan=True):
                            raise ValueError("VRT values differ from source pixels")
                fields = {
                    k: v for k, v in record.items() if k not in ("image", "label")
                }
                tiles.append(
                    {
                        **fields,
                        "image_path": str(image_out),
                        "label_path": str(label_out),
                        "source_image_path": str(image_path),
                        "source_label_path": str(label_path),
                        "source_image_sha256": transfer_index[str(image_path)][
                            "sha256"
                        ],
                        "source_label_sha256": transfer_index[str(label_path)][
                            "sha256"
                        ],
                        "prepared_image_sha256": sha256(image_out),
                        "prepared_label_sha256": sha256(label_out),
                        "source_window": [x, y, int(window.width), int(window.height)],
                        "bands": list(S2_BANDS),
                        "crs": str(image.crs),
                        "bounds_wgs84": box,
                        "valid_pixels": valid_count,
                        "water_pixels": int((mapped == 1).sum()),
                        "available_at_utc": None,
                        "replay_eligibility": "retrospective_training_only",
                    }
                )
                stats["kept"] += 1
    return tiles, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data/training/expanded_20261114")
    parser.add_argument("--sen1-root", default="/data/raw/sen1floods11/v1.1")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--min-valid-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if args.tile_size != 512 or not 0 <= args.min_valid_fraction <= 1:
        parser.error(
            "This intake contract requires 512-pixel tiles and valid fraction in [0,1]"
        )
    root, sen_root = Path(args.root), Path(args.sen1_root)
    audit = root / "audit"
    summary = json.loads((audit / "download_summary.json").read_text())
    if summary["failed_assets"]:
        raise RuntimeError(
            "Resolve failed downloads before publishing training manifests"
        )
    index = {}
    with (audit / "transfers.jsonl").open() as stream:
        for line in stream:
            transfer = json.loads(line)
            if transfer["status"] != "failed":
                index[transfer["path"]] = transfer
    plan = json.loads((root / "download_plan.json").read_text())
    if len(index) != len(plan["assets"]):
        raise RuntimeError("Transfer audit does not cover the entire plan")
    hand = hand_sources(sen_root)
    sen_metadata = {
        feature["properties"]["location"]: feature["properties"]
        for feature in json.loads(
            (sen_root / "Sen1Floods11_Metadata.geojson").read_text()
        )["features"]
    }
    for record in hand:
        meta = sen_metadata.get(record["event_id"], {})
        record.update(
            {"acquired_date": meta.get("s2_date"), "country_iso3": meta.get("ISO_CC")}
        )
    world = wf_sources(root / "worldfloods_v2/raw")
    world_heldout_events = {
        r["event_id"] for r in world if r["split"] in ("val", "test")
    }
    world_test_events = {r["event_id"] for r in world if r["split"] == "test"}
    if len(hand) != 446 or len(world) != 509:
        raise RuntimeError(
            "Source pair inventory differs from reviewed provider release"
        )
    protected = []
    test_boxes = []
    protected_records = []
    for record in hand + world:
        if record["split"] not in ("val", "test", "bolivia"):
            continue
        with rasterio.open(record["image"]) as image:
            box = geographic_bounds(image)
        protected.append(box)
        if record["split"] in ("test", "bolivia"):
            test_boxes.append(box)
        protected_records.append(
            {
                "dataset": record["dataset"],
                "scene_id": record["scene_id"],
                "split": record["split"],
                "bounds_wgs84": box,
            }
        )
    # Reserve the complete observed Bolivia event region, including weak imagery,
    # so another provider's train scenes cannot fill gaps between held-out chips.
    bolivia_paths = [r["image"] for r in hand if r["split"] == "bolivia"]
    bolivia_paths += sorted(
        (sen_root / "data/flood_events/WeaklyLabeled/S2Weak").glob(
            "Bolivia_*_S2Weak.tif"
        )
    )
    bolivia_boxes = []
    for path in bolivia_paths:
        with rasterio.open(path) as image:
            bolivia_boxes.append(geographic_bounds(image))
    bolivia_region = [
        min(b[0] for b in bolivia_boxes),
        min(b[1] for b in bolivia_boxes),
        max(b[2] for b in bolivia_boxes),
        max(b[3] for b in bolivia_boxes),
    ]
    protected.append(bolivia_region)
    test_boxes.append(bolivia_region)
    protected_records.append(
        {
            "dataset": "sen1floods11_bolivia_event",
            "scene_id": "union_bbox_of_all_original_bolivia_images",
            "split": "bolivia",
            "bounds_wgs84": bolivia_region,
            "source_image_count": len(bolivia_paths),
        }
    )
    write_json(audit / "protected_geographies.json", protected_records)
    sets = {
        "sen1_hand_official": [],
        "bolivia_holdout": [],
        "worldfloods_official": [],
        "optical_supervised": [],
        "weak_train_only": [],
    }
    scene_stats, rejected = [], []

    def process(record, protected_boxes, min_fraction):
        try:
            pairs, stats = process_scene(
                record,
                root / "prepared",
                args.tile_size,
                min_fraction,
                protected_boxes,
                index,
            )
            scene_stats.append(stats)
            return pairs
        except (ValueError, OSError, rasterio.errors.RasterioError) as exc:
            rejected.append(
                {
                    "dataset": record["dataset"],
                    "scene_id": record["scene_id"],
                    "reason": str(exc),
                }
            )
            print("REJECTED", record["scene_id"], str(exc), flush=True)
            return []

    for record in hand:
        pairs = process(record, [], 0)
        if record["split"] == "bolivia":
            for pair in pairs:
                pair["split"] = "test"
            sets["bolivia_holdout"].extend(pairs)
        else:
            sets["sen1_hand_official"].extend(pairs)
            for pair in pairs:
                if eligible_combined_pair(
                    pair, protected, test_boxes, world_heldout_events, world_test_events
                ):
                    sets["optical_supervised"].append(pair)
    print(
        "HAND_PREPARED",
        dict(Counter(p["split"] for p in sets["sen1_hand_official"])),
        flush=True,
    )
    for number, record in enumerate(world, 1):
        pairs = process(record, [], args.min_valid_fraction)
        sets["worldfloods_official"].extend(pairs)
        for pair in pairs:
            if eligible_combined_pair(
                pair, protected, test_boxes, world_heldout_events, world_test_events
            ):
                sets["optical_supervised"].append(pair)
        if number % 10 == 0:
            print(
                "WORLDFLOODS_PREPARED",
                number,
                "/",
                len(world),
                "tiles",
                len(sets["worldfloods_official"]),
                flush=True,
            )
    weak_base = sen_root / "data/flood_events/WeaklyLabeled"
    weak_images = sorted((weak_base / "S2Weak").glob("*.tif"))
    weak_labels = sorted((weak_base / "S2IndexLabelWeak").glob("*.tif"))
    if len(weak_images) != 4384 or len(weak_labels) != 4385:
        raise RuntimeError("Weak-label inventory differs from pinned provider release")
    weak_ids = {p.name.removesuffix("_S2Weak.tif") for p in weak_images}
    unpaired_labels = [
        str(p)
        for p in weak_labels
        if p.name.removesuffix("_S2IndexLabelWeak.tif") not in weak_ids
    ]
    for number, image in enumerate(weak_images, 1):
        chip = image.name.removesuffix("_S2Weak.tif")
        event = chip.rsplit("_", 1)[0]
        if event == "Bolivia":
            rejected.append(
                {
                    "dataset": "sen1floods11_weak",
                    "scene_id": chip,
                    "reason": "Bolivia event reserved as independent holdout",
                }
            )
            continue
        label = weak_base / "S2IndexLabelWeak" / (chip + "_S2IndexLabelWeak.tif")
        record = {
            "image": image,
            "label": label,
            "split": "train",
            "scene_id": chip,
            "event_id": event,
            "dataset": "sen1floods11_weak",
            "label_quality": "automatic_optical_threshold",
            "acquired_date": sen_metadata.get(event, {}).get("s2_date"),
            "country_iso3": sen_metadata.get(event, {}).get("ISO_CC"),
        }
        sets["weak_train_only"].extend(
            process(record, protected, args.min_valid_fraction)
        )
        if number % 200 == 0:
            print(
                "WEAK_PREPARED",
                number,
                "tiles",
                len(sets["weak_train_only"]),
                flush=True,
            )
    sets["optical_with_weak"] = sets["optical_supervised"] + sets["weak_train_only"]
    # Every input path is unique. Validation/test locations are absent from train.
    for name, pairs in sets.items():
        paths = [p["image_path"] for p in pairs]
        if len(set(paths)) != len(paths):
            raise RuntimeError("Duplicate image in manifest: " + name)
        write_json(root / "manifests" / (name + ".json"), pairs)
    report = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "tile_size": args.tile_size,
        "minimum_valid_fraction_worldfloods_and_weak": args.min_valid_fraction,
        "band_order": list(S2_BANDS),
        "units": "L1C TOA reflectance scaled by 10000; existing U-Net loader divides by 10000 and clips [0,1]",
        "label_mapping": {
            "-1": "invalid, cloud or unobserved",
            "0": "no_water",
            "1": "water, including permanent water",
        },
        "worldfloods_source_order": "13 Sentinel-2 bands followed by QA60 and cloud probability in 15-band exports; only B2,B3,B4,B8A,B11,B12 selected",
        "worldfloods_processing_level_verification": {
            "level": "L1C",
            "source": "https://www.nature.com/articles/s41598-023-47595-7",
            "note": "Verified after the immutable download plan was created",
        },
        "grid_policy": "Identical CRS/dimensions; at most 1e-6 image-pixel displacement at all four corners for provider numeric rounding; no resampling",
        "combined_split_policy": "Published split labels retained; remove WorldFloods train examples from any held-out EMS event, and WorldFloods validation examples from test EMS events; remove train tiles intersecting any validation/test/Bolivia bbox and validation tiles intersecting any test/Bolivia bbox; protect the union bbox of all Bolivia hand and weak images. Official Sen1 hand protocol retains within-event splits; complete cross-provider event independence is not claimed.",
        "worldfloods_heldout_event_ids": sorted(world_heldout_events),
        "bolivia_protected_region_wgs84": bolivia_region,
        "weak_policy": "Opt-in only; threshold-derived labels, train only, Bolivia withheld, protected geography excluded",
        "sar_policy": "Downloaded as separate original SAR resources; incompatible with the optical U-Net loader",
        "license_notes": plan["datasets"],
        "manifests": {
            name: {
                "path": str(root / "manifests" / (name + ".json")),
                "sha256": sha256(root / "manifests" / (name + ".json")),
                "pairs": len(pairs),
                "splits": dict(Counter(p["split"] for p in pairs)),
                "events": dict(Counter(p["event_id"] for p in pairs)),
                "datasets": dict(Counter(p["dataset"] for p in pairs)),
                "dataset_splits": {
                    dataset: dict(
                        Counter(p["split"] for p in pairs if p["dataset"] == dataset)
                    )
                    for dataset in sorted({p["dataset"] for p in pairs})
                },
                "water_pixels": sum(p["water_pixels"] for p in pairs),
                "valid_pixels": sum(p["valid_pixels"] for p in pairs),
            }
            for name, pairs in sets.items()
        },
        "rejected_scene_count": len(rejected),
        "unpaired_weak_labels": unpaired_labels,
        "source_pair_inventory": {
            "sen1_hand": len(hand),
            "worldfloods": len(world),
            "sen1_weak_images": len(weak_images),
        },
        "training_run_performed": False,
    }
    write_json(audit / "scene_qa.json", scene_stats)
    write_json(audit / "rejected_scenes.json", rejected)
    write_json(audit / "training_readiness.json", report)
    print(
        "TRAINING_READINESS",
        json.dumps(
            {name: value["splits"] for name, value in report["manifests"].items()}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
