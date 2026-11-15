#!/usr/bin/env python3
"""Prepare the verified official optical hand-label release for a local demo."""

import argparse
import json
import runpy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/raw/sen1floods11/v1.1")
    )
    parser.add_argument(
        "--intake", type=Path, default=Path("outputs/local_demo/intake")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/local_demo/data"))
    args = parser.parse_args()
    args.data_root, args.output = args.data_root.resolve(), args.output.resolve()
    summary = json.loads((args.intake / "download_summary.json").read_text())
    if summary["failed_assets"]:
        raise RuntimeError("Repair failed source downloads first")
    index = {}
    for line in (args.intake / "transfers.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["status"] != "failed":
            index[row["path"]] = row
    prepare = runpy.run_path(
        str(Path(__file__).with_name("download_training_prepare.py"))
    )
    sources = prepare["hand_sources"](args.data_root)
    if len(sources) != 446:
        raise RuntimeError("Unexpected official hand-label release inventory")

    def process(record):
        return prepare["process_scene"](
            record, args.output / "prepared", 512, 0, [], index
        )

    pairs, bolivia, qa, excluded = [], [], [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for record, (tiles, stats) in zip(
            sources, pool.map(process, sources), strict=True
        ):
            qa.append(stats)
            if not tiles:
                excluded.append(record["scene_id"])
            if record["split"] == "bolivia":
                for tile in tiles:
                    tile["split"] = "test"
                bolivia.extend(tiles)
            else:
                pairs.extend(tiles)
    paths = [p["image_path"] for p in pairs + bolivia]
    if len(paths) != len(set(paths)):
        raise RuntimeError("Duplicate image paths")
    for name, value in [
        ("pairs.json", pairs),
        ("bolivia_holdout.json", bolivia),
        ("raster_qa.json", qa),
    ]:
        prepare["write_json"](args.output / name, value)
    report = {
        "status": "passed",
        "source_pairs": len(sources),
        "splits": dict(Counter(p["split"] for p in pairs)),
        "bolivia_holdout_pairs": len(bolivia),
        "excluded_all_invalid": excluded,
        "pairs_sha256": prepare["sha256"](args.output / "pairs.json"),
        "split_policy": "Official within-event train/val/test preserved; Bolivia separate; no test-set model selection",
        "bands": list(prepare["S2_BANDS"]),
        "labels": {
            "-1": "invalid/cloud",
            "0": "no_water",
            "1": "water including permanent water",
        },
        "units": "Original L1C reflectance * 10000; U-Net loader scales /10000 and clips [0,1]",
        "training_performed": False,
    }
    prepare["write_json"](args.output / "readiness.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
