#!/usr/bin/env python3
"""Compute measured paired speedup only when matching workload manifests exist."""
import argparse
import json
from pathlib import Path


def compare(one: dict, two: dict) -> dict:
    required = ("tile_manifest_sha256", "model_sha256", "tile_count", "inference_seconds")
    if any(k not in item for item in (one, two) for k in required):
        raise ValueError("Missing measured benchmark fields")
    for key in ("tile_manifest_sha256", "model_sha256", "tile_count"):
        if one[key] != two[key]:
            raise ValueError(f"Workload mismatch: {key}")
    if not one.get("cuda_verified") or not two.get("cuda_verified"):
        raise ValueError("Cannot claim GPU benchmark without CUDA evidence")
    if one.get("nodes_used") != 1 or two.get("nodes_used") != 2:
        raise ValueError("Expected one vs two actual nodes")
    if len(set(two.get("node_ids", []))) != 2:
        raise ValueError("Two distinct node identities required")
    t1 = float(one["inference_seconds"])
    t2 = float(two["inference_seconds"])
    if t1 <= 0 or t2 <= 0:
        raise ValueError("Measured inference durations must be positive")
    return {"speedup_compute": t1 / t2, "parallel_efficiency": t1 / t2 / 2,
            "tile_manifest_sha256": one["tile_manifest_sha256"],
            "model_sha256": one["model_sha256"],
            "tile_count": one["tile_count"],
            "note": "Compute-only speedup, NOT end-to-end wall-clock speedup"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--one", required=True)
    p.add_argument("--two", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = compare(json.loads(Path(a.one).read_text()), json.loads(Path(a.two).read_text()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
