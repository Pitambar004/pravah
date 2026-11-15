"""Automatically compare newly completed compatible Nepal optical water runs.

Discovery/imagery acquisition remain separate. This observer only consumes
actual georeferenced segmentation artifacts with matching model, threshold,
channel semantics and explicit cloud-validity masks on identical grids.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from pravah.analysis.change import compare_runs
from pravah.replay.experiment import sha256


def identity(metadata):
    info = metadata.get("model_info") or {}
    digest = metadata.get("model_sha256") or info.get("checkpoint_sha256")
    bands = metadata.get("bands") or info.get("bands")
    return digest, bands, metadata.get("threshold"), metadata.get("input_processing_level")


def eligible(metadata):
    study = metadata.get("case_study") or {}
    place = str(study.get("region", "")).lower()
    digest, bands, threshold, level = identity(metadata)
    return (
        metadata.get("is_synthetic") is False
        and study.get("scope") != "benchmark"
        and ("nepal" in place or study.get("id", "").startswith("rasuwa"))
        and isinstance(digest, str) and len(digest) == 64
        and bands == ["B2", "B3", "B4", "B8A", "B11", "B12"]
        and threshold is not None
        and level == "Sentinel-2 L1C"
        and metadata.get("cloud_validity_verified") is True
    )


def watch_water(runs: Path, root: Path):
    candidates = []
    for path in sorted(runs.glob("*/observed_metadata.json"))[:500]:
        try:
            md = json.loads(path.read_text())
            date = datetime.fromisoformat(md["acquired_at_utc"].replace("Z", "+00:00"))
            if date.tzinfo is None or not eligible(md):
                continue
            if (path.parent / "observed_water.tif").is_file():
                candidates.append((date, path.parent, md))
        except (OSError, ValueError, KeyError):
            continue
    candidates.sort(key=lambda item: item[0])
    if len(candidates) < 2:
        return {"status": "unavailable", "records": [],
                "reason": "Need two compatible, cloud-valid Nepal optical water runs on the same grid",
                "forecast": None, "new_water": None}
    pre_date, pre, pre_md = candidates[-2]
    post_date, post, post_md = candidates[-1]
    if identity(pre_md) != identity(post_md) or pre_date >= post_date:
        return {"status": "incompatible", "records": [], "reason": "Model, bands, product, threshold or acquisition order differ"}
    files = [pre / "observed_metadata.json", pre / "observed_water.tif",
             post / "observed_metadata.json", post / "observed_water.tif"]
    input_hashes = {str(p): sha256(p) for p in files}
    digest = hashlib.sha256(json.dumps(input_hashes, sort_keys=True).encode()).hexdigest()[:24]
    destination = root / "water_changes" / digest
    try:
        if not (destination / "receipt.json").exists():
            summary = compare_runs(pre, post, destination)
            (destination / "receipt.json").write_text(json.dumps({"inputs": input_hashes, "comparison_id": digest}) + "\n")
        else:
            summary = json.loads((destination / "change_summary.json").read_text())
        new_area = summary["area_km2"]["new_water"]
        return {"status": "ok", "records": [{"id": digest, "new_water_area_km2": new_area}],
                "summary": summary, "new_water": json.loads((destination / "change_new_water.geojson").read_text()),
                "watch_threshold_km2": .01,
                "threshold_semantics": "Analyst watch at ≥0.01 km² newly classed water; not a validated disaster threshold"}
    except (OSError, ValueError, KeyError) as exc:
        return {"status": "incompatible", "records": [], "reason": str(exc), "new_water": None}
