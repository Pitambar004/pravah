"""Audited, immutable cutoff reconstructions; never fabricate a forecast.

Observation time alone does not establish historical availability. A prepared
context file, catalog listing or later report cannot become a predictor merely
because its filename describes a pre-event period.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pravah.replay.eligibility import parse_time

SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,96}\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or root.resolve() not in path.parents:
        raise ValueError("Asset paths must remain inside the replay package")
    return path


def asset_reason(asset: dict, cutoff: str, root: Path) -> str | None:
    """Return a concrete exclusion reason, or None for a verified input."""
    end = parse_time(cutoff)
    if end is None:
        raise ValueError("A timezone-aware cutoff is required")
    if asset.get("is_synthetic") or asset.get("test_fixture"):
        return "synthetic_fixture"
    if asset.get("role") != "input_candidate":
        return "evaluation_or_context_only"
    if asset.get("download_status") != "verified":
        return "not_a_verified_download"
    try:
        acquired = parse_time(asset.get("acquired_at_utc"))
        available = parse_time(asset.get("available_at_utc"))
        coverage_end = parse_time(asset.get("coverage_end_at_utc"))
    except (ValueError, TypeError, AttributeError):
        return "invalid_timestamp"
    if acquired is None or available is None:
        return "unknown_availability_or_acquisition"
    if acquired > end or (coverage_end is not None and coverage_end > end):
        return "observations_after_cutoff"
    if available > end:
        return "available_after_cutoff"
    if available < acquired:
        return "availability_precedes_acquisition"
    proof = asset.get("availability_evidence") or {}
    if asset.get("availability_verified") is not True or not proof.get("source_uri"):
        return "historical_availability_unverified"
    try:
        for relative, expected in [
            (asset["path"], asset["sha256"]),
            (proof["path"], proof["sha256"]),
        ]:
            if sha256(contained(root, relative)) != expected:
                return "integrity_mismatch"
    except (OSError, ValueError, KeyError, TypeError):
        return "missing_or_invalid_file_or_availability_proof"
    return None


def audit(manifest: dict, cutoff: str, root: Path) -> dict:
    if parse_time(cutoff) is None:
        raise ValueError("A timezone-aware cutoff is required")
    inputs, excluded, identities = [], [], set()
    for asset in manifest["assets"]:
        identity = asset["asset_id"]
        if identity in identities:
            raise ValueError(f"Duplicate asset_id: {identity}")
        identities.add(identity)
        reason = asset_reason(asset, cutoff, root)
        if reason:
            # Exclusion summaries contain no observation values from later data.
            excluded.append({"asset_id": identity, "reason": reason})
        else:
            inputs.append(asset)
    return {
        "cutoff_at_utc": parse_time(cutoff).astimezone(timezone.utc).isoformat(),
        "inputs": inputs,
        "excluded": excluded,
        "eligible_input_count": len(inputs),
        "excluded_count": len(excluded),
    }


def _seal(document: dict) -> dict:
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return {**document, "snapshot_sha256": hashlib.sha256(payload.encode()).hexdigest()}


def freeze(root: Path, manifest: dict, cutoff: str, run_id: str) -> dict:
    """Freeze only eligible inputs, independently of later evaluation evidence."""
    if not SAFE_ID.fullmatch(run_id):
        raise ValueError("Invalid experiment ID")
    result = audit(manifest, cutoff, root)
    document = _seal(
        {
            "schema_version": 1,
            "run_id": run_id,
            "event_id": manifest["event_id"],
            "experiment": "retrospective_cutoff_reconstruction",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "case_study_selection": "retrospective; not a blind prospective experiment",
            **result,
            "status": "inputs_frozen_model_missing"
            if result["inputs"]
            else "blocked_no_eligible_inputs",
            "prediction": None,
            "forecast_model": None,
            "unavailable_outputs": [
                "forecast_extent",
                "arrival_time",
                "water_depth",
                "river_width",
                "forecast_exposure",
                "forecast_accuracy",
            ],
        }
    )
    folder = contained(root, f"experiments/{run_id}")
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / "snapshot.json").open("x") as stream:
        stream.write(json.dumps(document, indent=2, allow_nan=False) + "\n")
    return document


def verify_snapshot(root: Path, run_id: str) -> dict:
    if not SAFE_ID.fullmatch(run_id):
        raise ValueError("Invalid experiment ID")
    document = json.loads(
        contained(root, f"experiments/{run_id}/snapshot.json").read_text()
    )
    recorded = document.pop("snapshot_sha256")
    sealed = _seal(document)
    if recorded != sealed["snapshot_sha256"]:
        raise ValueError("Frozen snapshot integrity mismatch")
    for asset in document["inputs"]:
        reason = asset_reason(asset, document["cutoff_at_utc"], root)
        if reason:
            raise ValueError(
                f"Frozen input {asset['asset_id']} no longer valid: {reason}"
            )
    return sealed


def register_evaluation(root: Path, manifest: dict, run_id: str) -> dict:
    """Reveal evaluation artifacts after freezing; never produce absent scores."""
    snapshot = verify_snapshot(root, run_id)
    if snapshot["event_id"] != manifest["event_id"]:
        raise ValueError("Evaluation belongs to a different event")
    references = []
    for asset in manifest["assets"]:
        if asset.get("role") != "evaluation_reference":
            continue
        if asset.get("is_synthetic") or asset.get("test_fixture"):
            raise ValueError("Synthetic references cannot enter an event evaluation")
        if asset.get("download_status") != "verified":
            continue
        if sha256(contained(root, asset["path"])) != asset["sha256"]:
            raise ValueError("Evaluation reference integrity mismatch")
        references.append(asset)
    document = {
        "run_id": run_id,
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "revealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "references": references,
        "metrics": None,
        "status": "blocked_no_forecast"
        if snapshot["prediction"] is None
        else "evaluation_pending",
        "reason": "No downstream forecast was produced; later evidence cannot supply a missing prediction.",
        "class_caveat": "UNOSAT mud/rockflow extent, optical water, gauge stage and confirmed damage are distinct targets.",
    }
    with (root / "experiments" / run_id / "evaluation.json").open("x") as stream:
        stream.write(json.dumps(document, indent=2, allow_nan=False) + "\n")
    return document
