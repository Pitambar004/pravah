"""Synthetic transfer fixtures only; never exported as satellite observations."""

import hashlib
import json
import runpy
import sys
from io import BytesIO
from pathlib import Path

import pytest

MODULE = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/download_training_datasets.py")
)


def asset(payload=b"synthetic transfer fixture"):
    return {
        "dataset": "synthetic",
        "destination": "training/sample.bin",
        "url": "https://huggingface.co/synthetic",
        "bytes": len(payload),
        "checksum": {
            "algorithm": "sha256",
            "value": hashlib.sha256(payload).hexdigest(),
        },
    }


def test_path_escape_rejected(tmp_path):
    for name in ("../outside", "/outside", "ok/../../outside"):
        with pytest.raises(ValueError):
            MODULE["target_path"](tmp_path, name)
    (tmp_path / "escape").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError):
        MODULE["target_path"](tmp_path, "escape/outside")


def test_budget_and_duplicate_rejected(tmp_path):
    item = asset()
    with pytest.raises(ValueError, match="budget"):
        MODULE["validate_plan"]({"assets": [item]}, tmp_path, 1)
    with pytest.raises(ValueError, match="Duplicate"):
        MODULE["validate_plan"]({"assets": [item, item]}, tmp_path, 9999)


def test_corrupt_existing_source_preserved(tmp_path):
    item = asset()
    path = tmp_path / item["destination"]
    path.parent.mkdir()
    path.write_bytes(b"existing research artifact")
    result = MODULE["download_asset"](item, tmp_path, retries=1)
    assert result["status"] == "failed"
    assert path.read_bytes() == b"existing research artifact"


def test_truncated_transfer_is_not_published(tmp_path, monkeypatch):
    item = asset()

    def open_fake(*args, **kwargs):
        stream = BytesIO(b"short")
        stream.headers = {"Content-Length": str(item["bytes"])}
        return stream

    monkeypatch.setitem(MODULE["download_asset"].__globals__, "urlopen", open_fake)
    result = MODULE["download_asset"](item, tmp_path, retries=1)
    assert result["status"] == "failed"
    assert not (tmp_path / item["destination"]).exists()
    assert not list(tmp_path.rglob("*.part"))


def test_verified_download_and_reuse(tmp_path, monkeypatch):
    payload = b"synthetic transfer fixture"
    item = asset(payload)

    def open_fake(*args, **kwargs):
        stream = BytesIO(payload)
        stream.headers = {"Content-Length": str(len(payload))}
        return stream

    monkeypatch.setitem(MODULE["download_asset"].__globals__, "urlopen", open_fake)
    first = MODULE["download_asset"](item, tmp_path, retries=1)
    second = MODULE["download_asset"](item, tmp_path, retries=1)
    assert first["status"] == "downloaded_integrity_verified"
    assert second["status"] == "verified_existing_transfer"
    assert first["sha256"] == second["sha256"] == item["checksum"]["value"]


def test_provider_object_spaces_are_encoded(tmp_path, monkeypatch):
    payload = b"synthetic transfer fixture"
    item = asset(payload)
    item["url"] = "https://storage.googleapis.com/example/a (1).tif?generation=123"

    def open_fake(request, **kwargs):
        assert (
            request.full_url
            == "https://storage.googleapis.com/example/a%20%281%29.tif?generation=123"
        )
        stream = BytesIO(payload)
        stream.headers = {"Content-Length": str(len(payload))}
        return stream

    monkeypatch.setitem(MODULE["download_asset"].__globals__, "urlopen", open_fake)
    assert (
        MODULE["download_asset"](item, tmp_path, retries=1)["status"]
        == "downloaded_integrity_verified"
    )


def test_retry_preserves_original_audit_and_repairs_only_failure(tmp_path, monkeypatch):
    payload = b"synthetic transfer fixture"
    good = asset(payload)
    bad = {**good, "destination": "training/retry.bin"}
    source = tmp_path / good["destination"]
    source.parent.mkdir()
    source.write_bytes(payload)
    report = tmp_path / "audit"
    report.mkdir()
    original = [
        {
            **good,
            "path": str(source),
            "status": "verified_existing_transfer",
            "sha256": good["checksum"]["value"],
        },
        {**bad, "path": str(tmp_path / bad["destination"]), "status": "failed"},
    ]
    original_log = "".join(json.dumps(row) + "\n" for row in original)
    (report / "transfers.jsonl").write_text(original_log)
    (report / "download_summary.json").write_text(
        json.dumps({"failed_assets": [original[1]]})
    )
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"assets": [good, bad]}))
    calls = []

    def open_fake(request, **kwargs):
        calls.append(request.full_url)
        stream = BytesIO(payload)
        stream.headers = {"Content-Length": str(len(payload))}
        return stream

    monkeypatch.setitem(MODULE["download_asset"].__globals__, "urlopen", open_fake)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "intake",
            "--plan",
            str(plan),
            "--root",
            str(tmp_path),
            "--report-dir",
            str(report),
            "--retry-failed",
            "--reserve-gib",
            "0",
        ],
    )
    MODULE["main"]()
    assert len(calls) == 1
    assert (report / "transfers.jsonl").read_text().startswith(original_log)
    assert len(list(report.glob("download_summary_before_retry_*.json"))) == 1
    summary = json.loads((report / "download_summary.json").read_text())
    assert summary["failed_assets"] == []
    assert summary["datasets"]["synthetic"]["verified_bytes"] == 2 * len(payload)
