"""Synthetic package/CLI fixtures never submit remote jobs."""

import io
import json
import runpy
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
CLI = runpy.run_path(str(ROOT / "scripts/fuzzball_morning.py"))
WORKER = runpy.run_path(str(ROOT / "scripts/fuzzball_run.py"))


def test_workflow_uses_team_volume_cuda_and_explicit_egress():
    doc = CLI["workflow"]("synthetic", "train", "a" * 64)
    assert doc["version"] == "v4"
    assert doc["volumes"]["dataset"] == {"use": "team12", "name": "Pravah-Dataset"}
    job = doc["jobs"]["pravah-model"]
    assert job["resource"]["devices"] == {"nvidia.com/gpu": 1}
    assert "--device cuda" in job["script"] and "--epochs 20" in job["script"]
    assert "--no-index --no-deps" in job["script"]
    assert (
        doc["volumes"]["transfer"]["egress"][0]["source"]["uri"]
        == "file://out/results.tar.gz"
    )


def test_uncertain_submission_id_is_not_guessed():
    first = "12345678-1234-1234-1234-123456789012"
    second = "87654321-1234-1234-1234-123456789012"
    assert CLI["workflow_id"](json.dumps({"id": first})) == first
    assert CLI["workflow_id"]("Started workflow " + first) == first
    with pytest.raises(ValueError, match="unambiguous"):
        CLI["workflow_id"](first + " " + second)


@pytest.mark.parametrize(
    "name,kind", [("../escape", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)]
)
def test_downloaded_archive_cannot_escape_destination(tmp_path, name, kind):
    path = tmp_path / "synthetic.tar"
    with tarfile.open(path, "w") as tar:
        info = tarfile.TarInfo(name)
        info.type = kind
        if kind == tarfile.SYMTYPE:
            info.linkname = "../escape"
        tar.addfile(info, io.BytesIO())
    with pytest.raises(ValueError, match="Unsafe"):
        CLI["safe_extract"](path, tmp_path / "out")


def test_bundle_corruption_is_detected(tmp_path):
    (tmp_path / "source.py").write_text("synthetic")
    (tmp_path / "bundle_manifest.json").write_text(
        json.dumps({"files": {"source.py": "0" * 64}})
    )
    with pytest.raises(ValueError, match="integrity"):
        WORKER["verify_bundle"](tmp_path)


def test_bootstrap_logs_are_allowed_but_completed_runs_are_preserved(tmp_path):
    for name in [
        "bootstrap_status.json",
        "wheel_integrity.log",
        "environment_install.log",
    ]:
        (tmp_path / name).write_text("synthetic bootstrap")
    WORKER["prepare_output"](tmp_path)
    (tmp_path / "status.json").write_text('{"status":"passed"}')
    with pytest.raises(RuntimeError, match="fresh result"):
        WORKER["prepare_output"](tmp_path)


def test_server_validation_failure_prevents_upload_and_submission(
    tmp_path, monkeypatch
):
    calls = []
    scope = CLI["launch"].__globals__
    monkeypatch.setitem(scope, "doctor", dict)
    monkeypatch.setitem(
        scope,
        "prepare",
        lambda mode: {
            "bundle": str(tmp_path / "bundle.tar.gz"),
            "workflow_file": "synthetic.yaml",
        },
    )

    def invalid(arguments, **kwargs):
        calls.append(arguments)
        raise RuntimeError("synthetic server validation failure")

    monkeypatch.setitem(scope, "call", invalid)
    with pytest.raises(RuntimeError, match="validation failure"):
        CLI["launch"]("inference")
    assert calls == [["workflow", "validate", "synthetic.yaml"]]
