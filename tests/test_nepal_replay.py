"""Synthetic fixtures for cutoff integrity; never exported as observations."""

import json
from copy import deepcopy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pravah.replay.experiment import (
    audit,
    freeze,
    register_evaluation,
    sha256,
    verify_snapshot,
)
from pravah.replay.service import attach_replay_routes

CUTOFF = "2026-08-26T08:30:00+05:45"


@pytest.fixture
def evidence(tmp_path):
    # Manufactured protocol examples stay entirely within pytest's temp folder.
    (tmp_path / "input.csv").write_text("fixture only\n")
    (tmp_path / "availability.json").write_text('{"synthetic_receipt": true}\n')
    asset = {
        "asset_id": "SYNTHETIC_PROTOCOL_EXAMPLE",
        "path": "input.csv",
        "sha256": sha256(tmp_path / "input.csv"),
        "role": "input_candidate",
        "download_status": "verified",
        "acquired_at_utc": "2026-08-25T00:00:00Z",
        "available_at_utc": "2026-08-25T01:00:00Z",
        "availability_verified": True,
        "availability_evidence": {
            "source_uri": "https://provider.example/fixture",
            "path": "availability.json",
            "sha256": sha256(tmp_path / "availability.json"),
        },
    }
    return {"event_id": "TEST_EVENT", "assets": [asset], "blockers": ["fixture only"]}


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"available_at_utc": None}, "unknown_availability_or_acquisition"),
        ({"availability_verified": False}, "historical_availability_unverified"),
        ({"acquired_at_utc": "2026-08-26T03:00:00Z"}, "observations_after_cutoff"),
        ({"available_at_utc": "2026-08-26T03:00:00Z"}, "available_after_cutoff"),
        ({"coverage_end_at_utc": "2026-08-27T00:00:00Z"}, "observations_after_cutoff"),
        ({"acquired_at_utc": "2026-08-25"}, "invalid_timestamp"),
        ({"role": "evaluation_reference"}, "evaluation_or_context_only"),
        ({"download_status": "catalog_only"}, "not_a_verified_download"),
        ({"test_fixture": True}, "synthetic_fixture"),
        ({"is_synthetic": True}, "synthetic_fixture"),
        ({"path": "../outside.csv"}, "missing_or_invalid_file_or_availability_proof"),
        ({"sha256": "00" * 32}, "integrity_mismatch"),
    ],
)
def test_excludes_ineligible_sources(evidence, tmp_path, change, reason):
    evidence["assets"][0].update(change)
    result = audit(evidence, CUTOFF, tmp_path)
    assert not result["inputs"]
    assert result["excluded"][0]["reason"] == reason


def test_cutoff_equality_and_nepal_utc(evidence, tmp_path):
    evidence["assets"][0].update(
        acquired_at_utc=CUTOFF, available_at_utc="2026-08-26T02:45:00Z"
    )
    assert audit(evidence, CUTOFF, tmp_path)["eligible_input_count"] == 1
    with pytest.raises(ValueError, match="timezone-aware"):
        audit(evidence, "2026-08-26T08:30:00", tmp_path)


def test_duplicate_ids_rejected(evidence, tmp_path):
    evidence["assets"].append(deepcopy(evidence["assets"][0]))
    with pytest.raises(ValueError, match="Duplicate asset_id"):
        audit(evidence, CUTOFF, tmp_path)


def test_freeze_is_immutable_and_detects_input_changes(evidence, tmp_path):
    result = freeze(tmp_path, evidence, CUTOFF, "TEST_SNAPSHOT")
    assert result["eligible_input_count"] == 1
    assert result["prediction"] is None
    assert verify_snapshot(tmp_path, "TEST_SNAPSHOT") == result
    with pytest.raises(FileExistsError):
        freeze(tmp_path, evidence, CUTOFF, "TEST_SNAPSHOT")
    (tmp_path / "input.csv").write_text("changed fixture\n")
    with pytest.raises(ValueError, match="no longer valid"):
        verify_snapshot(tmp_path, "TEST_SNAPSHOT")


def test_tampered_snapshot_and_path_rejected(evidence, tmp_path):
    freeze(tmp_path, evidence, CUTOFF, "TEST_SNAPSHOT")
    path = tmp_path / "experiments/TEST_SNAPSHOT/snapshot.json"
    snapshot = json.loads(path.read_text())
    snapshot["prediction"] = {"fabricated": True}
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_snapshot(tmp_path, "TEST_SNAPSHOT")
    with pytest.raises(ValueError, match="Invalid experiment ID"):
        freeze(tmp_path, evidence, CUTOFF, "../outside")


def test_later_reference_never_supplies_forecast(evidence, tmp_path):
    reference = {
        **evidence["assets"][0],
        "asset_id": "TEST_REFERENCE",
        "role": "evaluation_reference",
        "acquired_at_utc": "2026-08-27T00:00:00Z",
        "payload": "LATER_VALUE",
    }
    evidence["assets"] = [reference]
    snapshot = freeze(tmp_path, evidence, CUTOFF, "TEST_EMPTY")
    assert snapshot["status"] == "blocked_no_eligible_inputs"
    assert "LATER_VALUE" not in json.dumps(snapshot)
    evaluation = register_evaluation(tmp_path, evidence, "TEST_EMPTY")
    assert (
        evaluation["status"] == "blocked_no_forecast" and evaluation["metrics"] is None
    )
    assert evaluation["references"] == [reference]
    with pytest.raises(FileExistsError):
        register_evaluation(tmp_path, evidence, "TEST_EMPTY")


def test_api_requires_auth_and_freeze_before_reveal(evidence, tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(evidence))
    client = TestClient(attach_replay_routes(FastAPI(), tmp_path, "test-only-token"))
    request = {"run_id": "TEST_API", "cutoff": CUTOFF}
    assert client.post("/api/v1/replay/snapshots", json=request).status_code == 401
    headers = {"X-Pravah-Token": "test-only-token"}
    assert (
        client.post(
            "/api/v1/replay/snapshots/TEST_API/reveal", headers=headers
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/v1/replay/snapshots", json=request, headers=headers
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/v1/replay/snapshots", json=request, headers=headers
        ).status_code
        == 409
    )
    assert client.get("/api/v1/replay/snapshots/TEST_API").json()["prediction"] is None
    assert (
        client.post("/api/v1/replay/snapshots/TEST_API/reveal", headers=headers).json()[
            "metrics"
        ]
        is None
    )
    report = client.get("/api/v1/replay/readiness").json()
    assert report["forecast_model_loaded"] is False
    assert client.get("/api/v1/replay/readiness?cutoff=2026-08-26").status_code == 422


def test_nepal_page_renders_without_data(monkeypatch, tmp_path):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PRAVAH_REPLAY_ROOT", str(tmp_path))
    monkeypatch.setenv("PRAVAH_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PRAVAH_API_URL", "http://127.0.0.1:9")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/Home.py"))
    app.run().switch_page("views/experiment.py").run()
    assert not app.exception
    assert any("Prepared Nepal evidence is not available" in x.value for x in app.info)


def test_freeze_cannot_follow_external_experiment_symlink(evidence, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "_outside")
    outside.mkdir()
    (tmp_path / "experiments").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="inside the replay package"):
        freeze(tmp_path, evidence, CUTOFF, "TEST_ESCAPE")
    assert not list(outside.iterdir())


def test_nepal_page_freezes_then_reveals_without_accuracy(
    evidence, monkeypatch, tmp_path
):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    evidence["assets"][0]["role"] = "evaluation_reference"
    (tmp_path / "manifest.json").write_text(json.dumps(evidence))
    monkeypatch.setenv("PRAVAH_REPLAY_ROOT", str(tmp_path))
    monkeypatch.setenv("PRAVAH_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("PRAVAH_API_URL", "http://127.0.0.1:9")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/Home.py"))
    app.run().switch_page("views/experiment.py").run()
    app.button(key="pv_exp_freeze").click().run()
    assert not app.exception
    snapshot = next((tmp_path / "experiments").glob("*/snapshot.json"))
    assert json.loads(snapshot.read_text())["prediction"] is None
    app.button(key="pv_exp_reveal").click().run()
    assert not app.exception
    result = json.loads(snapshot.with_name("evaluation.json").read_text())
    assert result["metrics"] is None and result["references"]
