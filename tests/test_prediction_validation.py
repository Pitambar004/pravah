"""Prediction-vs-reality scoring. All rasters here are SYNTHETIC unit-test fixtures."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pravah.analysis.prediction_validation import (
    check_integrity,
    score_against_reference,
    validate_prediction_run,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import validate_prediction as cli

TR = from_origin(300000, 3100000, 10, 10)
CRS = "EPSG:32645"
INPUTS = [{"name": "rain_gauge_fixture", "acquired_at_utc": "2026-08-24T00:00:00Z",
           "available_at_utc": "2026-08-24T06:00:00Z"}]


def _tif(path: Path, arr, dtype="uint8", nodata=255):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=dtype, crs=CRS, transform=TR, nodata=nodata) as dst:
        dst.write(arr.astype(dtype), 1)


def _run(tmp, name, arr, acquired):
    d = tmp / name
    d.mkdir()
    _tif(d / "observed_water.tif", arr)
    (d / "observed_metadata.json").write_text(json.dumps({"run_id": name, "acquired_at_utc": acquired,
                                                          "scene_id": f"S_{name}"}))
    return d


def _pred(tmp, arr, cutoff="2026-08-25T00:00:00Z", inputs=INPUTS, prob=False):
    d = tmp / "pred"
    d.mkdir()
    if prob:
        _tif(d / "predicted_flood_probability.tif", arr, dtype="float32", nodata=-1)
    else:
        _tif(d / "predicted_flood.tif", arr)
    (d / "prediction_metadata.json").write_text(json.dumps({
        "run_id": "pred", "model_id": "fixture", "input_cutoff_utc": cutoff,
        "issued_at_utc": "2026-08-25T01:00:00Z", "inputs": inputs}))
    return d


def test_scores_and_agreement_codes():
    ref = np.array([[1, 1, 0, 0, 255]], dtype=np.uint8)
    pred = np.array([[1, 0, 1, 0, 1]], dtype=np.uint8)
    res, ag = score_against_reference(pred, ref)
    assert res["counts"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1, "valid": 4}
    assert res["metrics"]["iou"] == pytest.approx(1 / 3)
    assert res["metrics"]["far"] == pytest.approx(0.5)
    assert ag.tolist() == [[1, 3, 2, 0, 255]]


def test_new_water_excludes_already_wet_and_baseline_has_no_skill():
    base = np.array([[1, 0, 0, 0]], dtype=np.uint8)
    ref = np.array([[1, 1, 1, 0]], dtype=np.uint8)
    pred = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    res, ag = score_against_reference(pred, ref, base, "new_water")
    assert res["comparable_pixels"] == 3  # pixel 0 was already wet
    assert ag[0, 0] == 255
    assert res["persistence_baseline"]["metrics"]["recall"] == 0
    assert res["skill_vs_persistence"]["beats_baseline"] is True


def test_integrity_rejects_late_or_unknown_inputs():
    bad = [{"name": "late", "acquired_at_utc": "2026-08-26T00:00:00Z", "available_at_utc": "2026-08-26T01:00:00Z"},
           {"name": "unknown", "acquired_at_utc": "2026-08-20T00:00:00Z"}]
    r = check_integrity({"input_cutoff_utc": "2026-08-25T00:00:00Z", "inputs": bad},
                        [{"run_id": "r", "acquired_at_utc": "2026-08-27T00:00:00Z"}])
    assert not r["eligible"]
    assert any("late" in v for v in r["violations"]) and any("unknown" in v for v in r["violations"])


def test_integrity_rejects_reference_before_cutoff_and_reference_as_input():
    r = check_integrity({"input_cutoff_utc": "2026-08-28T00:00:00Z",
                         "inputs": [{**INPUTS[0], "scene_id": "S_post"}]},
                        [{"run_id": "post", "scene_id": "S_post", "acquired_at_utc": "2026-08-27T00:00:00Z"}],
                        "2026-08-26T00:00:00Z")
    assert not r["eligible"]
    assert r["claim_type"] == "post_event_nowcast"
    assert len(r["violations"]) == 2


def test_integrity_early_warning_lead_time():
    r = check_integrity({"input_cutoff_utc": "2026-08-25T00:00:00Z", "inputs": INPUTS},
                        [{"run_id": "post", "acquired_at_utc": "2026-08-27T00:00:00Z"}], "2026-08-26T00:00:00Z")
    assert r["eligible"] and r["claim_type"] == "early_warning" and r["lead_time_hours_before_event"] == 24


def test_end_to_end_run_writes_outputs(tmp_path):
    base = np.zeros((20, 20), np.uint8)
    base[:, 0] = 1
    post = base.copy()
    post[5:10, 5:10] = 1
    post[0, :] = 255
    pred = base.copy()
    pred[5:10, 5:12] = 1
    pre_d = _run(tmp_path, "pre", base, "2026-08-11T00:00:00Z")
    post_d = _run(tmp_path, "post", post, "2026-08-27T00:00:00Z")
    pdir = _pred(tmp_path, pred)
    s = validate_prediction_run(pdir, [post_d], [pre_d], "new_water", "2026-08-26T00:00:00Z")
    assert s["status"] == "scored"
    c = s["combined"]["counts"]
    assert (c["tp"], c["fp"], c["fn"]) == (25, 10, 0)
    assert s["combined"]["skill_vs_persistence"]["beats_baseline"]
    out = pdir / "validation" / "post"
    for f in ("agreement.tif", "agreement_hit.geojson", "agreement_false_alarm.geojson", "agreement_miss.geojson"):
        assert (out / f).exists()
    assert (pdir / "predicted_flood.geojson").exists()


def test_probability_prediction_needs_threshold(tmp_path):
    ref = np.zeros((4, 4), np.uint8)
    ref[0, 0] = 1
    post_d = _run(tmp_path, "post", ref, "2026-08-27T00:00:00Z")
    prob = np.zeros((4, 4), np.float32)
    prob[0, 0] = 0.9
    pdir = _pred(tmp_path, prob, prob=True)
    assert cli.main(["--prediction", str(pdir), "--reference", str(post_d)]) == 2
    assert cli.main(["--prediction", str(pdir), "--reference", str(post_d), "--threshold", "0.5"]) == 0
    s = json.loads((pdir / "prediction_validation.json").read_text())
    assert s["combined"]["metrics"]["iou"] == 1.0


def test_cli_rejects_leaky_prediction(tmp_path):
    post_d = _run(tmp_path, "post", np.zeros((4, 4), np.uint8), "2026-08-27T00:00:00Z")
    pdir = _pred(tmp_path, np.zeros((4, 4), np.uint8), cutoff="2026-08-28T00:00:00Z")
    assert cli.main(["--prediction", str(pdir), "--reference", str(post_d)]) == 3
    s = json.loads((pdir / "prediction_validation.json").read_text())
    assert s["status"] == "rejected_integrity" and "combined" not in s
