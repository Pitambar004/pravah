"""AppTest for Prediction vs Reality and Model Training with SYNTHETIC tmp-dir runs only."""
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from pravah.analysis.prediction_validation import validate_prediction_run

HOME = str((Path(__file__).parent.parent / "app" / "Home.py").resolve())
TR = from_origin(320000, 3110000, 10, 10)


def _tif(path, arr):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="uint8",
                       crs="EPSG:32645", transform=TR, nodata=255) as dst:
        dst.write(arr, 1)


def _obs(root: Path, name: str, arr, acquired: str):
    d = root / name
    d.mkdir()
    _tif(d / "observed_water.tif", arr)
    (d / "observed_metadata.json").write_text(json.dumps({"run_id": name, "acquired_at_utc": acquired}))
    return d


@pytest.fixture
def env(monkeypatch, tmp_path):
    import streamlit as st

    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("PRAVAH_OUTPUTS_DIR", str(out))
    monkeypatch.setenv("PRAVAH_TRAINING_DIR", str(tmp_path / "training"))
    monkeypatch.setenv("PRAVAH_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("PRAVAH_CONTEXT_DIR", str(tmp_path / "ctx"))
    st.cache_data.clear()
    yield tmp_path
    st.cache_data.clear()


def _open(page: str) -> AppTest:
    at = AppTest.from_file(HOME, default_timeout=60)
    at.run()
    at.switch_page(page)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def _text(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_validation_page_shows_measured_scores(env):
    out = env / "outputs"
    pre = np.zeros((30, 30), np.uint8)
    pre[:, 0] = 1
    post = pre.copy()
    post[10:20, 10:20] = 1
    _obs(out, "fx_20260811_T45RUL", pre, "2026-08-11T04:52:41Z")
    post_d = _obs(out, "fx_20260827_T45RUL", post, "2026-08-27T04:56:59Z")
    pdir = out / "fx_prediction"
    pdir.mkdir()
    pred = pre.copy()
    pred[10:20, 10:25] = 1
    _tif(pdir / "predicted_flood.tif", pred)
    (pdir / "prediction_metadata.json").write_text(json.dumps({
        "run_id": "fx_prediction", "model_id": "fixture-model", "input_cutoff_utc": "2026-08-24T00:00:00Z",
        "inputs": [{"name": "rain", "acquired_at_utc": "2026-08-23T00:00:00Z",
                    "available_at_utc": "2026-08-23T03:00:00Z"}]}))
    validate_prediction_run(pdir, [post_d], [out / "fx_20260811_T45RUL"], "new_water", "2026-08-25T18:15:00Z")
    at = _open("views/validation.py")
    txt = _text(at)
    assert "Result" in txt and "beats" in txt and "does <b>not</b>" not in txt
    assert "66.7%" in txt  # CSI = 100 / (100 + 50 false-alarm px)


def test_training_page_claims_speedup_only_when_equivalent(env):
    root = env / "training"
    common = {"status": "completed", "pairs_sha256": "abc", "global_batch_size": 8, "optimizer_steps": 100,
              "seed": 42, "lr": 1e-3, "cuda_verified": True, "final_validation": {"metrics": {"iou": 0.7}}}
    runs = {"gb10_1node": {**common, "world_size": 1, "distinct_hostnames": 1, "train_seconds": 100.0},
            "gb10_2node": {**common, "world_size": 2, "distinct_hostnames": 2, "train_seconds": 62.5}}
    for name, s in runs.items():
        (root / name).mkdir(parents=True)
        (root / name / "training_summary.json").write_text(json.dumps({**s, "run_label": name}))
        (root / name / "train_metrics.csv").write_text(
            "epoch,optimizer_step,loss_rank0,elapsed_sec,world_size,train_sec,global_samples\n0,1,0.9,1,1,1,8\n")
    at = _open("views/training.py")
    assert "1.60×" in _text(at)
    s = json.loads((root / "gb10_2node" / "training_summary.json").read_text())
    s["optimizer_steps"] = 50  # unequal workload -> no speedup claim
    (root / "gb10_2node" / "training_summary.json").write_text(json.dumps(s))
    at = _open("views/training.py")
    assert "1.60×" not in _text(at) and "Not claimed" in _text(at)


@pytest.mark.parametrize("page", ["views/validation.py", "views/gallery.py", "views/live.py",
                                  "views/alerts.py", "views/training.py"])
def test_model_pages_never_offer_fixtures_without_developer_flag(env, monkeypatch, page):
    """Without PRAVAH_ENABLE_TEST_FIXTURES=1 there is no fixture toggle and no fixture content."""
    monkeypatch.setenv("PRAVAH_ENABLE_TEST_FIXTURES", "0")
    at = AppTest.from_file(HOME, default_timeout=30)
    at.session_state["pv_demo"] = True  # stale state must not re-enable fixtures
    at.run()
    at.switch_page(page)
    at.run()
    assert not at.exception, at.exception
    assert not at.sidebar.toggle
    assert at.session_state["pv_demo"] is False
    assert "Synthetic fixture" not in " ".join(m.value for m in at.markdown)
