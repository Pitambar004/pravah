"""Inference engine + FastAPI tests.

SYNTHETIC ONLY: the raster is random noise written to a pytest tmp dir and the
checkpoint is a *randomly initialised* MiniUNet saved in the train_ddp.py format.
They exercise plumbing (georeferencing, class codes, API contract), never model
quality, and are never exported as observations or a trained model.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from pravah.api.main import create_app
from pravah.api.settings import Settings
from pravah.inference.engine import (
    BANDS,
    CheckpointError,
    InferenceEngine,
    resolve_device,
)
from pravah.ml.unet import MiniUNet

CRS = "EPSG:32645"  # UTM 45N — metric, so areas are computed
TRANSFORM = from_origin(330000.0, 3120000.0, 10.0, 10.0)


def write_raster(path, bands=6, width=300, height=280, crs=CRS, nodata_cols=20):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4000, size=(bands, height, width), dtype=np.uint16)
    data[:, :, :nodata_cols] = 0
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=bands, dtype="uint16",
                       crs=crs, transform=TRANSFORM, nodata=0) as dst:
        dst.write(data)
    return path


def write_checkpoint(path, **overrides):
    torch.manual_seed(0)
    payload = {"model": MiniUNet().state_dict(), "epoch": 0, "steps": 1, "architecture": "MiniUNet",
               "bands": list(BANDS), "source_data_pairs_sha256": "synthetic-test"}
    payload.update(overrides)
    torch.save(payload, path)
    return path


@pytest.fixture()
def ckpt(tmp_path):
    return write_checkpoint(tmp_path / "ckpt.pt")


# ---------------------------------------------------------------- engine
def test_engine_outputs_are_georeferenced_and_coded(tmp_path, ckpt):
    scene = write_raster(tmp_path / "scene.tif")
    eng = InferenceEngine(ckpt, device="cpu", batch_size=4)
    res = eng.predict(scene, tmp_path / "out", run_id="t1", threshold=0.5, tile_size=128, overlap=16)
    out = tmp_path / "out"
    for f in ("observed_water.tif", "invalid_mask.tif", "water_probability.tif",
              "observed_metadata.json", "observed_water.geojson"):
        assert (out / f).is_file(), f
    with rasterio.open(out / "observed_water.tif") as w:
        cls = w.read(1)
        assert w.crs.to_string() == CRS and w.transform == TRANSFORM and w.nodata == 255
    assert set(np.unique(cls)) <= {0, 1, 255}
    assert (cls[:, :20] == 255).all(), "nodata pixels must be invalid (255), never 'no water'"
    with rasterio.open(out / "water_probability.tif") as p:
        prob = p.read(1)
    valid = cls != 255
    assert ((prob[valid] >= 0) & (prob[valid] <= 1)).all() and (prob[~valid] == -9999).all()
    md = res.metadata
    assert md["runtime"]["cuda_verified"] is False and md["runtime"]["device"] == "cpu"
    assert md["model_info"]["checkpoint_sha256"] and md["tile_count"] > 1
    assert md["water_pixel_count"] == int((cls == 1).sum())
    assert md["water_area_m2"] == pytest.approx(md["water_pixel_count"] * 100.0)
    gj = json.loads((out / "observed_water.geojson").read_text())
    for feat in gj["features"][:20]:
        lon, lat = feat["geometry"]["coordinates"][0][0]
        assert 84 < lon < 90 and 25 < lat < 30  # reprojected to WGS84 near UTM 45N origin


def test_batch_size_does_not_change_prediction(tmp_path, ckpt):
    scene = write_raster(tmp_path / "scene.tif", width=200, height=180)
    a = InferenceEngine(ckpt, device="cpu", batch_size=1)
    b = InferenceEngine(ckpt, device="cpu", batch_size=8)
    a.predict(scene, tmp_path / "a", run_id="a", tile_size=64, overlap=8)
    b.predict(scene, tmp_path / "b", run_id="b", tile_size=64, overlap=8)
    with rasterio.open(tmp_path / "a/water_probability.tif") as pa, rasterio.open(tmp_path / "b/water_probability.tif") as pb:
        np.testing.assert_allclose(pa.read(1), pb.read(1), atol=1e-5)


def test_thirteen_band_input_selects_s2_bands(tmp_path, ckpt):
    scene = write_raster(tmp_path / "s13.tif", bands=13, width=96, height=96)
    res = InferenceEngine(ckpt, device="cpu").predict(scene, tmp_path / "o", run_id="s13", tile_size=64, overlap=0)
    assert res.metadata["bands_used"] == ["B2", "B3", "B4", "B8A", "B11", "B12"]


@pytest.mark.parametrize("bands", [2, 4])
def test_rejects_non_s2_band_counts(tmp_path, ckpt, bands):
    scene = write_raster(tmp_path / "bad.tif", bands=bands, width=64, height=64)
    with pytest.raises(ValueError, match="bands"):
        InferenceEngine(ckpt, device="cpu").predict(scene, tmp_path / "o", run_id="x")


def test_rejects_ungeoreferenced_input(tmp_path, ckpt):
    p = tmp_path / "nocrs.tif"
    with rasterio.open(p, "w", driver="GTiff", width=32, height=32, count=6, dtype="uint16") as dst:
        dst.write(np.ones((6, 32, 32), dtype=np.uint16))
    with pytest.raises(ValueError, match="CRS"):
        InferenceEngine(ckpt, device="cpu").predict(p, tmp_path / "o", run_id="x")


def test_checkpoint_validation(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        InferenceEngine(tmp_path / "missing.pt", device="cpu")
    with pytest.raises(CheckpointError, match="architecture"):
        InferenceEngine(write_checkpoint(tmp_path / "a.pt", architecture="Prithvi"), device="cpu")
    with pytest.raises(CheckpointError, match="band order"):
        InferenceEngine(write_checkpoint(tmp_path / "b.pt", bands=["VV", "VH"]), device="cpu")
    torch.save({"model": {"bogus": torch.zeros(1)}}, tmp_path / "c.pt")
    with pytest.raises(CheckpointError, match="MiniUNet"):
        InferenceEngine(tmp_path / "c.pt", device="cpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA refusal path")
def test_require_cuda_never_falls_back_to_cpu():
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("auto", require_cuda=True)


# ---------------------------------------------------------------- API
def _settings(tmp_path, checkpoint, token=None):
    scenes = tmp_path / "data/raw/sentinel2"
    scenes.mkdir(parents=True)
    return Settings(data_root=tmp_path / "data", scenes_dir=scenes, outputs_dir=tmp_path / "data/outputs/predictions",
                    checkpoint=checkpoint, device="cpu", require_cuda=False, batch_size=4,
                    api_token=token, max_queued_jobs=4)


def test_api_without_checkpoint_refuses_predictions(tmp_path):
    s = _settings(tmp_path, tmp_path / "none.pt")
    write_raster(s.scenes_dir / "scene.tif", width=64, height=64)
    with TestClient(create_app(s)) as c:
        st = c.get("/api/v1/status").json()
        assert st["model_loaded"] is False and "not found" in st["model_error"]
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        r = c.post("/api/v1/predictions", json={"scene_id": "scene.tif"})
        assert r.status_code == 503


def test_api_end_to_end(tmp_path, ckpt):
    s = _settings(tmp_path, ckpt)
    write_raster(s.scenes_dir / "scene.tif", width=128, height=128)
    write_raster(s.scenes_dir / "sar_like.tif", bands=2, width=32, height=32)
    with TestClient(create_app(s)) as c:
        assert c.get("/health").json() == {"status": "ok"}
        assert c.get("/ready").json() == {"status": "ready"}
        st = c.get("/api/v1/status").json()
        assert st["model_loaded"] and st["runtime"]["device"] == "cpu" and st["runtime"]["cuda_verified"] is False
        scenes = {x["scene_id"]: x for x in c.get("/api/v1/scenes").json()["scenes"]}
        assert scenes["scene.tif"]["model_compatible"] and not scenes["sar_like.tif"]["model_compatible"]
        assert c.post("/api/v1/predictions", json={"scene_id": "sar_like.tif"}).status_code == 422

        r = c.post("/api/v1/predictions", json={"scene_id": "scene.tif", "run_id": "demo_run", "threshold": 0.4})
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]
        for _ in range(400):
            j = c.get(f"/api/v1/predictions/{job_id}").json()
            if j["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.05)
        assert j["status"] == "succeeded", j
        assert j["tiles_done"] == j["tiles_total"]
        assert c.post("/api/v1/predictions", json={"scene_id": "scene.tif", "run_id": "demo_run"}).status_code == 409

        runs = c.get("/api/v1/runs").json()["runs"]
        assert runs[0]["run_id"] == "demo_run" and runs[0]["complete"] and runs[0]["threshold"] == 0.4
        md = c.get("/api/v1/runs/demo_run").json()
        assert md["scene_id"] == "scene.tif" and md["job_id"] == job_id
        f = c.get("/api/v1/runs/demo_run/files/observed_water.geojson")
        assert f.status_code == 200 and f.json()["type"] == "FeatureCollection"
        assert c.get("/api/v1/runs/demo_run/files/secret.txt").status_code == 404
        assert c.get("/api/v1/runs/..%2F..%2Fetc/files/job.json").status_code in (400, 404)
    # dashboard contract: run folder loads with the existing data loader
    from app.data_loader import load_run_artifacts

    a = load_run_artifacts(s.outputs_dir / "demo_run")
    assert not a.is_synthetic and a.observed_metadata["run_id"] == "demo_run"
    assert a.observed_water_tif is not None and a.water_geojson["type"] == "FeatureCollection"


def test_api_rejects_path_traversal(tmp_path, ckpt):
    s = _settings(tmp_path, ckpt)
    write_raster(tmp_path / "outside.tif", width=32, height=32)
    with TestClient(create_app(s)) as c:
        for sid in ("../../../outside.tif", "/etc/passwd", "../../outside.tif"):
            assert c.post("/api/v1/predictions", json={"scene_id": sid}).status_code == 400
        assert c.post("/api/v1/predictions", json={"scene_id": "x.tif", "run_id": "bad id!"}).status_code in (400, 404)


def test_api_token_guards_writes(tmp_path, ckpt):
    s = _settings(tmp_path, ckpt, token="s3cret-test-token")
    write_raster(s.scenes_dir / "scene.tif", width=64, height=64)
    with TestClient(create_app(s)) as c:
        assert c.get("/api/v1/status").status_code == 200  # reads stay open
        assert c.post("/api/v1/predictions", json={"scene_id": "scene.tif"}).status_code == 401
        assert c.post("/api/v1/predictions", json={"scene_id": "scene.tif"},
                      headers={"X-Pravah-Token": "wrong"}).status_code == 401
        ok = c.post("/api/v1/predictions", json={"scene_id": "scene.tif"},
                    headers={"X-Pravah-Token": "s3cret-test-token"})
        assert ok.status_code == 202
        assert c.post("/api/v1/model/reload").status_code == 401


def test_api_accepts_fuzzball_endpoint_prefix(tmp_path):
    s = _settings(tmp_path, tmp_path / "none.pt")
    from dataclasses import replace

    s = replace(s, root_path="/endpoints/endpoint-abc123")
    with TestClient(create_app(s)) as c:
        assert c.get("/endpoints/endpoint-abc123/health").json() == {"status": "ok"}
        assert c.get("/endpoints/endpoint-abc123/api/v1/status").status_code == 200
        assert c.get("/health").status_code == 200  # also works if the proxy strips the prefix


def test_root_path_from_fuzzball_env(monkeypatch):
    monkeypatch.delenv("PRAVAH_API_ROOT_PATH", raising=False)
    monkeypatch.setenv("FB_ENDPOINT_URL_API", "https://endpoints.example.net/endpoints/endpoint-xyz/")
    assert Settings.from_env().root_path == "/endpoints/endpoint-xyz"
    monkeypatch.setenv("PRAVAH_API_ROOT_PATH", "")
    assert Settings.from_env().root_path == ""


def test_dashboard_loads_run_over_http_from_api(tmp_path, ckpt, monkeypatch):
    """Real API + real dashboard client + remote loader (HTTP routed into the ASGI app)."""
    import httpx
    import streamlit as st

    from app import api_client
    from app.remote_runs import list_remote_runs, load_remote_run

    s = _settings(tmp_path, ckpt)
    write_raster(s.scenes_dir / "scene.tif", width=128, height=128)
    base = "http://fuzzball-endpoint.test/endpoints/endpoint-abc"
    monkeypatch.setenv("PRAVAH_API_URL", base)
    monkeypatch.setenv("PRAVAH_FB_ENDPOINT_TOKEN", "fb-test-token")
    from dataclasses import replace

    seen_headers = []
    with TestClient(create_app(replace(s, root_path="/endpoints/endpoint-abc"))) as c:
        def route(method, url, **kw):
            seen_headers.append(kw.get("headers") or {})
            return c.request(method, url.replace("http://fuzzball-endpoint.test", ""),
                             json=kw.get("json"), headers=kw.get("headers"))

        monkeypatch.setattr(httpx, "request", route)
        st.cache_data.clear()
        job, err = api_client.submit_prediction("scene.tif", 0.5, "remote_run_1")
        assert err is None
        for _ in range(400):
            if c.get(f"/api/v1/predictions/{job['job_id']}").json()["status"] == "succeeded":
                break
            time.sleep(0.05)
        runs, err = list_remote_runs(api_client.api_url())
        assert err is None and runs == ["remote_run_1"]
        a = load_remote_run("remote_run_1")
    assert a.remote_source == base and not a.is_synthetic and not a.errors
    assert a.observed_metadata["run_id"] == "remote_run_1"
    assert a.observed_metadata["runtime"]["device"] == "cpu"
    assert a.water_geojson["type"] == "FeatureCollection"
    assert all(h.get("FB-Authorization") == "Bearer fb-test-token" for h in seen_headers)


def test_api_location_prediction_fetches_then_predicts(tmp_path, ckpt):
    """Fake fetcher (synthetic raster, no network) stands in for pravah.catalog.s2_fetch.fetch_scene."""
    s = _settings(tmp_path, ckpt)
    calls = []

    def fake_fetch(lat, lon, scenes_dir, *, place, half_km, lookback_days, max_cloud, units, progress):
        calls.append({"lat": lat, "lon": lon, "place": place, "half_km": half_km, "units": units})
        progress("Downloading synthetic bands", 0.5)
        (scenes_dir / "auto").mkdir(exist_ok=True)
        write_raster(scenes_dir / "auto/synthetic_crop.tif", width=96, height=96)
        side = {"scene_id": "auto/synthetic_crop.tif", "product_id": "SYNTHETIC_NOT_A_PRODUCT",
                "acquired_at_utc": "2026-11-12T05:02:21Z", "sensor": "S2", "processing_level": "L1C"}
        (scenes_dir / "auto/synthetic_crop.json").write_text(json.dumps(side))
        return side

    with TestClient(create_app(s, scene_fetcher=fake_fetch)) as c:
        bad = c.post("/api/v1/locations/predictions", json={"lat": 95, "lon": 84})
        assert bad.status_code == 422
        r = c.post("/api/v1/locations/predictions", json={"lat": 27.9828, "lon": 84.2656, "place": "Synthetic Town, Nowhere"})
        assert r.status_code == 202, r.text
        job = r.json()
        assert job["run_id"].startswith("loc_Synthetic_Town_") and job["scene_id"] == ""
        for _ in range(400):
            j = c.get(f"/api/v1/predictions/{job['job_id']}").json()
            if j["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.05)
        assert j["status"] == "succeeded", j
        assert j["scene_id"] == "auto/synthetic_crop.tif" and j["stage"] == "predicting"
        assert j["stage_detail"] == "Downloading synthetic bands"
        md = c.get(f"/api/v1/runs/{job['run_id']}").json()
        assert md["scene_id"] == "auto/synthetic_crop.tif" and md["product_id"] == "SYNTHETIC_NOT_A_PRODUCT"
        assert md["case_study"]["scope"] == "live_location" and md["case_study"]["region"] == "Synthetic Town"
        assert md["case_study"]["event_date"] == "12 Nov 2026"
    assert calls == [{"lat": 27.9828, "lon": 84.2656, "place": "Synthetic Town, Nowhere", "half_km": 5.0, "units": "dn"}]


def test_api_location_prediction_reports_fetch_failure(tmp_path, ckpt):
    s = _settings(tmp_path, ckpt)

    def no_scene(*_a, **_k):
        from pravah.catalog.s2_fetch import FetchError
        raise FetchError("No Sentinel-2 L1C scene in the last 60 days")

    with TestClient(create_app(s, scene_fetcher=no_scene)) as c:
        job = c.post("/api/v1/locations/predictions", json={"lat": 10, "lon": 10}).json()
        assert job["run_id"].startswith("loc_10p000_10p000_")
        for _ in range(200):
            j = c.get(f"/api/v1/predictions/{job['job_id']}").json()
            if j["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.05)
        assert j["status"] == "failed" and "No Sentinel-2 L1C scene" in j["error"]
