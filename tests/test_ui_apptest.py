"""End-to-end Streamlit AppTest — every route in fixture mode and empty mode."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="Streamlit required for AppTest suite")
from streamlit.testing.v1 import AppTest

HOME_PATH = (Path(__file__).parent.parent / "app" / "Home.py").resolve()

PAGES = [
    "views/overview.py",
    "views/flood_map.py",
    "views/replay.py",
    "views/exposure.py",
    "views/inference.py",
    "views/compute.py",
    "views/methodology.py",
]


@pytest.fixture(autouse=True)
def _isolated_data(monkeypatch, tmp_path):
    """Never depend on the developer's outputs/ or a locally running API."""
    import streamlit as st

    empty = tmp_path / "outputs"
    empty.mkdir()
    monkeypatch.setenv("PRAVAH_OUTPUTS_DIR", str(empty))
    monkeypatch.setenv("PRAVAH_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("PRAVAH_CONTEXT_DIR", str(tmp_path / "context"))
    monkeypatch.delenv("PRAVAH_RUN_SOURCE", raising=False)
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _app(page: str, demo: bool) -> AppTest:
    at = AppTest.from_file(str(HOME_PATH), default_timeout=30)
    at.run()
    if not demo:
        at.sidebar.toggle[0].set_value(False)
        at.run()
    at.switch_page(page)
    at.run()
    return at


def _text(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def test_home_page_renders():
    at = AppTest.from_file(str(HOME_PATH), default_timeout=30)
    at.run()
    assert not at.exception


@pytest.mark.parametrize("page", PAGES)
def test_route_renders_with_fixtures(page):
    at = _app(page, demo=True)
    assert not at.exception, f"{page} failed in fixture mode: {at.exception}"
    assert "Synthetic fixture mode" in _text(at)


@pytest.mark.parametrize("page", PAGES)
def test_route_renders_empty(page):
    at = _app(page, demo=False)
    assert not at.exception, f"{page} failed in empty mode: {at.exception}"
    assert "Synthetic fixture" not in _text(at), f"{page} shows fixture content with fixtures disabled"


def test_empty_mode_never_shows_synthetic_scene_ids():
    """With fixtures off, replay must not fall back to synthetic scenes."""
    at = _app("views/replay.py", demo=False)
    text = _text(at)
    assert "S2A_MSIL1C_20260816" not in text
    assert "No source manifest available" in text


def test_strict_replay_hides_later_evidence():
    at = _app("views/replay.py", demo=True)
    text = _text(at)
    # Default cutoff 26 Aug 11:00 NPT: post-event scene and UNOSAT must be hidden.
    assert "S2B_MSIL1C_20260827T051219" not in text
    assert "UNOSAT_NEPAL_MUDFLOW_ROCKFLOW_EXTENT_AUG28" not in text
    # Pre-event baseline is eligible and shown.
    assert "S2A_MSIL1C_20260816T045701" in text
    # Unknown-availability event pass acquired before cutoff is shown as excluded.
    assert "S2A_MSIL1C_20260826T050122_EVENT_UNCONFIRMED_AVAILABILITY" in text


def test_inference_page_reports_unreachable_api(monkeypatch):
    monkeypatch.setenv("PRAVAH_API_URL", "http://127.0.0.1:9")  # discard port: nothing listens
    at = _app("views/inference.py", demo=False)
    assert not at.exception
    assert "Inference API unavailable" in _text(at)


def _fake_status(model_loaded: bool):
    return {
        "model_loaded": model_loaded, "require_cuda": True,
        "model_error": None if model_loaded else "Checkpoint not found: /data/models/checkpoints/checkpoint_last.pt",
        "model": {"architecture": "MiniUNet", "checkpoint_sha256": "ab" * 32, "epoch": 2, "steps": 10,
                  "bands": ["B2", "B3", "B4", "B8A", "B11", "B12"]} if model_loaded else None,
        "runtime": {"device": "cuda", "cuda_verified": True, "gpu_name": "TEST-GPU-NAME",
                    "torch_version": "x", "torch_cuda_version": "y", "hostname": "h", "machine": "aarch64"}
        if model_loaded else None,
        "storage": {"outputs_dir": {"path": "/data/outputs/predictions", "writable": True},
                    "checkpoint": {"path": "/data/models/checkpoints/checkpoint_last.pt", "exists": model_loaded}},
        "queue": {"pending": 0, "max": 8},
    }


@pytest.mark.parametrize("loaded", [True, False])
def test_inference_page_with_api(monkeypatch, loaded):
    """API responses are stubbed in-process; values are test strings, not hardware facts."""
    from app import api_client

    monkeypatch.setattr(api_client, "status", lambda: (_fake_status(loaded), None))
    monkeypatch.setattr(api_client, "scenes", lambda: ({"scenes": [
        {"scene_id": "s2/scene.tif", "model_compatible": True, "band_count": 13, "crs": "EPSG:32645"}]}, None))
    monkeypatch.setattr(api_client, "jobs", lambda: ({"jobs": []}, None))
    at = _app("views/inference.py", demo=False)
    assert not at.exception, at.exception
    text = _text(at)
    if loaded:
        assert "TEST-GPU-NAME" in text and "CUDA verified" in text
        assert at.button(key="FormSubmitter:pv_predict-Run on GPU").disabled is False
    else:
        assert "No trained checkpoint loaded" in text
        assert at.button(key="FormSubmitter:pv_predict-Run inference").disabled is True


def test_map_uses_api_runs_when_no_local_runs(monkeypatch, tmp_path):
    """Remote mode: run list + metadata + polygons come from the API (stubbed, test values)."""
    import streamlit as st

    from app import api_client

    monkeypatch.setenv("PRAVAH_OUTPUTS_DIR", str(tmp_path))  # empty local storage
    monkeypatch.setenv("PRAVAH_API_URL", "http://api.test")
    geo = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"label": "w"},
           "geometry": {"type": "Polygon", "coordinates": [[[85.3, 28.1], [85.31, 28.1], [85.31, 28.11],
                                                            [85.3, 28.1]]]}}]}
    monkeypatch.setattr(api_client, "runs", lambda: ({"runs": [{"run_id": "TEST_REMOTE_RUN", "complete": True}]}, None))
    monkeypatch.setattr(api_client, "run_metadata", lambda r: ({"run_id": r, "threshold": 0.5}, None))
    monkeypatch.setattr(api_client, "run_json_file", lambda r, n: (geo, None))
    st.cache_data.clear()
    at = _app("views/flood_map.py", demo=True)  # demo flag unused: fixtures default off with remote runs
    assert not at.exception, at.exception
    text = _text(at)
    assert "TEST_REMOTE_RUN" in text and "Run artifacts loaded" in text
    assert "Synthetic fixture mode" not in text
    st.cache_data.clear()
