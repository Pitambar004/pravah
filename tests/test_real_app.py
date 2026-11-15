"""Synthetic safeguards only; no test artifacts enter the live application."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.components.map_view import MapLayer
from app.data_loader import load_replay_catalog_with_source
from pravah.integration.real_engine import metric_crs


def test_metric_projection_follows_scene_country():
    assert metric_crs([93.7, 26.5, 93.8, 26.6]) == "EPSG:32646"
    assert metric_crs([-0.7, 38.1, -0.6, 38.2]) == "EPSG:32630"
    assert metric_crs([-95.0, 39.0, -94.9, 39.1]) == "EPSG:32615"
    assert metric_crs([-65, -18, -64.9, -17.9]) == "EPSG:32720"


def test_zero_exposure_layer_is_available_not_missing():
    layer = MapLayer("buildings", "Buildings", "Infrastructure", "derived", "blue",
                     data={"type": "FeatureCollection", "features": []})
    assert layer.available
    layer.data = None
    assert not layer.available


def test_stac_created_timestamp_is_not_verified_availability(tmp_path):
    p = tmp_path / "catalog.json"
    p.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "id": "SYNTHETIC_CATALOG_RECORD",
                        "properties": {
                            "datetime": "2020-01-01T00:00:00Z",
                            "created": "2020-01-02T00:00:00Z",
                            "published": "2020-01-02T00:00:00Z",
                        },
                    }
                ]
            }
        )
    )
    records, _ = load_replay_catalog_with_source(p)
    assert records[0]["available_at_utc"] is None


@pytest.mark.parametrize("stale_fixture_state", [False, True])
def test_production_has_no_fixture_toggle_or_fallback(
    monkeypatch, tmp_path, stale_fixture_state
):
    monkeypatch.setenv("PRAVAH_OUTPUTS_DIR", str(tmp_path))
    monkeypatch.setenv("PRAVAH_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("PRAVAH_RUN_SOURCE", "local")
    monkeypatch.setenv("PRAVAH_ENABLE_TEST_FIXTURES", "0")
    monkeypatch.setenv("PRAVAH_CONTEXT_DIR", str(tmp_path / "context"))
    monkeypatch.setenv("PRAVAH_USER_CONTEXT_DIR", str(tmp_path / "user_context"))
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "app/Home.py"), default_timeout=30
    )
    app.session_state["pv_demo"] = stale_fixture_state
    app.run()
    assert not app.exception
    assert not app.sidebar.toggle
    assert app.session_state["pv_demo"] is False
    assert "Synthetic fixture mode" not in " ".join(m.value for m in app.markdown)
