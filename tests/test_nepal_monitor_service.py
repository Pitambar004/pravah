"""API contracts exercised with synthetic provider responses in temporary storage."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pravah.replay.monitor_service import attach_monitor_routes


def test_nepal_routes_render_with_missing_sources(monkeypatch, tmp_path):
    from pathlib import Path

    import streamlit as st
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PRAVAH_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("PRAVAH_NEPAL_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("PRAVAH_OUTPUTS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("PRAVAH_NEPAL_GEOGRAPHY", str(tmp_path / "geography"))
    monkeypatch.setenv("PRAVAH_NATIONAL_ROOT", str(tmp_path / "national"))
    monkeypatch.setenv("PRAVAH_LAKE_ROOT", str(tmp_path / "lakes"))
    monkeypatch.setenv("PRAVAH_REPLAY_ROOT", str(tmp_path / "evidence"))
    monkeypatch.setenv("PRAVAH_RUN_SOURCE", "local")
    monkeypatch.setenv("PRAVAH_ENABLE_TEST_FIXTURES", "0")
    st.cache_data.clear()
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app/Home.py"), default_timeout=30).run()
    assert not app.exception  # default page is Live Watch: renders an honest "unavailable" with no API
    assert any("Live state unavailable" in w.value for w in app.warning)
    app.switch_page("views/nepal_monitor.py").run()
    assert not app.exception
    assert "Rasuwa Hydropower &amp; River Watch" in " ".join(m.value for m in app.markdown)
    app.switch_page("views/simulate.py").run()
    assert not app.exception
    assert "Prepared Nepal evidence is unavailable" in " ".join(m.value for m in app.info)
    st.cache_data.clear()


def test_monitor_routes_auth_missing_geography_and_no_forecast(tmp_path):
    app = attach_monitor_routes(FastAPI(), tmp_path / "live", tmp_path / "geo", token="test-only", enabled=False)
    def fetch(name, now):
        doc = {"is_synthetic": True, "records": [], "retrieved_at_utc": now.isoformat()}
        return doc, json.dumps(doc).encode()
    app.state.nepal_monitor.fetcher = fetch
    with TestClient(app) as client:
        state = client.get("/api/v1/nepal/live").json()
        assert state["forecast"] is None and state["forecast_model_loaded"] is False
        assert state["automatic_monitoring_enabled"] is False
        assert client.post("/api/v1/nepal/live/refresh").status_code == 401
        response = client.post("/api/v1/nepal/live/refresh", headers={"X-Pravah-Token": "test-only"})
        assert response.status_code == 200
        assert all(s["status"] == "empty" for s in response.json()["sources"].values())
        assert client.get("/api/v1/nepal/geography").status_code == 503
        assert client.get("/api/v1/nepal/lakes").status_code == 503
        assert client.get("/api/v1/nepal/rivers").status_code == 503
        assert client.get("/api/v1/nepal/imagery").status_code == 503
        assert client.post("/api/v1/nepal/lakes/change", json={}).status_code == 401
        assert client.post("/api/v1/nepal/scenarios", json={}).status_code == 401
        assert client.post("/api/v1/nepal/scenarios", json={}, headers={"X-Pravah-Token": "test-only"}).status_code == 503
        assert client.post("/api/v1/nepal/scenarios", json={"extra_margin_m": 251}, headers={"X-Pravah-Token": "test-only"}).status_code == 422
