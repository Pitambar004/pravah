"""Shared location bar logic. SYNTHETIC ONLY: fake jobs/run folders, no network or API."""
import json

import pytest

from app.components import location_bar as lb
from pravah.api.main import location_case_study

PLACE = {"name": "Synthetic Town, Nowhere", "lat": 44.51261, "lon": -88.01258, "bbox": [44.4, -88.1, 44.6, -87.9]}
LOC = {"lat": 44.51261, "lon": -88.01258, "place": PLACE["name"], "half_size_km": 5.0}


def test_job_progress_stages():
    assert lb.job_progress({"status": "queued"})[0] == 0.0
    f, text = lb.job_progress({"status": "running", "stage": "fetching", "stage_progress": 0.5, "stage_detail": "Downloaded 3/6 bands"})
    assert f == pytest.approx(0.3) and text == "Downloaded 3/6 bands"
    f, text = lb.job_progress({"status": "running", "stage": "predicting", "tiles_done": 5, "tiles_total": 10})
    assert f == pytest.approx(0.8) and "5/10 tiles" in text


def test_box_bounds_is_10km_square():
    (s, w), (n, e) = lb.box_bounds(PLACE)
    assert n - s == pytest.approx(10 / 111.32, rel=1e-6) and w < PLACE["lon"] < e


def test_same_point_rounds_to_4_decimals():
    assert lb.same_point({"lat": 44.51264, "lon": -88.01258}, PLACE)
    assert not lb.same_point({"lat": 44.6, "lon": -88.01258}, PLACE)
    assert not lb.same_point(None, PLACE)


def test_local_run_for_reads_case_study(tmp_path, monkeypatch):
    monkeypatch.setenv("PRAVAH_OUTPUTS_DIR", str(tmp_path))
    run = tmp_path / "loc_Synthetic_Town_20261115T000000Z"
    run.mkdir()
    (run / "observed_metadata.json").write_text(json.dumps({"case_study": location_case_study(LOC, "2026-11-10T16:45:21Z")}))
    (tmp_path / "loc_elsewhere").mkdir()
    (tmp_path / "loc_elsewhere/observed_metadata.json").write_text(json.dumps({"case_study": {"lat": 1, "lon": 2}}))
    assert lb.local_run_for(PLACE) == run.name


def test_place_state_submits_once_then_reports_pending(tmp_path, monkeypatch):
    monkeypatch.setenv("PRAVAH_OUTPUTS_DIR", str(tmp_path))
    jobs, submitted = [], []

    def submit(lat, lon, place, threshold=0.5):
        submitted.append(place)
        job = {"run_id": "loc_x", "status": "queued", "location": LOC}
        jobs.insert(0, job)
        return job, None

    monkeypatch.setattr(lb.api_client, "jobs", lambda: ({"jobs": jobs}, None))
    monkeypatch.setattr(lb.api_client, "submit_location_prediction", submit)
    assert lb.place_state(PLACE)["status"] == "pending"
    assert lb.place_state(PLACE)["status"] == "pending"
    assert submitted == [PLACE["name"]]
    jobs[0].update(status="failed", error="FetchError: too cloudy")
    state = lb.place_state(PLACE)
    assert state["status"] == "failed" and "too cloudy" in state["error"] and len(submitted) == 1
    jobs[0].update(status="succeeded")
    assert lb.place_state(PLACE) == {"status": "done", "job": jobs[0], "run_id": "loc_x"}


def test_location_case_study_labels():
    cs = location_case_study(LOC, "2026-11-10T16:45:21.024Z")
    assert cs["scope"] == "live_location" and cs["region"] == "Synthetic Town" and cs["event_date"] == "10 Nov 2026"
    assert location_case_study({**LOC, "place": None}, None)["region"] == "44.5126, -88.0126"
