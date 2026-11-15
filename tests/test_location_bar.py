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


def _synthetic_loc_run(root, name, acquired="2026-11-10T16:45:21Z"):
    """SYNTHETIC searched-place run: UTM 16N grid near the fake place, no real observation."""
    run = root / name
    run.mkdir()
    meta = {"case_study": location_case_study(LOC, acquired), "acquired_at_utc": acquired, "crs": "EPSG:32616",
            "transform": [10.0, 0.0, 414480.0, 0.0, -10.0, 4934340.0], "raster_width": 1008, "raster_height": 1008}
    (run / "observed_metadata.json").write_text(json.dumps(meta))
    return run


def test_searched_place_run_never_inherits_rasuwa_aoi_or_context(tmp_path, monkeypatch):
    from app.data_loader import load_run_artifacts
    from app.runs import run_role

    monkeypatch.chdir(tmp_path)  # config/aois/aoi_core.geojson stands in for the Rasuwa AOI
    (tmp_path / "config/aois").mkdir(parents=True)
    (tmp_path / "config/aois/aoi_core.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [85.3, 28.1]}}]}))
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "rivers.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    monkeypatch.setenv("PRAVAH_CONTEXT_DIR", str(ctx))

    a = load_run_artifacts(_synthetic_loc_run(tmp_path, "loc_Synthetic_Town_20261115T000000Z"))
    assert a.is_other_place and run_role(a) == "other_place" and "rivers" not in a.context_layers
    ring = a.aoi_geojson["features"][0]["geometry"]["coordinates"][0]
    lons, lats = [p[0] for p in ring], [p[1] for p in ring]
    assert min(lons) < LOC["lon"] < max(lons) and min(lats) < LOC["lat"] < max(lats)
    assert max(lats) - min(lats) == pytest.approx(10.08 / 111.0, rel=0.02)  # 1008 px × 10 m


def test_grid_footprint_rejects_missing_grid():
    from app.data_loader import grid_footprint_geojson

    assert grid_footprint_geojson(None) is None
    assert grid_footprint_geojson({"crs": "EPSG:32616", "transform": [10, 0, 0, 0, -10, 0]}) is None


def test_gallery_place_run_picks_matching_newest(tmp_path):
    from app.data_loader import load_run_artifacts
    from app.sections.gallery import place_run

    old = load_run_artifacts(_synthetic_loc_run(tmp_path, "loc_a", "2026-11-01T00:00:00Z"))
    new = load_run_artifacts(_synthetic_loc_run(tmp_path, "loc_b", "2026-11-10T00:00:00Z"))
    rasuwa = load_run_artifacts(tmp_path / "absent")
    assert place_run(PLACE, [old, new], rasuwa) is new
    assert place_run(PLACE, [old, new], old) is old  # the active run wins
    assert place_run({**PLACE, "lat": 1.0}, [old, new], rasuwa) is None
    assert place_run(None, [old, new], rasuwa) is None
