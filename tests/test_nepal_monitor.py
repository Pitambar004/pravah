"""Synthetic in-memory fixtures; never exported as Nepal observations."""

import json
from datetime import datetime, timedelta, timezone

import geopandas as gpd
import httpx
import pytest
from shapely.geometry import LineString, Point, mapping

from pravah.replay.live import LiveMonitor, changes, parse_gauges
from pravah.replay.scenario import explore, load_geography
from pravah.replay.water_watch import eligible, watch_water

NOW = datetime(2026, 11, 14, tzinfo=timezone.utc)


def station(value=None, date=None, **extra):
    return {"id": 1, "name": "Synthetic test station", "tags": ["RLS"], "latitude": 28., "longitude": 85.,
            "warning_level": "6", "danger_level": "7", "waterLevel": None if value is None else {"value": value, "datetime": date or NOW.isoformat()}, **extra}


def gauges(rows):
    return parse_gauges("var coordinates = " + json.dumps(rows) + ";", NOW)


def test_gauge_null_is_missing_and_schema_failure_is_not_outage():
    doc = gauges([station()])
    assert doc["records"][0]["level_m"] is None
    assert doc["records"][0]["quality"] == "missing"
    assert doc["empty_is_outage"] is False
    with pytest.raises(ValueError, match="schema changed"):
        parse_gauges("<html>provider error</html>", NOW)


def test_station_datums_stale_future_and_invalid_values():
    rows = gauges([station(359, warning_level="367"), station(100008),
                   station(6, (NOW - timedelta(hours=2)).isoformat()),
                   station(6, (NOW + timedelta(hours=1)).isoformat())])["records"]
    assert rows[0]["quality"] == "current" and rows[0]["level_m"] == 359
    assert rows[1]["level_m"] is None
    assert rows[2]["quality"] == "stale"
    assert rows[3]["quality"] == "future_timestamp"


def test_threshold_crossing_and_missing_sample_watch_dedupe():
    before = {**gauges([station(5.8)]), "retrieved_at_utc": NOW.isoformat()}
    after = {**gauges([station(6.2)]), "retrieved_at_utc": NOW.isoformat()}
    alerts = changes("gauges", before, after)
    assert len(alerts) == 1 and "provider warning" in alerts[0]["message"]
    assert changes("gauges", after, after) == []
    missing = {**gauges([station()]), "retrieved_at_utc": NOW.isoformat()}
    alert = changes("gauges", after, missing)[0]
    assert "unavailable" in alert["message"] and "failed" not in alert["message"]
    stale = {**gauges([station(8, (NOW - timedelta(hours=2)).isoformat())]), "retrieved_at_utc": NOW.isoformat()}
    assert changes("gauges", None, stale) == []


def test_catalog_baseline_is_not_new_event_and_preserves_landslide():
    doc = {"records": [{"id": "TEST", "event_type": "landslide", "observed_at_utc": NOW.isoformat()}], "retrieved_at_utc": NOW.isoformat()}
    assert changes("seismic", None, doc, first=True) == []
    assert "landslide" in changes("seismic", {"records": []}, doc)[0]["message"]


def test_monitor_persists_hashes_and_retains_evidence_after_failure(tmp_path):
    failed = False
    def fetch(name, now):
        if failed:
            raise httpx.ReadTimeout("Synthetic provider timeout")
        doc = {"records": [{"id": "TEST", "name": "Synthetic fixture", "quality": "missing"}], "retrieved_at_utc": now.isoformat()}
        return doc, json.dumps(doc).encode()
    monitor = LiveMonitor(tmp_path, fetch)
    assert monitor.refresh(force=True)
    before = monitor.snapshot()
    assert all(s["status"] == "ok" for s in before["sources"].values())
    assert all(len(s["response_sha256"]) == 64 for s in before["sources"].values())
    failed = True
    monitor.refresh(force=True)
    after = monitor.snapshot()
    assert all(s["status"] == "error" and s["records"] for s in after["sources"].values())
    assert len(after["alerts"]) == 4
    monitor.refresh(force=True)
    assert len(monitor.snapshot()["alerts"]) == 4
    assert LiveMonitor(tmp_path).snapshot()["updated_at_utc"] == monitor.snapshot()["updated_at_utc"]


def test_scenario_buffers_use_metres_and_missing_exposure_is_null():
    line = gpd.GeoSeries([LineString([(500000, 3100000), (500000, 3101000)])], crs="EPSG:32645").to_crs("EPSG:4326").iloc[0]
    polygon = gpd.GeoSeries([Point(500060, 3100500).buffer(5)], crs="EPSG:32645").to_crs("EPSG:4326").iloc[0]
    def doc(geom):
        return {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": mapping(geom), "properties": {"osm_id": "TEST"}}]}
    result = explore({"rivers": doc(line), "buildings": doc(polygon)}, 25, 50)
    assert result["exposure"]["building_footprints_intersecting"] == 1
    assert result["exposure"]["road_length_m_intersecting"] is None
    assert result["forecast"] is None and result["accuracy"] is None
    baseline = gpd.GeoDataFrame.from_features(result["baseline"]["features"], crs="EPSG:4326").to_crs("EPSG:32645").geometry.union_all()
    overflow = gpd.GeoDataFrame.from_features(result["overflow"]["features"], crs="EPSG:4326").to_crs("EPSG:32645").geometry.union_all()
    assert baseline.intersection(overflow).area < .001
    assert 49000 < baseline.area < 53000
    zero = explore({"rivers": doc(line)}, 25, 0)
    assert zero["overflow"]["features"] == []
    with pytest.raises(ValueError):
        explore({"rivers": doc(line)}, 25, 251)


def test_synthetic_geography_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"is_synthetic": True, "layers": {"rivers": {"file": "x.geojson", "sha256": "bad"}}}))
    (tmp_path / "x.geojson").write_text("{}")
    with pytest.raises(ValueError):
        load_geography(tmp_path)


def test_water_watch_never_consumes_benchmark_or_synthetic_or_unmasked(tmp_path):
    assert eligible({"is_synthetic": True, "case_study": {"region": "Nepal"}}) is False
    assert eligible({"is_synthetic": False, "case_study": {"region": "India", "scope": "benchmark"}}) is False
    assert eligible({"is_synthetic": False, "case_study": {"region": "Nepal"}}) is False
    assert watch_water(tmp_path, tmp_path)["status"] == "unavailable"


def test_osm_missing_tag_values_do_not_break_real_extract_processing():
    import runpy
    from pathlib import Path

    parse = runpy.run_path(str(Path(__file__).parents[1] / "scripts/nepal_replay/prepare_monitor.py"))["tags"]
    assert parse(float("nan")) == {} and parse(None) == {}
    assert parse('"bridge"=>"yes","name"=>"Synthetic fixture"')["bridge"] == "yes"
