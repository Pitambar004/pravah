"""Unit tests for PRAVAH artifact loading and validation layer."""
import json
from pathlib import Path

from app.data_loader import (
    discover_available_runs,
    load_run_artifacts,
    safe_read_geojson,
    safe_read_json,
)


def test_safe_read_json_nonexistent(tmp_path: Path):
    data, err = safe_read_json(tmp_path / "nonexistent.json")
    assert data is None
    assert err is None


def test_safe_read_json_empty_file(tmp_path: Path):
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("   \n")
    data, err = safe_read_json(empty_file)
    assert data is None
    assert "empty" in err.lower()


def test_safe_read_json_malformed(tmp_path: Path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{ unquoted_key: 123 ")
    data, err = safe_read_json(bad_file)
    assert data is None
    assert "JSON parse error" in err


def test_safe_read_json_valid(tmp_path: Path):
    good_file = tmp_path / "good.json"
    good_file.write_text(json.dumps({"key": "value", "count": 42}))
    data, err = safe_read_json(good_file)
    assert data == {"key": "value", "count": 42}
    assert err is None


def test_safe_read_geojson_valid(tmp_path: Path):
    geo_file = tmp_path / "test.geojson"
    geo_content = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [85.3, 28.1]},
                "properties": {"name": "Test Point"}
            }
        ]
    }
    geo_file.write_text(json.dumps(geo_content))
    data, err = safe_read_geojson(geo_file)
    assert data is not None
    assert data["type"] == "FeatureCollection"
    assert err is None


def test_load_run_artifacts_empty_directory(tmp_path: Path):
    empty_dir = tmp_path / "run_empty"
    empty_dir.mkdir()
    
    # Without synthetic fallback
    artifacts = load_run_artifacts(empty_dir, allow_synthetic_fallback=False)
    assert not artifacts.has_any_artifacts
    assert not artifacts.is_synthetic
    assert artifacts.observed_metadata is None
    assert artifacts.exposure_summary is None
    assert artifacts.benchmark is None

    # With synthetic fallback
    synthetic_artifacts = load_run_artifacts(empty_dir, allow_synthetic_fallback=True)
    assert synthetic_artifacts.is_synthetic
    assert synthetic_artifacts.has_any_artifacts
    assert synthetic_artifacts.observed_metadata is not None
    assert synthetic_artifacts.exposure_summary is not None
    assert synthetic_artifacts.benchmark is not None


def test_load_run_artifacts_real_files(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    meta = {
        "run_id": "run_001",
        "threshold": 0.5,
        "valid_pixel_count": 1000,
        "invalid_pixel_count": 10,
    }
    (run_dir / "observed_metadata.json").write_text(json.dumps(meta))

    exp = {
        "water_area_m2": 50000.0,
        "building_footprints_intersecting": 5,
        "road_length_m_intersecting": 1200.0,
        "bridge_tagged_features_intersecting": 1,
        "unavailable_fields": ["population"],
    }
    (run_dir / "exposure_summary.json").write_text(json.dumps(exp))

    artifacts = load_run_artifacts(run_dir, allow_synthetic_fallback=False)
    assert not artifacts.is_synthetic
    assert artifacts.has_any_artifacts
    assert artifacts.observed_metadata["run_id"] == "run_001"
    assert artifacts.exposure_summary["building_footprints_intersecting"] == 5
    assert artifacts.benchmark is None
    assert "benchmark.json" in artifacts.missing_files


def test_discover_available_runs(tmp_path: Path):
    runs_dir = tmp_path / "outputs"
    runs_dir.mkdir()
    (runs_dir / "run_20260826_01").mkdir()
    (runs_dir / "run_20260826_02").mkdir()
    (runs_dir / "latest").mkdir()
    (runs_dir / ".hidden_dir").mkdir()
    (runs_dir / "some_file.txt").write_text("hello")

    discovered = discover_available_runs(runs_dir)
    assert "run_20260826_01" in discovered
    assert "run_20260826_02" in discovered
    assert "latest" in discovered
    assert ".hidden_dir" not in discovered
    assert "some_file.txt" not in discovered
