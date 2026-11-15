"""Pre/post change and exposure feature export. SYNTHETIC rasters/vectors in tmp dirs only."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

from pravah.analysis.change import change_map, compare_runs

ROOT = Path(__file__).resolve().parent.parent
TR = from_origin(330000.0, 3120000.0, 10.0, 10.0)


def _run(d: Path, arr: np.ndarray, acquired: str) -> Path:
    d.mkdir(parents=True)
    with rasterio.open(d / "observed_water.tif", "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0],
                       count=1, dtype="uint8", crs="EPSG:32645", transform=TR, nodata=255) as dst:
        dst.write(arr, 1)
    (d / "observed_metadata.json").write_text(json.dumps({"run_id": d.name, "acquired_at_utc": acquired,
                                                          "model_sha256": "m"}))
    return d


def test_change_codes_only_where_both_valid():
    pre = np.array([[0, 0, 1, 1, 255, 0]], dtype=np.uint8)
    post = np.array([[0, 1, 1, 0, 1, 255]], dtype=np.uint8)
    assert change_map(pre, post).tolist() == [[0, 2, 1, 3, 255, 255]]
    with pytest.raises(ValueError):
        change_map(pre, np.array([[0, 7, 1, 0, 1, 255]], dtype=np.uint8))


def test_compare_runs_writes_layers_and_summary(tmp_path):
    pre = np.zeros((40, 40), np.uint8); pre[:, :5] = 255; pre[10:20, 10:20] = 1
    post = np.zeros((40, 40), np.uint8); post[30:, :] = 255; post[0:5, 25:35] = 1; post[10:20, 10:15] = 1
    a = _run(tmp_path / "pre", pre, "2026-08-11T04:52:41Z")
    b = _run(tmp_path / "post", post, "2026-08-27T04:56:59Z")
    s = compare_runs(a, b)
    pc = s["pixel_counts"]
    assert pc["new_water"] == 50 and pc["persistent_water"] == 50 and pc["water_no_longer_detected"] == 50
    assert pc["not_comparable"] == int(((pre == 255) | (post == 255)).sum())
    assert s["area_km2"]["new_water"] == pytest.approx(50 * 100 / 1e6)
    for f in ("change_vs_pre.tif", "change_summary.json", "change_new_water.geojson", "change_lost_water.geojson"):
        assert (b / f).is_file()
    with pytest.raises(ValueError, match="earlier"):
        compare_runs(b, a, tmp_path / "x")

    from app.components.map_view import build_layers
    from app.data_loader import load_run_artifacts
    art = load_run_artifacts(b)
    layers = {lyr.id: lyr for lyr in build_layers(art)}
    assert layers["change_new"].available and layers["change_lost"].available
    assert layers["change_new"].group == "Change vs pre-event"


def test_exposure_feature_export(tmp_path):
    spec = importlib.util.spec_from_file_location("exposure_cli", ROOT / "scripts/exposure.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    crs = "EPSG:32645"
    water = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=crs)
    buildings = gpd.GeoDataFrame({"osm_id": ["b1", "b2"], "building": ["yes", "house"]},
                                 geometry=[box(90, 90, 110, 110), box(500, 500, 510, 510)], crs=crs)
    roads = gpd.GeoDataFrame({"osm_id": ["r1"], "highway": ["primary"]},
                             geometry=[LineString([(-50, 50), (150, 50)])], crs=crs)
    n = mod.export_feature_layers(water, {"buildings": buildings, "roads": roads, "bridges": None}, crs, tmp_path)
    assert n == {"buildings": 1, "roads": 1}
    assert not (tmp_path / "exposure_bridges.geojson").exists()  # no input -> not claimed as "none"
    r = json.loads((tmp_path / "exposure_roads.geojson").read_text())["features"][0]["properties"]
    assert r["clipped_length_m"] == pytest.approx(100.0)
    b = json.loads((tmp_path / "exposure_buildings.geojson").read_text())["features"][0]
    assert b["properties"]["osm_id"] == "b1" and b["properties"]["intersection_area_m2"] == pytest.approx(100.0)
    assert "not damage" in b["properties"]["label"]


def test_context_layers_load_for_real_runs_not_fixtures(tmp_path, monkeypatch):
    from app.components.map_view import build_layers
    from app.data_loader import load_run_artifacts

    ctx = tmp_path / "context"
    ctx.mkdir()
    river = {"type": "FeatureCollection", "pravah_layer": {"source": "OpenStreetMap snapshot 2026-01-01 (Geofabrik)"},
             "features": [{"type": "Feature", "properties": {"label": "Test river"},
                           "geometry": {"type": "LineString", "coordinates": [[85.3, 28.1], [85.31, 28.12]]}}]}
    (ctx / "rivers.geojson").write_text(json.dumps(river))
    monkeypatch.setenv("PRAVAH_CONTEXT_DIR", str(ctx))
    run = _run(tmp_path / "r", np.zeros((8, 8), np.uint8), "2026-08-27T00:00:00Z")
    layers = {lyr.id: lyr for lyr in build_layers(load_run_artifacts(run))}
    assert layers["rivers"].available and "2026-01-01" in layers["rivers"].source
    assert not layers["admin"].available
    fixture = load_run_artifacts(tmp_path / "missing", allow_synthetic_fallback=True)
    assert fixture.is_synthetic and fixture.context_layers == {}


def test_osm_identifier_uses_way_id_for_closed_way_polygons():
    import pandas as pd

    spec = importlib.util.spec_from_file_location("p3_infra", ROOT / "scripts/p3_build_infrastructure.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    way_poly = pd.Series({"osm_id": float("nan"), "osm_way_id": "123", "other_tags": None})
    relation = pd.Series({"osm_id": "456", "osm_way_id": None, "other_tags": None})
    line = pd.Series({"osm_id": "789", "other_tags": None})
    assert mod.osm_identifier(way_poly) == "way/123"
    assert mod.osm_identifier(relation) == "456"
    assert mod.osm_identifier(line) == "789"
