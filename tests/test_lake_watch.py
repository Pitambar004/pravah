"""Synthetic masks/catalogs in temporary test storage; never event observations."""

import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from pravah.replay.cascade import acquisition_footprints, daily_history
from pravah.replay.lakes import surface_change


def test_lake_surface_change_measures_common_area_and_never_depth(tmp_path):
    profile = {"driver": "GTiff", "width": 4, "height": 4, "count": 1, "dtype": "uint8",
               "crs": "EPSG:32645", "transform": from_origin(500000, 3100000, 10, 10), "nodata": 255}
    a = np.zeros((4,4), dtype="uint8"); b = a.copy()
    a[0,0] = 1; b[0,:3] = 1; b[3,3] = 255
    files = [tmp_path / "pre.tif", tmp_path / "post.tif"]
    for path, values in zip(files, [a,b]):
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(values, 1)
    geo = gpd.GeoSeries([box(500000, 3099960, 500040, 3100000)], crs="EPSG:32645").to_crs("EPSG:4326").iloc[0]
    result = surface_change(*files, mapping(geo))
    assert result["area_before_m2_common_valid"] == 100
    assert result["area_after_m2_common_valid"] == 300
    assert result["area_change_m2_common_valid"] == 200
    assert result["comparable_fraction_of_search"] == 15/16
    assert result["depth_m"] is None and result["water_level_m"] is None and result["volume_m3"] is None
    with rasterio.open(files[1], "w", **profile) as dst:
        dst.write(np.full((4,4), 255, dtype="uint8"), 1)
    missing = surface_change(*files, mapping(geo))
    assert missing["area_after_m2_common_valid"] is None
    assert missing["area_change_m2_common_valid"] is None


def test_lake_change_rejects_grid_mismatch_and_geographic_units(tmp_path):
    profile = {"driver": "GTiff", "width": 2, "height": 2, "count": 1, "dtype": "uint8",
               "crs": "EPSG:4326", "transform": from_origin(85, 28, .01, .01), "nodata": 255}
    paths = [tmp_path / "a.tif", tmp_path / "b.tif"]
    for path in paths:
        with rasterio.open(path,"w",**profile) as dst: dst.write(np.zeros((2,2),dtype="uint8"),1)
    with pytest.raises(ValueError, match="metric grid"):
        surface_change(*paths, mapping(box(85,27,86,28)))


def test_15_day_replay_never_fills_missing_lake_or_slope_series(tmp_path):
    source = tmp_path / "sources"; source.mkdir()
    (source / "august_era5_rasuwa.json").write_text(json.dumps({"daily": {"time": ["2026-08-11", "2026-08-26"], "precipitation_sum": [2., 10.]}}))
    scene = {"id": "SYNTHETIC_CATALOG_TEST", "geometry": mapping(box(85,28,86,29)), "properties": {"datetime": "2026-08-12T04:00:00Z"}}
    (source / "rasuwa_l1c_catalog.json").write_text(json.dumps({"items": [scene]}))
    history = daily_history(tmp_path)
    assert len(history["rows"]) == 16 and history["start_date"] == "2026-08-11"
    assert history["rows"][0]["rainfall_mm_retrospective_reanalysis"] == 2.
    assert history["rows"][1]["optical_catalog_records"] == 1
    assert all(row["daily_slope_displacement_m"] is None and row["lake_surface_area_m2"] is None for row in history["rows"])
    assert acquisition_footprints(tmp_path,"2026-08-11T08:30:00+05:45")["features"] == []
    assert len(acquisition_footprints(tmp_path,"2026-08-12T12:00:00+05:45")["features"]) == 1
