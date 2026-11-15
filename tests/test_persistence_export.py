"""All rasters and metadata here are synthetic test fixtures, never disaster evidence."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from export_persistence_prediction import export


@pytest.fixture
def source(tmp_path):
    pre = tmp_path / "TEST_ONLY_pre"
    pre.mkdir()
    raster = pre / "observed_water.tif"
    with rasterio.open(raster, "w", driver="GTiff", count=1, dtype="uint8",
                       width=3, height=1, crs="EPSG:32645", nodata=255,
                       transform=from_origin(300000, 3100000, 10, 10)) as dst:
        dst.write(np.array([[0, 1, 255]], dtype=np.uint8), 1)
    meta = {"is_synthetic": False,  # deliberately exercises production admission in temporary test storage
            "source_temporal_role": "pre_event", "scene_id": "TEST_ONLY",
            "acquired_at_utc": "2026-08-11T00:00:00Z", "available_at_utc": "2026-08-12T00:00:00Z",
            "availability_basis": "TEST_ONLY", "catalog_uri": "https://example.invalid/fixture",
            "output_sha256": {raster.name: hashlib.sha256(raster.read_bytes()).hexdigest()}}
    (pre / "observed_metadata.json").write_text(json.dumps(meta))
    return pre, meta


def test_export_preserves_classes_grid_and_actual_issue_time(source, tmp_path):
    pre, _ = source
    out = tmp_path / "TEST_ONLY_prediction"
    meta = export(pre, out, "2026-08-25T23:59:59Z")
    assert meta["evaluation_mode"] == "retrospective_reconstruction"
    assert meta["issued_at_utc"] == meta["created_at_utc"]
    assert meta["issued_at_utc"] > meta["input_cutoff_utc"]
    with rasterio.open(out / "predicted_flood.tif") as dst, rasterio.open(pre / "observed_water.tif") as src:
        np.testing.assert_array_equal(dst.read(), src.read())
        assert dst.transform == src.transform and dst.crs == src.crs and dst.nodata == 255
    with pytest.raises(FileExistsError):
        export(pre, out, "2026-08-25T23:59:59Z")


@pytest.mark.parametrize("key,value", [
    ("is_synthetic", True), ("is_synthetic", None),
    ("source_temporal_role", "post_event"),
    ("available_at_utc", None), ("available_at_utc", "2026-08-26T00:00:00Z"),
    ("acquired_at_utc", "2026-08-26T00:00:00Z"),
    ("available_at_utc", "2026-08-10T00:00:00Z"),
    ("available_at_utc", "2026-08-12T00:00:00"),
    ("availability_basis", None), ("output_sha256", {}),
])
def test_rejects_ineligible_or_unaudited_sources(source, tmp_path, key, value):
    pre, meta = source
    meta[key] = value
    (pre / "observed_metadata.json").write_text(json.dumps(meta))
    out = tmp_path / "TEST_ONLY_prediction"
    with pytest.raises(ValueError):
        export(pre, out, "2026-08-25T23:59:59Z")
    assert not out.exists()
