"""Synthetic raster fixtures stay in pytest's temporary directory."""

import runpy
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

MODULE = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/render_local_results.py")
)


def test_cloud_mask_is_evaluation_only_and_dry_is_not_nodata(tmp_path):
    image = tmp_path / "synthetic_image.tif"
    label = tmp_path / "synthetic_label.tif"
    grid = from_origin(500000, 3000000, 10, 10)
    channels = np.ones((6, 2, 2), dtype=np.int16) * 1000
    channels[:, 0, 0] = 0  # source-invalid
    truth = np.array([[[-1, -1], [0, 1]]], dtype=np.int16)
    for path, values, nodata in [(image, channels, 0), (label, truth, -1)]:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=values.shape[0],
            dtype="int16",
            crs="EPSG:32645",
            transform=grid,
            nodata=nodata,
        ) as dest:
            dest.write(values)
    report = MODULE["export_prediction"](
        image,
        label,
        np.array([[1, 1], [0, 1]], dtype=np.uint8),
        tmp_path / "synthetic_output",
        {
            "name": "SYNTHETIC UNIT TEST — NOT OBSERVATIONS",
            "scene_id": "synthetic",
            "split": "unit test",
        },
    )
    assert report["counts"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 1, "valid": 2}
    with rasterio.open(tmp_path / "synthetic_output/model_water.tif") as exported:
        assert exported.nodata == 255
        assert exported.transform == grid
        assert exported.crs.to_epsg() == 32645
        # Reference-invalid/cloud remains a prediction in the raw image-only export.
        assert exported.read(1).tolist() == [[255, 1], [0, 1]]
    with rasterio.open(tmp_path / "synthetic_output/reference_valid_mask.tif") as valid:
        assert valid.read(1).tolist() == [[0, 0], [1, 1]]
