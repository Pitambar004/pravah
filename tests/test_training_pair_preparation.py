"""Synthetic rasters test geometry and labels; they never leave the test temp dir."""

import runpy
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

MODULE = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/download_training_prepare.py")
)


def test_cloud_invalid_and_land_water_semantics():
    source = np.array([[[0, 1, 1, 2, 1]], [[2, 1, 2, 2, 0]]], dtype=np.int16)
    assert MODULE["worldfloods_label"](source).tolist() == [[-1, 0, 1, -1, -1]]
    with pytest.raises(ValueError):
        MODULE["worldfloods_label"](np.array([[[3]], [[1]]]))


def test_holdout_intersection_distinguishes_shared_edges():
    overlap = MODULE["overlapping"]
    protected = [(0, 0, 1, 1)]
    assert overlap((0.5, 0.5, 1.5, 1.5), protected)
    assert not overlap((1, 0, 2, 1), protected)
    assert not overlap((2, 2, 3, 3), protected)
    assert not overlap((0, 0, 1, 1), [])


def test_known_heldout_event_excluded_even_without_spatial_overlap():
    allowed = MODULE["eligible_combined_pair"]
    pair = {
        "dataset": "worldfloods_v2",
        "event_id": "synthetic_heldout_event",
        "split": "train",
        "bounds_wgs84": (10, 10, 11, 11),
    }
    assert not allowed(pair, [(0, 0, 1, 1)], [], {"synthetic_heldout_event"}, set())
    assert allowed(
        {**pair, "event_id": "synthetic_train_event"},
        [(0, 0, 1, 1)],
        [],
        {"synthetic_heldout_event"},
        set(),
    )
    assert not allowed(
        {**pair, "split": "val"}, [], [], set(), {"synthetic_heldout_event"}
    )


def test_vrt_preserves_six_bands_grid_and_edge_padding(tmp_path):
    source = tmp_path / "synthetic15.tif"
    values = np.stack([np.full((8, 9), i, dtype=np.int16) for i in range(1, 16)])
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=9,
        height=8,
        count=15,
        dtype="int16",
        crs="EPSG:32645",
        transform=from_origin(500000, 3000000, 10, 10),
    ) as dest:
        dest.write(values)
    target = tmp_path / "edge.vrt"
    with rasterio.open(source) as image:
        MODULE["write_vrt"](target, image, Window(4, 3, 5, 5), 512)
    with rasterio.open(target) as image:
        assert image.count == 6
        assert image.descriptions == MODULE["S2_BANDS"]
        assert image.transform == from_origin(500040, 2999970, 10, 10)
        actual = image.read()
        assert actual[:, 0, 0].tolist() == [2, 3, 4, 9, 12, 13]
        assert (
            actual[:, :5, :5] == np.array([2, 3, 4, 9, 12, 13])[:, None, None]
        ).all()
        assert not actual[:, 5:, :].any()
        assert not actual[:, :, 5:].any()


def test_misaligned_labels_are_rejected(tmp_path):
    paths = [tmp_path / "image.tif", tmp_path / "label.tif"]
    for path, count, x in [(paths[0], 13, 500000), (paths[1], 1, 500010)]:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=8,
            height=8,
            count=count,
            dtype="int16",
            crs="EPSG:32645",
            transform=from_origin(x, 3000000, 10, 10),
        ) as dest:
            dest.write(np.ones((count, 8, 8), dtype=np.int16))
    with (
        rasterio.open(paths[0]) as image,
        rasterio.open(paths[1]) as label,
        pytest.raises(ValueError, match="grid mismatch"),
    ):
        MODULE["check_grid"](image, label, "sen1floods11_hand")


def test_provider_coordinate_roundoff_is_not_resampled(tmp_path):
    paths = [tmp_path / "image.tif", tmp_path / "label.tif"]
    for path, count, x in [
        (paths[0], 13, -1.7017684742360213),
        (paths[1], 1, -1.7017684742360215),
    ]:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=8,
            height=8,
            count=count,
            dtype="int16",
            crs="EPSG:4326",
            transform=from_origin(
                x, 8.69, 8.983152841195215e-05, 8.983152841195215e-05
            ),
        ) as dest:
            dest.write(np.ones((count, 8, 8), dtype=np.int16))
    with rasterio.open(paths[0]) as image, rasterio.open(paths[1]) as label:
        MODULE["check_grid"](image, label, "sen1floods11_hand")
        assert MODULE["grid_displacement_pixels"](image, label) < 1e-6


def test_nonfinite_image_cannot_enter_baseline_manifest(tmp_path):
    image_path, label_path = tmp_path / "image.tif", tmp_path / "label.tif"
    pixels = np.ones((13, 8, 8), dtype=np.float32)
    pixels[1, 0, 0] = np.nan
    for path, values in [
        (image_path, pixels),
        (label_path, np.zeros((1, 8, 8), dtype=np.int16)),
    ]:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=8,
            height=8,
            count=values.shape[0],
            dtype=values.dtype,
            crs="EPSG:32645",
            transform=from_origin(500000, 3000000, 10, 10),
        ) as dest:
            dest.write(values)
    record = {
        "dataset": "sen1floods11_hand",
        "image": image_path,
        "label": label_path,
        "split": "train",
        "scene_id": "synthetic_nan",
    }
    with pytest.raises(ValueError, match="Nonfinite"):
        MODULE["process_scene"](record, tmp_path / "prepared", 512, 0, [], {})
    assert not (tmp_path / "prepared").exists()
