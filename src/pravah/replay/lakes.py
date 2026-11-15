"""Glacial-lake context and surface-area change; depth/level remain unknown."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import mapping, shape

from pravah.replay.experiment import sha256


def load_lakes(root: Path):
    metadata = json.loads((root / "manifest.json").read_text())
    if metadata.get("is_synthetic") is not False:
        raise ValueError("Lake context must be verified non-synthetic data")
    layers = {}
    for name, item in metadata["layers"].items():
        path = root / item["file"]
        if root.resolve() not in path.resolve().parents or path.is_symlink() or sha256(path) != item["sha256"]:
            raise ValueError("Lake inventory integrity mismatch")
        layers[name] = json.loads(path.read_text())
    return {"metadata": metadata, "layers": layers, "water_level_m": None, "depth_m": None}


def surface_change(pre_path: Path, post_path: Path, roi: dict):
    """Compare valid water masks within a nominated lake search polygon.

    Caller must verify dates, model/product/bands and cloud-validity metadata.
    This function enforces metric grids, nodata and class semantics. Values
    describe the common observed subset, not lake storage or a burst forecast.
    """
    with rasterio.open(pre_path) as pre, rasterio.open(post_path) as post:
        if pre.crs != post.crs or pre.transform != post.transform or pre.shape != post.shape:
            raise ValueError("Lake water masks need identical grids")
        if not pre.crs or not pre.crs.is_projected or pre.crs.linear_units != "metre":
            raise ValueError("Lake area measurement requires an explicit projected metric grid")
        if pre.nodata != 255 or post.nodata != 255 or pre.count != 1 or post.count != 1:
            raise ValueError("Expected single-band water classes with nodata=255")
        a, b = pre.read(1), post.read(1)
        if not np.isin(a, [0, 1, 255]).all() or not np.isin(b, [0, 1, 255]).all():
            raise ValueError("Water classes must be 0, 1, 255")
        polygon = gpd.GeoSeries([shape(roi.get("geometry", roi))], crs="EPSG:4326").to_crs(pre.crs).iloc[0]
        search = geometry_mask([mapping(polygon)], a.shape, pre.transform, invert=True)
        valid = search & (a != 255) & (b != 255)
        count, observed = int(search.sum()), int(valid.sum())
        pixel_area = abs(pre.transform.a * pre.transform.e - pre.transform.b * pre.transform.d)
        before = float(((a == 1) & valid).sum() * pixel_area) if observed else None
        after = float(((b == 1) & valid).sum() * pixel_area) if observed else None
        increase = after - before if observed else None
        return {"kind": "model_derived_lake_surface_change", "metric_crs": pre.crs.to_string(),
                "area_before_m2_common_valid": before, "area_after_m2_common_valid": after,
                "area_change_m2_common_valid": increase,
                "comparable_fraction_of_search": observed / count if count else None,
                "water_level_m": None, "depth_m": None, "volume_m3": None,
                "limit": "Surface area on commonly valid pixels. Snow/ice/shadow and mixed pixels can confound change; level needs elevation/altimetry, depth needs bathymetry. No burst prediction."}
