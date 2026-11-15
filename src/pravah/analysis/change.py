"""Pre-event vs post-event water change on a shared pixel grid.

Both inputs are class rasters with the contract 0 = no water, 1 = model-inferred
water, 255 = invalid (cloud, nodata, outside AOI). Comparison is only made where
**both** dates are valid; everything else is ``not_comparable`` (never "dry").

Output change codes (uint8):
    0 stable_no_water   1 persistent_water   2 new_water (0 -> 1)
    3 water_no_longer_detected (1 -> 0)      255 not_comparable

This is a change map between two *observations*. It is not a forecast, not a
skill score, and "new water" is model-inferred optical water that may include
sediment-laden flow, misclassified cloud/terrain shadow, or seasonal change.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from pravah.inference.engine import _pixel_area_m2, water_polygons

CHANGE_CODES = {
    "0": "stable_no_water",
    "1": "persistent_water",
    "2": "new_water",
    "3": "water_no_longer_detected",
    "255": "not_comparable",
}
NOT_COMPARABLE = 255


def _read(path: Path):
    with rasterio.open(path) as src:
        return src.read(1), src.profile.copy(), src.crs, src.transform


def change_map(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    if pre.shape != post.shape:
        raise ValueError("pre and post rasters differ in shape")
    for name, arr in (("pre", pre), ("post", post)):
        if not np.isin(arr, (0, 1, 255)).all():
            raise ValueError(f"{name} raster has values outside {{0, 1, 255}}")
    out = np.full(pre.shape, NOT_COMPARABLE, dtype=np.uint8)
    both = (pre != 255) & (post != 255)
    out[both & (pre == 0) & (post == 0)] = 0
    out[both & (pre == 1) & (post == 1)] = 1
    out[both & (pre == 0) & (post == 1)] = 2
    out[both & (pre == 1) & (post == 0)] = 3
    return out


def compare_runs(pre_dir: Path, post_dir: Path, out_dir: Path | None = None,
                 max_polygons: int = 5000) -> dict[str, Any]:
    """Compare two run folders (each with observed_water.tif + observed_metadata.json).

    Writes ``change_vs_pre.tif``, ``change_summary.json`` and
    ``change_new_water.geojson`` / ``change_lost_water.geojson`` into ``out_dir``
    (default: the post-event run folder).
    """
    pre_dir, post_dir = Path(pre_dir), Path(post_dir)
    out = Path(out_dir) if out_dir else post_dir
    pre, profile, crs, tr = _read(pre_dir / "observed_water.tif")
    post, _, crs2, tr2 = _read(post_dir / "observed_water.tif")
    if crs != crs2 or tr != tr2 or pre.shape != post.shape:
        raise ValueError("pre/post grids differ (CRS, transform or shape); resample explicitly before comparing")
    pre_md = json.loads((pre_dir / "observed_metadata.json").read_text())
    post_md = json.loads((post_dir / "observed_metadata.json").read_text())
    if pre_md.get("acquired_at_utc") and post_md.get("acquired_at_utc") and \
            pre_md["acquired_at_utc"] >= post_md["acquired_at_utc"]:
        raise ValueError("'pre' acquisition is not earlier than 'post'")

    ch = change_map(pre, post)
    px = _pixel_area_m2(crs, tr)
    counts = {CHANGE_CODES[str(k)]: int((ch == k).sum()) for k in (0, 1, 2, 3, 255)}
    comparable = sum(v for k, v in counts.items() if k != "not_comparable")

    def km2(n: int) -> float | None:
        return round(n * px / 1e6, 4) if px is not None else None

    pre_valid, post_valid = int((pre != 255).sum()), int((post != 255).sum())
    summary: dict[str, Any] = {
        "analysis": "pre_post_water_change",
        "pre_run_id": pre_md.get("run_id", pre_dir.name),
        "post_run_id": post_md.get("run_id", post_dir.name),
        "pre_scene_id": pre_md.get("scene_id"),
        "post_scene_id": post_md.get("scene_id"),
        "pre_acquired_at_utc": pre_md.get("acquired_at_utc"),
        "post_acquired_at_utc": post_md.get("acquired_at_utc"),
        "model_id": post_md.get("model_id") or post_md.get("model"),
        "same_model": (pre_md.get("model_sha256") == post_md.get("model_sha256")) if pre_md.get("model_sha256") else None,
        "crs": crs.to_string(),
        "pixel_area_m2": px,
        "grid_pixels": int(ch.size),
        "pixel_counts": counts,
        "area_km2": {k: km2(v) for k, v in counts.items()},
        "comparable_fraction_of_grid": round(comparable / ch.size, 6),
        "pre_valid_pixels": pre_valid,
        "post_valid_pixels": post_valid,
        "water_km2_within_comparable": {
            "pre": km2(counts["persistent_water"] + counts["water_no_longer_detected"]),
            "post": km2(counts["persistent_water"] + counts["new_water"]),
        },
        "change_codes": CHANGE_CODES,
        "interpretation_limits": [
            "Change between two model-derived optical observations; not a forecast or model skill score.",
            "Only pixels valid (cloud-free, observed) on both dates are compared; others are not_comparable.",
            "L1C cloud masks do not remove all cloud or terrain shadow; shadow can be misclassified as water.",
            "Sediment-laden debris flow differs spectrally from clear water and may be missed or mislabelled.",
            "Different acquisition geometry/illumination between dates can create apparent change.",
            "No independent Nepal water labels; UNOSAT extent (if added) is a different class (mudflow/rockflow).",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    prof = {**profile, "dtype": "uint8", "nodata": NOT_COMPARABLE, "count": 1, "compress": "deflate"}
    with rasterio.open(out / "change_vs_pre.tif", "w", **prof) as dst:
        dst.write(ch, 1)
    for code, name in ((2, "change_new_water.geojson"), (3, "change_lost_water.geojson")):
        # water_polygons vectorises pixels equal to 1, so remap the selected change code to 1
        gj, n = water_polygons(np.where(ch == code, 1, 0).astype(np.uint8), tr, crs, max_polygons=max_polygons)
        for f in gj["features"]:
            f["properties"].update({"class": CHANGE_CODES[str(code)], "label": CHANGE_CODES[str(code)].replace("_", " ")})
        (out / name).write_text(json.dumps(gj) + "\n")
        summary[f"{CHANGE_CODES[str(code)]}_polygons"] = {"total": n, "written": len(gj["features"])}
    (out / "change_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
