"""prepare_s2_reflectance on tiny SYNTHETIC rasters: scaling, masking, checksum refusal."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

SPEC = importlib.util.spec_from_file_location(
    "prep", Path(__file__).parent.parent / "scripts" / "prepare_s2_reflectance.py")
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)

CRS = "EPSG:32645"
MTD = ('<x><PROCESSING_BASELINE>05.12</PROCESSING_BASELINE><QUANTIFICATION_VALUE unit="none">10000'
       '</QUANTIFICATION_VALUE>' + "".join(f'<RADIO_ADD_OFFSET band_id="{i}">-1000</RADIO_ADD_OFFSET>'
                                           for i in range(13)) + "</x>")


def _tif(path, arr, res):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[-2], width=arr.shape[-1],
                       count=1 if arr.ndim == 2 else arr.shape[0], dtype=arr.dtype, crs=CRS,
                       transform=from_origin(300000, 3100000, res, res)) as dst:
        dst.write(arr if arr.ndim == 3 else arr[None])


@pytest.fixture
def scene(tmp_path):
    safe, ref = tmp_path / "safe", tmp_path / "ref"
    safe.mkdir()
    ref.mkdir()
    dn = np.full((4, 4), 3000, dtype=np.uint16)
    dn[0, 0] = 0  # sensor nodata
    expected = {}
    for key in ["B02", "B03", "B04", "B8A", "B11", "B12"]:
        f = safe / f"T45XXX_20260101T000000_{key}.tif"  # GTiff stands in for jp2 (same reader path)
        _tif(f, dn, 10)
        f.rename(f.with_suffix(".jp2"))
        expected[key] = hashlib.sha256(f.with_suffix(".jp2").read_bytes()).hexdigest()
    mask = np.zeros((3, 4, 4), dtype=np.uint8)
    mask[0, 3, 3] = 1  # opaque cloud
    _tif(safe / "MSK_CLASSI_B00.tif", mask, 10)
    (safe / "MSK_CLASSI_B00.tif").rename(safe / "MSK_CLASSI_B00.jp2")
    (safe / "MTD_MSIL1C.xml").write_text(MTD)
    _tif(ref / "observed_water.tif", np.zeros((4, 4), dtype=np.uint8), 10)
    (ref / "observed_metadata.json").write_text(json.dumps({
        "run_id": "ref", "scene_id": "S", "expected_band_sha256": expected, "is_synthetic": False,
        "radiometry": {"quantification_value": 10000.0, "radio_add_offset": {"B2": -1000.0}}}))
    return safe, ref, tmp_path / "out.tif"


def test_scaling_and_invalid_mask(scene):
    safe, ref, out = scene
    report = prep.prepare(safe, ref, out)
    with rasterio.open(out) as src:
        arr = src.read()
        assert src.count == 6 and src.nodata == prep.NODATA
    assert arr[0, 1, 1] == pytest.approx(0.2)  # (3000 - 1000) / 10000
    assert arr[0, 0, 0] == prep.NODATA and arr[5, 3, 3] == prep.NODATA
    assert report["invalid_pixels"] == 2 and report["is_synthetic"] is False


def test_refuses_checksum_mismatch(scene):
    safe, ref, out = scene
    meta = json.loads((ref / "observed_metadata.json").read_text())
    meta["expected_band_sha256"]["B03"] = "0" * 64
    (ref / "observed_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="B03 checksum"):
        prep.prepare(safe, ref, out)


def test_refuses_synthetic_reference(scene):
    safe, ref, out = scene
    meta = json.loads((ref / "observed_metadata.json").read_text())
    meta["is_synthetic"] = True
    (ref / "observed_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="synthetic"):
        prep.prepare(safe, ref, out)
