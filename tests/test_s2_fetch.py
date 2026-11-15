"""Sentinel-2 L1C live-crop fetcher. SYNTHETIC ONLY, no network.

The "mirror" is a pytest tmp dir of GeoTIFFs named like L1C .jp2 files and the
catalog is an httpx.MockTransport. Values are made up to test grid alignment,
radiometry and masking; nothing here is a real observation or product.
"""
from __future__ import annotations

import json

import httpx
import numpy as np
import pytest
import rasterio
import shapely
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from pravah.catalog import s2_fetch
from pravah.catalog.s2_fetch import (
    BANDS,
    FetchError,
    aoi_box,
    candidates,
    check_stac_scaling,
    fetch_scene,
    mirror_base,
    read_radiometry,
    window_for,
)

CRS = "EPSG:32645"
ULX, ULY, SIZE_M = 300000.0, 3100020.0, 12000  # synthetic 12 km "tile"
PRODUCT = "S2X_MSIL1C_20261101T000000_N0513_R000_T45RTL_20261101T000000"  # synthetic ID
GRANULE = "GRANULE/L1C_T45RTL_SYNTH"
MTD = """<n1:Level-1C_User_Product><QUANTIFICATION_VALUE unit="none">10000</QUANTIFICATION_VALUE>
<PROCESSING_BASELINE>05.13</PROCESSING_BASELINE>""" + "".join(
    f'<RADIO_ADD_OFFSET band_id="{i}">-1000</RADIO_ADD_OFFSET>' for i in range(13)) + "</n1:Level-1C_User_Product>"


def _centre_latlon():
    return Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(ULX + SIZE_M / 2, ULY - SIZE_M / 2)[::-1]


def _item(cloud=5.0, when="2026-11-01T05:00:00Z", scale=1e-4, offset=-0.1):
    inv = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    footprint = shapely.transform(box(ULX, ULY - SIZE_M, ULX + SIZE_M, ULY), inv.transform, interleaved=False)
    href = f"s3://eodata/x/{PRODUCT}.SAFE/{GRANULE}/IMG_DATA/T45RTL_20261101T000000_"
    assets = {k: {"href": href + k + ".jp2", "raster:scale": scale, "raster:offset": offset}
              for k in ("B02", "B03", "B04", "B8A", "B11", "B12")}
    assets["B02"].update({"proj:code": CRS, "proj:transform": [10, 0, ULX, 0, -10, ULY],
                          "proj:bbox": [ULX, ULY - SIZE_M, ULX + SIZE_M, ULY]})
    return {"id": PRODUCT, "collection": "sentinel-2-l1c", "geometry": mapping(footprint), "assets": assets,
            "properties": {"datetime": when, "published": "2026-11-01T09:00:00Z", "eo:cloud_cover": cloud}}


def _write(path, res, count=1, value=2000, dtype="uint16", edit=None):
    n = SIZE_M // res
    data = np.full((count, n, n), value, dtype=dtype)
    if edit:
        edit(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=count, dtype=dtype, crs=CRS,
                       transform=from_origin(ULX, ULY, res, res)) as dst:
        dst.write(data)


@pytest.fixture()
def mirror(tmp_path, monkeypatch):
    root = tmp_path / "mirror"
    safe = root / "tiles/45/R/TL" / (PRODUCT + ".SAFE")
    img = safe / GRANULE / "IMG_DATA"
    for key, res, val in (("B02", 10, 2000), ("B03", 10, 2100), ("B04", 10, 2200),
                          ("B8A", 20, 3000), ("B11", 20, 2500), ("B12", 20, 1500)):
        edit = (lambda d: d.__setitem__((0, 600, 600), 0)) if key == "B02" else None  # one sensor-nodata pixel
        _write(img / f"T45RTL_20261101T000000_{key}.jp2", res, value=val, edit=edit)
    # cloud flag (band 1 = opaque cloud) in a 60 m block near the centre
    _write(safe / GRANULE / "QI_DATA/MSK_CLASSI_B00.jp2", 60, count=3, value=0, dtype="uint8",
           edit=lambda d: d.__setitem__((0, slice(95, 97), slice(95, 97)), 1))
    (safe / "MTD_MSIL1C.xml").write_text(MTD)
    monkeypatch.setenv("PRAVAH_GCP_BUCKET_URL", str(root) + "/")
    monkeypatch.setenv("PRAVAH_CDSE_SEARCH_URL", "https://catalog.invalid/search")
    return root


def _client(items):
    return httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"features": items})))


def test_aoi_box_snaps_to_60m_grid_and_windows_align():
    b = aoi_box(ULX + 1234.5, ULY - 4321.0, 2000, ULX, ULY)
    assert all((v - ULX) % 60 == 0 for v in (b[0], b[2])) and all((ULY - v) % 60 == 0 for v in (b[1], b[3]))
    assert b[0] <= ULX + 1234.5 - 2000 and b[2] >= ULX + 1234.5 + 2000
    w10, w20, w60 = (window_for(b, ULX, ULY, r) for r in (10, 20, 60))
    assert (w10.col_off, w10.width) == (w20.col_off * 2, w20.width * 2) == (w60.col_off * 6, w60.width * 6)


def test_mirror_path_from_product_id(monkeypatch):
    monkeypatch.setenv("PRAVAH_GCP_BUCKET_URL", "https://mirror.invalid/bucket")
    assert mirror_base(PRODUCT) == f"https://mirror.invalid/bucket/tiles/45/R/TL/{PRODUCT}.SAFE/"
    with pytest.raises(FetchError):
        mirror_base("not_a_product")


def test_radiometry_and_catalog_cross_check():
    quant, offsets, baseline = read_radiometry(MTD)
    assert quant == 10000 and set(offsets.values()) == {-1000.0} and baseline == "05.13"
    check_stac_scaling(_item(), quant, offsets)
    with pytest.raises(FetchError, match="offset"):
        check_stac_scaling(_item(offset=0.0), quant, offsets)


def test_candidates_filter_cloud_and_coverage():
    lat, lon = _centre_latlon()
    assert len(candidates([_item()], lat, lon, 2000, 30)) == 1
    assert candidates([_item(cloud=80)], lat, lon, 2000, 30) == []
    assert candidates([_item()], lat, lon, 7000, 30) == []  # 14 km box does not fit the 12 km tile


def test_fetch_scene_writes_aligned_reflectance_with_masks(mirror, tmp_path):
    lat, lon = _centre_latlon()
    scenes = tmp_path / "scenes"
    msgs = []
    side = fetch_scene(lat, lon, scenes, place="Synthetic", half_km=2, client=_client([_item()]),
                       progress=lambda m, f: msgs.append(m))
    out = scenes / side["scene_id"]
    assert side["scene_id"].startswith("auto/T45RTL_20261101_") and out.is_file()
    assert json.loads(out.with_suffix(".json").read_text())["product_id"] == PRODUCT
    with rasterio.open(out) as src:
        assert src.count == 6 and src.res == (10, 10) and src.nodata == -9999 and list(src.descriptions) == BANDS
        assert (src.bounds.left - ULX) % 60 == 0 and (ULY - src.bounds.top) % 60 == 0
        data = src.read()
    valid = data[0] != -9999
    expected = [(v - 1000) / 10000 for v in (2000, 2100, 2200, 3000, 2500, 1500)]
    assert np.allclose(data[:, valid].astype(np.float64).mean(axis=1), expected, atol=1e-6)
    cloud_px = 2 * 6 * 2 * 6  # 2×2 60 m mask cells = 144 10 m pixels
    assert side["invalid_pixels"] == cloud_px + 1  # + the single DN == 0 pixel
    assert side["acquired_at_utc"] == "2026-11-01T05:00:00Z" and side["available_at_utc"] == "2026-11-01T09:00:00Z"
    assert side["is_synthetic"] is False  # flag describes the pipeline output schema; this test's input is synthetic
    assert any("Downloading 6 bands" in m for m in msgs)
    # second call reuses the cached crop without reading pixels again
    again = fetch_scene(lat, lon, scenes, half_km=2, client=_client([_item()]), progress=lambda m, f: msgs.append(m))
    assert again["scene_id"] == side["scene_id"] and msgs[-1].startswith("Reusing")


def test_fetch_scene_dn_units_scale_back(mirror, tmp_path):
    lat, lon = _centre_latlon()
    side = fetch_scene(lat, lon, tmp_path / "s", half_km=1, units="dn", client=_client([_item()]))
    with rasterio.open(tmp_path / "s" / side["scene_id"]) as src:
        b2 = src.read(1, masked=True)
    assert np.isclose(b2.mean(), 1000.0)


def test_fetch_scene_skips_cloudy_box(mirror, tmp_path, monkeypatch):
    lat, lon = _centre_latlon()
    monkeypatch.setattr(s2_fetch, "mask_invalid", lambda url, plan, hw=None: np.ones((10, 10), bool))
    with pytest.raises(FetchError, match="too cloudy"):
        fetch_scene(lat, lon, tmp_path / "s", half_km=1, client=_client([_item()]))


def test_fetch_scene_no_candidates(mirror, tmp_path):
    with pytest.raises(FetchError, match="No Sentinel-2 L1C scene"):
        fetch_scene(*_centre_latlon(), tmp_path / "s", half_km=1, client=_client([_item(cloud=99)]))
