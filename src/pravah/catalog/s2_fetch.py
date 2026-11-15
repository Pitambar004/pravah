"""Fetch a small Sentinel-2 L1C crop around a point so the live model can run there.

Pipeline (no credentials needed):

1. **Discover** — the public CDSE STAC catalog (``PRAVAH_CDSE_SEARCH_URL``) is searched
   for ``sentinel-2-l1c`` items over the point within ``lookback_days``, newest first,
   with tile-wide ``eo:cloud_cover`` <= ``max_cloud``. Only items whose tile grid
   (``proj:bbox``) and data footprint fully contain the requested box are kept.
2. **Pre-check clouds** — the 60 m ``MSK_CLASSI_B00.jp2`` (opaque cloud, cirrus,
   snow/ice) is read for just the box; scenes with too much flagged area are skipped
   before any band pixels are downloaded.
3. **Download** — windowed HTTP range reads of B02, B03, B04 (10 m) and B8A, B11, B12
   (20 m, nearest-upsampled onto the 10 m grid) from the public Google Cloud mirror
   (``PRAVAH_GCP_BUCKET_URL``). The box is snapped to the tile's 60 m grid so every
   band and the mask align exactly; no reprojection happens.
4. **Radiometry** — TOA reflectance = (DN + RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE,
   read from the product's own ``MTD_MSIL1C.xml`` and cross-checked against the STAC
   ``raster:scale``/``raster:offset``; a mismatch refuses the scene.
5. **Write** a 6-band ``[B2, B3, B4, B8A, B11, B12]`` float32 GeoTIFF (nodata -9999 where
   DN == 0 or the mask flags cloud/cirrus/snow) plus a ``.json`` sidecar recording
   product ID, acquisition and publication times, URIs, scaling and pixel counts.

This is optical Sentinel-2 L1C only (never L2A or Sentinel-1). Cloud/shadow masking is
the provider's L1C classification and does not catch every cloud shadow. The output is
an input image for retrospective water segmentation, not a forecast.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import rasterio
import shapely
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import box, shape

from pravah.configuration import configured_url

BANDS = ["B2", "B3", "B4", "B8A", "B11", "B12"]
FILE_KEYS = {"B2": "B02", "B3": "B03", "B4": "B04", "B8A": "B8A", "B11": "B11", "B12": "B12"}
BAND_IDS = {"B2": 1, "B3": 2, "B4": 3, "B8A": 8, "B11": 11, "B12": 12}  # band_id in MTD_MSIL1C.xml
RES_M = {"B2": 10, "B3": 10, "B4": 10, "B8A": 20, "B11": 20, "B12": 20}
MASK_RES_M = 60
NODATA = -9999.0
COLLECTION = "sentinel-2-l1c"
TILE_RE = re.compile(r"_T(\d{2})([C-X])([A-Z]{2})_")
GDAL_HTTP_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".jp2",
                 "GDAL_HTTP_MULTIRANGE": "YES", "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
                 "GDAL_HTTP_MAX_RETRY": "3", "GDAL_HTTP_RETRY_DELAY": "1"}

Progress = Callable[[str, float], None]  # (message, fraction of the fetch done 0..1)


class FetchError(RuntimeError):
    """No usable scene, or a scene failed a provenance/radiometry check."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


# ------------------------------------------------------------------- geometry
def aoi_box(x: float, y: float, half_m: float, ulx: float, uly: float) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) around (x, y), snapped outward to the tile's 60 m grid."""
    xmin = ulx + math.floor((x - half_m - ulx) / MASK_RES_M) * MASK_RES_M
    xmax = ulx + math.ceil((x + half_m - ulx) / MASK_RES_M) * MASK_RES_M
    ymax = uly - math.floor((uly - (y + half_m)) / MASK_RES_M) * MASK_RES_M
    ymin = uly - math.ceil((uly - (y - half_m)) / MASK_RES_M) * MASK_RES_M
    return xmin, ymin, xmax, ymax


def window_for(bounds: tuple[float, float, float, float], ulx: float, uly: float, res: float) -> Window:
    xmin, ymin, xmax, ymax = bounds
    return Window(round((xmin - ulx) / res), round((uly - ymax) / res),
                  round((xmax - xmin) / res), round((ymax - ymin) / res))


def tile_parts(product_id: str) -> tuple[str, str, str]:
    m = TILE_RE.search(product_id)
    if not m:
        raise FetchError(f"Cannot read the MGRS tile from product ID {product_id}")
    return m.group(1), m.group(2), m.group(3)


def safe_relative(href: str) -> str:
    """Path inside the .SAFE folder from a STAC asset href (s3://… or https://…)."""
    if ".SAFE/" not in href:
        raise FetchError(f"Unexpected asset location {href}")
    return href.split(".SAFE/", 1)[1]


def mirror_base(product_id: str) -> str:
    zone, band, square = tile_parts(product_id)
    root = configured_url("PRAVAH_GCP_BUCKET_URL").rstrip("/") + "/"
    return f"{root}tiles/{int(zone)}/{band}/{square}/{product_id}.SAFE/"


def _item_grid(item: dict[str, Any]) -> tuple[str, float, float, list[float]]:
    a = item.get("assets", {}).get("B02", {})
    code, tr, bb = a.get("proj:code") or item.get("properties", {}).get("proj:code"), a.get("proj:transform"), a.get("proj:bbox")
    if not code or not tr or not bb:
        raise FetchError(f"{item.get('id')}: catalog item lacks proj:code/proj:transform/proj:bbox")
    return code, float(tr[2]), float(tr[5]), [float(v) for v in bb]


def plan_item(item: dict[str, Any], lat: float, lon: float, half_m: float) -> dict[str, Any] | None:
    """Box in the item's tile CRS when the tile grid and data footprint both contain it, else None."""
    try:
        crs, ulx, uly, bb = _item_grid(item)
    except FetchError:
        return None
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = fwd.transform(lon, lat)
    bounds = aoi_box(x, y, half_m, ulx, uly)
    if not (bb[0] <= bounds[0] and bb[1] <= bounds[1] and bounds[2] <= bb[2] and bounds[3] <= bb[3]):
        return None
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    aoi_wgs84 = shapely.transform(box(*bounds), inv.transform, interleaved=False)
    if item.get("geometry") and not shape(item["geometry"]).contains(aoi_wgs84):
        return None
    return {"crs": crs, "ulx": ulx, "uly": uly, "bounds": bounds, "bounds_wgs84": list(aoi_wgs84.bounds)}


# ------------------------------------------------------------------ discovery
def search_items(lat: float, lon: float, *, lookback_days: int, now: datetime,
                 client: httpx.Client) -> list[dict[str, Any]]:
    query = {"collections": [COLLECTION], "intersects": {"type": "Point", "coordinates": [lon, lat]},
             "datetime": f"{_iso(now - timedelta(days=lookback_days))}/{_iso(now)}",
             "limit": 100, "sortby": [{"field": "datetime", "direction": "desc"}]}
    r = client.post(configured_url("PRAVAH_CDSE_SEARCH_URL"), json=query)
    r.raise_for_status()
    feats = r.json().get("features", [])
    if any(f.get("collection") != COLLECTION for f in feats):
        raise FetchError("Catalog returned a collection other than Sentinel-2 L1C")
    return feats


def candidates(items: list[dict[str, Any]], lat: float, lon: float, half_m: float,
               max_cloud: float) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Newest-first (item, plan) pairs that fully cover the box and pass the tile-wide cloud filter."""
    out = []
    for it in items:
        cc = it.get("properties", {}).get("eo:cloud_cover")
        if cc is None or cc > max_cloud:
            continue
        plan = plan_item(it, lat, lon, half_m)
        if plan:
            out.append((it, plan))
    out.sort(key=lambda p: p[0]["properties"].get("datetime") or "", reverse=True)
    return out


# ----------------------------------------------------------------- radiometry
def read_radiometry(text: str) -> tuple[float, dict[str, float], str]:
    q = re.search(r"<QUANTIFICATION_VALUE[^>]*>([^<]+)<", text)
    if not q:
        raise FetchError("QUANTIFICATION_VALUE missing from MTD_MSIL1C.xml")
    offsets = {int(i): float(v) for i, v in re.findall(r'<RADIO_ADD_OFFSET band_id="(\d+)">([^<]+)<', text)}
    baseline = re.search(r"<PROCESSING_BASELINE>([^<]+)<", text)
    return float(q.group(1)), {b: offsets.get(BAND_IDS[b], 0.0) for b in BANDS}, (baseline.group(1) if baseline else "")


def check_stac_scaling(item: dict[str, Any], quant: float, offsets: dict[str, float]) -> None:
    """Refuse when the catalog's raster:scale/offset disagree with the product metadata."""
    for b in BANDS:
        a = item.get("assets", {}).get(FILE_KEYS[b], {})
        scale, off = a.get("raster:scale"), a.get("raster:offset")
        if scale is not None and not math.isclose(scale, 1.0 / quant, rel_tol=1e-6):
            raise FetchError(f"{b}: catalog scale {scale} != 1/{quant} from MTD_MSIL1C.xml")
        if off is not None and not math.isclose(off, offsets[b] / quant, abs_tol=1e-9):
            raise FetchError(f"{b}: catalog offset {off} != {offsets[b]}/{quant} from MTD_MSIL1C.xml")


# ------------------------------------------------------------------- reading
def _gdal_path(url: str) -> str:
    return "/vsicurl/" + url if url.startswith(("http://", "https://")) else url


def _get_text(url: str, client: httpx.Client) -> str:
    if not url.startswith(("http://", "https://")):
        return Path(url).read_text()
    r = client.get(url)
    r.raise_for_status()
    return r.text


def _read(url: str, plan: dict[str, Any], res: int, out_hw: tuple[int, int] | None = None) -> np.ndarray:
    win = window_for(plan["bounds"], plan["ulx"], plan["uly"], res)
    with rasterio.Env(**GDAL_HTTP_ENV), rasterio.open(_gdal_path(url)) as src:
        shape_ = (src.count, *out_hw) if out_hw else None
        arr = src.read(window=win, out_shape=shape_, boundless=False)
    if arr.shape[-2:] != (out_hw or (int(win.height), int(win.width))):
        raise FetchError(f"Window read returned {arr.shape} from {url}")
    return arr


def mask_invalid(mask_url: str, plan: dict[str, Any], out_hw: tuple[int, int] | None = None) -> np.ndarray:
    """True where MSK_CLASSI_B00 flags opaque cloud, cirrus or snow/ice."""
    return (_read(mask_url, plan, MASK_RES_M, out_hw) > 0).any(axis=0)


def scene_name(item: dict[str, Any], lat: float, lon: float, half_km: float, units: str) -> str:
    zone, band, square = tile_parts(item["id"])
    date = (item["properties"].get("datetime") or "")[:10].replace("-", "")
    return f"auto/T{zone}{band}{square}_{date}_{lat:.4f}_{lon:.4f}_{half_km:g}km_{units}.tif"


# ---------------------------------------------------------------------- main
def fetch_scene(lat: float, lon: float, scenes_dir: Path, *, place: str | None = None, half_km: float = 5.0,
                lookback_days: int = 60, max_cloud: float = 30.0, max_invalid: float = 0.5,
                units: str = "reflectance", max_attempts: int = 4, now: datetime | None = None,
                progress: Progress | None = None, client: httpx.Client | None = None) -> dict[str, Any]:
    """Prepare the newest usable L1C crop around (lat, lon); return its sidecar (incl. relative ``scene_id``)."""
    if units not in ("reflectance", "dn"):
        raise ValueError("units must be 'reflectance' or 'dn'")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("lat/lon out of range")
    say = progress or (lambda _m, _f: None)
    now = now or _now()
    half_m = half_km * 1000.0
    own = client is None
    client = client or httpx.Client(timeout=30, follow_redirects=True)
    try:
        say("Searching Sentinel-2 L1C catalog", 0.02)
        items = search_items(lat, lon, lookback_days=lookback_days, now=now, client=client)
        cands = candidates(items, lat, lon, half_m, max_cloud)
        if not cands:
            raise FetchError(f"No Sentinel-2 L1C scene in the last {lookback_days} days with ≤{max_cloud:g}% "
                             f"tile cloud fully covers a {2 * half_km:g} km box at ({lat:.4f}, {lon:.4f}) "
                             f"({len(items)} catalog items checked)")
        skipped = []
        for attempt, (item, plan) in enumerate(cands[:max_attempts]):
            rel = scene_name(item, lat, lon, half_km, units)
            out = scenes_dir / rel
            if out.is_file() and out.with_suffix(".json").is_file():
                say(f"Reusing {rel}", 1.0)
                return json.loads(out.with_suffix(".json").read_text())
            base = mirror_base(item["id"])
            granule = safe_relative(item["assets"]["B02"]["href"]).rsplit("/IMG_DATA/", 1)[0]
            mask_url = f"{base}{granule}/QI_DATA/MSK_CLASSI_B00.jp2"
            say(f"Checking clouds in {item['id']}", 0.05 + 0.03 * attempt)
            try:
                flagged = float(mask_invalid(mask_url, plan).mean())
            except rasterio.RasterioIOError as exc:
                skipped.append(f"{item['id']}: mask unreadable ({exc})")
                continue
            if flagged > max_invalid:
                skipped.append(f"{item['id']}: {flagged:.0%} cloud/cirrus/snow in box")
                continue
            return _prepare(item, plan, base, granule, mask_url, out, rel, lat=lat, lon=lon, place=place,
                            half_km=half_km, units=units, flagged=flagged, say=say, client=client)
        raise FetchError("All candidate scenes were too cloudy or unreadable: " + "; ".join(skipped))
    finally:
        if own:
            client.close()


def _prepare(item, plan, base, granule, mask_url, out: Path, rel: str, *, lat, lon, place, half_km, units,
             flagged, say, client) -> dict[str, Any]:
    quant, offsets, baseline = read_radiometry(_get_text(base + "MTD_MSIL1C.xml", client))
    check_stac_scaling(item, quant, offsets)
    win10 = window_for(plan["bounds"], plan["ulx"], plan["uly"], 10)
    hw = (int(win10.height), int(win10.width))
    urls = {b: base + granule + "/IMG_DATA/" + Path(safe_relative(item["assets"][FILE_KEYS[b]]["href"])).name
            for b in BANDS}
    say(f"Downloading 6 bands of {item['id']} ({hw[1]}×{hw[0]} px)", 0.2)
    dn: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_read, urls[b], plan, RES_M[b], hw): b for b in BANDS}
        for fut in as_completed(futures):
            dn[futures[fut]] = fut.result()[0]
            say(f"Downloaded {len(dn)}/6 bands of {item['id']}", 0.2 + 0.7 * len(dn) / len(BANDS))
    invalid = mask_invalid(mask_url, plan, hw)
    for b in BANDS:
        invalid |= dn[b] == 0
    factor = 1.0 if units == "reflectance" else quant
    stack = np.stack([(dn[b].astype(np.float32) + offsets[b]) / quant * factor for b in BANDS])
    stack[:, invalid] = NODATA
    xmin, _, _, ymax = plan["bounds"]
    transform = from_origin(xmin, ymax, 10, 10)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    profile = {"driver": "GTiff", "height": hw[0], "width": hw[1], "count": len(BANDS), "dtype": "float32",
               "crs": plan["crs"], "transform": transform, "nodata": NODATA, "compress": "deflate",
               "tiled": True, "blockxsize": 256, "blockysize": 256}
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(stack)
        for i, b in enumerate(BANDS, start=1):
            dst.set_band_description(i, b)
    props = item["properties"]
    sidecar = {
        "schema_version": "pravah-s2-l1c-live-crop-v1",
        "scene_id": rel, "product_id": item["id"],
        "acquired_at_utc": props.get("datetime"),
        "available_at_utc": props.get("published"),
        "availability_basis": "CDSE STAC 'published'" if props.get("published") else "unknown",
        "retrieved_at_utc": _iso(_now()),
        "sensor": "S2", "processing_level": "L1C", "processing_baseline": baseline,
        "bands": BANDS,
        "units": "dimensionless_TOA_reflectance" if units == "reflectance" else "TOA_DN_equivalent_(reflectance*10000)",
        "scaling": "(DN + RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE" + ("" if units == "reflectance" else " * QUANTIFICATION_VALUE"),
        "quantification_value": quant, "radio_add_offset": offsets,
        "resampling": "none for 10 m bands; nearest 2× upsampling for 20 m B8A/B11/B12; nearest 6× for the 60 m mask",
        "crs": plan["crs"], "transform": list(transform)[:6], "shape": list(hw), "nodata": NODATA,
        "invalid_rule": "DN == 0 in any band or MSK_CLASSI_B00 opaque_cloud|cirrus_cloud|snow_ice",
        "invalid_pixels": int(invalid.sum()), "valid_pixels": int((~invalid).sum()),
        "mask_flagged_fraction": flagged, "tile_cloud_cover_percent": props.get("eo:cloud_cover"),
        "request": {"lat": lat, "lon": lon, "place": place, "half_size_km": half_km},
        "aoi_bounds_wgs84": plan["bounds_wgs84"],
        "catalog_uri": configured_url("PRAVAH_CDSE_SEARCH_URL"),
        "source_uris": {**urls, "MSK_CLASSI_B00": mask_url, "MTD_MSIL1C": base + "MTD_MSIL1C.xml"},
        "is_synthetic": False,
        "limitations": ["L1C cloud classification misses some cloud and terrain shadow.",
                        "Tile-wide cloud cover filter; the box itself was checked with MSK_CLASSI_B00 only.",
                        "Water segmentation of a past acquisition, not a flood forecast."],
    }
    tmp.replace(out)
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    say(f"Saved {rel}", 1.0)
    return sidecar
