"""Microsoft Planetary Computer STAC search and windowed COG reads on a fixed metric grid.

Every scene of one AOI is warped to the *same* EPSG:32645 grid, so dates compare pixel for
pixel. SAS tokens are short-lived and are never written to disk; persisted records keep
only item ids, timestamps and cloud cover. Discovery time in live mode is the verified
availability time; in retrospective use availability is unknown and recorded as such.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
import numpy as np

from pravah.configuration import configured_url
from pravah.watch.config import METRIC_CRS

_TOKENS: dict[str, tuple[str, float]] = {}


@dataclass(frozen=True)
class Grid:
    """A fixed north-up metric raster grid."""
    left: float
    top: float
    res: float
    width: int
    height: int
    crs: str = METRIC_CRS

    @property
    def transform(self):
        from affine import Affine
        return Affine(self.res, 0, self.left, 0, -self.res, self.top)

    @classmethod
    def from_bbox(cls, bbox_wgs84, res: float) -> Grid:
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
        xs, ys = zip(*[t.transform(x, y) for x in (bbox_wgs84[0], bbox_wgs84[2]) for y in (bbox_wgs84[1], bbox_wgs84[3])])
        left, right = math.floor(min(xs) / res) * res, math.ceil(max(xs) / res) * res
        bottom, top = math.floor(min(ys) / res) * res, math.ceil(max(ys) / res) * res
        return cls(left, top, res, int((right - left) / res), int((top - bottom) / res))


def search(collection: str, bbox, start: datetime, end: datetime, limit: int = 200,
           max_cloud: float | None = None, client: httpx.Client | None = None) -> list[dict]:
    body = {"collections": [collection], "bbox": list(bbox),
            "datetime": f"{start.isoformat().replace('+00:00', 'Z')}/{end.isoformat().replace('+00:00', 'Z')}",
            "limit": min(limit, 250), "sortby": [{"field": "datetime", "direction": "asc"}]}
    if max_cloud is not None:
        body["query"] = {"eo:cloud_cover": {"lt": max_cloud}}
    own = client is None
    client = client or httpx.Client(timeout=60, follow_redirects=True)
    items, url = [], configured_url("PRAVAH_PC_STAC_SEARCH_URL")
    try:
        while url and len(items) < limit:
            r = client.post(url, json=body)
            r.raise_for_status()
            page = r.json()
            items += [f for f in page.get("features", []) if f.get("collection") == collection]
            nxt = next((link for link in page.get("links", []) if link.get("rel") == "next"), None)
            url, body = (nxt["href"], nxt.get("body", body)) if nxt else (None, None)
    finally:
        if own:
            client.close()
    return items[:limit]


def sign(href: str, collection: str, client: httpx.Client | None = None) -> str:
    token, expiry = _TOKENS.get(collection, ("", 0.0))
    if time.time() > expiry - 120:
        own = client is None
        client = client or httpx.Client(timeout=30)
        try:
            r = client.get(configured_url("PRAVAH_PC_SAS_URL", collection=collection))
            r.raise_for_status()
            doc = r.json()
        finally:
            if own:
                client.close()
        token = doc["token"]
        expiry = time.time() + 30 * 60
        if doc.get("msft:expiry"):
            expiry = datetime.fromisoformat(doc["msft:expiry"].replace("Z", "+00:00")).timestamp()
        _TOKENS[collection] = (token, expiry)
    return href + ("&" if "?" in href else "?") + token


def read_to_grid(href: str, grid: Grid, resampling: str = "bilinear", dtype="float32") -> np.ndarray:
    """Warp one COG asset onto ``grid``; pixels outside the source are NaN."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2",  # noqa: SIM117
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff"):
        with rasterio.open(href) as src, WarpedVRT(src, crs=grid.crs, transform=grid.transform, width=grid.width,
                                                   height=grid.height, resampling=getattr(Resampling, resampling),
                                                   src_nodata=src.nodata if src.nodata is not None else 0,
                                                   nodata=np.nan, dtype=dtype) as vrt:
            return vrt.read(1)


def item_time(item: dict) -> datetime:
    p = item["properties"]
    return datetime.fromisoformat((p.get("datetime") or p["start_datetime"]).replace("Z", "+00:00"))


def group_by_pass(items: list[dict], gap_minutes: float = 30) -> list[list[dict]]:
    """Tiles of one satellite pass (same platform, minutes apart) are mosaicked together."""
    groups: list[list[dict]] = []
    for it in sorted(items, key=item_time):
        plat = it["properties"].get("platform")
        if groups and groups[-1][0]["properties"].get("platform") == plat and \
                (item_time(it) - item_time(groups[-1][-1])).total_seconds() < gap_minutes * 60:
            groups[-1].append(it)
        else:
            groups.append([it])
    return groups


def _overlaps(item: dict, bbox) -> bool:
    b = item.get("bbox")
    return bbox is None or not b or not (b[2] < bbox[0] or b[0] > bbox[2] or b[3] < bbox[1] or b[1] > bbox[3])


def mosaic(items: list[dict], asset: str, grid: Grid, collection: str, resampling="bilinear",
           client: httpx.Client | None = None, bbox=None) -> np.ndarray:
    out = np.full((grid.height, grid.width), np.nan, dtype="float32")
    for it in items:
        if asset not in it.get("assets", {}) or not _overlaps(it, bbox):
            continue
        if not np.isnan(out).any():
            break
        arr = read_to_grid(sign(it["assets"][asset]["href"], collection, client), grid, resampling)
        fill = np.isnan(out) & np.isfinite(arr)
        out[fill] = arr[fill]
    return out


def pass_record(items: list[dict], collection: str, discovered_at: datetime | None = None) -> dict:
    """Sanitized provenance (no hrefs/tokens). ``discovered_at`` is only set in live mode."""
    first = items[0]
    clouds = [i["properties"].get("eo:cloud_cover") for i in items if i["properties"].get("eo:cloud_cover") is not None]
    return {"pass_id": f"{collection}:{first['id']}", "collection": collection,
            "item_ids": [i["id"] for i in items], "platform": first["properties"].get("platform"),
            "acquired_at_utc": item_time(first).astimezone(timezone.utc).isoformat(),
            "cloud_cover_percent": (sum(clouds) / len(clouds)) if clouds else None,
            "processing_baseline": first["properties"].get("s2:processing_baseline"),
            "orbit_state": first["properties"].get("sat:orbit_state"),
            "relative_orbit": first["properties"].get("sat:relative_orbit"),
            "available_at_utc": discovered_at.isoformat() if discovered_at else None,
            "availability_verified": discovered_at is not None,
            "provider": "Microsoft Planetary Computer"}
