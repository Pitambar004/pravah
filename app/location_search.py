"""Place-name search and scene-footprint matching for the Live Prediction page.

Geocoding goes to the endpoint in ``PRAVAH_GEOCODER_URL`` (OpenStreetMap
Nominatim by default). Only the typed place name is sent, with a generic
client name. Nominatim's policy allows ~1 request/s and forbids
type-ahead autocomplete, so the UI searches only on Enter and caches results.

Matching never downloads imagery: a location can only be run if a scene
already in the inference API's scenes directory covers it.
"""
from __future__ import annotations

from typing import Any

import httpx
from pyproj import Transformer

from pravah import __version__
from pravah.configuration import configured_url

USER_AGENT = f"PRAVAH-dashboard/{__version__}"


def geocode(query: str, limit: int = 8) -> tuple[list[dict[str, Any]], str | None]:
    """Return ([{name, lat, lon, bbox}], error). bbox is [south, west, north, east] in EPSG:4326."""
    query = query.strip()
    if not query:
        return [], None
    try:
        url = configured_url("PRAVAH_GEOCODER_URL")
    except ValueError as exc:
        return [], str(exc)
    try:
        r = httpx.get(url, params={"q": query, "format": "jsonv2", "limit": limit, "accept-language": "en"},
                      headers={"User-Agent": USER_AGENT}, timeout=10.0)
        r.raise_for_status()
        rows = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        return [], f"Location search failed: {exc}"
    out = []
    for row in rows if isinstance(rows, list) else []:
        try:
            s, n, w, e = (float(v) for v in row["boundingbox"])
            out.append({"name": row["display_name"], "lat": float(row["lat"]), "lon": float(row["lon"]),
                        "bbox": [s, w, n, e]})
        except (KeyError, TypeError, ValueError):
            continue
    return out, None


def footprint_wgs84(scene: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """(west, south, east, north) of a /api/v1/scenes entry, or None when CRS/bounds are missing."""
    crs, bounds = scene.get("crs"), scene.get("bounds")
    if not crs or not bounds or len(bounds) != 4:
        return None
    try:
        return Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform_bounds(*bounds)
    except Exception:  # noqa: BLE001 - pyproj raises several types for unusual CRSs
        return None


def scenes_covering(scenes: list[dict[str, Any]], lat: float, lon: float) -> list[str]:
    """IDs of model-compatible scenes whose footprint contains the point."""
    hits = []
    for sc in scenes:
        if not sc.get("model_compatible"):
            continue
        fp = footprint_wgs84(sc)
        if fp and fp[0] <= lon <= fp[2] and fp[1] <= lat <= fp[3]:
            hits.append(sc["scene_id"])
    return hits
