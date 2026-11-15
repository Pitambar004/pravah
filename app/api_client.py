"""Thin client for the PRAVAH inference API.

Every call returns ``(data, error)`` instead of raising, so a down or slow API
renders as an honest status message rather than a crashed page.

``PRAVAH_API_URL`` points at the API. Inside one Fuzzball workflow the
dashboard reaches the API service as ``http://api.service:8000`` (service
hostname + ``.service``), which never leaves the cluster network.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from pravah.configuration import configured_url


def api_url() -> str:
    return configured_url("PRAVAH_API_URL").rstrip("/")


def _headers() -> dict[str, str]:
    """PRAVAH write token, plus a Fuzzball endpoint token when reaching the API through a
    non-public Fuzzball endpoint (``fuzzball endpoint generate-token <id>``), sent as
    ``FB-Authorization`` so it never collides with other auth headers."""
    h: dict[str, str] = {}
    token = os.environ.get("PRAVAH_API_TOKEN")
    if token:
        h["X-Pravah-Token"] = token
    fb = os.environ.get("PRAVAH_FB_ENDPOINT_TOKEN")
    if fb:
        h["FB-Authorization"] = f"Bearer {fb}"
    return h


def _request(method: str, path: str, *, json: dict[str, Any] | None = None,
             timeout: float = 5.0, base_url: str | None = None) -> tuple[Any | None, str | None]:
    endpoint = (base_url or api_url()).rstrip("/")
    try:
        r = httpx.request(method, f"{endpoint}{path}", json=json, headers=_headers(), timeout=timeout)
    except httpx.HTTPError as exc:
        return None, f"API unreachable at {endpoint} ({type(exc).__name__})"
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        return None, f"HTTP {r.status_code}: {detail}"
    try:
        return r.json(), None
    except ValueError:
        return None, "API returned non-JSON response"


def status() -> tuple[dict[str, Any] | None, str | None]:
    return _request("GET", "/api/v1/status")


def scenes() -> tuple[dict[str, Any] | None, str | None]:
    return _request("GET", "/api/v1/scenes", timeout=15.0)


def submit_prediction(scene_id: str, threshold: float, run_id: str | None = None):
    body: dict[str, Any] = {"scene_id": scene_id, "threshold": threshold}
    if run_id:
        body["run_id"] = run_id
    return _request("POST", "/api/v1/predictions", json=body, timeout=15.0)


def submit_location_prediction(lat: float, lon: float, place: str | None, threshold: float = 0.5):
    """Fetch the newest usable Sentinel-2 L1C crop around the point on the API host, then predict."""
    body = {"lat": lat, "lon": lon, "place": place, "threshold": threshold}
    return _request("POST", "/api/v1/locations/predictions", json=body, timeout=15.0)


def jobs() -> tuple[dict[str, Any] | None, str | None]:
    return _request("GET", "/api/v1/predictions")


def runs() -> tuple[dict[str, Any] | None, str | None]:
    return _request("GET", "/api/v1/runs")


def run_metadata(run_id: str) -> tuple[dict[str, Any] | None, str | None]:
    return _request("GET", f"/api/v1/runs/{run_id}", timeout=15.0)


def run_json_file(run_id: str, name: str) -> tuple[Any | None, str | None]:
    """Fetch a JSON/GeoJSON artifact (e.g. observed_water.geojson) of a finished run."""
    return _request("GET", f"/api/v1/runs/{run_id}/files/{name}", timeout=60.0)
