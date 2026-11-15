"""Location search + lake surface-change tile. All inputs are synthetic; no network."""
import httpx

from app import location_search
from app.location_search import footprint_wgs84, geocode, scenes_covering
from app.sections.lakes import surface_change_tile

# Synthetic UTM 45N footprint roughly 85.30–85.40 E, 28.13–28.22 N (not a real scene).
UTM_SCENE = {"scene_id": "synthetic_utm.tif", "crs": "EPSG:32645", "model_compatible": True,
             "bounds": [333000.0, 3113000.0, 343000.0, 3123000.0]}


def test_footprint_projects_to_wgs84():
    w, s, e, n = footprint_wgs84(UTM_SCENE)
    assert 85.2 < w < e < 85.5 and 28.0 < s < n < 28.3


def test_footprint_missing_crs_is_none():
    assert footprint_wgs84({"scene_id": "x.tif", "bounds": [0, 0, 1, 1], "crs": None}) is None


def test_scenes_covering_point_and_compatibility():
    w, s, e, n = footprint_wgs84(UTM_SCENE)
    inside = ((s + n) / 2, (w + e) / 2)
    incompatible = {**UTM_SCENE, "scene_id": "sar_like.tif", "model_compatible": False}
    assert scenes_covering([UTM_SCENE, incompatible], *inside) == ["synthetic_utm.tif"]
    assert scenes_covering([UTM_SCENE], 40.0, -100.0) == []


def test_geocode_parses_and_skips_bad_rows(monkeypatch):
    monkeypatch.setenv("PRAVAH_GEOCODER_URL", "https://geocoder.invalid/search")
    seen = {}

    def fake_get(url, params, headers, timeout):
        seen.update(url=url, params=params, headers=headers)
        rows = [{"display_name": "Synthetic Place", "lat": "28.1", "lon": "85.3",
                 "boundingbox": ["28.0", "28.2", "85.2", "85.4"]}, {"display_name": "broken"}]
        return httpx.Response(200, json=rows, request=httpx.Request("GET", url))

    monkeypatch.setattr(location_search.httpx, "get", fake_get)
    places, err = geocode("  Synthetic Place ")
    assert err is None and seen["params"]["q"] == "Synthetic Place"
    assert seen["headers"]["User-Agent"].startswith("PRAVAH-dashboard/")
    assert places == [{"name": "Synthetic Place", "lat": 28.1, "lon": 85.3, "bbox": [28.0, 85.2, 28.2, 85.4]}]


def test_geocode_blank_query_makes_no_request(monkeypatch):
    monkeypatch.setattr(location_search.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert geocode("   ") == ([], None)


def test_geocode_http_error_is_reported(monkeypatch):
    monkeypatch.setenv("PRAVAH_GEOCODER_URL", "https://geocoder.invalid/search")

    def fail(*a, **k):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(location_search.httpx, "get", fail)
    places, err = geocode("anywhere")
    assert places == [] and "Location search failed" in err


def test_surface_change_tile_states():
    assert surface_change_tile(None)[0] == "Awaiting imagery"
    base = {"lake_id": "SYN_1", "pre_acquired_at_utc": "2026-06-01T05:00:00Z",
            "post_acquired_at_utc": "2026-07-01T05:00:00Z", "comparable_fraction_of_search": 0.8}
    value, note = surface_change_tile({**base, "area_change_m2_common_valid": 12500.0})
    assert value == "+0.0125 km²" and "SYN_1" in note and "80%" in note and "not water level" in note
    assert surface_change_tile({**base, "area_change_m2_common_valid": None})[0] == "No common valid pixels"
