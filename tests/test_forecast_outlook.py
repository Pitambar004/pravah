"""pravah.forecast.outlook with SYNTHETIC HTTP responses (httpx.MockTransport); no network."""
import json
from datetime import date

import httpx
import pytest

from pravah.forecast import outlook as ol


def _cell(lat, lon, values, start="2024-01-01"):
    days = [date.fromordinal(date.fromisoformat(start).toordinal() + i).isoformat() for i in range(len(values))]
    return {"latitude": lat, "longitude": lon, "daily_units": {"river_discharge": "m³/s"},
            "daily": {"time": days, "river_discharge": values}}


def test_map_station_picks_main_channel_and_builds_climatology():
    calls = []

    def handler(request):
        calls.append(request.url.params)
        lats = request.url.params["latitude"].split(",")
        if len(lats) == 9:  # neighbourhood search: centre cell is the big river
            return httpx.Response(200, json=[_cell(float(a), 85.0, [100.0 if i == 4 else 5.0] * 10) for i, a in enumerate(lats)])
        return httpx.Response(200, json=_cell(float(lats[0]), 85.0, [float(v) for v in range(1, 366)], start="2023-01-01"))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        row = ol.map_station(client, {"id": "1", "name": "Test", "latitude": 28.0, "longitude": 85.0})
    assert row["cell_latitude"] == 28.0 and row["cell_distance_km"] == 0.0
    assert len(calls) == 2  # light selection request, then one-cell climatology
    jan = row["monthly_climatology_m3s"]["01"]
    assert jan["days"] == 31 and jan["p90"] == pytest.approx(28.0)


@pytest.mark.parametrize("peak,expected", [(None, "unknown"), (5, "normal"), (90, "high"), (99, "very_high")])
def test_classify(peak, expected):
    assert ol.classify(peak, {"p90": 90, "p99": 99}) == expected


def _cells():
    clim = {f"{m:02d}": {"p90": 50.0, "p99": 80.0} for m in range(1, 13)}
    return {"stations": [{"station_id": "1", "station_name": "S", "gauge_latitude": 28.0, "gauge_longitude": 85.0,
                          "cell_latitude": 28.0, "cell_longitude": 85.0, "cell_distance_km": 0.0, "cell_mean_m3s": 30.0,
                          "monthly_climatology_m3s": clim}]}


def _transport(flood_unit="m³/s"):
    def handler(request):
        if "flood" in str(request.url):
            days = ["2026-11-14", "2026-11-15", "2026-11-16", "2026-11-17"]
            vals = [40.0, 42.0, 60.0, 45.0]
            daily = {"time": days, **{k: vals for k in ol.DISCHARGE_STATS}}
            daily["river_discharge_max"] = [40.0, 42.0, 85.0, 50.0]
            return httpx.Response(200, json={"daily_units": {"river_discharge": flood_unit}, "daily": daily})
        return httpx.Response(200, json={"daily_units": {"precipitation_sum": "mm"},
                                         "daily": {"time": ["2026-11-15", "2026-11-16", "2026-11-17"],
                                                   "precipitation_sum": [1.0, 2.0, 3.0]}})
    return httpx.MockTransport(handler)


def test_fetch_outlook_flags_peak_and_keeps_quantities_separate():
    with httpx.Client(transport=_transport()) as client:
        out = ol.fetch_outlook(_cells(), {"1": {"level_m": 2.5, "warning_level_m": 4.0, "quality": "current"}},
                               today=date(2026, 11, 15), client=client)
    s = out["stations"][0]
    assert s["discharge_today_m3s"] == 42.0 and s["peak_7d_m3s"] == 60.0 and s["peak_7d_date"] == "2026-11-16"
    assert s["outlook"] == "high" and s["outlook_ensemble_max"] == "very_high"
    assert s["rain_next_72h_mm"] == 6.0
    assert s["dhm_level_m"] == 2.5  # stage kept as provided, never converted to discharge
    assert out["pravah_forecast_model"] is None and out["is_synthetic"] is False
    assert len(out["sources"]["glofas"]["sha256"]) == 64


def test_fetch_outlook_refuses_unexpected_units():
    with httpx.Client(transport=_transport("ft³/s")) as client, pytest.raises(ValueError, match="units"):
        ol.fetch_outlook(_cells(), {}, today=date(2026, 11, 15), client=client)


def test_build_station_cells_records_failures(tmp_path):
    def handler(request):
        return httpx.Response(500)

    original = httpx.Client

    class _Client(original):
        def __init__(self, *a, **k):
            super().__init__(*a, transport=httpx.MockTransport(handler), **{x: y for x, y in k.items() if x != "transport"})

    ol.httpx.Client = _Client
    try:
        doc = ol.build_station_cells([{"id": "1", "latitude": 28.0, "longitude": 85.0}, {"id": "2", "latitude": None}],
                                     tmp_path / "cells.json", pause_s=0)
    finally:
        ol.httpx.Client = original
    assert doc["stations"] == [] and {f["station_id"] for f in doc["failures"]} == {"1", "2"}
    assert json.loads((tmp_path / "cells.json").read_text())["is_synthetic"] is False
