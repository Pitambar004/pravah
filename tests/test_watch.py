"""Live Watch unit tests. ALL fixtures are synthetic and must never be exported as observations."""
from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from pravah.watch import assess, glacier, hydro, optical, sar

pytest.importorskip("torch")
# Tests that run even a 1-epoch synthetic training loop are opt-in: real training runs only on GB10.
needs_training = pytest.mark.skipif(os.environ.get("PRAVAH_RUN_TRAINING_TESTS") != "1",
                                    reason="training-shaped test; set PRAVAH_RUN_TRAINING_TESTS=1 (GB10/CI)")


# ------------------------------------------------------------------------------ glacier
def test_phase_correlation_recovers_known_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(300, 300)).astype("float32")
    fy, fx = np.meshgrid(np.fft.fftfreq(300), np.fft.fftfreq(300), indexing="ij")
    spec = np.fft.fft2(base) * np.exp(-(fy ** 2 + fx ** 2) * 60)          # smooth texture
    base = np.fft.ifft2(spec).real.astype("float32")
    moved = np.fft.ifft2(spec * np.exp(-2j * np.pi * (fy * 2.5 + fx * -3.0))).real.astype("float32")
    r = glacier.track(base, moved, np.ones_like(base, bool), chip=64, step=48, device="cpu")
    k = r["keep"]
    assert k.sum() > 10
    assert np.median(r["dy"][k]) == pytest.approx(2.5, abs=0.3)
    assert np.median(r["dx"][k]) == pytest.approx(-3.0, abs=0.3)
    zones = np.ones_like(base, int)
    sp = glacier.zone_speeds(r, zones, ["Z"], pixel_m=10, days=10)[0]
    assert sp["median_speed_m_per_day"] == pytest.approx(math.hypot(2.5, 3.0), rel=0.15)


# ------------------------------------------------------------------------------ hydraulics
def _rect_sections(width_m=60.0, slope=0.01, n=40):
    off = np.arange(-600, 601, 10.0)
    z = np.where(np.abs(off) <= width_m / 2, 100.0, 200.0)[None, :].repeat(n, 0)
    ch = np.arange(n) * 500.0
    bed = np.full(n, 100.0) - ch * 0  # flat bed values; slope given explicitly
    return hydro.Sections(ch, ch * 0, ch, np.ones(n), np.zeros(n), off, z, bed, np.full(n, slope),
                          np.full(n, int(np.argmin(np.abs(off)))))


def test_manning_matches_rectangular_channel():
    sec = _rect_sections()
    Q = np.full((len(sec.chainage_m), 2), 500.0)
    Q[:, 1] = 1500.0
    res = hydro.normal_depth(sec, Q, np.array([0.05, 0.05]), device="cpu")
    for j, q in enumerate((500.0, 1500.0)):
        h = res["depth_m"][0, j]
        b = 70.0  # 7 wet points × 10 m (grid-discretised width)
        area = b * h
        R = area / (b + 2 * h)
        q_check = area * R ** (2 / 3) * math.sqrt(0.01) / 0.05
        assert q_check == pytest.approx(q, rel=0.08)
    assert (res["depth_m"][:, 1] > res["depth_m"][:, 0]).all()
    assert res["width_m"][0, 0] == pytest.approx(70.0)


def test_wet_region_does_not_fill_disconnected_valley():
    sec = _rect_sections()
    sec.z[:, :5] = 90.0  # a lower basin far left, separated by the 200 m bank
    res = hydro.normal_depth(sec, np.full((len(sec.chainage_m), 1), 300.0), np.array([0.05]), device="cpu")
    assert res["width_m"][0, 0] == pytest.approx(70.0)


def test_hydraulic_geometry_fit_and_glof_envelope():
    q = np.array([20, 50, 100, 200, 400, 800.0])
    fit = hydro.fit_hydraulic_geometry(12 * q ** 0.3, q)
    assert fit["a"] == pytest.approx(12, rel=1e-6) and fit["b"] == pytest.approx(0.3, rel=1e-6)
    small, big = hydro.glof_peak(5e4), hydro.glof_peak(5e5)
    assert small["peak_m3s_low"] <= small["peak_m3s_mid"] <= small["peak_m3s_high"]
    assert big["peak_m3s_mid"] > small["peak_m3s_mid"]
    att = hydro.attenuate(1000.0, np.full(3, 100.0), np.array([0, 60, 600.0]))
    assert att[0, 0] == pytest.approx(1100) and att[2, 0] < 101


def test_arrival_time_increases_downstream():
    sec = _rect_sections()
    v = np.full((len(sec.chainage_m), 1), 3.0)
    t = hydro.arrival_hours(sec, v, origin_chainage_m=5000)
    assert np.isnan(t[:10]).all() and np.all(np.diff(t[10:, 0]) > 0)


def test_main_stem_bridges_gap_and_reaches_sink():
    from pyproj import Transformer
    to_geo = Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True)
    pts = [(300000 + i * 1000, 3100000 - i * 1000) for i in range(21)]
    seg1 = [to_geo.transform(*p) for p in pts[:10]]
    seg2 = [to_geo.transform(*p) for p in pts[11:]]   # 1.4 km gap between segments
    doc = {"features": [{"type": "Feature", "properties": {"name": "Trisuli"}, "geometry": {"type": "LineString", "coordinates": s}}
                        for s in (seg1, seg2)]}
    line, meta = hydro.main_stem(doc, seg1[0], seg2[-1])
    assert meta["bridged_gaps"] == 1
    assert line.length == pytest.approx(math.hypot(20000, 20000), rel=0.01)


# ------------------------------------------------------------------------------ indices
def test_optical_water_excludes_snow_and_cloud():
    g = np.array([[0.10, 0.60, 0.10, 0.10]])
    nir = np.array([[0.05, 0.70, 0.05, 0.05]])
    sw = np.array([[0.02, 0.05, 0.02, 0.02]])
    scl = np.array([[6, 11, 9, 6]], float)
    c = optical.classify(g, nir, sw, scl)
    assert c["water"].tolist() == [[True, False, False, True]]
    assert c["snow_ice"][0, 1] and not c["valid"][0, 2]
    zones = np.array([[1, 1, 2, 2]])
    rows = optical.lake_areas(c["water"], c["valid"], zones, ["A", "B"], 400.0, min_valid=0.8)
    assert rows[0]["area_m2"] == 400.0 and rows[1]["area_m2"] is None  # half of B under cloud → withheld
    assert optical.reflectance(np.array([2000.0]), "05.10")[0] == pytest.approx(0.1)


def test_sar_new_water_counts_only_common_valid_pixels():
    before = sar.classify(np.array([[0.1, 0.1, np.nan]]))
    after = sar.classify(np.array([[0.001, 0.1, 0.001]]))
    r = sar.new_water(before, after, np.ones((1, 3), bool), 400.0)
    assert r["new_water_km2"] == pytest.approx(400 / 1e6) and r["compared_km2"] == pytest.approx(800 / 1e6)


# ------------------------------------------------------------------------------ rules
NOW = datetime(2026, 8, 25, 12, tzinfo=timezone.utc)


def _snap(**kw):
    base = {"now": NOW.isoformat(), "rain": {"points": [{"name": "a", "last72_mm": 10, "next24_mm": 5}]},
            "seismic": [], "gauges": [], "lakes": {}, "glacier": [], "widths": [], "discharge": [], "sar_change": None}
    return {**base, **kw}


def test_quiet_snapshot_is_ok_and_missing_evidence_is_unknown():
    a = assess.assess(_snap())
    assert a["level"] == "ok"
    assert "lake_change_s2" in a["unknown"] and "glacier_speed" in a["unknown"]


def test_compound_families_escalate_one_level():
    lakes = {"s2": [{"lake_id": "GLO_85.4_28.4", "area_m2": 130000, "baseline_m2": 100000, "acquired_at_utc": NOW.isoformat()}]}
    quake = [{"magnitude": 4.8, "latitude": 28.2, "longitude": 85.4, "observed_at_utc": (NOW - timedelta(hours=5)).isoformat()}]
    rain = {"points": [{"name": "a", "last72_mm": 80, "next24_mm": 20}]}
    a = assess.assess(_snap(lakes=lakes, seismic=quake, rain=rain))
    fams = set(a["families_elevated"])
    assert {"rain", "seismic", "cryosphere"} <= fams
    assert a["level"] == "severe"  # worst = warning (lake +30 %), +1 for ≥3 families


def test_upstream_gauge_going_silent_is_flagged():
    g = [{"id": "4913", "name": "Rasuwagadhi", "quality": "stale", "level_m": 3.0,
          "observed_at_utc": (NOW - timedelta(hours=2)).isoformat(), "history": []}]
    a = assess.assess(_snap(gauges=g))
    silent = next(i for i in a["indicators"] if i["id"] == "gauge_silent")
    assert silent["status"] == "warning"


def test_future_quake_is_ignored():
    quake = [{"magnitude": 6.5, "latitude": 28.2, "longitude": 85.4, "observed_at_utc": (NOW + timedelta(hours=1)).isoformat()}]
    a = assess.assess(_snap(seismic=quake))
    assert next(i for i in a["indicators"] if i["id"] == "seismic")["status"] == "ok"


# ------------------------------------------------------------------------------ emulator
@needs_training
def test_emulator_trains_on_synthetic_series(tmp_path):
    from pravah.watch.emulator import Emulator, train
    rng = np.random.default_rng(1)
    days = [date(2016, 1, 1) + timedelta(days=i) for i in range((date(2026, 6, 1) - date(2016, 1, 1)).days)]
    P = rng.gamma(0.5, 6, size=(len(days), 2))
    T = 10 + 10 * np.sin(np.arange(len(days)) / 58)[:, None].repeat(2, 1)
    Q = 50 + np.convolve(P[:, 0], np.exp(-np.arange(30) / 6), mode="full")[:len(days)][:, None] * 3
    table = {"days": [d.isoformat() for d in days], "precip": P.tolist(), "temp": T.tolist(),
             "discharge": Q.tolist(), "stations": ["S1"], "points": ["a", "b"], "is_synthetic": True}
    summary = train(table, tmp_path / "member_0", epochs=1, device="cpu", batch_size=512)
    assert summary["samples"]["test"] > 0 and "S1" in summary["metrics_same_day"]["test"]
    emu = Emulator(tmp_path, device="cpu")
    out = emu.predict(P[-90:], T[-90:], days[-90:])
    assert len(out["discharge_m3s"]) == 3 and out["members"] == 1


# ------------------------------------------------------------------------------ end to end
def _synthetic_world(tmp_path):
    """A straight synthetic valley: DEM GeoTIFF, river line, buildings near and far."""
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_origin
    to_geo = Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True)
    x0, y0 = 330000.0, 3120000.0
    river = [to_geo.transform(x0, y0 - i * 1000) for i in range(31)]          # 30 km due south
    lon0, lat0 = to_geo.transform(x0 - 3000, y0 + 1000)
    res = 0.0003
    W, H = 220, 1120
    tr = from_origin(lon0, lat0, res, res)
    rows, cols = np.mgrid[0:H, 0:W]
    lon, lat = tr * (cols + 0.5, rows + 0.5)
    to_m = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
    mx, my = to_m.transform(lon, lat)
    dem = (500 - (y0 - my) * 0.005 + np.clip(np.abs(mx - x0) - 40, 0, None) * 0.05).astype("float32")
    dem_p = tmp_path / "dem.tif"
    with rasterio.open(dem_p, "w", driver="GTiff", width=W, height=H, count=1, dtype="float32", crs="EPSG:4326", transform=tr) as d:
        d.write(dem, 1)
    geo = tmp_path / "geo"
    geo.mkdir()
    fc = lambda feats: {"type": "FeatureCollection", "features": feats, "is_synthetic": True}
    pt = lambda x, y, **p: {"type": "Feature", "properties": p, "geometry": {"type": "Point", "coordinates": list(to_geo.transform(x, y))}}
    (geo / "rivers.geojson").write_text(json.dumps(fc([{"type": "Feature", "properties": {"name": "Trisuli"},
                                                         "geometry": {"type": "LineString", "coordinates": river}}])))
    (geo / "buildings.geojson").write_text(json.dumps(fc([pt(x0 + 30, y0 - 10000), pt(x0 + 1200, y0 - 10000)])))
    (geo / "settlements.geojson").write_text(json.dumps(fc([pt(x0 + 200, y0 - 15000, name="Synthetic Bazar")])))
    (geo / "manifest.json").write_text(json.dumps({"is_synthetic": True, "source": {"note": "synthetic test fixture"}}))
    clim = {f"{m:02d}": {"p50": 100, "p90": 200, "p99": 400} for m in range(1, 13)}
    lon_s, lat_s = to_geo.transform(x0, y0 - 2000)
    lon_d, lat_d = to_geo.transform(x0, y0 - 28000)
    cells = {"stations": [{"station_id": "UP", "station_name": "Up", "gauge_latitude": lat_s, "gauge_longitude": lon_s,
                           "cell_latitude": lat_s, "cell_longitude": lon_s, "monthly_climatology_m3s": clim},
                          {"station_id": "DN", "station_name": "Down", "gauge_latitude": lat_d, "gauge_longitude": lon_d,
                           "cell_latitude": lat_d, "cell_longitude": lon_d, "monthly_climatology_m3s": clim}]}
    (tmp_path / "cells.json").write_text(json.dumps(cells))
    return river, dem_p, geo


def test_river_model_end_to_end_on_synthetic_valley(tmp_path, monkeypatch):
    from pravah.watch import river as river_mod
    river, dem_p, geo = _synthetic_world(tmp_path)
    monkeypatch.setattr(river_mod, "RIVER_SOURCE_HINT", river[0])
    monkeypatch.setattr(river_mod, "RIVER_SINK_HINT", river[-1])
    meta = river_mod.build_river_model(geo, [dem_p], tmp_path / "cells.json", tmp_path / "rm")
    assert meta["main_stem"]["length_km"] == pytest.approx(30, abs=0.5)
    assert {s["station_id"] for s in meta["stations"]} == {"UP", "DN"}
    rm = river_mod.RiverModel(tmp_path / "rm", device="cpu")
    low = rm.run({"glofas_median": {"UP": 50, "DN": 60}}, polygons=False)
    glof = hydro.glof_peak(4e5) | {"origin_chainage_m": 0.0}
    high = rm.run({"glofas_median": {"UP": 50, "DN": 60}}, glof=glof)
    w_low = np.median([r["hydraulic_width_m"][1] for r in low["reaches"]])
    w_high = np.median([r["hydraulic_width_m"][1] for r in high["reaches"]])
    assert w_high > w_low
    assert high["exposure"]["buildings"]["max"] >= low["exposure"]["buildings"]["max"]
    assert high["exposure"]["buildings"]["max"] <= 2
    assert high["extent_max"]["type"] in ("Polygon", "MultiPolygon")
    assert all(r["arrival_h"] is not None for r in high["reaches"])


def _synthetic_emulator(tmp_path, stations=("UP", "DN")):
    from pravah.watch.emulator import train
    rng = np.random.default_rng(2)
    days = [date(2016, 1, 1) + timedelta(days=i) for i in range((date(2026, 9, 1) - date(2016, 1, 1)).days)]
    P = rng.gamma(0.5, 6, size=(len(days), 2))
    Q = 80 + np.convolve(P[:, 0], np.exp(-np.arange(20) / 5), mode="full")[:len(days)][:, None].repeat(len(stations), 1) * 4
    table = {"days": [d.isoformat() for d in days], "precip": P.tolist(), "temp": np.full_like(P, 8).tolist(),
             "discharge": Q.tolist(), "stations": list(stations), "is_synthetic": True}
    train(table, tmp_path / "emu/member_0", epochs=1, device="cpu", batch_size=1024)
    return days, P


@needs_training
def test_replay_never_sees_future_imagery_and_triggers_glof(tmp_path, monkeypatch):
    from pravah.watch import river as river_mod
    from pravah.watch.emulator import Emulator
    from pravah.watch.replay import build_frames
    river, dem_p, geo = _synthetic_world(tmp_path)
    monkeypatch.setattr(river_mod, "RIVER_SOURCE_HINT", river[0])
    monkeypatch.setattr(river_mod, "RIVER_SINK_HINT", river[-1])
    river_mod.build_river_model(geo, [dem_p], tmp_path / "cells.json", tmp_path / "rm")
    rm = river_mod.RiverModel(tmp_path / "rm", device="cpu")
    days, P = _synthetic_emulator(tmp_path)
    emu = Emulator(tmp_path / "emu", device="cpu")
    start = datetime(2026, 8, 20, tzinfo=timezone.utc)
    lake = f"GLO_{river[0][0]:.5f}_{river[0][1]:.5f}"
    def s2(t, area):
        return {"pass": {"pass_id": f"s2:{t}", "acquired_at_utc": t.isoformat()}, "lakes": [{"lake_id": lake, "area_m2": area, "valid_fraction": 1}],
                "reaches": []}
    grow_at = start + timedelta(hours=10)
    pack = {"event": "synthetic", "start_utc": start.isoformat(), "end_utc": (start + timedelta(hours=20)).isoformat(),
            "s2": [s2(start - timedelta(days=3), 100000), s2(grow_at, 140000)], "s1": [], "glacier_pairs": [], "s1_changes": [],
            "rain_hourly": {"times": [(start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(-80, 30)],
                            "precip": [[1.0, 1.0]] * 110, "points": ["a", "b"]},
            "rain_daily": {"days": [d.isoformat() for d in days], "precip": P.tolist(), "temp": np.full_like(P, 8).tolist()},
            "seismic": [], "gauges": None, "glofas_reanalysis": {"days": [], "stations": [], "discharge": []},
            "availability": "synthetic", "is_synthetic": True}
    out = build_frames(pack, rm, emu, {"s2": {lake: {"median_m2": 100000}}}, {})
    frames = out["frames"]
    assert out["frame_count"] == 21
    for f in frames:
        lake_ind = next(i for i in f["assessment"]["indicators"] if i["id"] == "lake_change_s2")
        clock = datetime.fromisoformat(f["clock_utc"])
        if clock < grow_at:
            assert lake_ind["status"] == "ok" and f["downstream"]["glof"] is None
        else:
            assert lake_ind["status"] == "warning" and f["downstream"]["glof"]["lake_id"] == lake
    assert frames[10]["new_imagery"][0]["pass_id"] == f"s2:{grow_at}"
    assert all(i["status"] == "unknown" for f in frames for i in f["assessment"]["indicators"] if i["id"] == "rain_24h_forecast")


def test_watch_api_routes_without_network(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from pravah.watch.server import attach_watch_routes
    app = FastAPI()
    engine = attach_watch_routes(app, tmp_path, token="secret", start_engine=False)
    c = TestClient(app)
    assert c.get("/api/v1/watch/state").status_code == 503
    assert c.post("/api/v1/watch/refresh").status_code == 401
    engine.state = {"mode": "live", "assessment": {"level": "ok"}, "_flow": {"x": 1}}
    body = c.get("/api/v1/watch/state").json()
    assert body["assessment"]["level"] == "ok" and "_flow" not in body
    assert c.get("/api/v1/watch/replay").json()["status"] == "pending"
    assert c.get("/api/v1/watch/geography").status_code == 503


def test_watch_page_renders_live_and_replay_from_synthetic_api(tmp_path, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from app import api_client
    from pravah.watch import river as river_mod
    river, dem_p, geo = _synthetic_world(tmp_path)
    monkeypatch.setattr(river_mod, "RIVER_SOURCE_HINT", river[0])
    monkeypatch.setattr(river_mod, "RIVER_SINK_HINT", river[-1])
    river_mod.build_river_model(geo, [dem_p], tmp_path / "cells.json", tmp_path / "rm")
    rm = river_mod.RiverModel(tmp_path / "rm", device="cpu")
    down = rm.run({"glofas_median": {"UP": 50, "DN": 60}}, glof=hydro.glof_peak(4e5) | {"origin_chainage_m": 0.0, "lake_id": "GLO_1_2"})
    a = assess.assess(_snap())
    state = {"mode": "live", "updated_at_utc": NOW.isoformat(), "assessment": a, "downstream": down,
             "evidence": {"seismic": [], "lakes": {"s2": []}, "latest_passes": []}, "source_errors": {}, "tick_seconds_measured": 0.1}
    geo_doc = {"main_stem": rm.main_stem, "stations": rm.meta["stations"], "reaches": rm.meta["reaches"]}
    frame = {"clock_utc": NOW.isoformat(), "assessment": a, "downstream": down, "new_imagery": []}
    routes = {"/api/v1/watch/state": state, "/api/v1/watch/geography": geo_doc, "/api/v1/watch/timeline?limit=600": {"rows": []},
              "/api/v1/watch/replay": {"status": "ready", "frame_count": 1, "levels": ["ok"], "start_utc": NOW.isoformat()},
              "/api/v1/watch/replay/frames/0": frame}
    monkeypatch.setattr(api_client, "_request", lambda method, path, **kw: (routes.get(path), None if path in routes else "missing"))
    page = Path(__file__).parents[1] / "app/views/watch.py"
    at = AppTest.from_file(str(page), default_timeout=30).run()
    assert not at.exception, at.exception
    assert any("Downstream river response" in s.value for s in at.subheader)
    assert at.dataframe and len(at.dataframe[0].value) == len(down["reaches"])
    at.session_state["pv_watch_mode"] = "Rasuwa 26 Aug 2026 replay"
    at.run()
    assert not at.exception, at.exception


def test_source_fetchers_against_mocked_providers(tmp_path, monkeypatch):
    """Exercise URL formatting and response parsing of every watch source with fake (synthetic) payloads."""
    import httpx

    from pravah.watch import sources
    seen = []

    def handler(req: httpx.Request):
        seen.append(str(req.url))
        u = str(req.url)
        if "copernicus-dem" in u:
            return httpx.Response(404 if "E083" in u else 200, content=b"tif")
        if "archive" in u:
            n = len(req.url.params["latitude"].split(","))
            if "hourly" in req.url.params:
                one = {"hourly": {"time": ["2026-08-11T00:00"], "precipitation": [1.0], "temperature_2m": [5.0]}}
            else:
                one = {"daily": {"time": ["2020-01-01", "2020-01-02"], "precipitation_sum": [1.0, None], "temperature_2m_mean": [3.0, 4.0]}}
            return httpx.Response(200, json=[one] * n if n > 1 else one)
        if "flood" in u:
            n = len(req.url.params["latitude"].split(","))
            daily = {"time": ["2020-01-01", "2020-01-02"], "river_discharge": [10.0, 12.0],
                     "river_discharge_median": [10.0, 12.0], "river_discharge_max": [20.0, 22.0], "river_discharge_min": [5.0, 6.0]}
            return httpx.Response(200, json=[{"daily": daily}] * n if n > 1 else {"daily": daily})
        if "earthquake" in u:
            return httpx.Response(200, json={"features": [{"id": "x", "properties": {"place": "p", "mag": 4.6, "type": "earthquake", "time": 1786000000000},
                                                           "geometry": {"coordinates": [85.3, 28.2, 10]}}]})
        if "api.open-meteo.com" in u:
            n = len(req.url.params["latitude"].split(","))
            one = {"hourly": {"time": ["2026-11-15T00:00"], "precipitation": [0.5]},
                   "daily": {"time": ["2026-11-15"], "precipitation_sum": [2.0], "temperature_2m_mean": [6.0]}}
            return httpx.Response(200, json=[one] * n)
        return httpx.Response(500)

    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: real(*a, **{**kw, "transport": httpx.MockTransport(handler)}))
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    tiles = sources.download_dem((83.5, 27.5, 85.2, 28.2), tmp_path / "dem")
    assert {t["tile"] for t in tiles} == {"Copernicus_DSM_COG_10_N27_00_E084_00_DEM", "Copernicus_DSM_COG_10_N27_00_E085_00_DEM",
                                          "Copernicus_DSM_COG_10_N28_00_E084_00_DEM", "Copernicus_DSM_COG_10_N28_00_E085_00_DEM"}
    assert any(u.endswith("Copernicus_DSM_COG_10_N28_00_E085_00_DEM/Copernicus_DSM_COG_10_N28_00_E085_00_DEM.tif") for u in seen)
    era = sources.era5_daily(date(2020, 1, 1), date(2020, 1, 2))
    assert len(era["precip"][0]) == len(era["points"]) and era["precip"][1][0] is None
    assert len(sources.era5_hourly(date(2026, 8, 11), date(2026, 8, 11))["precip"]) == 1
    cells = [{"station_id": "A", "cell_latitude": 28.0, "cell_longitude": 85.0}, {"station_id": "B", "cell_latitude": 27.9, "cell_longitude": 85.1}]
    g = sources.glofas_daily(cells, date(2020, 1, 1), date(2020, 1, 2), ensemble=True)
    assert g["discharge"] == [[10.0, 10.0], [12.0, 12.0]] and g["river_discharge_max"][0] == [20.0, 20.0]
    assert sources.usgs_events(NOW - timedelta(days=1), NOW)[0]["magnitude"] == 4.6
    fc = sources.forecast_rain()
    assert fc["precip"][0][0] == 2.0 and len(fc["hourly_precip"][0]) == len(fc["points"])


def test_build_history_step_merges_sources(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pravah.watch import build
    days = ["2020-01-01", "2020-01-02", "2020-01-03"]
    monkeypatch.setattr(build.sources, "era5_daily", lambda s, e: {"days": days, "precip": [[1, None]] * 3, "temp": [[2, 3]] * 3,
                                                                    "points": ["a", "b"], "provenance": []})
    monkeypatch.setattr(build.sources, "glofas_daily", lambda c, s, e: {"days": days[1:], "discharge": [[5.0], [None]], "stations": ["S"], "provenance": []})
    (tmp_path / "assets/forecast").mkdir(parents=True)
    (tmp_path / "assets/forecast/glofas_cells.json").write_text(json.dumps({"stations": [{"station_id": "S"}]}))
    monkeypatch.setenv("PRAVAH_WATCH_ASSETS", str(tmp_path / "assets"))
    r = build.step_history(tmp_path / "out", SimpleNamespace())
    t = json.loads((tmp_path / "out/history/table.json").read_text())
    assert r["days"] == 2 and t["precip"][0] == [1, 0] and t["discharge"] == [[5.0], [None]]


def test_api_replaces_non_finite_numbers_with_null(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from pravah.watch.server import attach_watch_routes
    app = FastAPI()
    engine = attach_watch_routes(app, tmp_path, token=None, start_engine=False)
    engine.state = {"assessment": {"level": "ok", "x": float("inf")}, "downstream": {"r": [1.0, float("nan"), np.float32(2.5)]}}
    body = TestClient(app).get("/api/v1/watch/state").json()
    assert body["assessment"]["x"] is None and body["downstream"]["r"] == [1.0, None, 2.5]
    assert body["non_finite_fields"] == [".assessment.x", ".downstream.r[1]"]


def test_nan_station_discharge_is_ignored_and_exposure_is_relative_to_normal_flow(tmp_path, monkeypatch):
    from pravah.watch import river as river_mod
    river, dem_p, geo = _synthetic_world(tmp_path)
    monkeypatch.setattr(river_mod, "RIVER_SOURCE_HINT", river[0])
    monkeypatch.setattr(river_mod, "RIVER_SINK_HINT", river[-1])
    river_mod.build_river_model(geo, [dem_p], tmp_path / "cells.json", tmp_path / "rm")
    rm = river_mod.RiverModel(tmp_path / "rm", device="cpu")
    out = rm.run({"glofas_median": {"UP": float("nan"), "DN": 100.0}}, month=8)   # p50 = 100 in the fixture
    assert all(np.isfinite(r["discharge_m3s"][1]) for r in out["reaches"])
    assert out["exposure"]["buildings"]["max"] == 0       # normal flow exposes nothing new
    coords = json.dumps(out["extent_max"])
    assert "Infinity" not in coords and "NaN" not in coords
    big = rm.run({"glofas_median": {"UP": 100.0, "DN": 100.0}}, glof=hydro.glof_peak(4e5) | {"origin_chainage_m": 0.0}, month=8)
    assert big["exposure"]["buildings"]["max"] >= out["exposure"]["buildings"]["max"]
    assert big["normal_flow_reference"]["month"] == 8


def test_lake_percentage_rule_ignores_unmeasurably_small_baselines():
    tiny = {"s2": [{"lake_id": "GLO_85.6_28.3", "area_m2": 38000, "baseline_m2": 3000, "acquired_at_utc": NOW.isoformat()}],
            "s1": [{"lake_id": "GLO_85.7_28.2", "area_m2": 43000, "baseline_m2": 0.0, "acquired_at_utc": NOW.isoformat()}]}
    a = assess.assess(_snap(lakes=tiny))
    assert {i["id"]: i["status"] for i in a["indicators"]}["lake_change_s2"] == "ok"
    assert {i["id"]: i["status"] for i in a["indicators"]}["lake_change_s1"] == "ok"
