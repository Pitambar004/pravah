"""Static river model (built once on GB10) + the downstream ensemble run by the live engine.

``build_river_model`` writes ``river_model/``: main stem, cross-sections, 10 km reaches,
station chainages and an exposure index of every mapped OSM building / road point / bridge /
facility / settlement (nearest section, lateral offset, DEM ground elevation).

``RiverModel.run`` combines all three methods the team chose:
  hydraulic   — Manning normal depth on DEM sections (n ∈ prior range) for each discharge source
  empirical   — W = a·Q^b per reach from historical Sentinel-2 widths (if a fit exists)
  ML          — discharge from the GRU emulator ensemble (alongside GloFAS forecast discharge)
plus a GLOF member set when a lake indicator fires. Output: per-reach width/depth/arrival
ranges, method agreement, inundation extent polygons and exposed-asset counts.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from pravah.watch import hydro
from pravah.watch.config import (
    CORRIDOR_BBOX,
    MANNING_N,
    METRIC_CRS,
    RIVER_SINK_HINT,
    RIVER_SOURCE_HINT,
    UPSTREAM_BBOX,
)

ASSET_KINDS = ("buildings", "roads", "bridges", "facilities", "settlements")
REACH_KM = 10.0


def dem_sampler(paths: list[Path]):
    """Return f(x, y in EPSG:32645) -> z using a merged in-memory DEM mosaic (EPSG:4326 COP-30)."""
    import rasterio
    from pyproj import Transformer
    from rasterio.merge import merge
    srcs = [rasterio.open(p) for p in paths]
    mosaic, transform = merge(srcs, nodata=np.nan, dtype="float32")
    for s in srcs:
        s.close()
    arr = mosaic[0]
    inv = ~transform
    to_geo = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)

    def sample(x, y):
        lon, lat = to_geo.transform(np.asarray(x), np.asarray(y))
        col, row = inv * (np.asarray(lon), np.asarray(lat))
        r, c = np.floor(row).astype(int), np.floor(col).astype(int)
        ok = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
        out = np.full(r.shape, np.nan, "float32")
        out[ok] = arr[r[ok], c[ok]]
        return out
    return sample


def _points(doc: dict, kind: str, road_step_m: float = 50.0):
    """Asset points (EPSG:32645) with weights: roads sampled every 50 m (weight = metres)."""
    import geopandas as gpd
    if not doc or not doc.get("features"):
        return np.zeros(0), np.zeros(0), np.zeros(0), []
    g = gpd.GeoDataFrame.from_features(doc["features"], crs="EPSG:4326").to_crs(METRIC_CRS)
    xs, ys, w, names = [], [], [], []
    for geom, props in zip(g.geometry, doc["features"]):
        name = (props.get("properties") or {}).get("name")
        if geom is None or geom.is_empty:
            continue
        if kind == "roads":
            parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
            for p in parts:
                n = max(1, int(p.length // road_step_m))
                for i in range(n):
                    pt = p.interpolate((i + 0.5) * p.length / n)
                    xs.append(pt.x)
                    ys.append(pt.y)
                    w.append(p.length / n)
                    names.append(name)
        else:
            pt = geom.representative_point() if geom.geom_type != "Point" else geom
            xs.append(pt.x)
            ys.append(pt.y)
            w.append(1.0)
            names.append(name)
    return np.array(xs), np.array(ys), np.array(w), names


def build_river_model(geography: Path, dem_paths: list[Path], glofas_cells: Path, out: Path) -> dict:
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import Point
    t0 = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    rivers = json.loads((geography / "rivers.geojson").read_text())
    line, stem_meta = hydro.main_stem(rivers, RIVER_SOURCE_HINT, RIVER_SINK_HINT)
    sample = dem_sampler(dem_paths)
    sec = hydro.build_sections(line, sample)
    (out / "sections.json").write_text(json.dumps(sec.to_json()))
    gpd.GeoSeries([line], crs=METRIC_CRS).to_crs("EPSG:4326").to_file(out / "main_stem.geojson", driver="GeoJSON")
    # stations → chainage
    to_m = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
    cells = json.loads(glofas_cells.read_text())["stations"]
    stations = []
    for c in cells:
        x, y = to_m.transform(c["gauge_longitude"], c["gauge_latitude"])
        d = line.distance(Point(x, y))
        if d <= 3000:
            stations.append({**{k: c[k] for k in ("station_id", "station_name", "gauge_latitude", "gauge_longitude",
                                                  "cell_latitude", "cell_longitude")},
                             "chainage_m": float(line.project(Point(x, y))), "offset_from_stem_m": round(d, 1),
                             "monthly_climatology_m3s": c["monthly_climatology_m3s"]})
    stations.sort(key=lambda s: s["chainage_m"])
    # reaches
    n_reach = int(np.ceil(line.length / (REACH_KM * 1000)))
    reaches = []
    for i in range(n_reach):
        a, b = i * REACH_KM * 1000, min(line.length, (i + 1) * REACH_KM * 1000)
        reaches.append({"reach": i, "reach_id": f"R{i:02d}", "start_km": a / 1000, "end_km": b / 1000,
                        "sections": [int(k) for k in np.where((sec.chainage_m >= a) & (sec.chainage_m < b))[0]]})
    # exposure index
    assets, counts = {}, {}
    for kind in ASSET_KINDS:
        path = geography / f"{kind}.geojson"
        doc = json.loads(path.read_text()) if path.exists() else None
        xs, ys, w, names = _points(doc, kind)
        if len(xs) == 0:
            continue
        gz = sample(xs, ys)
        idx = hydro.index_assets(sec, xs, ys, gz)
        keep = idx["inside"]
        assets[kind] = {"x": xs[keep], "y": ys[keep], "w": w[keep], "section": idx["section"][keep],
                        "offset_m": idx["offset_m"][keep], "ground_z": gz[keep]}
        if kind in ("settlements", "facilities", "bridges"):
            assets[kind]["names"] = [n for n, k in zip(names, keep) if k]
        counts[kind] = int(keep.sum())
    np.savez_compressed(out / "assets.npz", **{f"{k}__{f}": v for k, d in assets.items() for f, v in d.items() if f != "names"})
    names = {k: d["names"] for k, d in assets.items() if "names" in d}
    # label reaches with the nearest mapped settlement
    if "settlements" in assets:
        s = assets["settlements"]
        for r in reaches:
            mid = len(r["sections"]) and r["sections"][len(r["sections"]) // 2]
            d = np.hypot(s["x"] - sec.cx[mid], s["y"] - sec.cy[mid])
            j = int(np.argmin(d)) if len(d) else None
            r["label"] = names["settlements"][j] if j is not None and d[j] < 6000 and names["settlements"][j] else r["reach_id"]
    meta = {"schema": "pravah-river-model-v1", "crs": METRIC_CRS, "main_stem": stem_meta, "sections": len(sec.chainage_m),
            "section_spacing_m": float(sec.chainage_m[1] - sec.chainage_m[0]) if len(sec.chainage_m) > 1 else None,
            "stations": stations, "reaches": reaches, "asset_counts_in_corridor": counts, "asset_names": names,
            "dem": [p.name for p in dem_paths], "geography_manifest": json.loads((geography / "manifest.json").read_text()).get("source"),
            "build_seconds_measured": round(time.perf_counter() - t0, 2), "bboxes": {"corridor": CORRIDOR_BBOX, "upstream": UPSTREAM_BBOX},
            "method": hydro.__doc__}
    (out / "river_model.json").write_text(json.dumps(meta, indent=2))
    return meta


class RiverModel:
    def __init__(self, root: Path, device: str | None = None):
        self.root = root
        self.meta = json.loads((root / "river_model.json").read_text())
        self.sec = hydro.Sections.from_json(json.loads((root / "sections.json").read_text()))
        z = np.load(root / "assets.npz")
        self.assets: dict = {}
        for key in z.files:
            kind, field = key.split("__", 1)
            self.assets.setdefault(kind, {})[field] = z[key]
        self.fits = {}
        fp = root / "width_fits.json"
        if fp.exists():
            self.fits = json.loads(fp.read_text()).get("reaches", {})
        self.device = device
        self.main_stem = json.loads((root / "main_stem.geojson").read_text())

    # discharge along the stem from station values (linear in chainage, flat beyond the ends)
    def along(self, station_q: dict[str, float]) -> np.ndarray:
        pts = [(s["chainage_m"], station_q[s["station_id"]]) for s in self.meta["stations"]
               if station_q.get(s["station_id"]) is not None]
        if not pts:
            raise ValueError("No discharge at any main-stem station")
        ch, q = zip(*pts)
        return np.interp(self.sec.chainage_m, ch, q)

    def members(self, sources: dict[str, dict[str, float]], glof: dict | None = None):
        """sources: {"glofas_median": {station: Q}, "emulator": {...}, ...} → (S, M) Q and member labels."""
        cols, labels = [], []
        for name, sq in sources.items():
            if not sq:
                continue
            base = self.along(sq)
            for n in MANNING_N:
                cols.append((base, n))
                labels.append({"source": name, "manning_n": n, "glof": None})
        if glof:
            base_src = next(iter(k for k in sources if sources[k]), None)
            base = self.along(sources[base_src]) if base_src else np.zeros_like(self.sec.chainage_m)
            dist_km = np.clip(self.sec.chainage_m - glof.get("origin_chainage_m", 0.0), 0, None) / 1000
            for pk in ("peak_m3s_mid", "peak_m3s_high"):
                Qs = hydro.attenuate(glof[pk], base, dist_km)
                for j, L in enumerate(hydro.GLOF_ATTENUATION_KM):
                    cols.append((Qs[:, j], MANNING_N[1]))
                    labels.append({"source": "glof+" + (base_src or "none"), "manning_n": MANNING_N[1],
                                   "glof": {"peak": pk, "attenuation_km": L}})
        if not cols:
            raise ValueError("No discharge sources")
        Q = np.stack([c[0] for c in cols], 1)
        n = np.array([c[1] for c in cols])
        return Q, n, labels

    def run(self, sources: dict, glof: dict | None = None, polygons: bool = True) -> dict:
        t0 = time.perf_counter()
        Q, n, labels = self.members(sources, glof)
        res = hydro.normal_depth(self.sec, Q, n, device=self.device)
        hyd_s = time.perf_counter() - t0
        origin = glof.get("origin_chainage_m", 0.0) if glof else 0.0
        arrival = hydro.arrival_hours(self.sec, res["velocity_ms"], origin)
        exposure = {}
        for kind, a in self.assets.items():
            idx = {"section": a["section"].astype(int), "offset_m": a["offset_m"], "inside": np.ones(len(a["section"]), bool),
                   "ground_z": a["ground_z"]}
            hit = hydro.flooded(idx, res)                                     # (N, M)
            w = a["w"][:, None]
            per_member = (hit * w).sum(0)
            exposure[kind] = {"median": float(np.median(per_member)), "max": float(per_member.max()),
                              "min": float(per_member.min()), "units": "km" if kind == "roads" else "count"}
            if kind == "roads":
                exposure[kind] = {k: (v / 1000 if isinstance(v, float) else v) for k, v in exposure[kind].items()}
            exposure[kind]["per_reach_max"] = {}
            worst = int(np.argmax(per_member))
            for r in self.meta["reaches"]:
                secs = set(r["sections"])
                sel = np.array([s in secs for s in idx["section"]]) if len(idx["section"]) else np.zeros(0, bool)
                v = float((hit[sel, worst] * a["w"][sel]).sum()) if sel.any() else 0.0
                exposure[kind]["per_reach_max"][r["reach_id"]] = round(v / 1000, 2) if kind == "roads" else int(v)
        reaches = []
        for r in self.meta["reaches"]:
            s = np.array(r["sections"], int)
            if len(s) == 0:
                continue
            q_med = float(np.median(Q[s]))
            hyd_w = res["width_m"][s]
            fit = self.fits.get(r["reach_id"])
            emp = hydro.empirical_width(fit, q_med) if fit and fit.get("r2", 0) >= 0.3 else None
            hyd_med = float(np.median(hyd_w))
            reaches.append({
                "reach_id": r["reach_id"], "label": r.get("label", r["reach_id"]), "start_km": r["start_km"], "end_km": r["end_km"],
                "discharge_m3s": [float(Q[s].min()), q_med, float(Q[s].max())],
                "hydraulic_width_m": [float(np.percentile(hyd_w, 10)), hyd_med, float(np.percentile(hyd_w, 90))],
                "empirical_width_m": None if emp is None else float(emp),
                "width_methods_agree": None if emp is None else bool(0.5 <= emp / max(hyd_med, 1) <= 2.0),
                "depth_m": [float(np.percentile(res["depth_m"][s], 10)), float(np.median(res["depth_m"][s])),
                            float(np.percentile(res["depth_m"][s], 90))],
                "velocity_ms": float(np.median(res["velocity_ms"][s])),
                "arrival_h": None if np.isnan(arrival[s]).all() else [float(np.nanmin(arrival[s])), float(np.nanmedian(arrival[s])),
                                                                     float(np.nanmax(arrival[s]))],
                "section_full_fraction": float(res["section_full"][s].mean()),
            })
        out = {"members": labels, "member_count": len(labels), "reaches": reaches, "exposure": exposure,
               "device": res["device"], "hydraulics_seconds_measured": round(hyd_s, 4),
               "glof": glof, "method": "Manning normal depth on DEM sections + W=aQ^b + emulator/GloFAS discharge"}
        if polygons:
            med = int(np.argsort(res["width_m"].sum(0))[len(labels) // 2])
            worst = int(np.argmax(res["width_m"].sum(0)))
            out["extent_median"] = hydro.extent_polygon(self.sec, res["left_m"][:, med], res["right_m"][:, med])
            out["extent_max"] = hydro.extent_polygon(self.sec, res["left_m"][:, worst], res["right_m"][:, worst])
            out["exposed_points_max"] = self.exposed_points(res, worst)
        out["total_seconds_measured"] = round(time.perf_counter() - t0, 4)
        return out

    def exposed_points(self, res, member, limit=3000):
        from pyproj import Transformer
        to_geo = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)
        feats = []
        for kind in ("buildings", "bridges", "facilities", "settlements"):
            a = self.assets.get(kind)
            if not a:
                continue
            idx = {"section": a["section"].astype(int), "offset_m": a["offset_m"], "inside": np.ones(len(a["section"]), bool),
                   "ground_z": a["ground_z"]}
            hit = hydro.flooded(idx, {k: v[:, member:member + 1] for k, v in res.items() if isinstance(v, np.ndarray)})[:, 0]
            names = self.meta.get("asset_names", {}).get(kind)
            for i in np.where(hit)[0][:limit]:
                lon, lat = to_geo.transform(a["x"][i], a["y"][i])
                feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                              "properties": {"kind": kind, "name": names[i] if names and i < len(names) else None,
                                             "chainage_km": round(float(self.sec.chainage_m[a["section"][i]]) / 1000, 1)}})
        return {"type": "FeatureCollection", "features": feats}

    def chainage_of(self, lon: float, lat: float) -> float:
        from pyproj import Transformer
        x, y = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True).transform(lon, lat)
        return float(self.sec.chainage_m[int(np.argmin(np.hypot(self.sec.cx - x, self.sec.cy - y)))])
