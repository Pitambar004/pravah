"""Turn one satellite pass into watch observations — identical code for live, history and replay.

Windows: each river reach, each upstream lake cluster and each glacier zone is read as its
own small metric grid, so a pass costs a few dozen windowed COG reads instead of a whole
tile. Optical (S2 L2A) and SAR (S1 RTC) produce separate records with their sensor named.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from pravah.watch import optical, sar, stac
from pravah.watch.config import (
    GLACIER_RES_M,
    S1_COLLECTION,
    S1_RES_M,
    S2_COLLECTION,
    S2_RES_M,
)


@dataclass
class Window:
    id: str
    kind: str                     # reach | lakes | glacier
    bbox: tuple                   # lon/lat
    zones: list                   # [(zone_id, geometry_wgs84)] rasterised to 1..n on the window grid
    length_m: list | None = None  # reach centreline length per zone (reach windows)


def rasterize_zones(window: Window, grid: stac.Grid) -> np.ndarray:
    import geopandas as gpd
    from rasterio.features import rasterize
    if not window.zones:
        return np.zeros((grid.height, grid.width), "int32")
    from shapely.geometry import shape
    g = gpd.GeoSeries([shape(z[1]) if isinstance(z[1], dict) else z[1] for z in window.zones],
                      crs="EPSG:4326").to_crs(grid.crs)
    return rasterize([(geom, i + 1) for i, geom in enumerate(g)], out_shape=(grid.height, grid.width),
                     transform=grid.transform, fill=0, dtype="int32")


def safe_name(pass_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", pass_id)[:150]


class Observer:
    def __init__(self, windows: list[Window], store: Path, threads: int = 8):
        self.windows, self.store, self.threads = windows, store, threads
        self.store.mkdir(parents=True, exist_ok=True)
        self._grids: dict = {}

    def grid(self, w: Window, res: float):
        key = (w.id, res)
        if key not in self._grids:
            g = stac.Grid.from_bbox(w.bbox, res)
            self._grids[key] = (g, rasterize_zones(w, g))
        return self._grids[key]

    # ------------------------------------------------------------------------ Sentinel-2 L2A
    def s2_pass(self, items: list[dict], discovered_at=None) -> dict:
        rec = stac.pass_record(items, S2_COLLECTION, discovered_at)
        baseline = rec["processing_baseline"]

        def one(w: Window):
            if w.kind == "glacier":
                g, zones = self.grid(w, GLACIER_RES_M)
                nir = stac.mosaic(items, "B08", g, S2_COLLECTION, bbox=w.bbox)
                scl = stac.mosaic(items, "SCL", g, S2_COLLECTION, resampling="nearest", bbox=w.bbox)
                green = stac.mosaic(items, "B03", g, S2_COLLECTION, bbox=w.bbox)
                swir = stac.mosaic(items, "B11", g, S2_COLLECTION, bbox=w.bbox)
                cls = optical.classify(optical.reflectance(green, baseline), optical.reflectance(nir, baseline),
                                       optical.reflectance(swir, baseline), scl)
                return w, {"nir": optical.reflectance(nir, baseline), "usable": cls["snow_ice"] | (cls["valid"] & ~cls["water"])}
            g, zones = self.grid(w, S2_RES_M)
            bands = {b: stac.mosaic(items, b, g, S2_COLLECTION, bbox=w.bbox) for b in ("B03", "B08", "B11")}
            scl = stac.mosaic(items, "SCL", g, S2_COLLECTION, resampling="nearest", bbox=w.bbox)
            cls = optical.classify(*(optical.reflectance(bands[b], baseline) for b in ("B03", "B08", "B11")), scl)
            px = g.res * g.res
            if w.kind == "reach":
                return w, {"reaches": [dict(r, reach_id=w.zones[r["reach"]][0]) for r in
                                       optical.reach_widths(cls["water"], cls["valid"], zones, w.length_m, px)]}
            return w, {"lakes": optical.lake_areas(cls["water"], cls["valid"], zones, [z[0] for z in w.zones], px)}

        out = {"pass": rec, "sensor": "sentinel-2-l2a (optical)", "lakes": [], "reaches": [], "glacier_windows": []}
        arrays = {}
        with ThreadPoolExecutor(self.threads) as pool:
            for w, res in pool.map(one, self.windows):
                if w.kind == "glacier":
                    arrays[w.id + "__nir"] = res["nir"].astype("float32")
                    arrays[w.id + "__usable"] = res["usable"]
                    out["glacier_windows"].append(w.id)
                else:
                    out["lakes"] += res.get("lakes", [])
                    out["reaches"] += res.get("reaches", [])
        name = safe_name(rec["pass_id"])
        if arrays:
            np.savez_compressed(self.store / f"{name}.npz", **arrays)
        (self.store / f"{name}.json").write_text(json.dumps(out))
        return out

    def glacier_pair(self, before: dict, after: dict) -> list[dict]:
        """Feature-track every glacier window between two stored S2 passes (GPU)."""
        from pravah.watch.glacier import track, zone_speeds
        a = np.load(self.store / f"{safe_name(before['pass']['pass_id'])}.npz")
        b = np.load(self.store / f"{safe_name(after['pass']['pass_id'])}.npz")
        days = (datetime.fromisoformat(after["pass"]["acquired_at_utc"])
                - datetime.fromisoformat(before["pass"]["acquired_at_utc"])).total_seconds() / 86400
        results = []
        for w in self.windows:
            if w.kind != "glacier" or f"{w.id}__nir" not in a or f"{w.id}__nir" not in b:
                continue
            _g, zones = self.grid(w, GLACIER_RES_M)
            usable = a[f"{w.id}__usable"] & b[f"{w.id}__usable"]
            r = track(a[f"{w.id}__nir"], b[f"{w.id}__nir"], usable)
            for z in zone_speeds(r, zones, [zz[0] for zz in w.zones], GLACIER_RES_M, days):
                results.append({**z, "window": w.id, "days": round(days, 2), "device": r.get("device")})
        return results

    # ------------------------------------------------------------------------- Sentinel-1 RTC
    def s1_pass(self, items: list[dict], discovered_at=None) -> dict:
        rec = stac.pass_record(items, S1_COLLECTION, discovered_at)

        def one(w: Window):
            if w.kind == "glacier":
                return w, None
            g, zones = self.grid(w, S1_RES_M)
            vv = stac.mosaic(items, "vv", g, S1_COLLECTION, bbox=w.bbox)
            cls = sar.classify(vv)
            px = g.res * g.res
            if w.kind == "reach":
                return w, {"water": cls["water"], "valid": cls["valid"], "region": zones > 0}
            return w, {"lakes": optical.lake_areas(cls["water"], cls["valid"], zones, [z[0] for z in w.zones], px)}

        out = {"pass": rec, "sensor": "sentinel-1-rtc (SAR, VV)", "lakes": [], "corridor_water_windows": []}
        arrays = {}
        with ThreadPoolExecutor(self.threads) as pool:
            for w, res in pool.map(one, self.windows):
                if res is None:
                    continue
                if w.kind == "reach":
                    arrays[w.id + "__water"], arrays[w.id + "__valid"] = res["water"], res["valid"]
                    out["corridor_water_windows"].append(w.id)
                else:
                    out["lakes"] += [dict(x, sensor="sentinel-1") for x in res["lakes"]]
        name = safe_name(rec["pass_id"])
        np.savez_compressed(self.store / f"{name}.npz", **arrays)
        (self.store / f"{name}.json").write_text(json.dumps(out))
        return out

    def s1_change(self, before: dict, after: dict) -> dict:
        """New SAR water in the corridor between two passes of the same relative orbit."""
        if before["pass"].get("relative_orbit") != after["pass"].get("relative_orbit"):
            raise ValueError("SAR change needs the same relative orbit (identical viewing geometry)")
        a = np.load(self.store / f"{safe_name(before['pass']['pass_id'])}.npz")
        b = np.load(self.store / f"{safe_name(after['pass']['pass_id'])}.npz")
        total = {"new_water_km2": 0.0, "lost_water_km2": 0.0, "compared_km2": 0.0}
        for w in self.windows:
            if w.kind != "reach" or f"{w.id}__water" not in a or f"{w.id}__water" not in b:
                continue
            g, zones = self.grid(w, S1_RES_M)
            r = sar.new_water({"water": a[f"{w.id}__water"], "valid": a[f"{w.id}__valid"]},
                              {"water": b[f"{w.id}__water"], "valid": b[f"{w.id}__valid"]}, zones > 0, g.res * g.res)
            for k in total:
                total[k] += r[k]
        return {**total, "before": before["pass"]["pass_id"], "after": after["pass"]["pass_id"],
                "sensor": "sentinel-1-rtc", "threshold_db": sar.WATER_DB}
