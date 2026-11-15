"""Observation windows: river reaches, upstream lake clusters and glacier zones."""
from __future__ import annotations

import json
from pathlib import Path

from pravah.watch.config import METRIC_CRS, UPSTREAM_BBOX
from pravah.watch.observe import Window


def build_windows(river_dir: Path, lakes_dir: Path, reach_half_width_m=300.0, lake_buffer_m=200.0,
                  glacier_radius_m=2500.0, max_glacier_zones=10, cluster_deg=0.1) -> list[Window]:
    import geopandas as gpd
    from shapely.geometry import box, mapping
    from shapely.ops import substring
    meta = json.loads((river_dir / "river_model.json").read_text())
    stem = gpd.read_file(river_dir / "main_stem.geojson").to_crs(METRIC_CRS).geometry.iloc[0]
    to_geo = lambda geoms: gpd.GeoSeries(geoms, crs=METRIC_CRS).to_crs("EPSG:4326")
    windows: list[Window] = []
    for r in meta["reaches"]:
        seg = substring(stem, r["start_km"] * 1000, r["end_km"] * 1000)
        poly = to_geo([seg.buffer(reach_half_width_m)]).iloc[0]
        windows.append(Window(id=f"reach_{r['reach_id']}", kind="reach", bbox=tuple(poly.bounds),
                              zones=[(r["reach_id"], mapping(poly))], length_m=[seg.length]))
    lakes = gpd.read_file(lakes_dir / "s2_lake_inventory.geojson")
    up = lakes[lakes.intersects(box(*UPSTREAM_BBOX))].copy()
    if up.empty:
        return windows
    up_m = up.to_crs(METRIC_CRS)
    buf = to_geo(list(up_m.geometry.buffer(lake_buffer_m)))
    up["cell"] = [(int(c.x // cluster_deg), int(c.y // cluster_deg)) for c in up.geometry.representative_point()]
    for cell, idx in up.groupby("cell").groups.items():
        geoms = [buf.loc[i] for i in idx]
        bounds = gpd.GeoSeries(geoms).total_bounds
        windows.append(Window(id=f"lakes_{cell[0]}_{cell[1]}", kind="lakes", bbox=tuple(bounds),
                              zones=[(str(up.loc[i, "GLO_ID"]), mapping(g)) for i, g in zip(idx, geoms)]))
    fed = up_m[up.get("CONNECTIVITY", "").astype(str).str.contains("Glacier", case=False)] if "CONNECTIVITY" in up else up_m
    fed = fed.assign(area=fed.geometry.area).sort_values("area", ascending=False).head(max_glacier_zones)
    for i, row in fed.iterrows():
        zone = to_geo([row.geometry.buffer(glacier_radius_m)]).iloc[0]
        windows.append(Window(id=f"glacier_{up.loc[i, 'GLO_ID']}", kind="glacier", bbox=tuple(zone.bounds),
                              zones=[(f"G:{up.loc[i, 'GLO_ID']}", mapping(zone))]))
    return windows


def save(windows: list[Window], path: Path):
    path.write_text(json.dumps([w.__dict__ for w in windows]))


def load(path: Path) -> list[Window]:
    return [Window(**{**d, "bbox": tuple(d["bbox"]), "zones": [tuple(z) for z in d["zones"]]})
            for d in json.loads(path.read_text())]
