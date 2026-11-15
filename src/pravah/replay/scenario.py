"""Exploratory river-corridor buffers over actual mapped OSM geometry.

Metric buffers screen exposure to an assumed lateral corridor. They do not
model flow, measured river width, terrain confinement, arrival time or damage.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
from shapely import make_valid
from shapely.geometry import mapping

from pravah.gis.exposure import summarize_exposure
from pravah.replay.experiment import sha256

METRIC_CRS = "EPSG:32645"
CATEGORIES = ["rivers", "riverbanks", "buildings", "roads", "bridges", "facilities", "settlements"]


def load_geography(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    layers = {}
    for name in CATEGORIES:
        record = manifest.get("layers", {}).get(name)
        if record is None:
            layers[name] = None
            continue
        path = root / record["file"]
        if root.resolve() not in path.resolve().parents or path.is_symlink():
            raise ValueError("Invalid geography member")
        if sha256(path) != record["sha256"]:
            raise ValueError("Geography checksum mismatch: " + name)
        doc = json.loads(path.read_text())
        if doc.get("is_synthetic") or manifest.get("is_synthetic"):
            raise ValueError("Synthetic fixtures cannot become Nepal geography")
        layers[name] = doc
    return manifest, layers


def load_national_rivers(root: Path):
    metadata = json.loads((root / "manifest.json").read_text())
    path = root / metadata["file"]
    if metadata.get("is_synthetic") is not False or root.resolve() not in path.resolve().parents or path.is_symlink() or sha256(path) != metadata["sha256"]:
        raise ValueError("National river source failed integrity verification")
    return {"metadata": metadata, "layer": json.loads(path.read_text())}


def frame(doc):
    if doc is None:
        return None
    if not doc.get("features"):
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame.from_features(doc["features"], crs="EPSG:4326")


def collection(geometry, properties):
    return {"type": "FeatureCollection", "features": [] if geometry.is_empty else [
        {"type": "Feature", "geometry": mapping(geometry), "properties": properties}]}


def explore(layers, assumed_half_width_m=25, extra_margin_m=100):
    for value, lo, hi in [(assumed_half_width_m, 5, 100), (extra_margin_m, 0, 250)]:
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError("Scenario width outside supported metric range")
    rivers = frame(layers.get("rivers"))
    if rivers is None or rivers.empty:
        raise ValueError("Mapped river geometry is unavailable")
    projected = rivers.to_crs(METRIC_CRS)
    if not projected.geometry.geom_type.isin(["LineString", "MultiLineString"]).all():
        raise ValueError("River scenario needs line geometry")
    line = projected.geometry.map(make_valid).union_all()
    baseline = line.buffer(assumed_half_width_m)
    expanded = line.buffer(assumed_half_width_m + extra_margin_m)
    overflow = expanded.difference(baseline)
    water = gpd.GeoDataFrame(geometry=[expanded], crs=METRIC_CRS)
    buildings, roads, bridges = (frame(layers.get(k)) for k in ["buildings", "roads", "bridges"])
    exposure = summarize_exposure(water, buildings, roads, bridges, METRIC_CRS,
                                 settlements=frame(layers.get("settlements")))
    intersections = {}
    for name in ["buildings", "roads", "bridges", "facilities", "settlements"]:
        source = frame(layers.get(name))
        if source is None:
            intersections[name] = None
            continue
        source = source.to_crs(METRIC_CRS)
        selected = source[source.intersects(expanded)].copy()
        if name in {"roads", "bridges"} and not selected.empty:
            selected.geometry = selected.geometry.intersection(expanded)
        intersections[name] = json.loads(selected.to_crs("EPSG:4326").to_json())
    exposure["facilities_intersecting"] = None if intersections["facilities"] is None else len(intersections["facilities"]["features"])
    properties = {"kind": "exploratory_scenario", "assumed_half_width_m": assumed_half_width_m,
                  "extra_margin_m": extra_margin_m,
                  "semantics": "Uniform projected buffer of mapped river centrelines; not measured overflow or calibrated flow"}
    def geographic(geom):
        return gpd.GeoSeries([geom], crs=METRIC_CRS).to_crs("EPSG:4326").iloc[0]
    return {"kind": "exploratory_scenario", "metric_crs": METRIC_CRS,
            "assumptions": properties, "forecast": None, "accuracy": None,
            "baseline": collection(geographic(baseline), properties),
            "overflow": collection(geographic(overflow), properties),
            "exposure": exposure, "intersections": intersections,
            "limitation": "Current OSM context and assumed buffers; no hydrodynamic propagation, water depth, arrival time or confirmed damage"}
