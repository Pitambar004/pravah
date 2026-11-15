"""Group exposed infrastructure by nearest named settlement, for an alert *preview*.

Input: the ``exposure_<category>.geojson`` layers written by ``scripts/exposure.py``
(features intersecting a predicted or observed water/flood mask) and OSM
settlement points. Each feature's centroid is assigned to the nearest settlement
within ``max_km``; otherwise to ``UNASSIGNED``.

Alert tiers here are ILLUSTRATIVE thresholds for a prototype interface. They are not
an official warning standard; official flood warnings in Nepal are issued by the
Department of Hydrology and Meteorology (DHM) and local authorities. Nothing in
this module sends a message.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import shape

UNASSIGNED = "No named settlement within range"
DEFAULT_TIERS = (("Warning", 50), ("Watch", 10), ("Advisory", 1))  # min exposed buildings, illustrative
EARTH_KM = 6371.0088


@dataclass
class Settlement:
    name: str
    lon: float
    lat: float
    kind: str = ""


@dataclass
class AreaImpact:
    settlement: str
    kind: str = ""
    lon: float | None = None
    lat: float | None = None
    buildings: int = 0
    road_m: float = 0.0
    bridges: int = 0
    tier: str = "None"
    feature_ids: list[str] = field(default_factory=list)


def _centroids(fc: dict[str, Any] | None) -> list[tuple[float, float, dict[str, Any]]]:
    out = []
    for f in (fc or {}).get("features") or []:
        g = f.get("geometry")
        if not g:
            continue
        try:
            c = shape(g).centroid
        except (ValueError, TypeError, AttributeError):
            continue
        if not c.is_empty:
            out.append((c.x, c.y, f.get("properties") or {}))
    return out


def _nearest(points: np.ndarray, sets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of and haversine distance (km) to the nearest settlement for each point."""
    lon1, lat1 = np.radians(points[:, 0])[:, None], np.radians(points[:, 1])[:, None]
    lon2, lat2 = np.radians(sets[:, 0])[None, :], np.radians(sets[:, 1])[None, :]
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    d = 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    idx = d.argmin(axis=1)
    return idx, d[np.arange(len(idx)), idx]


def tier_for(buildings: int, tiers=DEFAULT_TIERS) -> str:
    for name, minimum in tiers:
        if buildings >= minimum:
            return name
    return "None"


def group_by_settlement(exposure: dict[str, dict[str, Any]], settlements: list[Settlement],
                        max_km: float = 3.0, tiers=DEFAULT_TIERS) -> list[AreaImpact]:
    """``exposure`` maps category ('buildings'|'roads'|'bridges') to a FeatureCollection."""
    areas: dict[str, AreaImpact] = {}
    named = [s for s in settlements if s.name and s.name.lower() != "nan"]
    sets = np.array([[s.lon, s.lat] for s in named]) if named else None
    for cat in ("buildings", "roads", "bridges"):
        pts = _centroids(exposure.get(cat))
        if not pts:
            continue
        if sets is not None:
            idx, dist = _nearest(np.array([[x, y] for x, y, _ in pts]), sets)
        for i, (_, _, props) in enumerate(pts):
            if sets is not None and dist[i] <= max_km:
                s = named[int(idx[i])]
                key = f"{s.name}|{s.lon:.5f}|{s.lat:.5f}"
                area = areas.setdefault(key, AreaImpact(s.name, s.kind, s.lon, s.lat))
            else:
                area = areas.setdefault(UNASSIGNED, AreaImpact(UNASSIGNED))
            if cat == "buildings":
                area.buildings += 1
            elif cat == "bridges":
                area.bridges += 1
            else:
                length = props.get("clipped_length_m") or props.get("length_m")
                area.road_m += float(length) if isinstance(length, (int, float)) and math.isfinite(length) else 0.0
            if props.get("osm_id"):
                area.feature_ids.append(str(props["osm_id"]))
    for a in areas.values():
        a.tier = tier_for(a.buildings, tiers)
    return sorted(areas.values(), key=lambda a: (a.settlement == UNASSIGNED, -a.buildings, -a.road_m, a.settlement))


def draft_message(area: AreaImpact, source_kind: str, issued_for: str) -> str:
    """Plain-language draft for a human operator to review. Never sent by PRAVAH."""
    basis = ("a model PREDICTION made before the event" if source_kind == "prediction"
             else "water mapped from a satellite image taken AFTER the event")
    road = f", {area.road_m / 1000:.1f} km of road" if area.road_m else ""
    bridge = f", {area.bridges} bridge section(s)" if area.bridges else ""
    return (f"[SIMULATED {area.tier.upper()} — NOT AN OFFICIAL WARNING] {issued_for}: {basis} shows possible "
            f"flooding near {area.settlement} affecting {area.buildings} mapped building(s){road}{bridge}. "
            "Follow instructions from DHM Nepal and local authorities.")
