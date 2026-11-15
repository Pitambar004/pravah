"""Downloaded Nepal map context for the Model evidence pages.

Reuses the verified bundles the Rasuwa operations pages read (each loader checks
its manifest checksums and refuses synthetic data):

* ``data/processed/nepal_monitor``  OSM rivers, riverbanks, roads, bridges,
  facilities, settlements, buildings (Geofabrik Nepal extract, ODbL 1.0)
* ``data/processed/nepal_national`` named major rivers of Nepal
* ``data/processed/nepal_lakes``    Glacial Lake Observatory 2017–2024 inventory

Layers are clipped to the Rasuwa study region so every map stays light. They are
current mapped context, not pre-event infrastructure and not flood evidence.
Missing bundles simply yield no layers.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st
from shapely.geometry import box, mapping, shape

from app.components.map_view import MapLayer, _slim_geojson
from app.theme import TOKENS

# lon/lat box covering both Rasuwa Sentinel-2 tiles (T45RUL, T45RUM) with a margin
RASUWA_BBOX = (85.05, 27.80, 85.85, 28.75)
OSM_LIMIT = "Current mapped geometry (OSM coverage is incomplete); not verified pre-event infrastructure."

# (key, label, group, color, fill, visible, light) — "light" layers also go on side-by-side maps
_OSM = [
    ("rivers", "Mapped rivers (OSM)", "Geographic context", "map_water", False, True, True),
    ("riverbanks", "Mapped riverbank water (OSM)", "Geographic context", "map_water", True, False, True),
    ("roads", "Roads (OSM)", "Infrastructure", "text_secondary", False, False, False),
    ("bridges", "Bridges (OSM)", "Infrastructure", "map_exposure", False, False, True),
    ("facilities", "Public facilities (OSM)", "Infrastructure", "warning", True, False, True),
    ("settlements", "Settlements (OSM)", "Infrastructure", "text_secondary", True, False, True),
    ("buildings", "Building footprints (OSM)", "Infrastructure", "map_exposure", True, False, False),
]


def _mtime(root: Path) -> float:
    manifest = root / "manifest.json"
    return manifest.stat().st_mtime if manifest.exists() else 0.0


def _clip(doc: dict[str, Any] | None, bbox: tuple[float, float, float, float]) -> dict[str, Any] | None:
    if not doc:
        return None
    region = box(*bbox)
    feats = []
    for f in doc.get("features") or []:
        if not f.get("geometry"):
            continue
        geom = shape(f["geometry"])
        if not geom.intersects(region):
            continue
        geom = geom.intersection(region)
        if not geom.is_empty:
            feats.append({"type": "Feature", "properties": f.get("properties") or {}, "geometry": mapping(geom)})
    return _slim_geojson({"type": "FeatureCollection", "features": feats}) if feats else None


@st.cache_data(show_spinner=False, max_entries=4)
def _geography(root: str, version: float, bbox: tuple[float, ...]) -> tuple[dict[str, Any], str] | None:
    from pravah.replay.scenario import load_geography

    try:
        meta, layers = load_geography(Path(root))
    except (OSError, ValueError, KeyError):
        return None
    return {k: _clip(layers.get(k), bbox) for k, *_ in _OSM}, meta.get("retrieved_at_utc", "—")


@st.cache_data(show_spinner=False, max_entries=4)
def _national(root: str, version: float) -> tuple[dict[str, Any], str] | None:
    from pravah.replay.scenario import load_national_rivers

    try:
        doc = load_national_rivers(Path(root))
    except (OSError, ValueError, KeyError):
        return None
    return _slim_geojson(doc["layer"]), doc["metadata"].get("retrieved_at_utc", "—")


@st.cache_data(show_spinner=False, max_entries=4)
def _lakes(root: str, version: float, bbox: tuple[float, ...]) -> tuple[dict[str, Any] | None, dict[str, Any]] | None:
    from pravah.replay.lakes import load_lakes

    try:
        doc = load_lakes(Path(root))
    except (OSError, ValueError, KeyError):
        return None
    return _clip(doc["layers"].get("s2"), bbox), doc.get("metadata") or {}


def downloaded_context_layers(light: bool = False, bbox: tuple[float, float, float, float] = RASUWA_BBOX) -> list[MapLayer]:
    """MapLayers for the downloaded Nepal context. ``light=True`` keeps only small layers."""
    geo_root = Path(os.environ.get("PRAVAH_NEPAL_GEOGRAPHY", "data/processed/nepal_monitor"))
    nat_root = Path(os.environ.get("PRAVAH_NATIONAL_ROOT", "data/processed/nepal_national"))
    lake_root = Path(os.environ.get("PRAVAH_LAKE_ROOT", "data/processed/nepal_lakes"))
    layers: list[MapLayer] = []

    national = _national(str(nat_root), _mtime(nat_root))
    if national and not light:
        data, stamp = national
        layers.append(MapLayer(
            id="ctx_national_rivers", name="Major rivers of Nepal (OSM)", group="Geographic context", kind="context",
            color=TOKENS["map_water"], data=data, fill=False, visible=False, opacity=0.8,
            source="OpenStreetMap contributors · ODbL 1.0", timestamp=stamp,
            description="Named major rivers from OSM waterway=river records, simplified for country view.",
            limitation="Centrelines only; not measured width or flow routing."))

    geography = _geography(str(geo_root), _mtime(geo_root), bbox)
    if geography:
        clipped, stamp = geography
        for key, label, group, color, fill, visible, is_light in _OSM:
            if (light and not is_light) or not clipped.get(key):
                continue
            layers.append(MapLayer(
                id=f"ctx_{key}", name=label, group=group, kind="context", color=TOKENS[color],
                data=clipped[key], fill=fill, visible=visible, opacity=0.7,
                source="OpenStreetMap contributors · ODbL 1.0 (Geofabrik Nepal extract)", timestamp=stamp,
                description="Rasuwa study region, clipped from the Bhote Koshi–Trishuli corridor bundle.",
                limitation=OSM_LIMIT))

    lakes = _lakes(str(lake_root), _mtime(lake_root), bbox)
    if lakes and lakes[0]:
        data, meta = lakes
        layers.append(MapLayer(
            id="ctx_glacial_lakes", name="Glacial lakes 2017–2024 (GLO)", group="Geographic context", kind="context",
            color=TOKENS["map_water"], data=data, fill=True, visible=False, opacity=0.6,
            source="Glacial Lake Observatory · CC BY 4.0" + (f" · {meta['doi']}" if meta.get("doi") else ""),
            timestamp="Merged maximum outlines, annual mosaics 2017–2024",
            description="Sentinel-2 lake inventory baseline.",
            limitation="Inventory baseline; not 2026 lake levels or an identified flood source."))
    return layers


def with_downloaded_context(layers: list[MapLayer], light: bool = False) -> list[MapLayer]:
    """Append downloaded context; it replaces the older single ``rivers`` context layer when present."""
    extra = downloaded_context_layers(light=light)
    if any(lyr.id == "ctx_rivers" for lyr in extra):
        layers = [lyr for lyr in layers if lyr.id != "rivers"]
    return list(layers) + extra
