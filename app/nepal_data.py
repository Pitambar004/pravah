"""Nepal-first data access, independent of the selected benchmark run."""

import os
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from app import api_client
from app.components.map_view import MapLayer
from app.theme import TOKENS
from pravah.configuration import configured_url
from pravah.replay.scenario import load_geography, load_national_rivers


def monitor_api_url():
    explicit = os.environ.get("PRAVAH_NEPAL_API_URL")
    if explicit:
        return explicit.rstrip("/")
    endpoint = api_client.api_url()
    parsed = urlparse(endpoint)
    # Older local benchmark launchers still target the API without Nepal routes.
    # The prepared monitor has its own local service; hosted APIs stay explicit.
    if parsed.hostname in {"localhost", "127.0.0.1"} and parsed.port == 8000 and Path("outputs/nepal_monitor/live/latest.json").exists():
        return configured_url("PRAVAH_NEPAL_API_URL")
    return endpoint


def request_monitor(method, path, **kwargs):
    return api_client._request(method, path, base_url=monitor_api_url(), **kwargs)


@st.cache_data(ttl=300, show_spinner=False)
def geography(root: str, api_url: str, local_version: float = 0):
    try:
        metadata, layers = load_geography(Path(root))
        return {"metadata": metadata, "layers": layers}, None
    except (OSError, ValueError, KeyError):
        return api_client._request("GET", "/api/v1/nepal/geography", timeout=3)


def load_nepal_geography():
    root = os.environ.get("PRAVAH_NEPAL_GEOGRAPHY", "data/processed/nepal_monitor")
    manifest = Path(root) / "manifest.json"
    version = manifest.stat().st_mtime if manifest.exists() else 0
    return geography(root, api_client.api_url(), version)


@st.cache_data(ttl=300, show_spinner=False)
def national_context(root, version, endpoint):
    try:
        return load_national_rivers(Path(root)), None
    except (OSError, ValueError, KeyError):
        return api_client._request("GET", "/api/v1/nepal/rivers", timeout=3)


def load_national_context():
    root = os.environ.get("PRAVAH_NATIONAL_ROOT", "data/processed/nepal_national")
    manifest = Path(root) / "manifest.json"
    return national_context(root, manifest.stat().st_mtime if manifest.exists() else 0, api_client.api_url())


def geography_layers(document):
    layers = (document or {}).get("layers", {})
    metadata = (document or {}).get("metadata", {})
    stamp = metadata.get("retrieved_at_utc", "—")
    result = []
    for name, label, color, fill, visible in [
        ("rivers", "Mapped river centrelines", TOKENS["map_water"], False, True),
        ("riverbanks", "Mapped riverbank water polygons", TOKENS["map_water"], True, True),
        ("buildings", "OSM building footprints", TOKENS["map_exposure"], True, False),
        ("roads", "OSM roads", TOKENS["text_secondary"], False, False),
        ("bridges", "OSM bridge-tagged features", TOKENS["map_exposure"], False, False),
        ("facilities", "OSM public facilities", TOKENS["warning"], True, False),
        ("settlements", "Mapped settlements", TOKENS["text_secondary"], True, False),
    ]:
        result.append(MapLayer(id=name, name=label, group="Geographic context" if name in {"rivers", "riverbanks"} else "Infrastructure",
                               kind="context", color=color, data=layers.get(name), fill=fill, visible=visible,
                               opacity=.7, source="OpenStreetMap contributors · ODbL 1.0", timestamp=stamp,
                               description="Bhote Koshi–Trishuli–Narayani corridor mapped context",
                               limitation="Current mapped geometry. River centrelines do not measure normal width; OSM coverage is incomplete.",
                               unavailable_reason="Nepal layer not acquired"))
    return result
