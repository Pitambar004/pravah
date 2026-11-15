"""Verified satellite comparison access for the operator and historical replay."""

import os
from pathlib import Path

import streamlit as st

from app.nepal_data import monitor_api_url, request_monitor
from pravah.replay.image_pairs import load_image_pairs


@st.cache_data(ttl=300, show_spinner=False)
def image_context(root, version, endpoint):
    try:
        return load_image_pairs(Path(root)), None
    except (OSError, ValueError, KeyError):
        return request_monitor("GET", "/api/v1/nepal/imagery", timeout=4)


def get_image_pairs():
    root = os.environ.get("PRAVAH_IMAGERY_ROOT", "data/processed/nepal_imagery")
    path = Path(root) / "pairs.json"
    return image_context(root, path.stat().st_mtime if path.exists() else 0, monitor_api_url())
