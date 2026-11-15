"""Index of every run in the outputs folder, with its role relative to the event.

Roles come only from recorded timestamps: a run is ``pre_event`` / ``post_event`` by
its Sentinel-2 ``acquired_at_utc`` relative to the start of the event day, and a
``prediction`` when it carries ``prediction_metadata.json``. Nothing is inferred
from folder names except a display label.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from app.components.map_view import short_run_label
from app.context import CASE_STUDY
from app.data_loader import RunArtifacts, load_all_runs, load_training_runs

ROLE_LABELS = {"pre_event": "Before event", "post_event": "After event", "prediction": "Prediction",
               "unknown": "Undated"}
ROLE_VARIANTS = {"pre_event": "neutral", "post_event": "accent", "prediction": "warning", "unknown": "neutral"}


def outputs_dir() -> Path:
    return Path(os.environ.get("PRAVAH_OUTPUTS_DIR", "outputs"))


def training_dir() -> Path:
    return Path(os.environ.get("PRAVAH_TRAINING_DIR", str(outputs_dir() / "training")))


def models_dir() -> Path:
    """Finished model folders copied from laptops/clusters (checkpoint + final_evaluation.json)."""
    return Path(os.environ.get("PRAVAH_MODELS_DIR", str(outputs_dir() / "models")))


def local_models_dir() -> Path:
    """Runs written by the local trainer (scripts/train_local.py → outputs/local_demo/<run>/)."""
    return Path(os.environ.get("PRAVAH_LOCAL_MODELS_DIR", str(outputs_dir() / "local_demo")))


def _signature(root: Path) -> tuple:
    """Changes whenever a run folder is added or rewritten, so the cache never goes stale."""
    if not root.is_dir():
        return ()
    return tuple(sorted((d.name, d.stat().st_mtime_ns) for d in root.iterdir() if d.is_dir()))


@st.cache_data(show_spinner="Indexing runs…", max_entries=4)
def _all_runs(root: str, _sig: tuple) -> list[RunArtifacts]:
    return load_all_runs(root)


def all_runs() -> list[RunArtifacts]:
    root = outputs_dir()
    return _all_runs(str(root), _signature(root))


def run_role(a: RunArtifacts) -> str:
    if a.is_prediction:
        return "prediction"
    acq = a.acquired_at
    if not acq:
        return "unknown"
    return "pre_event" if acq < CASE_STUDY["event_day_start_utc"] else "post_event"


def model_tag(a: RunArtifacts) -> str:
    """Which model made the map: 'Your model' (MiniUNet via pravah.inference) or 'Prithvi'."""
    md = a.observed_metadata or {}
    if md.get("engine") == "pravah.inference":
        return "Your model"
    if "prithvi" in str(md.get("model_id") or md.get("model") or "").lower():
        return "Prithvi"
    return ""


def run_title(a: RunArtifacts) -> str:
    tag = model_tag(a)
    return f"{ROLE_LABELS[run_role(a)]} · {short_run_label(a.run_id)}" + (f" · {tag}" if tag else "")


def training_runs() -> list[dict]:
    # small files; re-read every refresh so live runs update
    return load_training_runs(training_dir(), models_dir(), local_models_dir())
