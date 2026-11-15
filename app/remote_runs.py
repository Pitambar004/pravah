"""Load run artifacts from the inference API instead of the local disk.

Lets a dashboard on a laptop show runs produced by the API on Fuzzball: run list,
``observed_metadata.json`` and the ``observed_water.geojson`` map layer are fetched
over HTTP. Rasters stay on the volume (the map uses polygons). Nothing is
synthesised: a missing artifact stays missing and is reported.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from app import api_client
from app.data_loader import RunArtifacts


@st.cache_data(ttl=15, show_spinner=False)
def list_remote_runs(api_url: str) -> tuple[list[str], str | None]:
    """Completed run IDs on the API (``api_url`` is part of the cache key)."""
    data, err = api_client.runs()
    if err:
        return [], err
    return [r["run_id"] for r in data.get("runs", []) if r.get("complete")], None


@st.cache_data(ttl=300, show_spinner="Loading run from inference API…")
def _fetch(api_url: str, run_id: str) -> dict[str, Any]:
    meta, meta_err = api_client.run_metadata(run_id)
    geo, geo_err = api_client.run_json_file(run_id, "observed_water.geojson")
    return {"meta": meta, "meta_err": meta_err, "geo": geo, "geo_err": geo_err}


def load_remote_run(run_id: str) -> RunArtifacts:
    url = api_client.api_url()
    got = _fetch(url, run_id)
    a = RunArtifacts(run_id=run_id, folder=Path(run_id), remote_source=url)
    if got["meta"]:
        a.observed_metadata = got["meta"]
    else:
        a.missing_files.append("observed_metadata.json")
        if got["meta_err"]:
            a.errors.append(f"observed_metadata.json: {got['meta_err']}")
    if isinstance(got["geo"], dict) and got["geo"].get("type") == "FeatureCollection":
        a.water_geojson = got["geo"]
    else:
        a.missing_files.append("observed_water.geojson")
        if got["geo_err"]:
            a.errors.append(f"observed_water.geojson: {got['geo_err']}")
    a.missing_files += ["exposure_summary.json", "benchmark.json"]
    return a
