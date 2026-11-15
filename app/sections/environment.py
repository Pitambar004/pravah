"""Display actual imported Nepal environmental context, never model predictions."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import streamlit as st

from app.components.layout import render_page_header


def checked(path, expected=None):
    path = Path(path)
    if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Source checksum mismatch: " + path.name)
    return path


def render_environment():
    render_page_header(
        "Nepal environmental data",
        "Measured and reanalysis context from your imported archives.",
    )
    root = Path(
        os.environ.get("PRAVAH_USER_CONTEXT_DIR", "data/processed/user_context")
    )
    catalog_file = root / "context_catalog.json"
    if not catalog_file.exists():
        st.info("No imported environmental data is configured.")
        return
    catalog = json.loads(catalog_file.read_text())
    st.caption(
        "These datasets provide retrospective context. They are separate from the optical U-Net inputs and do not generate flood forecasts."
    )
    a, b, c = st.columns(3)
    a.metric("Weather tables", len(catalog["weather"]))
    b.metric("Rainfall days", len(catalog["rainfall"]))
    c.metric("Complete terrain tiles", len(catalog["terrain"]))
    weather, rain, terrain, sources = st.tabs(
        ["Weather", "Rainfall", "Terrain", "Source provenance"]
    )
    with weather:
        names = [f"{r['model']} · {r['location']}" for r in catalog["weather"]]
        chosen = st.selectbox("Provider and location", names)
        record = catalog["weather"][names.index(chosen)]
        frame = pd.read_csv(checked(record["path"], record["sha256"]))
        variables = list(record["units"])
        field = st.selectbox("Variable", variables)
        numeric = pd.to_numeric(frame[field], errors="coerce")
        time_column = next(
            (k for k in ["time_utc", "date_utc", "time", "date"] if k in frame),
            frame.columns[0],
        )
        st.caption(
            f"{record['first_time']} to {record['last_time']} · {record['units'][field]} · {record['time_basis']}"
        )
        if numeric.notna().any():
            chart = pd.DataFrame(
                {"Time": frame[time_column], field: numeric}
            ).set_index("Time")
            st.line_chart(chart, y_label=record["units"][field])
        else:
            st.info(
                "This variable is entirely missing in the supplied provider response; no values are filled in."
            )
        st.caption(
            f"{int(numeric.isna().sum()):,} missing of {len(numeric):,} records. Reanalysis grids are not local gauge measurements."
        )
        st.download_button(
            "Download selected weather CSV",
            Path(record["path"]).read_bytes(),
            file_name=Path(record["path"]).parent.name
            + "_"
            + Path(record["path"]).name,
        )
    with rain:
        path = root / "chirps_point_series.csv"
        if path.exists():
            frame = pd.read_csv(path)
            location = st.selectbox(
                "Rainfall location", sorted(frame.location.unique())
            )
            subset = frame[frame.location == location]
            st.line_chart(
                subset.set_index("date_utc")[["precipitation_mm_per_day"]],
                y_label="mm/day",
            )
            st.caption(
                "CHIRPS v3 RNL · February–July 2026. This archive does not cover the August 26 event. Daily timing derives from ERA5 and is not an independent validation of ERA5."
            )
            st.dataframe(subset, hide_index=True)
            st.download_button(
                "Download rainfall series",
                subset.to_csv(index=False),
                file_name=f"{location}_CHIRPS.csv",
            )
    with terrain:
        coverage_path = root / "terrain_point_coverage.json"
        if coverage_path.exists():
            coverage = json.loads(coverage_path.read_text())
            location = st.selectbox(
                "Terrain location",
                sorted(coverage),
                index=sorted(coverage).index("rasuwa") if "rasuwa" in coverage else 0,
            )
            point = coverage[location]
            if point.get("complete_tile_available"):
                record = next(
                    r for r in catalog["terrain"] if r["path"] == point["source"]
                )
                st.metric(
                    "DSM surface height at requested point",
                    f"{point['surface_height_m']:.1f} m",
                )
                with rasterio.open(checked(record["path"], record["sha256"])) as src:
                    grid = src.read(1, out_shape=(360, 360), masked=True)
                low, high = float(grid.min()), float(grid.max())
                preview = np.clip((grid.filled(low) - low) / max(high - low, 1), 0, 1)
                st.image(
                    preview,
                    caption=f"Copernicus DSM tile · {low:.0f}–{high:.0f} m display range · preview downsampled only",
                    width=480,
                )
                st.caption(
                    f"Point lon/lat: {point['requested_lon_lat']} · source grid {record['crs']}. Surface elevation is not water depth or a flood-risk score."
                )
            else:
                st.info(
                    "No complete, verified terrain tile covers this requested point."
                )
            st.dataframe(
                [
                    {
                        "Location": k,
                        "Tile available": v["complete_tile_available"],
                        "DSM height (m)": v.get("surface_height_m"),
                    }
                    for k, v in coverage.items()
                ],
                hide_index=True,
            )
    with sources:
        st.caption(
            "Historical availability is unverified; these records are excluded from strict historical cutoff inputs."
        )
        st.download_button(
            "Download context provenance",
            catalog_file.read_bytes(),
            file_name="context_catalog.json",
            mime="application/json",
        )
        st.dataframe(
            [
                {
                    "Provider": r["provider"],
                    "Model": r["model"],
                    "Location": r["location"],
                    "Rows": r["rows"],
                    "Start": r["first_time"],
                    "End": r["last_time"],
                }
                for r in catalog["weather"]
            ],
            hide_index=True,
        )
