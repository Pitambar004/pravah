"""Sentinel-1 RTC water and backscatter change. SAR only — never mixed with optical bands.

Input is radiometrically terrain-corrected gamma0 in *linear power* (VV). Open water is
specular and dark: VV < -18 dB is a fixed literature-style prior, not a locally calibrated
threshold. Steep Himalayan terrain causes radar shadow/layover; RTC nodata and very dark
shadow areas outside lake footprints are reported, not interpreted.
"""
from __future__ import annotations

import numpy as np

WATER_DB = -18.0


def to_db(linear: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(linear > 0, 10 * np.log10(linear), np.nan).astype("float32")


def classify(vv_linear: np.ndarray) -> dict[str, np.ndarray]:
    db = to_db(vv_linear)
    valid = np.isfinite(db)
    return {"valid": valid, "water": valid & (db < WATER_DB), "vv_db": db}


def new_water(before: dict, after: dict, region: np.ndarray, pixel_area_m2: float) -> dict:
    """Pixels dry→water inside ``region`` where both passes are valid (same orbit geometry required)."""
    both = before["valid"] & after["valid"] & region
    gained = both & ~before["water"] & after["water"]
    lost = both & before["water"] & ~after["water"]
    return {"new_water_km2": float(gained.sum() * pixel_area_m2 / 1e6),
            "lost_water_km2": float(lost.sum() * pixel_area_m2 / 1e6),
            "compared_km2": float(both.sum() * pixel_area_m2 / 1e6)}
