"""Sentinel-2 L2A water / snow indices, glacial-lake areas and river effective width.

Optical only. Units: surface reflectance (0–1) after the L2A offset correction. Water is
MNDWI > 0.1 with low NIR (snow and ice share high MNDWI but are bright in NIR). Pixels
under SCL cloud, shadow, cirrus, saturation or nodata are invalid and never counted as
dry. Lake/width values are withheld when too little of the footprint is valid.
"""
from __future__ import annotations

import numpy as np

INVALID_SCL = {0, 1, 3, 8, 9, 10}   # nodata, saturated, cloud shadow, cloud medium/high, cirrus
SCL_SNOW = 11


def reflectance(dn: np.ndarray, processing_baseline: str | None) -> np.ndarray:
    """L2A DN → reflectance. Baseline ≥ 04.00 (Jan 2022+) adds a +1000 offset."""
    offset = 1000.0 if processing_baseline and float(processing_baseline) >= 4.0 else 0.0
    return (dn.astype("float32") - offset) / 10000.0


def classify(green, nir, swir1, scl) -> dict[str, np.ndarray]:
    finite = np.isfinite(green) & np.isfinite(nir) & np.isfinite(swir1) & np.isfinite(scl)
    scl_i = np.where(np.isfinite(scl), scl, 0).astype("int16")
    valid = finite & ~np.isin(scl_i, list(INVALID_SCL))
    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = (green - swir1) / (green + swir1)
    water = valid & (mndwi > 0.1) & (nir < 0.15)
    snow_ice = valid & ~water & ((scl_i == SCL_SNOW) | ((mndwi > 0.4) & (nir >= 0.15)))
    return {"valid": valid, "water": water, "snow_ice": snow_ice, "mndwi": mndwi}


def zonal_area(mask: np.ndarray, valid: np.ndarray, zones: np.ndarray, n_zones: int, pixel_area_m2: float):
    """Per-zone (1..n) area of ``mask`` and valid fraction, using integer zone raster (0 = none)."""
    z = zones.ravel()
    total = np.bincount(z, minlength=n_zones + 1)[1:].astype(float)
    good = np.bincount(z, weights=valid.ravel().astype(float), minlength=n_zones + 1)[1:]
    hit = np.bincount(z, weights=(mask & valid).ravel().astype(float), minlength=n_zones + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total > 0, good / total, 0.0)
    return hit * pixel_area_m2, frac


def lake_areas(water, valid, lake_zones, lake_ids, pixel_area_m2, min_valid=0.8) -> list[dict]:
    """Water area inside each buffered lake footprint; withheld if valid fraction < min_valid."""
    area, frac = zonal_area(water, valid, lake_zones, len(lake_ids), pixel_area_m2)
    return [{"lake_id": lid, "area_m2": float(a) if f >= min_valid else None, "valid_fraction": round(float(f), 3)}
            for lid, a, f in zip(lake_ids, area, frac)]


def reach_widths(water, valid, reach_zones, reach_lengths_m, pixel_area_m2, min_valid=0.8) -> list[dict]:
    """Effective width = water area in reach corridor / reach centreline length (Pavelsky & Smith style)."""
    area, frac = zonal_area(water, valid, reach_zones, len(reach_lengths_m), pixel_area_m2)
    return [{"reach": i, "width_m": float(a / L) if f >= min_valid and L > 0 else None,
             "valid_fraction": round(float(f), 3)} for i, (a, f, L) in enumerate(zip(area, frac, reach_lengths_m))]
