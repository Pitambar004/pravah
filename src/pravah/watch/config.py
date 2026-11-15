"""Fixed geography, sources and analyst rules for the Rasuwa Live Watch.

Every threshold here is an *analyst watch rule*: a transparent trigger for human review,
not a validated flood threshold. DHM issues official warnings in Nepal.
"""
from __future__ import annotations

import os
from pathlib import Path

METRIC_CRS = "EPSG:32645"  # UTM 45N: metres for every length/area/width computation

# Upstream cryosphere watch box (Bhote Koshi / Kyirong headwaters + Langtang), lon/lat.
UPSTREAM_BBOX = (85.05, 28.05, 85.80, 28.80)
# Downstream corridor: Rasuwagadhi (Tibet border) to the Trishuli–Narayani confluence at Devghat.
CORRIDOR_BBOX = (84.38, 27.68, 85.45, 28.55)
RIVER_SOURCE_HINT = (85.30, 28.50)    # northern end of the mapped main stem
RIVER_SINK_HINT = (84.42, 27.74)      # Devghat confluence
MAIN_STEM_NAMES = {"Trishuli Ganga River", "त्रिशुली नदी", "吉隆藏布", "Trishuli River", "भोटे कोशी", "Trisuli"}

# DHM main-stem stations upstream→downstream (ids as served by DHM river-watch).
MAIN_STEM_GAUGES = ["4913", "191", "52", "5705", "4781", "265"]
UPSTREAM_GAUGES = {"4913", "191"}   # a silent upstream gauge during high flow is a GLOF warning sign

# Catchment rainfall points (name, lat, lon): gridded model estimates, not rain gauges.
CATCHMENT_POINTS = [
    ("Kyirong headwaters", 28.85, 85.30), ("Rasuwagadhi", 28.27, 85.38),
    ("Langtang", 28.21, 85.55), ("Gosaikunda", 28.08, 85.41), ("Dhunche", 28.11, 85.30),
    ("Betrawati", 27.97, 85.18), ("Trishuli Bazar", 27.92, 85.15), ("Lower Trishuli", 27.80, 84.80),
]

S2_COLLECTION = "sentinel-2-l2a"   # optical, surface reflectance; SCL cloud/snow classes
S1_COLLECTION = "sentinel-1-rtc"   # SAR, radiometrically terrain-corrected gamma0 (linear power)
S2_RES_M = 20                      # B03/B08/B11/SCL resampled to one 20 m grid for water/lake/width
GLACIER_RES_M = 10                 # B08 at native 10 m for feature tracking
S1_RES_M = 20

# Analyst watch rules (value, units, meaning). Status order: ok < watch < warning < severe.
RULES = {
    "rain_72h_mm": {"watch": 60.0, "warning": 120.0, "units": "mm / 72 h catchment mean"},
    "rain_24h_max_mm": {"watch": 50.0, "warning": 100.0, "units": "mm / 24 h at any catchment point"},
    "seismic_radius_km": 100.0, "seismic_window_h": 72.0,
    "seismic_mag": {"watch": 4.5, "warning": 5.5, "units": "Mw / mb within radius"},
    "gauge_rise_m_per_h": {"warning": 0.5, "units": "m in ≤ 1 h"},
    "gauge_silent_after_s": 3600, "gauge_recent_s": 6 * 3600,
    "lake_growth_fraction": {"watch": 0.10, "warning": 0.25, "units": "area change vs sensor baseline"},
    "lake_min_change_km2": 0.02, "lake_drain_fraction": 0.30, "lake_min_valid_fraction": 0.80,
    # A % change is only meaningful when the baseline itself is measurable: 0.02 km² = 50 px at 20 m.
    "lake_min_baseline_km2": 0.02,
    "glacier_speedup_ratio": {"watch": 2.0, "warning": 3.0, "units": "zone median speed / baseline"},
    "width_anomaly_ratio": {"watch": 1.5, "warning": 2.0, "units": "observed / expected width at current Q"},
    "sar_new_water_km2": {"watch": 0.05, "warning": 0.5, "units": "km² new SAR-dark water in corridor"},
}
RULES_VERSION = "2026-11-15.2"   # bump when any rule changes: cached replay frames are rebuilt
FAMILIES = {"rain": "Rainfall", "seismic": "Seismic", "gauge": "River gauges", "cryosphere": "Glaciers & lakes",
            "river": "River imagery & discharge"}

# Downstream hydraulics priors (ensemble members span them; calibration may narrow them).
MANNING_N = (0.035, 0.050, 0.070)          # mountain gravel/boulder channels and floodplains
GLOF_ATTENUATION_KM = (30.0, 60.0, 120.0)  # e-folding distance of outburst peak; assumption
SECTION_SPACING_M = 500.0
SECTION_HALF_WIDTH_M = 1500.0
SECTION_STEP_M = 30.0


def watch_root() -> Path:
    return Path(os.environ.get("PRAVAH_WATCH_ROOT", "outputs/pravah_watch"))


def data_root() -> Path:
    """Geography/lake/forecast assets (bundled into the Fuzzball code bundle under data/processed)."""
    return Path(os.environ.get("PRAVAH_WATCH_ASSETS", "data/processed"))
