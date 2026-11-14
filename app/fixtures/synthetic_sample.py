"""Explicitly labeled synthetic test fixtures for UI component verification only.

IMPORTANT SCIENTIFIC INTEGRITY CONSTRAINT:
These geometries, exposure counts, and benchmark numbers are synthetic test fixtures
generated solely to verify Streamlit UI components when live model inference outputs
have not yet run. They MUST NEVER be interpreted, exported, or presented as real
disaster observations or actual cluster benchmark results.
"""

SYNTHETIC_OBSERVED_METADATA = {
    "is_synthetic": True,
    "synthetic_disclaimer": "SYNTHETIC TEST FIXTURE ONLY - DO NOT USE FOR DISASTER ASSESSMENT",
    "run_id": "synthetic_fixture_test_01",
    "threshold": 0.5,
    "source_raster": "data/raw/sentinel2/SYNTHETIC_S2_CHIP_RASUWA.tif",
    "tile_manifest_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "valid_pixel_count": 262144,
    "invalid_pixel_count": 12400,
    "class_codes": {
        "0": "no_water",
        "1": "model_inferred_water",
        "255": "invalid"
    },
    "science_note": "Model-derived water, not calibrated flood forecast or structural damage"
}

SYNTHETIC_EXPOSURE_SUMMARY = {
    "is_synthetic": True,
    "synthetic_disclaimer": "SYNTHETIC TEST FIXTURE ONLY - DO NOT USE FOR DISASTER ASSESSMENT",
    "water_area_m2": 348250.0,
    "building_footprints_intersecting": 14,
    "road_length_m_intersecting": 1845.5,
    "bridge_tagged_features_intersecting": 2,
    "modelled_population_estimate": 85,
    "source_type": "model_inferred_water",
    "mask_artifact": "outputs/synthetic_fixture_test_01/observed_water.tif",
    "osm_snapshot_timestamp": "2026-01-01T00:00:00Z (Geofabrik historical Nepal extract)",
    "valid_pixel_count": 262144,
    "inference_is_not_damage": True,
    "unavailable_fields": []
}

SYNTHETIC_BENCHMARK = {
    "is_synthetic": True,
    "synthetic_disclaimer": "SYNTHETIC TEST FIXTURE ONLY - AWAITING VERIFIED SUNDAY GB10 RUN",
    "speedup_compute": 1.88,
    "parallel_efficiency": 0.94,
    "tile_manifest_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "model_sha256": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "tile_count": 64,
    "one_node_seconds": 18.4,
    "two_node_seconds": 9.8,
    "throughput_one_node_tiles_per_sec": 3.48,
    "throughput_two_node_tiles_per_sec": 6.53,
    "nodes_used": 2,
    "node_ids": ["pgx-worker-01-synthetic", "pgx-worker-02-synthetic"],
    "cuda_verified": True,
    "note": "Compute-only speedup, NOT end-to-end wall-clock speedup"
}

# Synthetic Rasuwa / Trishuli River polygon GeoJSON for testing UI maps
SYNTHETIC_WATER_GEOJSON = {
    "type": "FeatureCollection",
    "is_synthetic": True,
    "features": [
        {
            "type": "Feature",
            "properties": {
                "class": "model_inferred_water",
                "area_m2": 182400.0,
                "label": "Trishuli River corridor - Inferred Water [Synthetic Fixture]"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [85.321, 28.125],
                        [85.328, 28.134],
                        [85.334, 28.148],
                        [85.342, 28.162],
                        [85.345, 28.160],
                        [85.337, 28.146],
                        [85.330, 28.132],
                        [85.323, 28.123],
                        [85.321, 28.125]
                    ]
                ]
            }
        },
        {
            "type": "Feature",
            "properties": {
                "class": "model_inferred_water",
                "area_m2": 165850.0,
                "label": "Bhote Koshi confluence - Inferred Water [Synthetic Fixture]"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [85.345, 28.160],
                        [85.352, 28.175],
                        [85.358, 28.190],
                        [85.362, 28.188],
                        [85.355, 28.173],
                        [85.348, 28.158],
                        [85.345, 28.160]
                    ]
                ]
            }
        }
    ]
}

# Synthetic Study Area / Analysis Coverage bounding polygon
SYNTHETIC_AOI_GEOJSON = {
    "type": "FeatureCollection",
    "is_synthetic": True,
    "features": [
        {
            "type": "Feature",
            "properties": {
                "name": "Rasuwa-Nuwakot Core Analysis AOI",
                "crs": "EPSG:4326",
                "description": "Corridor bounding box for Bhote Koshi / Trishuli"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [85.250, 28.080],
                        [85.420, 28.080],
                        [85.420, 28.250],
                        [85.250, 28.250],
                        [85.250, 28.080]
                    ]
                ]
            }
        }
    ]
}

# Retrospective UNOSAT comparison polygon (Post-Event published August 28, 2026)
SYNTHETIC_UNOSAT_GEOJSON = {
    "type": "FeatureCollection",
    "is_synthetic": True,
    "features": [
        {
            "type": "Feature",
            "properties": {
                "layer": "retrospective_unosat_affected",
                "published_date": "2026-08-28T00:00:00Z",
                "classification": "satellite_detected_mudflow_rockflow_affected",
                "citation": "UNOSAT / UNESCO IHP-WINS 10.63253/1y26ihdq",
                "note": "Published post-event; not an input known before cutoff"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [85.318, 28.120],
                        [85.330, 28.136],
                        [85.338, 28.152],
                        [85.350, 28.170],
                        [85.364, 28.195],
                        [85.368, 28.191],
                        [85.353, 28.167],
                        [85.341, 28.149],
                        [85.332, 28.133],
                        [85.320, 28.118],
                        [85.318, 28.120]
                    ]
                ]
            }
        }
    ]
}

# Synthetic Satellite Scenes & Datasets catalog for Historical Replay verification
SYNTHETIC_REPLAY_ASSETS = [
    {
        "asset_id": "S2A_MSIL1C_20260816T045701_N0511_R119_T45RVP_PRE_01",
        "platform": "Sentinel-2A",
        "product_type": "S2MSI1C",
        "acquired_at_utc": "2026-08-16T04:57:01+00:00",
        "available_at_utc": "2026-08-16T08:12:30+00:00",
        "cloud_cover_percent": 14.2,
        "provider": "Copernicus Data Space Ecosystem (CDSE)",
        "license": "CC-BY-4.0 Copernicus Sentinel Data",
        "description": "Pre-event baseline optical observation (Rasuwa tile 45RVP)"
    },
    {
        "asset_id": "S2B_MSIL1C_20260821T050649_N0511_R119_T45RVP_PRE_02",
        "platform": "Sentinel-2B",
        "product_type": "S2MSI1C",
        "acquired_at_utc": "2026-08-21T05:06:49+00:00",
        "available_at_utc": "2026-08-21T09:45:10+00:00",
        "cloud_cover_percent": 68.5,
        "provider": "Copernicus Data Space Ecosystem (CDSE)",
        "license": "CC-BY-4.0 Copernicus Sentinel Data",
        "description": "Pre-event optical observation with monsoon cloud occlusion"
    },
    {
        "asset_id": "S1A_IW_GRDH_1SDV_20260824T121500_SAR_PRE_01",
        "platform": "Sentinel-1A",
        "product_type": "SAR IW GRDH",
        "acquired_at_utc": "2026-08-24T12:15:00+00:00",
        "available_at_utc": "2026-08-24T16:30:00+00:00",
        "cloud_cover_percent": 0.0,
        "provider": "Copernicus Data Space Ecosystem / ASF",
        "license": "CC-BY-4.0 Copernicus Sentinel Data",
        "description": "Pre-event SAR ground range detected backscatter"
    },
    {
        "asset_id": "OSM_NEPAL_EXTRACT_HISTORICAL_PBF_20260101",
        "platform": "OpenStreetMap",
        "product_type": "Vector OSM PBF",
        "acquired_at_utc": "2026-01-01T00:00:00+00:00",
        "available_at_utc": "2026-01-01T06:00:00+00:00",
        "cloud_cover_percent": None,
        "provider": "Geofabrik Historical Snapshot",
        "license": "ODbL 1.0",
        "description": "Pre-event baseline infrastructure extract (buildings, roads, bridges)"
    },
    {
        "asset_id": "S2A_MSIL1C_20260826T050122_EVENT_UNCONFIRMED_AVAILABILITY",
        "platform": "Sentinel-2A",
        "product_type": "S2MSI1C",
        "acquired_at_utc": "2026-08-26T05:01:22+00:00",
        "available_at_utc": None,  # Availability unknown! Must be EXCLUDED from cutoff gate
        "cloud_cover_percent": 82.1,
        "provider": "Copernicus STAC Catalog",
        "license": "CC-BY-4.0 Copernicus Sentinel Data",
        "description": "Morning event pass with unverified catalog publication timestamp"
    },
    {
        "asset_id": "S2B_MSIL1C_20260827T051219_N0511_R119_T45RVP_POST_01",
        "platform": "Sentinel-2B",
        "product_type": "S2MSI1C",
        "acquired_at_utc": "2026-08-27T05:12:19+00:00",
        "available_at_utc": "2026-08-27T09:30:45+00:00",
        "cloud_cover_percent": 35.6,
        "provider": "Copernicus Data Space Ecosystem (CDSE)",
        "license": "CC-BY-4.0 Copernicus Sentinel Data",
        "description": "Post-event primary optical scene (Retrospective reveal only)"
    },
    {
        "asset_id": "UNOSAT_NEPAL_MUDFLOW_ROCKFLOW_EXTENT_AUG28",
        "platform": "UNOSAT / UNESCO",
        "product_type": "Derived Vector Shapefile",
        "acquired_at_utc": "2026-08-27T10:00:00+00:00",
        "available_at_utc": "2026-08-28T14:00:00+00:00",
        "cloud_cover_percent": None,
        "provider": "UNESCO IHP-WINS / UNITAR-UNOSAT",
        "license": "CC-BY-NC-SA 3.0 IGO",
        "description": "Retrospective satellite-detected mudflow and rockflow extent report"
    }
]
