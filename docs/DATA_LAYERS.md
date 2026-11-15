# Dashboard data layers — sources, regeneration and the 14 Nov Rasuwa results

**Audience:** Team 12 (P1–P5), especially P3 (GIS) and whoever presents the demo.
**Date:** 14 Nov 2026. All numbers below were computed from files on disk; commands are reproducible.

## 1. Layer status

| Map layer | Status | Produced by | Source |
|---|---|---|---|
| Study area (AOI) | **Missing** | P1/P3 | Needs UNOSAT *Analysis Extent* → `config/aois/analysis_coverage.geojson` (runbook P1-01). The provisional bbox 85.10–85.70 E, 27.85–28.35 N is deliberately **not** drawn as an AOI. |
| Administrative boundaries | Available (30) | `scripts/build_context_layers.py` | OSM 2026-01-01, `boundary=administrative`, admin_level 6–7. Not an official survey dataset. |
| River network | Available (101) | `scripts/build_context_layers.py` | OSM 2026-01-01, `waterway=river` (Bhote Koshi, Trishuli, Langtang, Chilime …). |
| Model-derived water | Available (4 runs) | P3 Prithvi workflows on Fuzzball → `scripts/import_observation_bundle.py` | Prithvi-EO-2.0-300M-TL-Sen1Floods11 on GB10; S2 L1C. |
| Sentinel-2 scene preview | **Missing** | P1/P5 | Needs RGB quicklooks from the B04/B03/B02 JP2s on `Pravah-Dataset`. |
| Water newly detected / no longer detected | Available (post runs) | `scripts/compare_observations.py` | Pixel-wise change between same-tile pre/post masks. |
| Buildings / roads / bridges | Available (4 runs) | `scripts/exposure.py --features-dir` | OSM 2026-01-01 via P3 pipeline. Potential exposure only. |
| Terrain susceptibility / hypothetical inundation | **Not built** | P3 (lowest priority) | GLO-30 + HydroRIVERS HAND scenario; runbook stop rules rank last. |
| UNOSAT affected extent | **Missing** | P1 | Manual browser download of the UNESCO IHP-WINS shapefiles (runbook §3.1). |

The inference API is **not** required for any of these; the dashboard reads run folders and
`data/processed/context/` directly (runbook §12.3). The API is only for GPU inference and for a laptop
dashboard that cannot see the volume.

## 2. Regenerate everything (local, ~2 min after downloads)

```bash
# 1. Observation bundles from Fuzzball (copied from P3's job logs: base64 tar chunks)
python scripts/import_observation_bundle.py <bundle_dir>... --outputs outputs

# 2. Pre/post change on identical grids (same MGRS tile)
python scripts/compare_observations.py --pre outputs/p3_prithvi_nepal_20260811_T45RUL --post outputs/p3_prithvi_nepal_20260827_T45RUL
python scripts/compare_observations.py --pre outputs/p3_prithvi_nepal_20260812_T45RUM --post outputs/p3_prithvi_nepal_20260903_T45RUM

# 3. OSM snapshot (pre-event, strict-replay eligible), verified against Geofabrik's .md5
curl -fL -o data/raw/osm/nepal-260101.osm.pbf https://download.geofabrik.de/asia/nepal-260101.osm.pbf
#    md5 2a771ff334eae3fd1ec92f0115366ecf
#    sha256 c90d8c81a7c0b57a7220799fddcd4b90f6c0597c13034b9f2eb5653353aa7bd0
python scripts/extract_osm_bbox.py --pbf data/raw/osm/nepal-260101.osm.pbf --out data/processed/osm   # or p3_extract_osm.sh
python scripts/p3_build_infrastructure.py --root data
python scripts/build_context_layers.py --osm-dir data/processed/osm --snapshot 2026-01-01

# 4. Exposure per run (+ map feature layers)
T=data/processed/osm/thematic
python scripts/exposure.py --mask outputs/<run>/observed_water.tif --buildings $T/buildings.gpkg \
  --roads $T/roads.gpkg --bridges $T/bridges.gpkg --settlements $T/settlements.gpkg --schools $T/schools.gpkg \
  --healthcare $T/healthcare.gpkg --emergency-facilities $T/emergency_facilities.gpkg --osm-date 2026-01-01 \
  --out outputs/<run>/exposure_summary.json --polygons-out outputs/<run>/exposure_water_polygons.geojson \
  --features-dir outputs/<run>
```

All outputs are under gitignored `outputs/` and `data/`.

## 3. What the Rasuwa data shows

Inputs: four Prithvi runs made by P3 on a GB10 (`gpu_name: NVIDIA GB10`, model sha256 `76eed77d…`).
Pre-event scenes are strict-cutoff eligible (`available_at_utc` before 2026-08-25T23:59:59Z); post-event
scenes are retrospective only. Water masks were copied with their recorded `output_sha256` verified.

| Tile | Pre → post | Comparable (cloud-free both dates) | Water pre → post (comparable) | New | Lost | Persistent |
|---|---|---|---|---|---|---|
| T45RUL | 11 Aug → 27 Aug | 398 km² (35.8 %) | 26.32 → 11.40 km² | 10.22 km² | 25.14 km² | 1.19 km² |
| T45RUM | 12 Aug → 3 Sep | 56 km² (2.0 %) | 0.79 → 7.74 km² | 7.43 km² | 0.48 km² | 0.31 km² |

Exposure (OSM 2026-01-01, whole valid area of each date):

| Run | Valid km² | Water km² | Buildings ∩ water | Road km ∩ water | Bridges ∩ water |
|---|---|---|---|---|---|
| 11 Aug T45RUL (pre) | 660.4 | 62.71 | 2,296 | 117.6 | 13 |
| 27 Aug T45RUL (post) | 746.9 | 24.17 | 1,780 | 67.1 | 57 |
| 12 Aug T45RUM (pre) | 2,500.7 | 38.79 | 197 | 10.7 | 24 |
| 3 Sep T45RUM (post) | 58.7 | 8.46 | 6 | 1.5 | 0 |

**Interpretation — no credible flood signal yet:**
- On T45RUL only 1.19 km² is water on both dates; real rivers should persist. Newly detected water is
  807 patches, median 800 m² (8 pixels), 630 under 0.5 ha, and **0.0 %** of it lies within 200 m of the
  three largest post-event water bodies — a speckle pattern consistent with shadow / cloud-edge
  misclassification, not a flood plume. The two dates are also different relative orbits (R076 vs R119).
- The pre-event image already intersects 2,296 buildings and 117.6 km of road. That baseline is higher
  than post-event, so post-event intersections are not evidence of impact.
- T45RUM 3 Sep is ~98 % cloud; 2 % comparable area cannot support conclusions.
- No independent Nepal water labels exist; no accuracy is claimed. UNOSAT (when added) maps
  mudflow/rockflow, a different class from optical water.

## 4. Bug fixed in this pass (needs P3 review)

`scripts/p3_build_infrastructure.py` wrote `osm_id = str(row.get("osm_id") or "")`. For closed-way
polygons GDAL's OSM driver leaves `osm_id` NaN (`osm_way_id` holds the id), so all ~139 k buildings got the
id `"nan"` and `summarize_exposure` de-duplicated them to 7 → building counts in `exposure_summary.json`
were wrong (0 instead of 2,296 for 11 Aug). Now uses `way/<osm_way_id>` when present; verified 139,191
distinct ids and summary counts equal the exported features for all four runs.

## 5. Next actions (owners)

1. **P1:** download UNOSAT Mudflow/Rockflow Extent + Analysis Extent + AOI frame (runbook §3.1) → unlocks AOI,
   UNOSAT layers and a qualitative post-event reference.
2. **P1/P5:** RGB quicklooks per scene on Fuzzball (B04/B03/B02), copied back like the bundles.
3. **P2/P3:** shadow handling before any flood claim — terrain-shadow mask from a DEM + sun geometry, or SAR
   (S1 pre/post passes 19 Aug–2 Sep are on the volume) which is insensitive to cloud.
4. **P2:** `outputs/benchmark/A/checkpoint_last.pt` (MiniUNet, 30 steps, val IoU 0.0) is a throughput test, not
   a usable model; train longer before serving it.
