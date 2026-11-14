# P3 — GIS, Validation and Exposure (Claude/Codex)

**Working branch:** `feat/p3-gis`. Read `AGENTS.md`, `README.md`, `docs/IMPLEMENTATION_RUNBOOK.md` (sections 3, 5, 8, 10, 13, 18–21), `docs/contracts.md`, and `docs/P5_INTEGRATION_HANDOFF.md` before edits. Own `src/pravah/gis/`, `scripts/exposure.py`, new `scripts/gis_validate.py`, GIS tests, and GIS source metadata. Coordinate output contracts with P2/P4/P5. Do not modify other owners' modules without approval; do not merge directly to main.

## P3's verified local progress — 2026-11-14

These facts are based on P3's actual terminal logs provided in the team conversation, **not on files in Git**. Files remain on the contributor's computer. Revalidate artifacts in Codex's working environment; never claim the cloud has access to WSL's C: drive.

- Environment: Windows PowerShell 5.1 + WSL Ubuntu user `preshack`; functioning WSL `rasterio`, `numpy`, `geopandas`, `shapely`, `pyproj`, `ogr2ogr`, `ogrinfo`. Windows Python 3.14 Rasterio DLL was blocked by Windows Application Control; use WSL. SciPy unnecessary.
- **Windows data root:** `C:\PRAVAH_DATA`; **WSL data root:** `/mnt/c/PRAVAH_DATA`.
- **Existing scripts/venv:** `~/pravah/{.venv,check_data_wsl.py,process_rainfall.py,process_terrain.py,extract_all_osm.sh,build_infrastructure.py}` (file existence per execution logs; verify locally). **Repository clone must be `~/pravah-repo`**, not `~/pravah`; protect existing scripts and `.venv`.
- Original source archives are private/local. Do not commit large .pbf, TIFFs, model checkpoints, outputs, credentials, or private signed URLs.

### Real data inventory

1. `raw/rainfall/chirps/`: 26 downloaded CHIRPS v3 FINAL daily EPSG:4326 GeoTIFFs, 2026-08-01 through 2026-08-26; global 0.05 degree cells. Raster inspection found `-9999` sample values even though dataset nodata tag reported None. **Mask `-9999` explicitly.** CHIRPS FINAL is retrospective; original historical publication availability unverified.
2. `raw/dem/`: two Copernicus GLO-30 DSM tiles covering ~85–86°E, 27–29°N; both EPSG:4326, 3600 × 3600. DSM is not ground-truth bare earth.
3. `raw/osm/nepal-260101.osm.pbf`: Geofabrik Nepal 2026-01-01 source, 407,696,403 bytes (~389 MiB), recorded MD5 pass. Source snapshot pre-dates event but may omit later roads/buildings.
4. `raw/models/dhara/model.onnx`: 97,783,082 bytes; and `raw/models/prithvi/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt`: 1,276,843,350 bytes. Three Prithvi 13-band 512×512 example TIFFs (India, Spain, USA), **not actual Rasuwa scenes**. No confirmed Rasuwa Sentinel scenes, validated water segmentation or UNOSAT affected/analysis polygons yet. P2/P1 own those.
5. `raw/earthquakes/`: a local file exists; contents not inspected.

### Local processing already executed

- **Provisional AOI bbox:** `[85.10, 27.85, 85.70, 28.35]` in EPSG:4326. This is **NOT** a verified Rasuwa administrative, watershed or UNOSAT analysis boundary. All statistics/layers are for this temporary selection only.
- `processed/rainfall/`: generated 26 `rainfall_YYYY-MM-DD.tif`, `rainfall_accumulation_aug01_26.tif`, `daily_summary.csv`. CHIRPS invalid -9999 masked. Reported highest *daily area-mean* 45.533 mm and highest sampled AOI cell 86.417 mm on 2026-08-10. These are NOT measured flood triggers or verified pre-event-known measurements. Review temporal leakage (Aug 26 included).
- `processed/terrain/`: `rasuwa_area_dem.tif`, `rasuwa_area_dem_utm.tif`, `rasuwa_area_slope_degrees.tif`. Metric reprojection to EPSG:32645, unsmoothed NumPy-gradient exploratory slope. Valid elevation 454.5–7214.4775 m; 3,935,188 valid slope pixels; median slope 31.34963°, p95 51.148422°, p99 59.746136°, max 83.93866°; 37,240 pixels >60° (~0.95%). Scientific terrain/hydrological QA NOT complete.
- `processed/osm/osm_points.gpkg`: 4,191 features, all valid.
- `processed/osm/osm_lines.gpkg`: 14,506, all valid.
- `processed/osm/osm_multipolygons.gpkg`: 146,075, **4 invalid** in source, 0 missing geometry.
- `processed/osm/osm_multilinestrings.gpkg`: 80, all valid.
- `processed/osm/osm_other_relations.gpkg`: 10, all valid.
- OSM source extraction = bbox spatial filter, not necessarily clipped geometry. Downstream thematic builder intersects with AOI bbox and repairs selected invalid polygons without overwriting source.

### Derived thematic GIS outputs already generated on local disk

Root `processed/osm/thematic/`, files `{name}.gpkg`, plus `infrastructure_summary.json`:

| Layer | Count | Qualification |
|---|---:|---|
| roads | 9,939 | mapped road segments, not closures |
| bridges | 324 | bridge-tagged line features, not necessarily unique bridges |
| waterways | 2,050 | mapped streams/rivers |
| buildings | 139,191 | mapped building footprint geometries, 1 selected invalid geometry repaired |
| settlements | 382 | place-tagged points |
| schools | 330 | amenity-tagged points from available attributes |
| healthcare | 25 | amenity-tagged points |
| emergency_facilities | 11 | available mapped emergency/amenity points |
| waterbodies | 181 | water-tagged polygons |
| landuse | 4,625 | landuse-tagged polygons |

**Never equate these counts with Rasuwa district totals or disaster impacts.** Features may overlap across categories, source completeness varies, and point amenity extraction may miss polygon-mapped POIs. Need inspect `other_tags`, geometries, tag coverage, and dataset licensing. Validate repairs, clipping and dedup before exposure calculations. 10 layers were successfully emitted and all selected features had nonempty clipped geometry.

## Implement P3 scope in repo now

1. Audit existing `src/pravah/gis/exposure.py`, `scripts/exposure.py`, `tests/test_geospatial.py`, and UI contract. Build `scripts/gis_validate.py` (CLI flags for configurable data root/AOI; metadata/CRS/nodata/shape/date and vector geometry QA; JSON summary, nonzero exits for fatal errors). **Do not hardcode a fake verified event AOI.**
2. Integrate the local outputs without adding large data to git: optionally symlink small files from `/mnt/c/PRAVAH_DATA/processed/osm/thematic` for local testing, or use explicit CLI absolute paths. Offer a portable `PRAVAH_DATA_ROOT` variable. Only commit scripts, unit tests and metadata with defensible source citations.
3. Implement/validate exact AOI coverage + raster/vector alignment, strict valid pixel semantics (model mask 0=no inferred water, 1=inferred water, 255=invalid), real metric area in checked projected CRS, counted **unique** building footprints, intersected **line length** for roads, bridge-tag counts with explicit potential duplication. Never confuse intersection with damage.
4. Handle geometry errors, empty layers, CRS mismatch, no-observation, duplicate features, boundaries crossing box, invalid polygons; add automated tests to `tests/`. Avoid including UNOSAT post-event affected extent in historical input.
5. Prepare `exposure_summary.json`, optional derived `exposure.geojson` and verified output metadata in `outputs/<run_id>/` once a real P2 water mask exists. Until then do not invent exposure numbers. Coordinate with P5's `app/data_loader.py`.
6. Read `docs/contracts.md`; maintain scientific provenance: OSM Jan-01 snapshot, CHIRPS FINAL retrospective, actual acquisition/availability when known, SHA256 and CRS/nodata. Document limitations and how to rerun from local paths.
7. Run `pytest -q tests/test_geospatial.py` and relevant tests; record exact results. Implement, commit and push only to `feat/p3-gis`; request pull request review.

## Immediate Codex prompt

"I am P3 GIS owner on branch feat/p3-gis. Read this file and AGENTS.md, README.md, docs/contracts.md, docs/IMPLEMENTATION_RUNBOOK.md, docs/P5_INTEGRATION_HANDOFF.md. My real WSL data are under /mnt/c/PRAVAH_DATA and existing preprocessing scripts are in ~/pravah, while this clone is ~/pravah-repo. Inspect existing GIS code/tests and implement a robust GIS validation CLI and exposure improvements using only P3-owned files. Treat the bbox as provisional, enforce nodata and CRS semantics, do not fabricate observed flood/water masks, do not commit large data. Run tests, show all changes and commit on my branch; do not push or merge without my approval."
