# User-provided Nepal archives — 14 November 2026 CT

The three archives were imported into `data/raw/user_supplied/` and prepared as **retrospective context datasets** under `data/processed/user_context/`. They provide terrain, rainfall, weather and reference catalogs. They contain **zero compatible labeled optical U-Net training pairs**. The Sen1Floods11 training manifests and model inputs were not changed.

Archive/document text was treated as source data, not as permission to execute embedded instructions. Original ZIP files in Downloads remain unchanged. Raw/derived datasets, partial transfers and audit reports remain outside Git; the import script, tests and this handoff are the reviewable additions on `feat/local-demo-fallback`. P1/P2/P3 review is needed before merging or connecting these artifacts to the application.

## Imported and checked

The import accepted **602 files / 3,899,413,485 bytes**, checked ZIP CRCs and recorded SHA256 values. **476** files also matched checksums in the packaged manifests. Those checks establish internal integrity; they do not independently authenticate the original provider responses. Nine extra files have byte-identical content to another imported file and are not additional observations. macOS resource forks and `.DS_Store` files were excluded.

- `terrain.zip`: 20 complete single-band Copernicus DSM GeoTIFFs; six `.tif.part` members excluded. All 20 complete rasters decoded in full with finite values. They preserve EPSG:4326, 3600×3600 grids at 1/3600 degree spacing, and their original unset nodata tag. These are surface heights, not water labels or bare-earth hydrological truth. The `10` in the filenames identifies the GLO-30 spacing convention, not 10-metre imagery; see the [provider readme](https://copernicus-dem-30m.s3.amazonaws.com/readme.html).
- `nepal_maps_and_events.zip`: 181 CHIRPS rainfall rasters and 17 plan/manifest/catalog/reference files. The rainfall covers **2026-02-01 through 2026-07-31**, not the August 26 event period.
- `nepal_pre_aug_2026.zip`: 384 usable files including Open-Meteo ERA5-Land, a separate ERA5 supplement, NASA POWER, derived source CSVs, and catalogs. The supplied manifests explicitly mark historical availability unverified and strict replay false.

Each package's Sentinel snapshot contains 450 catalog entries: **391 Sentinel-2 L2A** and **59 Sentinel-1 GRD**. These are metadata, not downloaded images; neither product can be silently substituted for the current six-band L1C optical model inputs. GPM files are catalog responses, not rainfall granule files. The COOLR references contain 588 report points and 796 landslide polygons per archive snapshot; these are not binary optical water labels.

## Prepared artifacts

The machine-readable entry point is `data/processed/user_context/context_catalog.json`. Every derived dataset has original file paths and hashes. The catalog explicitly sets `u_net_training_compatible: false` and identifies the retrospective role.

**Rainfall:** `chirps_nepal_context/` contains 181 single-band 100×180 GeoTIFF crops. The context window is 80–89°E, 26–31°N, explicitly a rectangle rather than Nepal's boundary or a flood footprint. Crops retain source pixel values and grid transforms without resampling. Negative/nonfinite rainfall is invalid, exported as nodata `-9999`; zero remains valid rainfall. Source rasters have no nodata metadata even though they contain `-9999` values.

`chirps_point_series.csv` contains 2,534 date/location samples for the 14 requested points. Requested coordinates and the actual containing source-cell center are kept separately. Units are mm/day, with UTC day labels. CHIRPS `rnl` derives daily timing from ERA5; agreement with ERA5 is not independent rainfall validation. The [producer's daily-product documentation](https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/readme.txt) explains the timing and dependence; the [dataset catalog](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHC_CHIRPS_V3_DAILY_RNL) documents units.

**Weather:** `weather/{era5_land,era5,NASA_POWER}/<location>.csv` contains **42 tables / 124,152 rows**. There are 60,816 hourly rows for each ERA5 family and 2,520 NASA daily rows. Hourly series have 4,344 rows per location, February 1–July 31 in Nepal local time; UTC timestamps are also recorded. NASA has 180 daily UTC rows per location, February 1–July 30. Each `.metadata.json` preserves variable units, source files, returned grid coordinates where supplied, requested coordinates, missing counts and time conventions. Different provider/model grids and units remain separate.

In these archived ERA5-Land responses, precipitation, rain, snowfall, surface pressure and wind columns are entirely missing. The ERA5 supplement supplies those values, but its snow-depth column is entirely missing. Missing values remain blank; they were not changed to zero or silently filled from another model. All 42 combined series passed timestamp-continuity and missing-count checks.

**Terrain:** `terrain_point_coverage.json` records whether a complete imported tile covers each requested location. The Rasuwa point is 85.35°E, 28.15°N. Its tile, `N28_00_E085_00`, was one of the incomplete members. A separate repair was started from the [official public Copernicus GLO-30 store](https://registry.opendata.aws/copernicus-dem/), reusing the archived prefix. It is accepted only after all 38,402,839 bytes match provider ETag/MD5 `0e1522e26e1a768bdaa827aa64436a0b`; full SHA256 and raster checks are then recorded. Consult `outputs/user_dataset_intake_20261114/rasuwa_terrain_repair.json` for completion, rather than assuming the repair finished. The repair refreshes the terrain catalog and coverage automatically. The other five partial tiles remain excluded.

## Using the data in PRAVAH

These artifacts are ready for the GIS/UI owner to load as context: elevation coverage, rainfall history and weather at declared sample points. The catalog is the integration boundary; no P3/P5 application modules were modified. A separate reviewed predictive model would need appropriate event targets and a time-valid experimental design before these covariates could support any forecast claim.

For the current U-Net, continue training on paired six-band Sentinel-2 L1C imagery and water masks. Do not append DSM heights, hourly weather, SAR bands, L2A catalog statistics or landslide polygons to that training manifest. To demonstrate Nepal segmentation, obtain a real compatible Nepal optical image and retain its acquisition, availability, clouds, units and scene identity; these ZIPs do not supply that image.

All context currently has **unknown verified historical availability** and is excluded from strict cutoff replay. A filename containing “pre_aug” and old observation dates do not establish that this particular version was available then. Some CHIRPS TIFF generation tags are from November. These datasets do not contain August 26 rainfall and cannot explain that event's rainfall from their February–July records alone.

Fuzzball upload has not been performed during the API connectivity outage. The artifacts are prepared on this Mac; existing Fuzzball data remain a separate inventory.

## Reproduction and evidence

```bash
.venv/bin/python -u scripts/import_user_context.py --archives \
  /Users/sachinshrestha/Downloads/terrain.zip \
  /Users/sachinshrestha/Downloads/nepal_maps_and_events.zip \
  /Users/sachinshrestha/Downloads/nepal_pre_aug_2026.zip \
  > outputs/user_dataset_intake_20261114/import.log 2>&1

.venv/bin/ruff check scripts/import_user_context.py tests/test_user_context.py
# All checks passed!
.venv/bin/python -m pytest -q tests/test_user_context.py
# 5 passed in 0.10s

.venv/bin/python -m pytest -q tests/test_user_context.py \
  tests/test_local_training.py tests/test_local_result_export.py \
  tests/test_training_data_intake.py tests/test_training_pair_preparation.py
# 25 passed in 0.87s
```

Do not rerun the import concurrently with the terrain repair. Per-archive receipts, source hashes and excluded partial filenames are in `outputs/user_dataset_intake_20261114/*.intake.json`. `summary.json` is the current intake summary; `prepared_validation.json` records real checks on all 42 weather tables and 181 prepared rainfall rasters. Synthetic tests cover unsafe ZIP paths/symlinks, exclusion of partial files, checksum mismatches, missing-versus-zero values and Nepal/UTC day boundaries. No test fixtures enter these datasets.
