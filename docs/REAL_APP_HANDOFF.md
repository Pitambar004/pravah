# Real-data application — 14 November 2026 CT

The working presentation path uses the completed PRAVAH MiniUNet and real Sen1Floods11 official-test images from India, Spain and the USA. Normal application launches have **no synthetic-data toggle or fixture fallback**. Missing inputs remain missing. The Nepal page uses the imported environmental records separately; it does not turn those records into water predictions.

## Open and run

On this Mac the running dashboard is **http://127.0.0.1:8502** and its API is **http://127.0.0.1:8000**. Port 8501 was already in use, so that existing process was preserved.

To restart this application later:

```bash
cd /Users/sachinshrestha/Projects/pravah
.venv/bin/python scripts/serve_real_app.py --ui-port 8502
```

Keep the command running. It starts both services on localhost, reports their URLs and stops its own child processes on Ctrl-C. Logs are `outputs/live_app/api.log` and `dashboard.log`. Running this locally executes predictions on the CPU; the model was trained using the Mac's MPS GPU. Device labels come from the actual process, never a fabricated cluster record.

Model inference and the prepared scene/OSM/environmental data are local. The interactive map currently loads Leaflet, fonts and basemap tiles from external providers, so keep internet access for the complete browser presentation.

Presentation flow:

1. Open **Run Inference**, select a real scene and click **Run on CPU**. Each click queues a new model forward pass; it is not a recorded animation. The configured checkpoint must match the completed run's SHA256, and imagery must match its staged source hash.
2. Click **Open run** after success. The map opens that result, showing the actual Sentinel-2 RGB preview, image footprint, model-derived water, OSM waterways and available exposure geometries. Empty-but-evaluated layers remain distinct from missing inputs. No hypothetical terrain/inundation scenarios are offered.
3. Inspect the reference/prediction/error comparison. The reference label is only used after inference for scoring; it is never a model input. Image-specific scores are not aggregate test performance.
4. Open **Infrastructure Exposure**. Counts and lengths are calculated from the real predicted mask and downloaded OSM geometries in the scene's local UTM projection. India uses EPSG:32646, Spain EPSG:32630, USA EPSG:32615. The current OSM extracts are explicitly later context over historical imagery, not historical flood damage or event-time exposure estimates.
5. Download GeoJSON and run rasters/metadata from the map page. Provenance, source hashes, model hash, exact dates available from the source and actual runtime accompany the outputs.
6. Open **Nepal Data** for 42 weather tables, 181 CHIRPS rainfall days and 21 complete terrain tiles, with source units and missing values preserved. Rainfall covers February–July, not the August 26 event. DSM surface height is not water depth or a risk score.
7. **Historical Replay** handles the 450 imported satellite catalog entries. They are labelled metadata-only, their images are not downloaded, and unverified availability excludes them from strict cutoff inputs. The application no longer interprets an unverified STAC `created`/`published` field as verified historical availability.
8. **Compute & Performance** shows measured prediction times and the completed training evaluation. It does not invent a one-node/two-node speedup.

## Real artifacts and preparation

- Checkpoint: `outputs/local_demo/unet_mps_20261115T021950Z/checkpoint_best.pt`, SHA256 `9ad57b8ac40c52e635591bf22407fb17747a7000c718726c01147b02385b4a67`.
- Completed training: 251 train / 86 validation / 88 test images, 15 separate Bolivia holdouts, 20 epochs. Validation selected epoch 13. Official-test F1 0.9004 / IoU 0.8188; Bolivia F1 0.8766 / IoU 0.7803. See `LOCAL_DEMO_HANDOFF.md` for full scope.
- Staged scenes: `data/processed/live_app/scenes/`; source identities and original-image hashes in `scene_catalog.json`. Six-band pixel values and georeferencing are preserved; imagery-only validity is written as a mask. RGB stretching affects display only.
- OSM: `data/raw/app_osm/<scene_id>/`. Each extract contains the raw response, exact bounded query, provider snapshot time, file hashes, categories and ODbL attribution. Conversion preserves relation-based building polygons and ways. Source: [OpenStreetMap Overpass QL](https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL), [OSM copyright](https://www.openstreetmap.org/copyright).
- Runtime configuration: `outputs/live_app/config.json`; every completed prediction is under `outputs/live_app/runs/<run_id>/`. All raw data, generated files, model weights and screenshots remain ignored by Git.

For a checkout with the previously downloaded source data and completed checkpoint available:

```bash
python -m pip install -e '.[live,dev]'
python scripts/prepare_real_app.py
python scripts/download_app_osm.py
python scripts/serve_real_app.py --ui-port 8502
```

Preparation uses the same fixed test examples already selected for the completed run. OSM downloads reuse cached responses on subsequent calls. A source-only Git clone does not include the ignored model/datasets; the scripts refuse to invent replacements. Future input scenes need an explicit compatible optical source manifest and checksum.

## Verification and boundaries

`outputs/live_app/http_verification.json` records successful real predictions for all three scenes, projection/class/grid checks, calculated exposure and downloaded-artifact hash checks. `real_ui_verification.json` records all eight routes exercised against actual data. `browser_verification.json` records clicking the real inference button, following the job to its exact map, exporting the water GeoJSON and viewing actual exposure and Nepal terrain. Browser screenshots are alongside these reports.

Final verification commands and recorded output:

```bash
.venv/bin/ruff check app/ tests/ src/ scripts/prepare_real_app.py scripts/download_app_osm.py scripts/serve_real_app.py scripts/infer_prithvi_local.py
# All checks passed!
.venv/bin/python -m pytest -q
# 118 passed, 1 skipped, 3 warnings in 4.23s
.venv/bin/python outputs/live_app/verify_browser.py
# status: passed; actual prediction India_1018327_20261115T034319Z
# exact-run navigation and export passed; browser_errors: []
```

Full output is retained in `lint_final.log`, `tests_final.log` and `browser_verification.log` under `outputs/live_app`. The skipped test requires the optional STAC catalog dependency. Synthetic tests stay separate from the real HTTP/browser verification. The last browser run measured 0.2184 seconds of model compute and 0.6687 seconds for input checks, inference, exports, exposure and reference comparison; queue wait and model startup are excluded. These are one CPU run's timings, not GPU or scaling measurements.

The separate published Prithvi reference also passed strict checkpoint loading and actual inference on its three upstream examples. Its results and rerun command are documented in `LOCAL_DEMO_HANDOFF.md`; this does not change the model used by the live U-Net service.

Core model architecture, inference engine, API queue and GIS algorithms were reused through a new integration adapter. This is P5 integration work on `feat/real-data-app`; P2/P3/P5 review is required before merge. No merge into main is part of this handoff.

The currently working pipeline is the agreed first priority. **Not yet verified:** live Fuzzball execution from this network, multi-node benchmarks, Nepal optical model inference, and pre/post Nepal change analysis. The API endpoint timed out in a new connection check (`outputs/live_app/fuzzball_connection.json`). Nepal imagery/UNOSAT inputs are not available in this local workspace. The supplied archives contain no compatible Nepal L1C imagery. Flood forecasts, depth, casualties and damage are not trained or validated features and are not generated.

Tomorrow's real GPU batch remains available through [FUZZBALL_MORNING.md](FUZZBALL_MORNING.md). Its server validation and execution must succeed on Team 12 v4.3.1 before claiming GPU deployment.
