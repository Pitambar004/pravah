# Nepal experimental replay and Fuzzball preparation — 14 November 2026 CT

The VS Code thread `01a1dc31-53db-7442-947d-bc6a6b5f259f` ended at its usage limit after authorizing this work. This feature branch continues its P5 integration task. It preserves the working real-data U-Net application and adds an audited Nepal experiment, portable serving package and fresh water-training package. **A trained downstream flood forecast is still missing.** Preparing these artifacts does not establish Fuzzball deployment or forecast accuracy.

## Implemented

Open the existing dashboard's **Nepal Experiment** page at `http://127.0.0.1:8502/experiment`. It audits source eligibility, freezes an immutable snapshot and reveals evaluation references separately. Its default cutoff is **26 August 2026, 08:30 Nepal time / 02:45 UTC**. This is an analyst-selected cutoff preceding the reviewed source-failure marker, not an independently justified warning threshold.

Inputs require timezone-aware acquisition and availability timestamps, explicit verified availability evidence, local file checksums and an input role. Unknown availability, later observations, catalogs, context-only files, references and marked synthetic fixtures are excluded. Complete time-series files cannot qualify if their coverage extends beyond the cutoff. Snapshots record their actual creation time in November and disclose retrospective case selection; they do not claim to have been created before the flood. Duplicate IDs, changed input bytes, altered snapshots and path traversal are rejected.

Forecast extent, arrival time, depth, river width, forecast exposure and accuracy remain null when no forecast exists. Later observations cannot create an absent prediction. Existing optical water segmentation and exposure remain available through the existing API; their class semantics and benchmark scope are unchanged.

The new API factory is `scripts.serve_nepal_replay:api`. It wraps the existing inference engine and adds:

- `GET /api/v1/replay/readiness?cutoff=...`
- `POST /api/v1/replay/snapshots` with `run_id` and `cutoff`
- `GET /api/v1/replay/snapshots/{run_id}`
- `POST /api/v1/replay/snapshots/{run_id}/reveal`

Mutation routes use the configured `X-Pravah-Token` when an API token is set. Hosting uses the existing Fuzzball group-scoped endpoint template. Credentials are supplied at runtime, not bundled.

## Actual source preparation

`outputs/nepal_replay/prepared/manifest.json` contains **51 staged source records**: the 42 imported weather tables, the complete Rasuwa Copernicus DSM, the imported satellite catalog and seven newly acquired source responses/catalogs. `acquisition_attempts.json` records each attempt and its actual hashes. At the default cutoff there are **zero eligible inputs**. Archived acquisition dates and filenames do not establish historical availability.

New acquisitions were:

- The [USGS event `us7000tbwb`](https://earthquake.usgs.gov/earthquakes/eventpage/us7000tbwb) current detail JSON. It identifies a reviewed **landslide**, origin time **08:37:10 NPT**, rather than an independently established tectonic precursor. The current catalog revision is later evidence; no first-publication receipt was recovered.
- The [official hydrology technical report](https://hydrology.gov.np/cm/files/Bhotekoshi-Trishuli-Narayani_Flood_Report_CHWRR_1%20_1787809133090.pdf), retained as retrospective evaluation context. It places Syabrubesi telemetry loss at **08:50**, following a final 08:40 reading. That outage occurs after the reviewed source-failure marker. Reported station stages are not interchangeable with water depth or flow discharge.
- The [NDRRMA 8 July 2025 situation report](https://ndrrma.gov.np/mediafiles/publications/Rasuwagadhi_Flood_Sitrep1_08072025_.pdf), as earlier-event context. A PDF narrative is not a labeled neural-routing training set.
- ERA5 daily precipitation and mean temperature for **1–28 August 2026** at Rasuwa and the supplied upstream point, from [Open-Meteo's historical API](https://open-meteo.com/en/docs/historical-weather-api). These are retrospective reanalysis responses, not event-time forecasts or river-gauge measurements. Returned units and missing values are preserved.
- A historical rendered view from [DHM station 4913](https://dhm.gov.np/hydrology/hms-Single/4913). The public request succeeded but its embedded sample array and table were **empty**. This is not a downloaded gauge time series. Public-session CSRF values/cookies were not bundled.
- **13 actual Sentinel-2 L1C catalog items** from the [Copernicus collection](https://stac.dataspace.copernicus.eu/v1/collections/sentinel-2-l1c), covering the bounded August 10–September 1 search. The listing is complete for that query. Its rectangle is a search extent, not a flood footprint. Item IDs, footprints, dates, cloud fields and asset keys are preserved; imagery was not downloaded and historical availability was not verified. Signed asset URLs are omitted from the transfer catalog.

Prepared source bytes are verified for the documented format/integrity checks. This does not independently establish historical availability, scientific accuracy or redistribution licenses. Source archives, reports, generated data, snapshots and checkpoints remain ignored by Git. Keep source attribution with any permitted team transfer; confirm source-specific redistribution terms before public data publication.

## Reproduce the preparation

These commands were executed. Use a new `--root` or `--run-id` when repeating a completed preparation or snapshot, because finished artifacts are preserved.

```bash
.venv/bin/python scripts/prepare_nepal_replay.py prepare --download
.venv/bin/python scripts/prepare_nepal_replay.py audit
.venv/bin/python scripts/prepare_nepal_replay.py freeze
.venv/bin/python scripts/prepare_nepal_replay.py reveal
.venv/bin/python scripts/package_nepal_replay.py --with-training
```

`outputs/nepal_replay/packages/latest_prepared.json` identifies the latest approximately 48 MB serving/evidence archive, archive checksum, persistent destination, object prefix, staging workflow, hosting template and training receipt. Its packaged source files have individual hashes. Machine-specific paths in copied application catalogs/OSM receipts are rewritten to relative paths; the original JSON hashes are retained. Raster pixels, georeferencing, labels and model weights are copied unchanged.

`scripts/run_nepal_bundle.py --bundle PATH --service verify` verifies all listed members. The same entry point configures either `--service api` or `--service dashboard` after unpacking. GPU API deployment uses `--device cuda`, which requires CUDA and cannot silently switch to CPU. Local portability verification uses `--device cpu` and is labeled accordingly. Both services share the confirmed dataset volume after staging; the new staging and hosting workflows remain **unverified on Team 12 v4.3.1**.

## Tomorrow: training, image and hosting

Start with the read-only connection check:

```bash
.venv/bin/python scripts/fuzzball_morning.py doctor
```

The existing [Fuzzball morning launcher](FUZZBALL_MORNING.md) prepares and launches the already tested optical U-Net training path. A fresh training bundle/receipt is included in the replay package receipt. This trains **water segmentation** on the fixed Sen1Floods11 splits, not a physics-informed downstream forecast and not Prithvi fine-tuning. Keep test/Bolivia labels out of model selection. Collect actual CUDA, job and environment evidence before claiming GPU execution.

The registry image name is still unset; a question was sent to the user. A Docker client is present but the local Docker daemon was not running, so the new image was **not built**. The base-image manifest was actually inspected and contains `linux/arm64/v8`. Its inspected ARM64 digest is:

```text
nvcr.io/nvidia/pytorch:25.10-py3@sha256:5c8302e4628ac326c412368675cc462b8aee2f96326a3bf817304a83816f179a
```

Build the reviewable image on a working builder using the team-approved image destination:

```bash
docker buildx build --platform linux/arm64 \
  -f scripts/nepal_replay/Dockerfile \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3@sha256:5c8302e4628ac326c412368675cc462b8aee2f96326a3bf817304a83816f179a \
  -t "$PRAVAH_SERVICE_IMAGE" --push .
```

`requirements-service.txt` pins the actual locally installed library versions; Linux/ARM64 dependency resolution and CUDA execution remain to be tested. PyTorch is supplied by the base image. The image contains the launcher and dependencies; the private volume supplies code, imagery, evidence and weights.

On the authorized event connection, validate the generated `stage.yaml` on the real server before uploading its exact archive to the receipt's object prefix and starting it. Record the workflow ID and confirm completion/volume readability before starting services. Render the receipt's `serve.yaml.tmpl` with `scripts/fuzzball/render_workflow.py`, supplying `SERVICE_IMAGE=docker://<actual built image>` and `ENDPOINT_SCOPE=group`. Validate that rendered workflow against **v4.3.1** before submission. The existing morning launcher's `workflow validate`, `object put --ttl 3d` and `workflow start` sequence documents the observed CLI argument forms. Do not infer success from local YAML/shell checks.

After a real water-training run, choosing its checkpoint for serving requires its downloaded/volume-resident evaluation and checksum, updated scene/config identity and a fresh package. Never replace a completed checkpoint in place or describe a training smoke run as a converged forecast model.

## Tested / blocked

Final source checks:

```bash
.venv/bin/python -m pytest -q
# 143 passed, 3 warnings
.venv/bin/ruff check app/ src/ tests/ scripts/prepare_nepal_replay.py \
  scripts/package_nepal_replay.py scripts/run_nepal_bundle.py scripts/serve_nepal_replay.py
# All checks passed!
.venv/bin/python -u outputs/nepal_replay/verify_portable.py
# status: passed; real CPU forward pass, replay HTTP checks, browser freeze/reveal/download
```

Full outputs and screenshots are under `outputs/nepal_replay/`. Automated unit fixtures are synthetic and remain in temporary test folders. Separate real verification unpacked the archive in a different directory, checked its hashes, loaded the completed checkpoint, performed fresh India benchmark inference with real OSM intersections, verified CRS/nodata/class codes, rejected a duplicate snapshot, and clicked/exported the actual replay in Chrome. These are CPU portability results, not Nepal forecasts, GPU measurements or deployment proof.

Still blocked: verified historical-availability receipts; raw timestamped gauge/outage series; downloaded matched Nepal optical images; independent Nepal water targets; a calibrated downstream model and suitable event-held-out forecast training data; registry/builder configuration; live Fuzzball service/CUDA validation; and multi-node benchmarks. UNOSAT mud/rockflow interpretation cannot replace binary-water labels or a missing forecast.

P2/P3/P4 peer review is requested before merge for scientific contracts, model reuse, source handling and Fuzzball compatibility. Work is on `feat/p5-nepal-replay`; no merge is authorized by this handoff.

## Single continuation prompt

> Resume `docs/NEPAL_REPLAY_HANDOFF.md` and `outputs/nepal_replay/packages/latest_prepared.json`. Fuzzball is available on the authorized event network. Use our approved registry/image destination, verify Team 12 v4.3.1 and CUDA, then stage the verified bundle, run the existing real water-training path, and host the API/dashboard through group endpoints. Preserve receipts, checkpoints and actual job IDs. Continue the Nepal experiment only with verified cutoff-eligible sources and an appropriate forecast model; report missing evidence instead of generating simulated predictions or invented accuracy.
