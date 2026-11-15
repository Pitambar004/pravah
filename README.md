# PRAVAH (प्रवाह) — Team 12 starter repository

**14 November implementation update:** The trained MiniUNet now runs real local predictions on verified Sen1Floods11 examples, with maps, OSM intersections and downloadable artifacts. Start and verification instructions are in [REAL_APP_HANDOFF.md](docs/REAL_APP_HANDOFF.md). This is a retrospective benchmark demonstration; Nepal optical inference and live Fuzzball deployment remain unverified. The original scaffold description below records the starting point.

**Nepal experiment preparation:** The [replay and Fuzzball handoff](docs/NEPAL_REPLAY_HANDOFF.md) adds verified source collection, immutable cutoff snapshots, separate evaluation reveal and portable API/dashboard packaging. Downstream forecasting remains blocked by missing time-eligible inputs and forecast training targets.

**Nepal dashboard:** The [monitor and demo handoff](docs/NEPAL_MONITOR_HANDOFF.md) adds **Rasuwa Operations** with functional gauge readings, water levels, timestamped satellite previews, weather/seismic records and optional browser notifications. Nepal's mapped major rivers appear in blue; infrastructure appears during source watches or separately recorded exploratory scenarios. **Glacier & Lake Watch** and the 15-day **Simulate Rasuwa flood** keep observations, assumptions and later evaluation distinct. Start with `.venv/bin/python scripts/serve_nepal_replay.py`; the local dashboard uses port 8506. Forecast accuracy and Fuzzball hosting require further measured validation.

For judges, `/simulate` has a one-click **2-minute demo**: 15-day catalog review, explicitly exploratory upstream/office alerts, mapped downstream asset review and a separate later-evidence reveal. CARTO basemaps use a locally configured credential excluded from Git and transfer bundles.

The demo now opens real Sentinel-2 before/current image dialogs during glacier
and lake review, animates regional and gauge map focuses, briefly locks the
gauge view, and reveals real pre/post imagery with downloadable comparison
results. Use **Start 2-minute judges demo**; detailed manual controls are in
their expander. Appearance highlights include cloud changes and do not measure
lake depth, glacier motion or forecast accuracy.

This is a **starter codebase** accompanying `docs/IMPLEMENTATION_RUNBOOK.md`, created during the Nov 13–15, 2026 hackathon. It provides locally testable **CPU-first** foundation components (strict temporal replay, tile manifest, small U-Net, label-aware metrics, geospatial exposure, distributed training script, tiled inference and stitching) plus branch/AI-agent instructions. It is NOT proof that a Rasuwa satellite scene or UNOSAT data has been downloaded, a trained model exists, a CUDA device has been used, a CIQ Fuzzball workflow has succeeded, or any benchmark has been measured.

**Important:** The Prithvi pretrained inference adapter, scene-specific downloads, audited Fuzzball 4.3.1 YAML, GB10 CUDA image, real dataset ingest and dashboard polish are **assigned implementation tasks**, not complete in this scaffold. Do not present this sample code's unit-test fixtures as scientific output.

## Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,ml]'       # may require platform-specific GDAL / torch resolution
pytest -q
python scripts/make_tile_manifest.py --help
python scripts/train_ddp.py --help
```

On macOS / Linux, install system GDAL/PROJ as needed, or use conda-forge. CPU-first dependencies may need platform-specific resolution.

Run the minimal local dashboard with `pip install -e '.[ui]' && streamlit run app/Home.py`; it deliberately shows **no event results** until verified artifact paths exist.

Read `docs/IMPLEMENTATION_RUNBOOK.md` **before** asking Claude/Codex to modify anything. Five role prompts are in `prompts/P1_*.md` through `prompts/P5_*.md`. Copy `AGENTS.md` and `CLAUDE.md` as-is into your shared GitHub repo.

**Source availability, licenses, timestamp leakage, multi-node allocation and benchmark integrity must be verified before judge-facing use.**

## Actual CLI commands available in this starter

```bash
python scripts/check_s2_pair.py --image /path/to/13band_S2Hand.tif --label /path/to/LabelHand.tif
python scripts/train_ddp.py --pairs data/manifests/s2_pairs.json --output-dir outputs/train_cpu --device cpu --epochs 1 --max-steps 2
python scripts/make_tile_manifest.py --raster /path/to/real_S2Hand.tif --run-id baseline01 --out data/manifests/tiles.json --tile-size 256 --overlap 32
python scripts/infer_tiles.py --manifest data/manifests/tiles.json --checkpoint outputs/train_cpu/checkpoint_last.pt --partition-index 0 --partition-count 1 --output-dir outputs/worker_one --device cpu
python scripts/stitch.py --manifest data/manifests/tiles.json --worker-dir outputs/worker_one --output-dir outputs/stitched
```

**These commands need real existing data and a trained checkpoint**. The dataset acquisition scripts, Prithvi adapter, historical AOI, live Fuzzball workflows and mature map UI are **implementation assignments** in the runbook, not completed deliverables in the starter.

## Dataset intake, catalog and exposure CLIs (work from verified real sources)

```bash
python scripts/discover_stac.py --collections
python scripts/discover_stac.py --collection VERIFIED_L1C_COLLECTION --aoi config/aois/VERIFIED.geojson --out data/manifests/stac_rasuwa.json
python scripts/download_assets.py --manifest data/manifests/reviewed_downloads_private.json --dest data/raw/downloads --out data/manifests/download_audit_private.json --dry-run
python scripts/exposure.py --mask outputs/stitched/observed_water.tif --buildings data/processed/buildings.gpkg --roads data/processed/roads.gpkg --osm-date VERIFIED_OSM_SNAPSHOT --out outputs/stitched/exposure_summary.json --polygons-out outputs/stitched/model_water.geojson
```

The `download_assets.py` manifest contains real provider URLs; do not commit signed URLs or secrets. These commands are templates that will fail until source-verified paths exist. `scripts/discover_stac.py` requires optional `.[catalog]`

## Prediction vs Reality, live maps, alerts and GB10 training

```bash
python scripts/validate_prediction.py --prediction outputs/<pred_run> --reference outputs/<post_run> --baseline outputs/<pre_run> --target new_water --event-time <UTC>
PRAVAH_TRAINING_DIR=/data/outputs/training streamlit run app/Home.py
```

- `docs/PREDICTION_CONTRACT.md` — the files a forecast model must write so its prediction can be scored against post-event Sentinel-2 water, with cutoff/leakage checks and a persistence baseline.
- `docs/TRAINING_GB10.md` — 1-node vs 2-node GB10 training templates (`workflows/fuzzball/05_train_*.yaml.tmpl`, unverified on v4.3.1) and how the dashboard compares them.
- `docs/WORLD_TRAINING_PIPELINE.md` — `make world-pipeline`: download WorldFloods v2 + Sen1Floods11 to Fuzzball, prepare split-protected manifests, then benchmark 1-node vs 2-node GB10 training (unverified on v4.3.1).
- `docs/USING_YOUR_MODEL.md` — running a trained checkpoint on a Sentinel-2 scene and loading the run in the dashboard.
- `docs/DATA_LAYERS.md` — where each dashboard map layer comes from and how to regenerate it.
- Sidebar pages: **Rasuwa operations** (Rasuwa Operations, Simulate Rasuwa flood, Glacier & Lake Watch) and **Model evidence** (Prediction vs Reality, Maps, Live Prediction, Alerts (simulated, never sent), Training). Other pages stay reachable by URL.
- The dashboard never shows synthetic fixtures by default. The "Show synthetic fixtures" toggle exists only when `PRAVAH_ENABLE_TEST_FIXTURES=1` (developer tests), and runs marked `is_synthetic` are refused.
