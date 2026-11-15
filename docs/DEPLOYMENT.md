# PRAVAH deployment — inference API, dashboard, Fuzzball

**Audience:** the Team 12 member operating Fuzzball on Sunday 15 Nov (P4) and the integrator (P5).
**Status (14 Nov 2026):** code, tests and local end-to-end run are done on a Mac (CPU). Nothing below has
run on Team 12 Fuzzball v4.3.1 or a GB10 yet. Workflow files are **templates** written against the
public Fuzzball 4.4 docs; each gate in §4 must pass on 4.3.1 before any claim is made.

## 1. What runs where

```text
teammate browser (venue Wi-Fi)
   │ HTTPS → Fuzzball endpoint "dashboard"  (scope: group by default)
   ▼
dashboard service  (CPU, Streamlit, image pravah-dashboard)
   │ http://api.service:8000   (in-workflow service hostname; never leaves the cluster)
   ▼
api service  (1× nvidia.com/gpu, FastAPI + MiniUNet on CUDA, image pravah-api, PRAVAH_REQUIRE_CUDA=1)
   │ reads  /data/raw/sentinel2/*.tif, /data/models/checkpoints/checkpoint_last.pt
   │ writes /data/outputs/predictions/<run_id>/
   ▼
Pravah-Dataset volume (provisioner team12) mounted at /data in both services
   ▲
dashboard reads the same run folders → map, exposure, compute pages
```

All model computation happens in the `api` service. The dashboard only renders.

## 2. Components added

| Path | Purpose |
|---|---|
| `src/pravah/inference/engine.py` | Loads a `train_ddp.py` checkpoint **strictly** (architecture, band order, state-dict keys, SHA-256); tiles a whole S2 raster, batches tiles to the device, averages overlaps, writes `observed_water.tif` (0/1/255), `invalid_mask.tif`, `water_probability.tif`, `observed_metadata.json`, `observed_water.geojson` (EPSG:4326). Timings are measured (`perf_counter`, `cuda.synchronize`). |
| `src/pravah/inference/__main__.py` | CLI: `python -m pravah.inference --scene … --checkpoint … --output-dir … --require-cuda` |
| `src/pravah/api/` | FastAPI app (`create_app`), env settings, single-GPU job queue |
| `app/sections/inference.py`, `app/api_client.py` | Dashboard **Run Inference** page |
| `infra/docker/Dockerfile.api`, `Dockerfile.dashboard` | Images (API base image must be supplied — see §3) |
| `workflows/fuzzball/00_gpu_smoke.yaml.tmpl` | Gate C: CUDA inside the API image |
| `workflows/fuzzball/06_inference_batch.yaml.tmpl` | One-shot GPU inference job (no web) |
| `workflows/fuzzball/07_serve.yaml.tmpl` | API + dashboard services with endpoints |
| `scripts/fuzzball/render_workflow.py` | Fills `{{TOKENS}}`, refuses unfilled/invalid values, structural checks → `build/fuzzball/` |
| `scripts/fuzzball/gpu_smoke.py`, `start_dashboard.sh` | Smoke test record; Streamlit start with endpoint base path |

### API

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | liveness (readiness probe) |
| GET | `/api/v1/status` | device, `cuda_verified`, GPU name, torch/CUDA versions, host, checkpoint SHA-256, storage writability, queue |
| GET | `/api/v1/scenes` | GeoTIFFs under `PRAVAH_SCENES_DIR`, with bands/CRS and `model_compatible` |
| POST | `/api/v1/predictions` | `{"scene_id", "threshold", "run_id"?}` → 202 job; 503 without checkpoint; 422 for SAR/non-S2; 409 duplicate run; 429 queue full |
| GET | `/api/v1/predictions[/{job_id}]` | status, tile progress, measured timings |
| GET | `/api/v1/runs[/{run_id}]` | finished runs / metadata |
| GET | `/api/v1/runs/{run_id}/files/{name}` | whitelisted artifacts only |
| POST | `/api/v1/model/reload` | after staging a new checkpoint |

Writes require header `X-Pravah-Token` when `PRAVAH_API_TOKEN` is set. Path traversal in `scene_id`/`run_id` is rejected.

### Environment

| Variable | Default | Used by |
|---|---|---|
| `PRAVAH_DATA_ROOT` | `/data` | API |
| `PRAVAH_SCENES_DIR` | `$ROOT/raw/sentinel2` | API |
| `PRAVAH_OUTPUTS_DIR` | `$ROOT/outputs/predictions` (API) / `outputs` (dashboard) | both |
| `PRAVAH_CHECKPOINT` | `$ROOT/models/checkpoints/checkpoint_last.pt` | API |
| `PRAVAH_DEVICE` / `PRAVAH_REQUIRE_CUDA` | `auto` / off | API — set `cuda` / `1` on Fuzzball |
| `PRAVAH_BATCH_SIZE` | `8` | API |
| `PRAVAH_API_TOKEN` | unset | API + dashboard |
| `PRAVAH_API_URL` | `http://localhost:8000` | dashboard |
| `PRAVAH_DASHBOARD_PASSCODE` | unset | dashboard (recommended if scope is `public`) |

## 3. Images

```bash
# API: BASE_IMAGE has no default on purpose. Use a linux/arm64 CUDA PyTorch image that supports GB10,
# confirmed with `docker buildx imagetools inspect <image:tag>` and then Gate C. Record the digest.
docker buildx build --platform linux/arm64 -f infra/docker/Dockerfile.api \
  --build-arg BASE_IMAGE=<verified-arm64-cuda-pytorch-image> -t <registry>/pravah-api:<tag> --push .
docker buildx build --platform linux/arm64,linux/amd64 -f infra/docker/Dockerfile.dashboard \
  -t <registry>/pravah-dashboard:<tag> --push .
```

The API Dockerfile fails the build if the base image's torch has no CUDA build. Ask CIQ which registries
Fuzzball may pull from; a private registry needs a Fuzzball secret (`--image-secret` in the renderer).
**Not yet built** — no images were pushed while preparing this.

## 4. Sunday gates (in order; stop at the first failure)

1. **Network/auth.** CIQ-approved network; `fuzzball` CLI context works. (Runbook P4-04 Gate A.)
2. **Storage.** Run existing `01_inspect.yaml`. Mounts use `/data: {volume: dataset}` — the form proven by
   Team 12's finished 4.3.1 workflows on 14 Nov (the 4.4 docs' `/data: dataset` shorthand is not used).
3. **Stage inputs** on `Pravah-Dataset`: the trained checkpoint at
   `/data/models/checkpoints/checkpoint_last.pt` and at least one real S2 L1C GeoTIFF under
   `/data/raw/sentinel2/` with its source manifest entry (scene ID, acquisition + availability times).
4. **GPU smoke (Gate C).**
   ```bash
   python scripts/fuzzball/render_workflow.py workflows/fuzzball/00_gpu_smoke.yaml.tmpl \
     --set API_IMAGE=docker://<registry>/pravah-api:<tag>
   fuzzball workflow start build/fuzzball/00_gpu_smoke.yaml   # confirm exact 4.3.1 command
   ```
   Pass = exit 0, JSON with `cuda_verified: true`, `machine: aarch64`, real device name, saved under
   `/data/outputs/evaluation/`. Keep the log.
5. **Batch inference (Gate D).** Render `06_inference_batch.yaml.tmpl` with `RUN_ID`, `SCENE_ID`; check
   `observed_metadata.json` shows `runtime.cuda_verified: true` and the expected `checkpoint_sha256`.
6. **Serve.**
   ```bash
   python scripts/fuzzball/render_workflow.py workflows/fuzzball/07_serve.yaml.tmpl \
     --set API_IMAGE=docker://<registry>/pravah-api:<tag> \
     --set DASHBOARD_IMAGE=docker://<registry>/pravah-dashboard:<tag> \
     --set ENDPOINT_SCOPE=group
   fuzzball workflow start build/fuzzball/07_serve.yaml
   fuzzball endpoint list          # dashboard URL
   ```
   Pass = the dashboard URL opens on a teammate's phone/laptop on venue Wi-Fi, Run Inference shows
   `CUDA verified` with the GB10 name, a submitted scene finishes, and the map shows the run.
7. **Things to confirm on 4.3.1** (not knowable from docs alone):
   - endpoint `scope: group` lets logged-in teammates open the URL in a browser (else use `public` +
     `--env dashboard:PRAVAH_DASHBOARD_PASSCODE=…`);
   - path endpoints forward the `/endpoints/<id>/` prefix unchanged and `FB_ENDPOINT_URL_DASHBOARD` is
     injected (`start_dashboard.sh` relies on both; override with `PRAVAH_BASE_URL_PATH`);
   - services resolve each other as `api.service`;
   - `persist`, `readiness-probe` and `depends-on` keys are accepted;
   - the endpoint domain is reachable from venue Wi-Fi (the earlier blueprint recorded a UI timeout).

Secrets (`PRAVAH_API_TOKEN`, passcode) must be passed at render time or via Fuzzball secrets, never committed.
`build/` is gitignored.

## 5. Local run (verified 14 Nov 2026, macOS arm64, CPU)

```bash
.venv/bin/pip install -e '.[ui,api,ml,dev]'
make api          # uvicorn on 127.0.0.1:8000, data root ./data
make serve        # dashboard on :8501 using the API and ./data/outputs/predictions
```

Without a checkpoint the API starts, `/api/v1/status` reports `model_loaded: false` with the reason, and
predictions return 503 — it never serves random weights.

Verification done for this change (plumbing only; random-noise raster and an **untrained** checkpoint
created in a temporary directory, not committed, not results):
- `pytest -q` → 74 passed; `ruff check app/ tests/ src/ scripts/fuzzball/render_workflow.py
  scripts/fuzzball/gpu_smoke.py` → "All checks passed!".
- CLI: `python -m pravah.inference --scene … --checkpoint … --output-dir … --device cpu` → wrote a run folder.
- Browser (Chrome via Playwright): Run Inference page → Run → job succeeded → Open run → map renders the
  run's polygons at the raster's UTM 45N location; no page errors.
- Measured on the Mac CPU for a 1024×1024 13-band raster, 25 tiles: model compute 1.04 s, end-to-end
  1.39 s. These are laptop CPU numbers, not GB10 numbers, and are not benchmark results.

## 6. Laptop dashboard reading runs from the Fuzzball API

The dashboard can show runs produced on Fuzzball without mounting the volume: it lists completed runs from
`GET /api/v1/runs` and fetches `observed_metadata.json` + `observed_water.geojson` over HTTP. Rasters stay on
the volume. The sidebar labels such runs `Source: API`.

```bash
PRAVAH_API_URL=https://endpoints.<domain>/endpoints/<api-endpoint-id> \
PRAVAH_FB_ENDPOINT_TOKEN=$(fuzzball endpoint generate-token <api-endpoint-id> --expiration 8h) \
PRAVAH_RUN_SOURCE=api \
  .venv/bin/python -m streamlit run app/Home.py --server.headless true
```

- `PRAVAH_RUN_SOURCE`: `auto` (default: local runs if any hold artifacts, else API runs), `local`, `api`.
- `PRAVAH_FB_ENDPOINT_TOKEN` is sent as `FB-Authorization: Bearer …` for non-public endpoints. Never commit it.
- The API accepts requests with or without the `/endpoints/<id>` prefix (`PRAVAH_API_ROOT_PATH`, else derived
  from Fuzzball's `FB_ENDPOINT_URL_API`), so it works whether or not the proxy strips the prefix.

Verified locally (14 Nov 2026) with the API served under a mock `/endpoints/endpoint-local` prefix and an
empty local outputs folder: the map loaded the API's run and polygons; `pytest -q` → 78 passed.

## 7. Scientific limits that still apply

MiniUNet is Team 12's small baseline, not Prithvi; its quality is unknown until evaluated on held-out
Sen1Floods11 splits. Output is model-inferred optical water (0/1, 255 invalid), not flood forecast, depth,
debris flow or damage. Replay eligibility rules are unchanged: a scene processed here is post-event
evidence unless its acquisition **and** verified availability precede the chosen cutoff.
