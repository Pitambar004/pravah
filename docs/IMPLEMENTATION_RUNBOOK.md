# PRAVAH (प्रवाह) — Team 12 Complete Five-Person Implementation Runbook

**See upstream. Protect downstream.**  
**Hackathon:** CIQ Supercomputing Local AI Hackathon, Chicago, 13–15 November 2026  
**Primary track:** Scientific AI  
**Technical demonstration:** Inference at Scale; optional Multi-Node Training  
**Study:** 26 August 2026 Rasuwa–Nuwakot / Bhote Koshi–Trishuli flood and debris-flow cascade, Nepal  
**Prepared:** Saturday 14 November 2026, Central Time  
**Planning assumption:** Team 12 cannot use its two Lenovo ThinkStation PGX / NVIDIA GB10 resources or Fuzzball until **Sunday 15 November at 09:00 CT**. **This is team-supplied, not guaranteed by CIQ.** The official event ends **Sunday 15 November at 16:00 CT**, leaving at most seven hours for cluster integration, measurement, demo, and submission after access returns.

> **Status:** A researched implementation and collaboration plan with a **companion locally tested starter code repository**, not a complete scientific implementation. The accompanying starter includes Python scripts for baseline training, 2-process CPU DDP, partitioned inference, stitching, provenance and exposure; some commands in the runbook remain *proposed* for contributors and are not yet present or validated. The starter has passed local synthetic-data smoke tests, **not** real-scene/model/GB10 validation. No genuine satellite scenes, model checkpoints or UNOSAT archives have been downloaded in preparing the deliverables. As of the earlier team blueprint, Team 12's Fuzzball web login previously worked, but network access, CLI context, actual Team 12 node identities, persistent volume and successful GPU workflows were unverified.
>
> **Do not confuse this runbook with the canonical preceding design:** `PRAVAH_Team12_Complete_Project_Blueprint.md` is the historical science/provenance reference; this document is the implementation and coordination plan. If they conflict, prefer verified data/model requirements and source evidence, update both documents via pull request, and record the decision.

---

## 0. START HERE — the ten decisions that should not change casually

1. **Exactly one central claim:** PRAVAH produces *georeferenced satellite-derived water/flood evidence*, overlays potentially exposed infrastructure, and measures performance of distributed scientific AI. It **does not forecast this historical disaster** without a separately demonstrated, time-eligible predictive model.
2. **Two replay modes:** `Evidence available at historical cutoff` and `Retrospective analysis / later reveal`. No post-cutoff imagery, UNOSAT labels, revised OSM snapshots or later reports are allowed in the first mode.
3. **Primary model candidate:** IBM/NASA **Prithvi EO 2.0 300M flood fine-tune**, using **Sentinel-2 Level-1C** six-band optical inputs. Do **not** feed it Sentinel-1 VV/VH or blindly substitute Sentinel-2 L2A. Run the upstream sample before Rasuwa imagery.
4. **Primary independent training project:** a **small, transparent 6-channel Sentinel-2 U-Net** trained on the official **Sen1Floods11 hand-labeled subset**, first locally/CPU for a smoke test, then with PyTorch `DistributedDataParallel` across the two GB10 systems if networking permits. This is not the same checkpoint as Prithvi and its results must be labeled separately.
5. **Cluster minimum:** same real tile manifest processed on **one node and two distinct nodes**, with timing and proof of GPU use. Two Fuzzball tasks is not by itself proof of two separate GPU nodes; inspect allocation and logs.
6. **First geographic scope:** UNOSAT's **actual analysis extent** and an explicitly bounded, smaller `aoi_core.geojson`, not an invented rectangular 'flood footprint'. The UNOSAT *affected extent* is post-event validation context, not a pre-event input.
7. **Data security:** separate GitHub accounts, separate Claude/Codex accounts, personal SSH keys/PATs, no shared credentials. Commit only scripts, schemas, manifests, small redistributable examples (if licensed), tests, screenshots of nonsecret logs, and reports—not multi-GB imagery or passwords.
8. **Five clear owners:** P1 data/evidence, P2 AI/model/training, P3 GIS/validation, P4 GPU/Fuzzball/benchmark, P5 integration/UI/demo. Everyone owns a branch and pull requests; no AI agent writes directly to `main`.
9. **Stop rules:** prioritize a working, defensible end-to-end pipeline over HAND terrain scenarios, LLM reports, new model architectures or polished animations. Never use fake counts, fabricated speedups or dummy GPU logs in judge-facing demo.
10. **Availability is not guaranteed:** Sunday 09:00 is a planning assumption; check it with CIQ. Fuzzball 4.4 online examples are **not automatically compatible with the event's observed 4.3.1 deployment**. Export/obtain a verified v4.3.1 example and adjust it.

### Official rule and timing reminders

- Event: 13 November 17:00 CT through 15 November 16:00 CT; maximum five competitors; *all project code must be written during the hackathon starting at kickoff*. Source: [official CIQ event page](https://ciq.com/events/local-ai-hackathon-2026).
- The official tracks are Multi-Node Training, Inference at Scale, Scientific AI and Cluster & Scheduling. The key requirement is showing what the **multi-node cluster** unlocks, not just demonstrating a flood app.
- As of 14 November, code created now is after Friday's kickoff; continue to comply with any more detailed organizer rules on pre-existing libraries, examples, and AI-generated work. Save provenance of new commits.
- **Do not assume Sunday 09:00–16:00 is entirely usable:** onboarding, image pulls, scheduling, staging and submission can consume it. Preserve the local demo and freeze it early.

---

## 1. What judges will see: defined scientific and computational outputs

### 1.1 Five-minute demo storyboard

| Demo clock | Proof shown | Source of truth | Owner |
|---|---|---|---|
| 0:00–0:40 | Rasuwa/Nuwakot map, Aug 26 event, analysis footprint; what could/could not have been known | Provenance and sourced event notes | P3/P5 |
| 0:40–1:30 | Real Sentinel input with acquisition time, available time, cloud/valid mask and exact model identity | `assets.json`, scene metadata, sample imagery | P1/P2 |
| 1:30–2:25 | Model-derived georeferenced water segmentation; toggle UNOSAT retrospective affected extent **without equating the class definitions** | `observed_water.tif`, per-run metadata, UNOSAT geometry | P2/P3 |
| 2:25–3:10 | Computed buildings/roads/bridge-tag overlap and coverage, uncertainty/caveats | `exposure_summary.json`, vector features | P3 |
| 3:10–4:15 | Actual one-node and two-node Fuzzball runs; job IDs, distinct node IDs, throughput, timings and output-equivalence test | Run logs and `benchmarks.csv` | P4/P2 |
| 4:15–4:45 | Optional distributed training: actual epochs, loss and held-out metrics; show only if completed | `training_metrics.csv`, checkpoints, data split manifest | P2/P4 |
| 4:45–5:00 | Repeatable repository, external-source attributions and limitations | GitHub, README, test summary | P5 |

### 1.2 Claims hierarchy

**Observed input:** a real source image at a stated acquisition date.  
**Model-derived evidence:** predicted optical *water* label; not guaranteed water depth, flood arrival or debris-damage class.  
**Retrospective external interpretation:** UNOSAT *mudflow and rockflow / flood-associated affected areas*, acquired and published after the event; not identical to water masks.  
**Exposure:** OSM feature intersections, not observed bridge failures, casualties, road closure or building damage.  
**Scenario (optional):** height-above-drainage threshold, not calibrated hydraulics.  
**Hardware claims:** only reproducibly logged runs on allocated Team 12 hardware.

### 1.3 Definition of done

**Minimum local end-to-end product, before GPU:** (a) one genuine source GeoTIFF opens and is georeferenced; (b) model-compatible sample runs in a local environment or a documented external-compatible CPU environment, or a clearly named rule baseline exists; (c) valid raster output maps correctly; (d) OSM overlay and computed summary; (e) Streamlit timeline and source metadata; (f) tests pass.  
**Strong hackathon submission:** minimum plus one successful Team 12 GPU run and paired same-workload one-node/two-node inference benchmark.  
**Exceptional stretch:** all of the above plus genuine 2-node synchronized DDP training, fixed-split test metrics and an independently checked scientific limitations report.

---

## 2. System architecture, data flow, and two separate experiments

```text
OFFICIAL SOURCES / CATALOGS                      MODEL TRAINING DATA
  Sentinel-2 L1C Rasuwa scenes                    Sen1Floods11 official hand labels
  Sentinel-1 SAR backup                           S2 13-band TOA and 0/1/-1 labels
  UNOSAT affected and analysis polygons                   |
  pre-cutoff OSM / optional DEM                    same train/val/test lists
              |                                           |
       P1 source audit + manifest                    P2 prepare 6-band chips
              |                                           |
  core AOI + target projected grid                     P2 small U-Net
              |                                           |
      P2 Prithvi pretrained inference          1-GPU train / 2-GPU torchrun DDP
              |                                   |
       deterministic tile manifest                  checkpoint + loss + IoU
              |
    +---------+----------+
    |                    |
  P4 GB10 #1           P4 GB10 #2
  Fuzzball task A      Fuzzball task B
    |                    |
    +--- shared volume --+
              |
        stitch / QA and provenance
              |
        water GeoTIFF + invalid mask
              |
      P3 OSM exposure + UNOSAT visual comparison
              |
       P5 Streamlit UI / replay / reporting
              |
       timestamped run and judge/demo export
```

**Experiment A (must succeed): independent-data-parallel inference.** Both nodes load the *same* compatible weights and process different non-overlapping lists of tiles, then store georeferenced predictions. There is **no cross-node gradient synchronization**, which makes this path comparatively resilient to interconnect limitations. Measure compute-only and true end-to-end throughput. More total tiles may demonstrate *throughput scaling* but **speedup** must use the identical fixed total workload.

**Experiment B (optional but fully prepared): synchronized distributed training.** Both nodes run a *single* coordinated optimization with DDP over the **same global training experiment** and each sees a different shard (`DistributedSampler`). Collective gradient exchange requires functioning NCCL and authorized inter-node connectivity. A pair of unrelated single-node trainings is *parallel hyperparameter search*, **not** multi-node DDP. Training a 300M-parameter Prithvi checkpoint end-to-end in the remaining window is not the primary plan.

**Technology choices:** Python 3.11 if supported; `rasterio`, `numpy`, `geopandas`, `shapely`, `pyproj`, `pystac-client`, `httpx`, `pandas`, `streamlit`, `pytest`, PyTorch (with verified aarch64/CUDA image on GB10), `segmentation-models-pytorch` or project-owned small U-Net, `huggingface_hub`, optionally TerraTorch. Pin actual installed versions *after testing*, in separate CPU and GPU locks. Prefer reproducibility over maximizing dependency count.

---

## 3. Verified source inventory, priority and acquisition rules

**Source status vocabulary:** `confirmed_listing` = provider documents item, **not downloaded**; `candidate` = find/check actual scene; `downloaded` = local file opens and checksum is recorded; `verified` = schema/license/date/coverage pass tests; `blocked` = recorded access failure. Do not label a listed provider asset as `downloaded` before using the file.

| Priority | Asset | What we will actually use | Verification and caveats | Official link |
|---|---|---|---|---|
| **A0** | UNOSAT Rasuwa/Nuwakot 26–27 Aug 2026 vectors | **Three different layers:** affected mud/rock flow, *analysis extent*, large map-frame AOI | Dataset dated 28 Aug 2026, **retrospective only**; distinct affected-area figures appear on summary and resource pages (about 30 vs 37 km²) and analysis 330 vs 363 km²—compute real polygon areas and cite exact product/version, **do not hardcode** | [UNESCO IHP-WINS DOI/dataset](https://ihp-wins.unesco.org/en/dataset/nepal-satellite-detected-mudflow-and-rockflow-extent-nepal) |
| **A0** | Official Sen1Floods11 **hand-labeled** S2 data and masks | Local training/validation/test; exact 13-band Sentinel-2 **L1C**, select bands for 6-channel network | Official repository says 446 manually labeled 512×512 chips and bucket `gs://sen1floods11`; repo prose has inconsistent bucket spelling—test path; retain original splits | [Official repo](https://github.com/cloudtostreet/Sen1Floods11) · [structure](https://github.com/cloudtostreet/Sen1Floods11/blob/master/docs/README.md) |
| **A0** | Prithvi flood fine-tune sample and checkpoint | First model smoke test and main pretrained scientific AI inference | 300M; six bands **B2,B3,B4,B8A,B11,B12**; model's provided inference script defaults to 0-based `[1,2,3,8,11,12]` from S2L1C 13-band sample. Check weights/config revision and scaling | [Model](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11) · [upstream inference.py](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/blob/main/inference.py) |
| **A1** | Rasuwa real Sentinel-2 L1C scenes, pre and post Aug 26 | Actual case-study image for Prithvi | UNOSAT reports a 27 Aug S2 source, but **scene ID and exact obtainable product are not yet independently audited**. Query STAC first; verify cloud, spatial overlap, licensing, ready-to-read bands, acquired and available times | [CDSE STAC docs](https://documentation.dataspace.copernicus.eu/APIs/STAC.html) · [endpoint](https://stac.dataspace.copernicus.eu/v1/) |
| **A1** | Pre-cutoff OSM Nepal extract | Buildings, roads, bridge-tagged segments | Geofabrik **2026-01-01 historical PBF** is listed; current/latest OSM is **post-event**, so not a strict replay input. Jan snapshot may miss features that existed in Aug; quantify coverage | [Geofabrik Nepal](https://download.geofabrik.de/asia/nepal.html) |
| **A1** | UN/WHO event sources | Event chronology and *later reveal* citations | Keep publication/as-of dates for each source; retrospective accounts cannot be fed into pre-event predictions | [WHO](https://www.who.int/nepal/emergencies/2026-rasuwa-flash-floods) · [UN Flash Update](https://nepal.un.org/sites/default/files/2026-09/Nepal_Rasuwa_Flood_Flash_Update_3_28_August_2026.pdf) |
| **B** | Sentinel-1 SAR/RTC | Backup optical-cloud independent analysis or contextual imagery | SAR has terrain shadow/layover; only run Dhara if two suitable pre + one post scenes, correct sigma0 dB, same geometry and normalization exist | [ASF](https://search.asf.alaska.edu/) · [HyP3 RTC guide](https://hyp3-docs.asf.alaska.edu/guides/rtc_product_guide/) · [CDSE](https://documentation.dataspace.copernicus.eu/APIs/STAC.html) |
| **B** | Dhara flood U-Net ONNX | Backup pretrained SAR model | Requires six channels: post VV/VH + pre1 VV/VH + pre2 VV/VH, **not** single radar image; known educational/non-operational model; GPU ONNX provider on ARM64 must be tested | [Dhara model card](https://huggingface.co/adityabavadekar/dhara-flood-unet) |
| **B** | Copernicus GLO-30 DSM | Hillshade/optional susceptibility scenarios | 30 m *digital surface model*, not bare-earth hydrological truth; verify tile and EPSG | [AWS registry](https://registry.opendata.aws/copernicus-dem/) |
| **B** | HydroRIVERS/HydroSHEDS | River reference for AOI and possible HAND | Hydrologically conditioned DEM / network if available and compatible; not direct observed flash-flood route | [HydroRIVERS](https://www.hydrosheds.org/products/hydrorivers) · [HydroSHEDS](https://www.hydrosheds.org/hydrosheds-core-downloads) |
| **C** | WorldPop population grid | Optional modelled resident population in intersected area | Verify *modelled reference year*, release date, units, nodata and licensing. Use actual pixel sums/fractional coverage; cannot claim persons evacuated/injured | [WorldPop portal](https://www.worldpop.org/) · [API](https://api.worldpop.org/v3/documentation) |
| **C** | JRC Global Surface Water | Permanent/historical-water context | Check source-year coverage and whether selected version existed at historical cutoff; not proof of event water | [JRC GSW](https://global-surface-water.appspot.com/download) |
| **C** | PlanetScope scene reported by UNOSAT | Source context *only if licensed and obtainable* | UNOSAT's right to analyze does not grant Team 12 permission to redistribute PlanetScope imagery. Do not download from unauthorized mirrors | [UNOSAT metadata](https://ihp-wins.unesco.org/en/dataset/nepal-satellite-detected-mudflow-and-rockflow-extent-nepal) |

### 3.1 Special UNOSAT discovery (crucial)

At the [UNESCO dataset](https://ihp-wins.unesco.org/en/dataset/nepal-satellite-detected-mudflow-and-rockflow-extent-nepal), download these **separately**, using the page's own Download links:

1. **Mudflow and Rockflow Extent** (SHP ZIP): actual affected polygon—use *only* as `retrospective_unosat_affected` and qualitative cross-class comparator.
2. **Analysis Extent** (SHP ZIP): shows where interpretation was performed; this becomes `analysis_coverage.geojson` and the default evaluation coverage mask. Outside coverage is **unknown**, never negative.
3. **Area of Interest Frame** (SHP ZIP): wide map rectangle; not an analysed or affected area; may be useful for a static overview map but **must not** be used as the evaluation denominator.

**Critical quality issue discovered during research:** The landing page's summary approximate areas differ from the more detailed resource descriptions (affected extent about 30 vs 37 km²; analysis extent about 330 vs 363 km²). Both are provider statements. After download, reproject to a metric CRS, calculate exact geometry areas and put the actual values and version discrepancy in `docs/data_quality.md`. Do not present either rounded number as Team 12's result. The affected resource says EPSG:32645; analysis resource says EPSG:4326; check actual embedded `.prj` rather than assuming.

**Important download warning:** The download link may redirect to an expiring signed cloud-storage URL. Keep the **stable dataset page and resource links**, not copied short-lived token-bearing URLs. Download in the browser or use the page's stable download endpoint; handle auth/errors normally. Dataset page DOI `10.63253/1y26ihdq` is a citation, not a guarantee that the file can be retrieved automatically. Check the resource's licence terms before redistribution.

### 3.2 Official Sen1Floods11 dataset details

- Labels: one band, `-1` invalid/no data, `0` non-water, `1` water. **Always use `ignore_index=-1`** in loss and metrics; never relabel `-1` as non-water.
- S2 source: 13 bands, L1C TOA reflectance typically stored as integers scaled by 10,000. Original source band mapping (zero-based): `0:B1`, `1:B2`, `2:B3`, `3:B4`, `4:B5`, `5:B6`, `6:B7`, `7:B8`, `8:B8A`, `9:B9`, `10:B10`, `11:B11`, `12:B12`.
- Prithvi-compatible six input channels in **this order:** `[B2, B3, B4, B8A, B11, B12]`, indices `[1,2,3,8,11,12]`. Confirm `inference.py` and configuration on the *pinned* model revision before any normalization changes.
- S1 chips are VV/VH dB and are *not* interchangeable with S2 channels.
- Official dataset whole bucket is described as approximately 14 GB; hand-labeled S2, labels and splits are materially smaller. Download **only hand-labeled S2 + hand labels + split CSVs** first; add S1 later if needed. Historical third-party size estimates for individual subsets are not guaranteed download sizes.
- Use the provider's split files to prevent leakage. Additional `event_id` grouping is required for an event-held-out experiment. Never tune on the test set. A pretrained Prithvi checkpoint already fine-tuned on Sen1Floods11 **cannot automatically be described as an independently unseen benchmark on the same dataset**.

### 3.3 What qualifies as a usable Rasuwa optical scene

Make a scene shortlist covering the **pre-event** window (e.g. 10–25 Aug 2026) and **post-event** window (26 Aug–5 Sep 2026); these are *search windows*, not claims about available acquisitions. For each candidate collect:

- STAC `id`, collection, platform, acquisition start/end, publication/created time if provider gives it, provider, processing level L1C, tile IDs, cloud percentage, orbital metadata, footprint, intersection area with AOI and usable asset URLs.
- Local open checks: file exists, nonzero bytes, hash, `rasterio.open` succeeds, correct bands, original pixel grid, nodata, footprints, masks, pixel ranges, licence/attribution, channel alignment and any reprojection history.
- **Pre-cutoff eligibility:** both acquisition and verified public availability timestamp must be before the selected decision cutoff. If provider doesn't expose reliable publication/availability time, set `availability_unknown` and exclude from strict replay.
- Prefer **a real post-event 27 Aug Sentinel-2 L1C image** when catalog verifies it. This is retrospective mapping only. If it's too cloudy, select another valid scene/date within the recorded window, or use the official labeled sample for the GPU proof while showing the Rasuwa evidence gap explicitly.
- Keep L1C and L2A distinct. Planetary Computer conveniently serves Sentinel-2 **L2A**, but do not silently use L2A with an L1C checkpoint or infer that their numeric normalization is identical.

---

## 4. Five-person ownership, nonoverlapping folders and handoffs

Choose one human as integration coordinator (P5). People can use **Claude Code or Codex independently**, but instructions and all authoritative results live in the **same GitHub repository**.

| Person | Responsible area | Exclusive code/write ownership (unless PR reviewed) | Hard deliverables | Handoff to |
|---|---|---|---|---|
| **P1 — Satellite/Data Evidence** | Scene discovery; download; metadata; AOI data contracts | `src/pravah/catalog/`, `scripts/discover_*`, `scripts/download_*`, `data/manifests/`, `docs/data_sources.md` | `assets.json`, `scenes.json`, a clean real S2 L1C sample, UNOSAT zip/geometry metadata, license table | P2, P3, P5 |
| **P2 — ML/Training** | Prithvi compatible inference; small U-Net; datasets; single-/DDP-training script | `src/pravah/ml/`, `scripts/prepare_chips.py`, `scripts/train_ddp.py`, `config/models/`, `tests/test_ml_*` | checkpoint run, tile predictions, model card+hash, correct loss/metrics, training script and one CPU forward/backward smoke test | P4, P3, P5 |
| **P3 — Geospatial/Science QA** | CRS, masks, OSM extraction, exposure, UNOSAT evaluation caveats | `src/pravah/gis/`, `scripts/process_osm.py`, `scripts/exposure.py`, `scripts/validate_reference.py`, `tests/test_gis_*` | OSM GPKG, georeferenced stitched water mask, overlay JSON and coverage, real map layers, tests | P5 |
| **P4 — Platform/GB10/Benchmarks** | Fuzzball, validated workflow examples, ARM64 image, storage, 1-/2-node and optional DDP launch | `infra/`, `workflows/`, `scripts/infer_tiles.py`, `scripts/stitch.py`, `scripts/make_tile_manifest.py`, `scripts/run_worker.py`, `scripts/benchmark.py`, `benchmarks/`, `docs/platform.md` | image+runtime smoke test, Team12 allocation evidence, two-node job logs, `benchmarks.csv`, reproducible commands | P2, P5 |
| **P5 — Integration/UI/Project lead** | contracts, CI, GitHub governance, Streamlit, replay and judge materials | `app/`, `src/pravah/contracts/`, `src/pravah/replay/`, `.github/`, `docs/demo_*`, root configs and `README.md` | runnable demo, cutoff enforcement, contract fixtures clearly marked, passing CI, final submission and slide/README narrative | All |

**Ownership rule:** P5 creates initial canonical API/schema and sample *test fixtures*, then owners implement those interfaces. Every contract change requires P5 + consumer's review. For overlapping files, coordinate a short PR instead of allowing two AI agents to rewrite the same module.

### 4.1 Dependencies to communicate in the first 30 minutes

- P1 ↔ P3: AOI/CRS, UNOSAT source quality, geospatial format.
- P1 ↔ P2: `scene_id`, source band order, radiometry, quality masks, acquisition/availability times.
- P2 ↔ P4: deterministic tile manifest, model files/hash, command-line runner, output directory, GPU libraries and container requirements.
- P2 ↔ P3: georeferenced pixel/class/raster contract and nodata meaning.
- P1/P2/P3/P4 → P5: versioned **JSON/GeoTIFF/GPKG artifacts**, not manually typed results.

### 4.2 Team operating cadence

- Start: 15-minute agreement on paths, schemas, issue ownership and source-of-truth branch.
- Each member posts a short check-in in a single GitHub issue/Discussion or team chat every 90–120 minutes: `working`, `done`, `blocked`, `PR`, `next`.
- Integrate a working thin vertical slice **before** adding extra models.
- P5 merges small, tested PRs continuously; do not defer integration to Sunday morning.
- The person operating the real GPU captures logs immediately and posts run IDs; do not trust screenshots alone.

---

## 5. Shared GitHub repo + separate Claude/Codex accounts — exact setup

### 5.1 One-time repo creation (P5)

Create **one repository** under a GitHub organization if possible (`ORG/pravah`), or under one member's personal account. Repository name: `pravah`. Private until team verifies licensing and redacts secrets, then public if submission permits. Invite the four other **individual GitHub accounts** as collaborators with write access. Only P5/team maintainers should alter repo settings or approve emergency overrides. GitHub documents that personal repos can invite collaborators on both public and private repositories: [GitHub collaborator instructions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository).

Example **after** P5 creates the empty repository in GitHub UI:

```bash
# P5 local setup (replace with the actual real URL; do not use an invented org)
git clone git@github.com:YOUR_ORG_OR_USER/pravah.git
cd pravah
git config user.name 'YOUR_GITHUB_DISPLAY_NAME'
git config user.email 'YOUR_VERIFIED_GITHUB_EMAIL'
git switch -c chore/project-foundation
# Add the directory tree and canonical contracts shown below.
git add .
git commit -m 'chore: initialize PRAVAH project contracts and structure'
git push -u origin chore/project-foundation
# Open a PR and merge it. Never commit password/token/credentials.
```

Once the foundation PR is merged, **every person**, independently:

```bash
git clone git@github.com:YOUR_ORG_OR_USER/pravah.git
cd pravah
git config user.name 'YOUR_OWN_GITHUB_NAME'
git config user.email 'YOUR_OWN_GITHUB_EMAIL'
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c feat/p1-data-catalog        # P1, change branch prefix for your own role
# Work and test only within your owned files.
git add <only-your-owned-paths>
git commit -m 'feat(data): add audited STAC scene discovery'
git push -u origin feat/p1-data-catalog
# Open a PR; review, fix comments, merge.
```

**Branch naming:** `feat/p1-*`, `feat/p2-*`, `feat/p3-*`, `feat/p4-*`, `feat/p5-*`, `fix/*`, `docs/*`. Use `git pull --ff-only` for main and `git fetch origin && git rebase origin/main` (only on your unshared feature branch) to update. Never force-push shared `main`.

### 5.2 Repo protection and workflow (P5)

1. Enable `main` branch protection or ruleset if repo permissions/plan allow it: no direct pushes; require PR; at least one human review; require CI status checks once working; disallow force pushes and deletions. Do not configure impossible requirements that block final submission.
2. Assign each issue and PR to exactly one owner; use labels `P1-data`, `P2-ml`, `P3-gis`, `P4-hpc`, `P5-ui`, `priority:p0`, `blocked`, `stretch`.
3. PR template must state: paths changed, issue number, commands executed, test result, source/data IDs, fixtures vs real data, known blockers, screenshot/log link **without secrets**.
4. `.gitignore`: `.env*` except `.env.example`, credentials, `.venv/`, `__pycache__/`, `data/raw/**`, `data/processed/**`, `models/weights/**`, `outputs/**`, `benchmarks/raw_logs/**`, `*.pt`, `*.pth`, `*.onnx`, `*.ckpt`, `*.tif`, `*.tiff`, `*.pbf`, `*.zip`. Keep small manifest JSON and approved report previews tracked by allowlist (`git add -f` *only if licensing permits*).
5. Put large assets in **local disk → shared authorized volume** or a team-approved object store; commit **metadata/URL/checksum/scripts** rather than Git LFS by default. Git LFS only if team approves quotas/credentials.
6. Use GitHub Actions on every PR for a small CPU dependency set, `ruff check`, format checks and `pytest`. Do **not** expect GitHub CI to have your GB10 or private Fuzzball network; GPU jobs are a separate run stage.
7. All generated outputs include commit SHA and `run_id`; do not overwrite finished runs. Distinguish `tests/fixtures/` from event data at all times.

### 5.3 Using Claude or Codex without conflicts

- Put **the same concise rules** in root `AGENTS.md` for Codex and `CLAUDE.md` for Claude Code. Codex explicitly recognizes `AGENTS.md` ([OpenAI documentation](https://learn.chatgpt.com/docs/agent-configuration/agents-md)); Claude Code automatically uses `CLAUDE.md` ([Anthropic documentation](https://support.claude.com/en/articles/14553240-give-claude-context-claude-md-and-better-prompts)).
- Each human stays signed into their *own* Claude/Codex/GitHub account. A tool login does not transfer GitHub permissions to another contributor; authorize each locally as needed. Personal credentials must never be copied to the repo.
- Give each coding agent only the owner's bounded task/branch and ask it to **inspect the repository first**, implement, write unit tests, run them, commit or prepare a PR, and report failures. Never ask five agents to 'build the whole app' at once.
- A person wanting multiple parallel coding sessions on a **single machine** uses separate `git worktree` directories and independent branches so two agents never share the same working tree. Across five human laptops, ordinary separate clones suffice.
- If an agent generates a broad patch touching another owner's files, reject/split the PR and ask the respective owner to review the interface change.
- An AI agent's claimed success is not evidence; terminal output, test logs, file hashes and actual GPU job records are evidence.

### 5.4 Root `AGENTS.md` / `CLAUDE.md` reference content (companion files are ready)

```markdown
# PRAVAH Team 12 engineering rules

Read docs/IMPLEMENTATION_RUNBOOK.md and docs/contracts.md first.
Goal: real, reproducible satellite-derived water evidence and measured 1-/2-node GB10 scaling.
Never claim pre-event forecasts based on post-event data, masks as confirmed damage,
or fake CUDA/benchmark results. Historical cutoff requires BOTH acquired_at and
verified available_at <= cutoff; unknown availability is excluded.

Architecture: P1 catalog, P2 ML, P3 GIS, P4 HPC, P5 integration/UI.
Work only in your assigned issue's paths and feature branch; no direct main commits.
Do not edit someone else's contract without proposing the change and seeking review.
Write tests; run the smallest meaningful test suite; report exact commands/results.
Preserve CRS, affine transform, nodata, radiometry, units and source timestamps.
Never invent scene IDs, URLs, measured metrics, GPU node identities or licenses.
Never commit secrets, raw archives or model weights. Treat unverified downloads as failed.
State implemented, tested, proposed, blocked and not attempted in each PR.
Use reproducible run IDs and immutable outputs; preserve source citation and license.
If a dependency fails, maintain a truthful CPU/rule-based fallback and flag it.
```

**Note:** Create `docs/IMPLEMENTATION_RUNBOOK.md` by copying this document into the repo, and put the schema definitions into `docs/contracts.md` before coding begins.

### 5.5 Recommended GitHub issue/PR format

```markdown
## Issue ID / Owner / Deadline
## Goal: one sentence
## Inputs / exact real sources / dependencies
## Files permitted to edit
## CLI contract (input flags, output paths)
## Acceptance tests (commands + expected invariants)
## Evidence (sample run logs, checksums, source IDs)
## Risks/fallback
## Handoff to downstream owner
```

Each PR description has an `Implemented / Tested / Not tested / Blocked` table; never substitute 'works' for evidence.

---

## 6. Proposed repository (create paths now; work is divided by owner)

```text
pravah/
├── README.md
├── AGENTS.md                       # Codex context (same rules as CLAUDE.md)
├── CLAUDE.md                       # Claude Code context
├── pyproject.toml                  # CPU/core tools, tests, Streamlit
├── requirements-cpu.txt            # pinned after validation
├── requirements-gb10.txt           # pinned after real aarch64 CUDA validation
├── .gitignore
├── .env.example                     # NAMES ONLY; no live values
├── Makefile
├── .github/
│   ├── workflows/ci.yml
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── CODEOWNERS
│   └── ISSUE_TEMPLATE/
├── config/
│   ├── project.yaml
│   ├── replay.yaml
│   ├── models/prithvi.yaml
│   ├── models/unet_s2.yaml
│   ├── train_s2_unet.yaml
│   └── aois/{aoi_core,aoi_corridor,analysis_coverage}.geojson
├── data/
│   ├── raw/{sentinel2,sen1floods11,unosat,osm,dem}/   # gitignored
│   ├── processed/                                      # gitignored
│   └── manifests/{assets,scenes,tiles,splits}.json
├── models/
│   ├── manifest_prithvi.json
│   ├── manifest_unet.json
│   └── weights/                              # gitignored
├── src/pravah/
│   ├── __init__.py
│   ├── contracts/{schema.py,validation.py}
│   ├── catalog/{stac.py,downloads.py,provenance.py}
│   ├── ml/{datasets.py,unet.py,prithvi.py,trainer.py,inference.py,metrics.py}
│   ├── gis/{crs.py,tiles.py,stitch.py,exposure.py,reference.py}
│   ├── replay/{eligibility.py,timeline.py}
│   └── reporting/{summary.py}
├── scripts/
│   ├── audit_environment.py
│   ├── discover_scenes.py
│   ├── download_assets.py
│   ├── download_sen1floods11.sh
│   ├── preprocess_s2.py
│   ├── prepare_chips.py
│   ├── make_tile_manifest.py
│   ├── infer_tiles.py
│   ├── run_worker.py
│   ├── stitch.py
│   ├── process_osm.py
│   ├── exposure.py
│   ├── validate_reference.py
│   ├── train_ddp.py
│   ├── benchmark.py
│   └── audit_replay.py
├── infra/
│   ├── Dockerfile.arm64                 # finalize from tested base, not guessed
│   ├── build-and-push.sh
│   ├── environment-report.sh
│   └── README.md
├── workflows/
│   ├── smoke_test.fz                   # must be validated against real 4.3.1
│   ├── one_node_infer.fz
│   ├── two_node_infer.fz
│   └── two_node_train.fz               # optional, not fabricated YAML
├── benchmarks/
│   ├── README.md
│   ├── benchmark_schema.json
│   └── benchmarks.csv                  # only measured records
├── outputs/                             # gitignored per run, manifest export tracked if safe
│   └── <run_id>/
│       ├── observed_water.tif
│       ├── observed_metadata.json
│       ├── invalid_mask.tif
│       ├── exposure.geojson
│       ├── exposure_summary.json
│       ├── benchmark.json
│       └── report.json
├── app/{Home.py,pages/,components/,fixtures/}
├── tests/
│   ├── fixtures/                       # clearly labeled synthetic unit-test data
│   ├── test_replay.py
│   ├── test_contracts.py
│   ├── test_tiles.py
│   ├── test_gis.py
│   ├── test_ml.py
│   └── test_benchmark.py
└── docs/
    ├── IMPLEMENTATION_RUNBOOK.md
    ├── contracts.md
    ├── data_sources.md
    ├── data_quality.md
    ├── model_card.md
    ├── science_limitations.md
    ├── platform.md
    ├── demo_script.md
    └── submission_checklist.md
```

### 6.1 Minimal environment setup (each laptop)

macOS (Homebrew already installed) or Ubuntu/WSL: ensure `git`, `python3`, `python3-venv`, `gdalinfo`, `ogrinfo`, `curl`, `unzip` and optional `osmium` are available. Install GDAL/PROJ from your platform's package manager before attempting fragile Python GDAL builds. On macOS use `brew install git python gdal proj osmium-tool` **if Homebrew is available**; on Ubuntu use `sudo apt install git python3 python3-venv python3-pip gdal-bin libgdal-dev proj-bin osmium-tool`. If the available Python/GDAL versions conflict, use `micromamba` or `conda-forge` and pin them.

```bash
cd pravah
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# P5 first creates requirements-cpu.txt with tested, pinned dependencies.
pip install -r requirements-cpu.txt
python -c 'import rasterio, geopandas, shapely, pyproj; print("GIS imports OK")'
pytest -q
```

**Do not require ARM64 CUDA PyTorch on the laptop.** A Mac MPS backend, if available, is local smoke testing, not a substitute for measured GB10 execution. It is normal for P2 to complete an initial CPU forward pass before Sunday.

### 6.2 Suggested starter CPU requirements (P5 to install-test and lock)

```text
numpy
pandas
rasterio
geopandas
shapely
pyproj
pystac-client
httpx
pydantic
python-dateutil
pyyaml
streamlit
folium
pytest
ruff
huggingface_hub
# torch/terratorch/segmentation-models-pytorch: add to a separate tested ML env
```

These are *unversioned planning entries only*; replace with a resolved lock and Python/platform versions after installation. Avoid installing massive model dependencies in CI unnecessarily.

---

## 7. P1 — satellite, geospatial source intake, scene discovery and auditing

### P1-01: Verify UNOSAT archive and construct study coverage

**Do first:** Open the three resource links on [UNESCO](https://ihp-wins.unesco.org/en/dataset/nepal-satellite-detected-mudflow-and-rockflow-extent-nepal). Download each shapefile ZIP through the official browser into `data/raw/unosat/`. Note download URL *without temporary signed tokens*, actual filename, file size, SHA256 and terms. Use:

```bash
find data/raw/unosat -type f -maxdepth 2 -print
shasum -a 256 data/raw/unosat/*.zip   # macOS; Linux sha256sum
unzip -l data/raw/unosat/NAME_FROM_PROVIDER.zip | head -30
ogrinfo -so -al data/raw/unosat/NAME_FROM_PROVIDER.zip
# For SHP inside ZIP, unzip to a directory before ogrinfo if necessary.
```

Do not assume `ogrinfo` can read ZIP paths the same way on all builds. Record actual layer CRS, field schema, feature count, union area, and invalid geometry repairs. Convert to GeoJSON EPSG:4326 for sharing and GeoPackage EPSG:32645 for meter calculations. For example, after identifying an actual `.shp`:

```bash
ogr2ogr -f GeoJSON -t_srs EPSG:4326 \
  config/aois/analysis_coverage.geojson data/raw/unosat/analysis_extent/ACTUAL_FILE.shp
ogr2ogr -f GPKG -t_srs EPSG:32645 \
  data/processed/unosat_reference.gpkg data/raw/unosat/affected/ACTUAL_FILE.shp \
  -nln affected_unosat
```

**Do not use this transformed retrospective affected mask for pre-cutoff model input.** P1 sends P3 the actual geometry paths and metadata; P3 checks geospatial validity and size mismatch.

**Acceptance:** `analysis_coverage.geojson` reads with GeoPandas, has a legitimate CRS, nonempty valid geometries, checksums and UNOSAT 28 Aug publication tag; affected geometry is separate, never interchangeable.

### P1-02: Download Sen1Floods11 *subset*, not entire archive

Use official bucket according to repo's **example commands**, after confirming with `gsutil ls`; install Google Cloud CLI/gsutil if missing. The README contains an inconsistent spelling (`senfloods11` in one sentence, `sen1floods11` in the examples). Validate before copying.

```bash
# Dry-check public bucket before bulk transfer
gsutil ls gs://sen1floods11/v1.1/
gsutil ls gs://sen1floods11/v1.1/splits/flood_handlabeled/
gsutil ls gs://sen1floods11/v1.1/data/flood_events/HandLabeled/

# Confirm these actual paths exist, then run separately (the structure is documented
# by downstream users of the official dataset; official bucket is the authority):
mkdir -p data/raw/sen1floods11/v1.1/splits/flood_handlabeled
mkdir -p data/raw/sen1floods11/v1.1/data/flood_events/HandLabeled/{S2Hand,LabelHand}
gsutil -m rsync -r gs://sen1floods11/v1.1/splits/flood_handlabeled/ \
  data/raw/sen1floods11/v1.1/splits/flood_handlabeled/
gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/ \
  data/raw/sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/
gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/ \
  data/raw/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/
```

If any folder differs, **do not brute-force sync the entire 14 GB bucket**; inspect `gsutil ls` and adjust documented actual names. Download a small sample first to validate 512×512, 13 bands, labels and split mapping. Record which split CSV filenames were found; do not invent them. Leave S1Hand (~additional download) for optional fallback. Count train/validation/test chips *from actual split files* and check zero overlap.

**Acceptance:** one S2 TIFF and its true label open, label unique values subset `{-1,0,1}`, 13 input bands, matching geometry, dataset split CSVs usable; source path/version logged. P2 uses exactly these files.

### P1-03: Retrieve Prithvi upstream inference sample and exact checkpoint

P1 or P2 can use Hugging Face CLI; coordinate to avoid duplicate large downloads. The model repo lists `config.yaml`, `inference.py`, examples, and a ~1.28 GB `.pt` checkpoint; example file `examples/India_900498_S2Hand.tif`. Pin a known commit hash (not `main` permanently).

```bash
python -m pip install 'huggingface_hub[cli]'
# Check 'hf --help' and the model files list first.
MODEL_ID=ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11
hf download "$MODEL_ID" inference.py --local-dir models/weights/prithvi
hf download "$MODEL_ID" config.yaml --local-dir models/weights/prithvi
hf download "$MODEL_ID" requirements.txt --local-dir models/weights/prithvi
hf download "$MODEL_ID" Prithvi-EO-V2-300M-TL-Sen1Floods11.pt --local-dir models/weights/prithvi
hf download "$MODEL_ID" examples/India_900498_S2Hand.tif --local-dir models/weights/prithvi
# Check hashes; record the resolved revision from HF metadata/cache as well.
```

If Hugging Face requires authorization or a revised filename, inspect the repository UI and record the reason; never fabricate checkpoint hashes. The model repo is Apache-2.0; verify the actual files' licences and redistribution rules.

### P1-04: Catalog Rasuwa Sentinel-2 L1C scenes (before downloading large products)

Discover provider STAC collections instead of assuming provider IDs; CDSE documents a public STAC endpoint but available collections may change. Use `pystac_client` with actual AOI geometry:

```python
from pystac_client import Client
from shapely.geometry import shape
import json

aoi = json.load(open('config/aois/analysis_coverage.geojson'))
geometry = aoi['features'][0]['geometry']  # replace with *union* if multi-feature
catalog = Client.open('https://stac.dataspace.copernicus.eu/v1/')
collections = {c.id: c.title for c in catalog.get_collections()}
print('Available collections:', collections)
# Choose the actual Sentinel-2 Level-1C collection ID from this output.
# search = catalog.search(collections=[REAL_S2_L1C_ID],
#                         intersects=geometry,
#                         datetime='2026-08-10/2026-09-05', max_items=100)
# items = list(search.items())
```

Use correct geometry even if UNOSAT coverage is multi-polygon; `shapely.ops.unary_union` / `union_all` and `mapping` can be used. Store **every returned STAC item JSON**, source asset names and class. Compare imagery overlap and quality; select small, usable AOI windows. Download only required S2 bands and associated quality layers if provider supports asset-level range requests; otherwise budget for larger SAFE archives and do not falsely claim a partial download if unavailable.

**If CDSE download auth blocks you:** (a) save the catalog metadata; (b) check documented public Planetary Computer STAC for alternatives, **noting it commonly provides L2A**; (c) prefer upstream Sen1Floods11 sample for inference smoke while maintaining an explicit Rasuwa data gap; (d) ask CIQ for data access help if necessary.

### P1-05: Build resilient download auditor

`download_assets.py` must have `--manifest`, `--dest`, `--max-concurrency 3`, `--timeout-seconds`, `--retry`, `--dry-run`; stream responses to `.part` then atomic rename, bounded memory, `Retry-After` handling, exponential backoff, content-length comparison when available, sha256, raster open, and an error record that **cannot** be mistaken for a completed asset. Check free disk space. Do not dump signed URLs/tokens to public logs. Record provider request IDs and licensing in manifest if permissible.

**Acceptance:** successful asset has actual file hash/open metadata; failed URL is `status=failed`, never a placeholder zero-byte success. A unit test simulates HTTP error/timeout and verifies failure behavior.

### P1-06: Preprocess Sentinel-2 with model-specific radiometry

- Start with the original **13-band upstream sample**, verify output from the original inference script before coding custom preprocessing.
- For new multi-asset L1C scenes, resample physical S2 B2/B3/B4/B8A/B11/B12 onto a **single explicitly chosen grid**. B8A/B11/B12 are 20 m native; resample to chosen target and **do not claim added real resolution**. For rapid output, use 20 m target or document the inference method's 10 m upsampling.
- Preserve reflectance scaling, band names, original `nodata` and quality masks. Do **not** min-max normalize each tile unless that exactly follows upstream inference/model preprocessing; changing preprocessing silently invalidates the checkpoint.
- Cloud/terrain shadow is a critical quality issue. Mark cloud/invalid separately, and do not interpret dark mountain shadow as confirmed water.
- Cut tightly around AOI, use an explicit projected CRS and affine transform; keep original image/asset lineage.

**Deliverables:** `data/processed/s2_rasuwa_l1c_6band.tif` if scene actually exists; `assets.json`, `scenes.json`, `docs/data_sources.md`, `data/manifests/scene_selection.md` (selection rationale). Failure-safe fallback is an official labeled sample **clearly labeled `Sen1Floods11 sample, not Rasuwa`.**

---

## 8. P2 — pretrained Prithvi inference and GPU-ready training

### P2-01: Reference model smoke test **before custom wrapper**

Use P1's exact downloaded model repo files in a compatible local Python environment, and follow model's included `requirements.txt`/`inference.py`. Try the upstream command, with correct relative paths and `--checkpoint`/`--config`:

```bash
# Execute from folder containing verified upstream inference.py, config and weight.
# The repo sample input is a 13-band S2 L1C hand-labeled chip.
python inference.py \
  --data_file examples/India_900498_S2Hand.tif \
  --config config.yaml \
  --checkpoint Prithvi-EO-V2-300M-TL-Sen1Floods11.pt \
  --output_dir ../../outputs/prithvi_reference_smoke
```

**Important:** The exact source code expects certain relative paths and TerraTorch versions. If it fails due to CPU memory or dependency, preserve the traceback and move to P2-02 with a smaller U-Net; do not claim Prithvi ran successfully. `inference.py`'s default `--input_indices` are 0-based `[1,2,3,8,11,12]` from the 13-band L1C source. Record the output's class IDs and `nodata` behavior; upstream example may set metadata nodata to `0`, which can collide with background class—adapt your *own* wrapper to use a distinct invalid code and test it before a publishable output.

### P2-02: Build small 6-band U-Net smoke model independent of Prithvi

Use Sen1Floods11 S2 hand-labeled subset and a small segmentation model (e.g. `segmentation_models_pytorch.Unet(encoder_name='resnet18', encoder_weights=None, in_channels=6, classes=2)`). This is a **separately trained baseline**, with 6 channels ordered to match training data. Verify package compatibility before adopting; write a project-owned lightweight U-Net if packaging is difficult.

**Dataset requirements:** `float32 [6,H,W]` input; label `int64 [H,W]` with values `{-1,0,1}`; validity mask; `H=W=512` initially or deterministic `256×256` crops for speed; optional flip/90-degree rotation training augmentation only; capture crop transforms for georeferenced inference. Preprocess radiometry with a stated fixed dataset strategy, separate from the upstream Prithvi normalization. Never infer 'flood-specific' classes from these binary labels.

**Loss:** cross-entropy with `ignore_index=-1`; optional Dice IoU on valid pixels. **Metrics:** for water class compute precision, recall, F1 and IoU on valid pixels with confusion counts `TP,FP,FN,TN` and label support; handle no-valid-pixel cases as `NA`, not perfect accuracy. Split CSV files govern train/validation/test; no test-set hyperparameter tuning.

### P2-03: Implement CLI and CPU unit smoke

`python scripts/train_ddp.py --config config/train_s2_unet.yaml --dry-run` must validate path/splits/model args without training. A smoke command `--max-steps 2 --device cpu --num-workers 0 --output-dir outputs/training_smoke` must do one real forward, loss, backward, optimizer step and write measured loss. Unit-test `ignore_index`, deterministic seed, split overlap=0 and checkpoint reload.

Proposed CLI contract:

```text
python scripts/train_ddp.py \
  --data-root data/raw/sen1floods11/v1.1 \
  --split-root data/raw/sen1floods11/v1.1/splits/flood_handlabeled \
  --config config/train_s2_unet.yaml \
  --output-dir outputs/TRAIN_RUN_ID \
  [--max-steps N] [--device cpu|cuda] [--resume CKPT]
```

Output: `config_resolved.yaml`, `data_splits.json`, `train_metrics.csv`, `val_metrics.csv`, `checkpoint_last.pt` (rank 0 only), `checkpoint_best.pt` (validation IoU only), `environment.json`, `source_hashes.json`. Final test metrics should be generated once after selection from validation—never leak the test labels into training or model choice.

### P2-04: Implement DDP correctness *before* touching Fuzzball

The script must work in three modes: ordinary CPU, ordinary `cuda:0`, and `torchrun` DDP with `RANK/WORLD_SIZE/LOCAL_RANK` provided by PyTorch. Standard DDP design:

```python
# PSEUDOCODE / design sketch, not a complete runnable training script
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

world_size = int(os.environ.get('WORLD_SIZE', '1'))
use_ddp = world_size > 1
if use_ddp:
    dist.init_process_group(backend='nccl')  # only after CUDA/NCCL verified
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = build_small_unet().to(device)
if use_ddp:
    model = DDP(model, device_ids=[local_rank])
train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
train_loader = DataLoader(train_dataset, sampler=train_sampler,
                          shuffle=not use_ddp, batch_size=PER_RANK_BATCH)
for epoch in range(epochs):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    # train; aggregate numerator/denominator metrics across ranks;
    # no duplicate samples silently scored; only rank 0 writes checkpoints
if use_ddp:
    dist.barrier()
    dist.destroy_process_group()
```

**DDP details easy to miss:** `DistributedSampler` can pad/duplicate samples to make equal per-rank lengths; aggregate validation predictions carefully, de-duplicate by chip ID or use a no-padding evaluation sampler. Set seeds, sampler epochs, `num_workers`, data-loader worker seed, `torch.backends` deterministic flags if useful. Synchronize gradients and scalar metrics appropriately. For fair 1-node vs 2-node comparison **hold total number of unique chips, global effective batch, optimizer-step budget, precision and train/val split fixed**; use gradient accumulation if one node cannot support matching batch, and report changed accumulation overhead. Do not claim scaling improvements from comparing different work.

### P2-05: Inference tile worker design

`python scripts/infer_tiles.py --tiles data/manifests/tiles.json --partition-index 0 --partition-count 2 --model MODEL_PATH --output-dir outputs/RUN_ID/tiles/worker0 --device cuda` reads deterministic tile IDs, skips invalid-only tiles with explicit records, writes a probability tile and per-tile JSON containing `tile_id`, source scene, bounds/transform/grid, valid fraction, model hash, worker identity, runtime, output SHA256. Each worker writes to a **distinct output prefix** and exits nonzero on a failed required tile. `partition_index = tile_index % partition_count`. The single-node benchmark runs `partition-count=1`, the two-node benchmark runs `0/2` and `1/2` over the **same full tile list**.

Do not average final class IDs. Stitch probability/logits in overlap with valid-mask-weighted averaging; then apply a documented class threshold. Preserve output georeferencing and invalid pixels. Keep `observed_water` distinct from `potential_new_water` derived from a historical baseline.

### P2-06: Model selection decision deadline

- If Prithvi reference sample runs + Rasuwa L1C works in the available environment: **Prithvi for event inference**, U-Net for training demo.
- If Prithvi sample requires long ARM64 dependency repair: **small U-Net for training and case-study inference**, clearly name it as trained on Sen1Floods11 and note domain transfer uncertainty.
- If S2 has unusable cloud/shadow: use official sample for honest GPU benchmark; map contextual Rasuwa scenes and UNOSAT data separately. Do **not** claim remote Rasuwa model output exists.
- If only threshold works: explicit `MNDWI/rule-based baseline`; make no claim that rule output proves pretrained AI.

---

## 9. P3 — geospatial overlay, scientifically defensible validation and quality tests

### P3-01: Freeze geographic contracts

- Shared GeoJSON/WGS84 (`EPSG:4326`) for web, raster/metric processing in verified local projected CRS **EPSG:32645** (Rasuwa UNOSAT affected source uses this) unless the actual scene/AOI analysis justifies another choice.
- `aoi_core.geojson` is a **small clearly documented derived polygon**, ideally intersection of one verified image footprint with a selected portion of UNOSAT analysis corridor, expanded only by an explicitly logged buffer. Do not use UNOSAT *post-event affected* shape to define a supposedly pre-event target without disclosing the retrospective choice of location.
- `aoi_corridor.geojson` = broader verified river corridor for context, not necessarily an AI inference footprint. Document geographic source and pixel/meter area.
- Reproject all actual geometries before area, buffer or length; **never compute kilometers from EPSG:4326 degrees**.

### P3-02: Obtain historical and current OSM separately

For historical replay prefer the **Jan 1, 2026** Nepal extract listed by Geofabrik (verify from source listing):

```bash
curl -fL --retry 3 \
  'https://download.geofabrik.de/asia/nepal-260101.osm.pbf' \
  -o data/raw/osm/nepal-260101.osm.pbf
# Official .md5 companion is available on listing; also record your own SHA256.
osmium fileinfo -e data/raw/osm/nepal-260101.osm.pbf
# First create a compact AOI .poly or polygon from verified aoi_core bounds.
osmium extract -p config/aois/aoi_core.geojson \
  data/raw/osm/nepal-260101.osm.pbf \
  -o data/processed/core_osm_20260101.osm.pbf --overwrite
```

**Guardrail:** `osmium extract -p` requires supported polygon input and a valid polygon geometry; test your installed version and simplify multi-polygons as required. Use `pyrosm`, `ogr2ogr`/OSM driver, or `osmium tags-filter` to extract `building`, `highway`, `bridge`, `amenity=hospital`, `amenity=school` as separate layers; test that lines/polygons and tags are preserved. **PBF does not directly become a valid footprint GeoJSON just by renaming it.**

Optionally download a **current** extract and label `retrospective_osm_snapshot` in dashboard. Note OSM attribution and ODbL obligations. The Jan 2026 snapshot can omit August 2026 features: incomplete historical map ≠ no buildings.

### P3-03: Stitch and validate water mask

- Verify each predicted tile shares the reference CRS, pixel grid, model revision and class mapping.
- Validate that union of processed tile IDs equals manifest tile set, no missing/duplicate tiles; allow exact comparison of single-/two-node outputs within floating-point tolerance if nondeterminism exists.
- Mosaic **probabilities** before thresholding; mask invalid/zero-coverage pixels; polygonize only valid water area; remove artifacts only using disclosed minimum mapping unit.
- Output `observed_water.tif`, `invalid_mask.tif`, optionally `observed_water.geojson`, `observed_metadata.json`. Clearly label this **model-inferred water**, not a field-confirmed flood perimeter.
- Calculate water area in square meters/ha from metric pixels (or geodesic equivalent) and record the area inside valid observational coverage. Do not include nodata as 'not flooded'.

### P3-04: Exposure methods and exact definitions

1. **Buildings:** count *unique* valid OSM building polygons intersecting the water-mask polygon, optionally compute intersection fraction. Name field `building_footprints_intersecting`, never `houses_destroyed`.
2. **Roads:** clip actual *line length* to water polygon in meters, sum kilometers by OSM highway tag. Name `road_length_m_intersecting`, never `damaged_road_km`.
3. **Bridges:** count distinct OSM `bridge=*` tagged linear features intersecting, de-duplicate by stable ID and source type; not confirmed bridge failures.
4. **Population:** optional; aggregate appropriate WorldPop raster *population counts* inside mask with fractionally weighted pixels or source-recommended zonal method. Preserve reference year, scaling, nodata, input release date, coverage and uncertainty.
5. **Data coverage:** number of valid observed pixels / target analysis footprint, and OSM source date. A zero exposure count within complete coverage is different from `null` due to missing/invalid input.

### P3-05: UNOSAT scientific evaluation protocol

Because UNOSAT maps **land affected by floodwater, mudflow, rockflow and deposits** and our Prithvi/U-Net mostly detects **water**, computing a single IoU and calling it `flood accuracy` would be scientifically misleading. We can:

- display both layers as transparent, **different legend entries**;
- calculate geometric overlap, omission and commission **only with an explicit 'cross-class spatial agreement, not classification accuracy' warning**;
- restrict comparisons to the intersection of model-valid area with **UNOSAT analysis extent**;
- calculate per-AOI coverage and invalid fraction; report `not comparable` if mask semantics/geometry diverge too strongly;
- show case-study qualitative crops, especially debris-filled valley, shadow, snow and dry scoured surfaces;
- use held-out **Sen1Floods11 binary-water labels** instead for genuine model water-IoU/precision/recall/F1, noting that Prithvi pretrained on the same dataset may not be independent.

### P3-06: Required synthetic unit tests (fixtures clearly identified)

- EPSG:4326 → EPSG:32645 distance/area sanity.
- Geometry polygon covers raster true-positive pixel with known pixel area.
- Road line crosses 10 m of mask: intersection length exactly testable.
- Duplicate bridge OSM IDs count once.
- Missing vector source leads to `null`/not available, not zero.
- Invalid/cloud pixel does **not** count as water or dry land.
- UNOSAT reference outside coverage is excluded from denominator.
- Background label `0` and nodata label distinct.

---

## 10. P4 — no-GPU preparation and Sunday GPU integration

### P4-01: What is already known vs unverified

From Team 12's preceding blueprint: Fuzzball UI/CLI observed `v4.3.1`; the local WSL CLI binary was installed, but its active context and real **API/OIDC URLs are unknown**; web access to `ui.fb-hack.civilfritz.net` private `172.16.0.133:443` timed out on event network, and the team resource definition was believed to be `team12` with cluster `fb-hack`. A visible NFS *provisioner* was not evidence of a mounted persistent volume. Event materials say **two** Lenovo ThinkStation PGX / GB10 systems are allocated per team; **Team 12's specific nodes must be confirmed**. A node belonging to some other resource definition must not be used.

**Do not 'fix' private routing by assigning your own IP, scanning internal hosts, disabling security protections, assuming SSH to hardware, or reusing another team's resources.** Ask authorized CIQ mentors for service health and network instructions.

### P4-02: Prepare files locally before Sunday 09:00

1. Read [Fuzzball docs](https://docs.ciq.com/fuzzball), especially [task arrays](https://docs.ciq.com/fuzzball/parallel-and-distributed/parallel), [generic multinode](https://docs.ciq.com/fuzzball/parallel-and-distributed/generic), [NCCL](https://docs.ciq.com/fuzzball/parallel-and-distributed/nccl), [persistent volume example](https://docs.ciq.com/fuzzball/volumes/vols-examples/vols-ex-persistent), [GPU jobs](https://docs.ciq.com/fuzzball/advanced-hardware/gpus), and [v4.3.1 release notes](https://docs.ciq.com/fuzzball/release-notes/v4.3.1).
2. Prepare `scripts/run_worker.py`: identical tested Python entrypoint for local one-worker and GPU workers, deterministic `--partition-index`/`--partition-count`, output files and timing JSON, graceful missing GPU failure if `--require-cuda`.
3. Prepare `scripts/benchmark.py` with a strict JSON/CSV schema and single/two run comparison. Ensure no demo data can be mistaken for measured runs.
4. Prepare `Dockerfile.arm64` **starting from a demonstrated/CIQ-provided ARM64 CUDA/PyTorch image**, or determine compatible NVIDIA NGC image manifest offline from public docs. Validate availability via `docker buildx imagetools inspect IMAGE:TAG`; select only when actual manifest contains `linux/arm64`, PyTorch has CUDA support for GB10 Blackwell and Fuzzball container policy permits the registry. Don't invent image tags. The container should include **all Python packages at build time** (Fuzzball workloads may run unprivileged).
5. Make `workflows/*.fz` templates marked `UNVERIFIED` until P4 has downloaded/obtained an actual successful **4.3.1** sample, then adapt its resource, shared-storage, job and multinode syntax. Keep stage names and resource definition explicit. Do not copy a v4.4 Fuzzfile blindly.
6. Create a short `.env.example` of names: `FUZZBALL_CLUSTER`, `FUZZBALL_GROUP`, `FUZZBALL_API_URL`, `FUZZBALL_OIDC_URL`, `CONTAINER_IMAGE`, `SHARED_VOLUME_NAME`, `PRAVAH_RUN_ID`; no real values/credentials.
7. Prepare the exact questions below in one message to organizers before 09:00 if they can respond.

### P4-03: Ask CIQ these precise questions

- Is `ui.fb-hack.civilfritz.net` expected to be reachable from our present Wi-Fi/VPN? What authorized network and route must we use?
- What are the **real Fuzzball CLI API endpoint** and **OIDC issuer/realm endpoint** for v4.3.1? The UI's CLI panel currently showed placeholders. Is client ID `fuzzball-cli` valid?
- Are **both** Team 12 GB10 nodes ready and allocated with Resource Definition `team12`? What are actual node IDs and `nvidia.com/gpu` resource counts?
- What are the current GPU/container base image(s) and registry credentials supported for **aarch64 / GB10 Blackwell**? Any CIQ-provided smoke-test Fuzzfile?
- Is there a **writable Team 12 persistent shared volume** and quota? How is it created/mounted? Is the shared filesystem truly VAST-backed?
- Does this cluster support Fuzzball `generic` multinode or task arrays pinned to two distinct Team 12 nodes? What networking/rendezvous ports and NCCL/GPU container settings are authorized?
- Can we upload a prepared image or staged data from laptops? Is there object ingress/volume ingress, or a network-provided job staging route?
- Is the exact judge submission deadline 16:00 CT, and must a live Fuzzball run be demonstrated before it?

### P4-04: Sunday 09:00 GPU access ladder (do not skip gates)

**Gate A — network/auth (09:00 onward):** connect using CIQ-approved network, verify HTTPS reachability, get actual endpoints and login. From earlier blueprint the CLI binary is installed on one WSL machine, but context is unconfigured. Command pattern below is a template requiring mentor-provided values:

```bash
fuzzball --version
fuzzball context create hackathon REAL_API_URL REAL_OIDC_REALM_URL fuzzball-cli
fuzzball context use hackathon
fuzzball context login --group team12
fuzzball context list
fuzzball workflow --help
# Read online help for exact syntax; do not invent CLI flags.
```

**Gate B — authorized nodes, GPU and storage:** list nodes in UI/CLI, confirm two distinct IDs **in Team 12** and readiness; run CIQ's known-good printer/sample; create/verify the actual persistent volume, mount it and test from two workflows; check `df -h` and write/rename/read test. Record node IDs and logs.

**Gate C — ARM64 CUDA smoke:** in **the actual scheduled container** run:

```bash
uname -m
nvidia-smi
python3 - <<'PY'
import torch
print('torch=', torch.__version__)
print('cuda_available=', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu=', torch.cuda.get_device_name(0))
    a=torch.ones((512,512),device='cuda'); print('cuda_sum=',a.sum().item())
PY
```

Expect aarch64 architecture, **actual** GPU visible, CUDA tensor operation succeeds. Do not assume previous inspection of another team's node proves our allocation. Log container image digest, device name, driver/CUDA version, allocations and node ID.

**Gate D — one GPU real inference:** stage one tile manifest, model weight and inputs to confirmed volume; run worker with `--partition-count 1`; verify model device is really CUDA, artifact hashes and georeferencing. Capture inference compute time distinct from queue/initialization.

**Gate E — two independent GB10 workers:** schedule two jobs/tasks with partitions `0/2` and `1/2` and require **two distinct Team 12 node IDs** in logs. Output to separate unique prefixes on shared volume, then stitch and compare coverage with one-node output. If scheduler places both jobs on same node, stop and correct the allocation; it is not two-node proof.

**Gate F — DDP (only if Gate E reliable):** use a validated Fuzzball multinode `generic` sample plus container with working PyTorch NCCL; launch `torchrun` across the two nodes with one GPU process per node. Check both ranks participate; train for a bounded number of steps, write rank-zero checkpoint and all-rank metrics. See §11.

### P4-05: Container portability pitfalls

- Team laptop may be macOS ARM64 or WSL x86_64, **but GB10 runs Linux aarch64**; host binaries and Python wheels are not interchangeable.
- `torch.cuda.is_available()` must return true **inside the allocated GPU container**, not just on host. PyTorch CPU-only builds may import while CUDA fails.
- NVIDIA GB10 GPU code needs a supported CUDA/Blackwell target; official [NVIDIA DGX Spark resources](https://build.nvidia.com/spark/nccl) explain multi-system NCCL, but a CIQ-managed Lenovo PGX environment is *not* an unrestricted DGX Spark host. Do not run host-level network/sudo/NCCL setup playbooks unless explicitly authorized.
- Fuzzball public docs currently show 4.4 examples; the team's observed deployment 4.3.1 may differ. Use mentor-provided 4.3.1 syntax and test it.
- VAST support must be empirically confirmed; a generic NFS mount isn't proof of its backing vendor.

### P4-06: Benchmark schema, methodology and pass/fail

For each run write `run_id`, `commit_sha`, `dataset_manifest_sha256`, `model_id/revision/hash`, `nodes_requested`, `nodes_used`, actual `node_ids`, `gpu_name`, image digest, `tile_count`, `tile_ids_hash`, `queue_seconds`, `pull_seconds`, `init_seconds`, `inference_seconds`, `stitch_seconds`, `end_to_end_seconds`, `valid_pixel_count`, `bytes_read`, `bytes_written`, output hash, failures, result status. If not measurable, value = `null` and state why.

- Start with one single-node and one two-node matched run; if time, repeat 3 times and show median + range.
- Report **inference compute speedup** and **wall-clock speedup separately**. `speedup=T1/T2`; `parallel_efficiency=speedup/2`; fixed identical total workload, consistent cold/warm status.
- Report **throughput** = valid tiles/sec or valid megapixels/sec, including denominators.
- Check `set(single_node_tile_ids)==set(two_node_all_tile_ids)`, no duplicate processed IDs, output-equivalence tolerances and consistent nodata masks.
- If a 2-node run is slower because of queue/I/O/startup, **report regression honestly** and explain bottlenecks. This is still scientific evidence.
- Do not use a fake screenshot or a calculation with example timings as a real benchmark.

---

## 11. Sunday optional genuine multi-node training with both GB10 machines

### 11.1 Why the primary trainable target is a small U-Net

Prithvi's 300M weights can be excellent for inference; fine-tuning it end-to-end with a large geospatial pipeline adds dependency, batch-memory and time risks. With only seven hours of GPU access, prepare a **minimal, honest, bounded 2-GPU DDP training experiment on hand-labeled Sentinel-2 water segmentation**, using the small 6-channel U-Net from §8. A 2-node checkpoint and synced gradients show actual Multi-Node Training. If successful and time remains, **optional LoRA/partial fine-tuning Prithvi** can use [IBM `peft-geofm`](https://github.com/IBM/peft-geofm), but only if its exact S2/Sen1Floods11 setup is tested. Do not claim the small U-Net is Prithvi or vice versa.

### 11.2 Hardware/multi-node preconditions

1. Both allocated GPU nodes visible, authorized and accessible through Fuzzball.
2. Correct aarch64 image with CUDA, PyTorch and NCCL, working single-GPU step.
3. A Fuzzball **generic multinode** job template compatible with v4.3.1 and an allowed rendezvous port/interface; get mentor guidance.
4. Dataset and configs readable from both nodes, identical model code and seed; only rank 0 writes checkpoints.
5. `NCCL_DEBUG=INFO` logging and authorized networking. Avoid indiscriminately changing NCCL variables or opening ports without CIQ direction.
6. `torchrun` launched **on each node**, same rendezvous ID/endpoint, `--nnodes=2`, `--nproc-per-node=1`, each with the correct rank. CIQ 4.4 generic docs use `$RANK`, `$MULTINODE_NODE_IP` and `exec_all` but **these must be validated for 4.3.1**.

### 11.3 Reference `torchrun` design (illustrative shell, NOT a Fuzzball Fuzzfile)

On *each* authorized node within the same **validated generic multi-node job**, when the host/port/rank are genuinely supplied by scheduler and verified:

```bash
# Variables set by approved launcher; these are placeholders, not known addresses.
# MASTER_ADDR must refer to the actual rendezvous node, not always local node.
# NODE_RANK must be 0 on one machine and 1 on the other.
export NCCL_DEBUG=INFO
torchrun \
  --nnodes=2 \
  --nproc-per-node=1 \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  scripts/train_ddp.py \
  --config config/train_s2_unet.yaml \
  --data-root /SHARED_REAL_PATH/sen1floods11/v1.1 \
  --output-dir /SHARED_REAL_PATH/runs/TRAIN_RUN_ID \
  --max-steps 40 --device cuda
```

**This is not necessarily the best Fuzzball launcher:** in Fuzzball 4.4, `generic` uses environment helpers and `exec_all` to launch processes on nodes. Team 12 must adapt the working 4.3.1 Fuzzfile, *not* assume two manually opened shells or SSH to GB10 hosts. PyTorch [torchrun docs](https://docs.pytorch.org/docs/stable/elastic/run.html) document rendezvous requirements; [DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) explains gradient synchronization.

### 11.4 Experiment matrix for training

| Experiment | Workers | Data / batch | Purpose | Required |
|---|---:|---|---|---|
| `train_smoke_cpu` | 1 CPU | 2 batches | Test dataset/labels/loss/backprop and checkpoint without GPUs | **Before Sunday** |
| `train_smoke_cuda_1` | 1 GB10 GPU | 2–5 steps | Verify real CUDA/PyTorch plus dataset reads | **Sunday** |
| `train_1gpu_fixed` | 1 GB10 GPU | Same unique chip set, effective global batch and steps as 2-GPU | Baseline | Strongly recommended |
| `train_2gpu_ddp_fixed` | 2 separate GB10 GPUs | Same unique chip set, effective global batch/steps, genuine synchronized gradients | Multi-node claim | Stretch if NCCL works |
| `train_prithvi_lora` | 2 GB10 GPUs | separate validated PEFT config, declared split | Advanced scientific enhancement | **Only if everything else done** |

**Do not declare two-node training just because both GPUs each run a training command.** The training run must show `WORLD_SIZE=2`, both ranks, a shared process group, synchronized gradients and a single coherent checkpoint.

### 11.5 Training model quality vs compute metric

- **Scientific metrics**: held-out IoU/F1 on Sen1Floods11 water labels, per-event breakdown if feasible; report sample counts and limitations. Use a held-out **event** split when making claims about unseen flood events. If official splits mix events, report this distinction explicitly. No Rasuwa accuracy without independent, semantics-matched labels.
- **Compute metrics**: same fixed workload, steps/sec, training samples/sec, wall time, NCCL communication/CPU I/O observations, GPU and node count. Compare models on equal optimization budgets, not a one-epoch vs ten-epoch apples-to-oranges chart.
- A couple of GPU minutes may prove multi-node wiring and throughput but is not enough to claim a scientifically converged model. The UI should show the status `DDP smoke / preliminary training`, not `fully trained` unless convergence/evaluation substantiates it.

### 11.6 Distributed failure fallbacks

- If NCCL rendezvous fails: do not waste the full seven-hour window. Preserve two-node independent inference and measured throughput, mark DDP `blocked`, optionally execute **independent single-node hyperparameter runs** on both nodes and label them correctly (`parallel experiment sweep`, not DDP).
- If model is CPU-only in the container: fix aarch64 CUDA base/compatibility with mentor. Do not measure CPU runs as GB10 GPU acceleration.
- If shared volume is slow: copy stage assets into each job's authorized local scratch while keeping durable results on persistent storage; measure staging separately and obey storage quotas.
- If GPU memory runs out: reduce per-rank batch, crop size or select the small U-Net; record changes to the benchmark config.

---

## 12. P5 — Streamlit dashboard, timeline integrity, integration and presentation

### 12.1 Build UI against schema-only fixtures tonight

Pages: (1) **Overview & Science** with location/event and methodology; (2) **Map & Layers** with base image, AOIs, model-derived water, invalid areas, OSM roads/buildings and UNOSAT *retrospective* affected extent; (3) **Evidence Timeline** with acquisition/availability date and 'pre-cutoff vs reveal' gating; (4) **Multi-Node Compute** with actual job IDs, node IDs and measured table; (5) **Sources & Limitations**.

Display honest labels: `not yet computed`, `no qualifying scene`, `invalid/obscured`, `no exposure inside verified coverage`; do not display unpopulated fields as zero. At initial UI state show a prominent **demo/test fixture** annotation if using made-up unit-test shapes for layout. Remove from judge presentation when showing real artifacts.

### 12.2 Historical replay integrity

The UI must differentiate at least five times: `event_time` (if documented), `acquired_at_utc`, `available_at_utc`, `ingested_at_utc`, `cutoff_at_utc`. For pre-cutoff visibility, **both `acquired_at` and `available_at` must be present and <= cutoff**. An unknown `available_at` means *not eligible*. Dataset publication dated Aug 28 (UNOSAT), reports dated Aug 28 or later, and Nov 2026 OSM cannot be visible as allegedly known before Aug 26.

Suggested UI control: `Mode = Evidence Available at Cutoff | Retrospective Reveal`, with strict record filtering and a separate banner for retrospective data. **No 'predict event' button** unless a genuine forecast model is added/validated. Reference report text must use source links and as-of date.

### 12.3 Integration API/data files

```
GET /runs                  # optional FastAPI; Streamlit can read immutable files directly
GET /runs/<run_id>/summary
GET /runs/<run_id>/tiles
GET /runs/<run_id>/benchmark
GET /runs/<run_id>/provenance
GET /runs/<run_id>/assets
```

**Implement no backend server until it adds value.** A simple Streamlit app that reads `outputs/<run_id>/` is more reliable for the judges, with data contract unit tests.

### 12.4 Run locally and demo fallback

```bash
python -m streamlit run app/Home.py
# Ensure app catches missing assets and displays an explanatory status, not a stack trace.
```

Prepare a static local copy of **real** finished outputs in one portable demo folder, subject to licence/size, and a local `app` launch command. If Fuzzball is down during presentation, show captured **genuine** workflow logs and completed results with IDs. Never claim a recorded run is live.

---

## 13. Canonical data contracts — MUST be agreed before parallel implementation

### 13.1 Asset manifest, one record per actual external asset

```json
{
  "asset_id": "ACTUAL_PROVIDER_ID",
  "source_name": "Copernicus Sentinel-2",
  "source_uri": "ACTUAL_NON_SECRET_URI_OR_CANONICAL_URL",
  "sensor": "S2",
  "processing_level": "L1C",
  "acquired_at_utc": null,
  "available_at_utc": null,
  "ingested_at_utc": null,
  "coverage_geom_id": "aoi_core_v1",
  "bands": ["B2", "B3", "B4", "B8A", "B11", "B12"],
  "band_units": "document_from_provider",
  "crs": null,
  "bounds": null,
  "nodata": null,
  "cloud_or_invalid_fraction": null,
  "license_id": null,
  "license_verified": false,
  "file_bytes": null,
  "sha256": null,
  "download_status": "candidate",
  "raster_open_status": "not_tested",
  "replay_eligibility": "unknown_availability",
  "notes": "Schema example: none of these are claimed real file measurements"
}
```

**Allowed statuses:** `candidate`, `downloading`, `downloaded_unaudited`, `verified`, `failed`, `not_licensed`, `blocked`; and replay `eligible`, `excluded_after_cutoff`, `unknown_availability`, `retrospective_only`.

### 13.2 Tile manifest

```json
{
  "run_id": "ACTUAL_RUN_ID",
  "model_id": "ACTUAL_PINNED_MODEL",
  "scene_asset_ids": ["ACTUAL_SCENE_ID"],
  "target_crs": "EPSG:32645",
  "source_width_px": 0,
  "source_height_px": 0,
  "tile_size_px": 0,
  "tile_overlap_px": 0,
  "tiles": [
    {
      "tile_id": "UNIQUE_TILE_ID",
      "index": 0,
      "window": [0, 0, 0, 0],
      "affine": [0, 0, 0, 0, 0, 0],
      "valid_fraction": null,
      "partition_count": 2,
      "partition_index": 0
    }
  ]
}
```

These zero fields are placeholders in a **schema example only** and must never appear in final artifacts as genuine pixel geometry. `partition_index` is computed deterministically from real tile ordering. Record hashes of JSON and input raster.

### 13.3 Water GeoTIFF, metadata and invalid semantics

- `observed_water.tif`: output class raster **with project-defined class codes**, e.g. `0=no_water`, `1=water`, `255=invalid`; this is an *output remapping* from source model's classes, explicitly documented, not a claim about the original checkpoint. Do **not** set GeoTIFF nodata to `0` if zero is valid no-water.
- `water_probability.tif`: optional `float32 0–1` per water probability, with actual validity and calibration caveats.
- `invalid_mask.tif`: `1=invalid / unobserved`, `0=valid`; include cloud/shadow/no-data sources where reliable. Do not equate low confidence with no data.
- `observed_metadata.json`: model URL, commit, weight hash, class source mapping, band/normalization details, acquisition/availability, output CRS/grid, threshold, valid coverage, test status, pipeline code SHA.
- `observed_water.geojson`: polygons created from water class *only*, explicit minimum mapped area if filtering is used.

### 13.4 Exposure JSON

```json
{
  "run_id": "ACTUAL_RUN_ID",
  "model_water_artifact": "outputs/ACTUAL_RUN_ID/observed_water.tif",
  "source_snapshot": "ACTUAL_OSM_VERSION_OR_NULL",
  "source_track": "retrospective_or_cutoff_eligible",
  "covered_area_m2": null,
  "valid_observation_fraction": null,
  "water_area_m2": null,
  "building_footprints_intersecting": null,
  "road_length_m_intersecting": null,
  "bridge_tagged_features_intersecting": null,
  "estimated_resident_population_intersecting": null,
  "unavailable_fields": ["values_not_yet_computed"],
  "caveats": ["Exposure intersection is not confirmed damage"]
}
```

### 13.5 Replay record and parser logic

```python
from datetime import datetime, timezone

def strict_replay_eligible(acquired, available, cutoff):
    if acquired is None or available is None or cutoff is None:
        return False
    if any(x.tzinfo is None or x.utcoffset() is None for x in (acquired, available, cutoff)):
        raise ValueError('All replay times must be timezone-aware')
    return acquired <= cutoff and available <= cutoff
```

Tests: before/before succeeds, after/before fails, before/after fails, unknown availability fails, naive datetime raises, equality passes. Nepal time UTC+05:45 must be represented correctly in UI along with UTC; Chicago demo time zone is Central Time with DST applied by date.

### 13.6 Integration validity rules

- All summary numbers are recomputed from artifacts or shown null; never hardcoded in UI.
- All maps use proper geometry and event CRS, and the legend is explicit about `observed model water`, `reference debris/rock/mud area`, `invalid`, `terrain scenario`.
- Input checksum + code SHA + model hash must identify a run. A path alone is not provenance.
- Train/validation/test manifests and tile manifests immutable after run starts.
- No test fixture enters final run registry unless explicitly marked `test_fixture=true` and excluded from judge metrics.

---

## 14. Exact execution timeline — before 09:00 Sunday, during cluster access, and deadline

**All times Central Time (America/Chicago), local to the hackathon. Adjust team rest times and organizer instructions as necessary.** These are *coordination checkpoints* rather than guaranteed task durations.

### Saturday 14 Nov: 00:45–02:00 — initialize and assign

**Everyone:** join one communication channel; confirm five GitHub usernames, personal access, time zone, hardware available, data/storage capacity and owner assignments. P5 creates repo and foundation PR; copies this runbook; posts P1–P5 GitHub issues; agrees on inputs/contracts. P1 opens UNOSAT resources and sample dataset. P2 checks exact Prithvi files. P3 confirms correct AOI/CRS strategy. P4 records access blockers and reads public Fuzzball docs. **Deliver:** foundation merged, an agreed schema, no double ownership.

### Saturday 02:00–06:00 — data and smoke tests

- **P1:** download the three UNOSAT resources, subset Sen1Floods11 labels/S2 and first example; begin STAC discovery; record checksums/status.
- **P2:** implement small U-Net dataset loader and 1–2-batch CPU smoke; download/configure exact Prithvi reference sample if dependency accessible; log any framework issues.
- **P3:** validate UNOSAT analysis area geometries, historical OSM extraction feasibility, GIS unit tests.
- **P4:** write worker CLI and benchmark contract, inspect image manifests and public docs; leave Fuzzfiles tagged unverified.
- **P5:** working Streamlit scaffolding, replay timestamp tests and CI.

**06:00 checkpoint:** at least one official real labeled sample locally, valid metadata, CPU smoke code paths, dashboard scaffold PRs. If data blocked, decide an alternate official sample, not synthetic event output.

### Saturday 06:00–12:00 — integrate first real vertical slice

Work/rest shifts can overlap: P1 selects the actual Rasuwa L1C scene if available; P2 runs upstream Prithvi sample or trains small U-Net, generating georeferenced test predictions; P3 creates OSM features and intersection logic; P4 tests local partition worker with CPU; P5 merges contracts and renders real sample/explicit fixture flags. **12:00 checkpoint:** sample image → segmentation → real raster → GIS overlay → dashboard. If not, freeze other features and repair the first broken step.

### Saturday 12:00–18:00 — Rasuwa evidence and GPU-ready artifact pack

P1 stages Rasuwa acquisition or records why unavailable. P2 creates deterministic tile prediction path and exact model manifest. P3 builds valid Rasuwa water/UNOSAT/OSM layers, checks comparisons. P4 packages aarch64 candidate container with verified architecture and prepares final 1-/2-node YAML *skeletons*. P5 verifies run registry and replay modes. **18:00 checkpoint:** source manifests, exact chosen model, pinned input data, shared tile list and one-command local demo.

### Saturday 18:00–23:00 — complete GPU preflight bundle

- P1 ensure all assets staged on **team-controlled media or approved transfer location**, not GitHub blobs; checksum list and size budget.
- P2 ensure GPU training script has CPU smoke, resume/checkpoint logic, DDP code path and 6-band data.
- P3 ensure GIS coverage/exposure tests and layer labels.
- P4 dry-run input partitioning, model/inference CPU compatibility, container build/push if allowed, benchmark scripts and exact questions for CIQ.
- P5 run full local demo rehearsal, fix broken workflows, merge open PRs. **23:00 checkpoint:** all local dependencies ready, and GPU commands described in `docs/platform.md`.

### Sunday 15 Nov 07:30–09:00 — immediate pre-cluster review

**Everyone:** pull main, run `pytest`, check asset manifests, ensure data transfer path and model checkpoint ready, confirm 09:00 access instruction and mentor contact, prepare demo screenshots of *real* local work. P5 records candidate final commit and source provenance. **Do not start major new feature development at 08:50.**

### Sunday 09:00–10:00 — gates A/B/C

P4 + P2 dominate: authorized network, v4.3.1 API/OIDC, Team12 nodes/storage, sample Fuzzfile, real ARM64 GPU test; P1 handles data staging; P3/P5 continue polishing and integration. **10:00 checkpoint:** at least one actual GPU job. If not, escalate immediately to mentors and preserve plan B.

### Sunday 10:00–11:30 — one-node and two-node inference

P4 launches matched one-node job then two-node split workers; P2 monitors model and tile output; P3 verifies stitch/equivalence and updates geospatial results; P5 displays live/past run metadata. **11:30 checkpoint:** genuine measured paired run or documented blocker.

### Sunday 11:30–12:45 — optional DDP training

Only if previous gates work: P2 and P4 run 1-node training smoke then synchronized DDP for fixed steps, capture ranks/metrics/checkpoint. Otherwise use time to strengthen inference replicability and validation. **12:45 checkpoint:** stop experimenting with new dependencies; convert logs into report and screenshots.

### Sunday 12:45–14:00 — verification, interpretability and benchmark table

P3 cross-checks geometry, source dates, UNOSAT semantics; P2 validates predictions and no-label leakage; P4 calculates benchmark from actual logs and probes reproducibility; P5 polishes dashboard and citations. **14:00 integration freeze:** merge essential PRs and pin final commit. Future fixes require approval.

### Sunday 14:00–15:00 — judge demo rehearsal and submission package

P5 runs full five-minute demo from clean checkout/local artifacts; all owners verify exact claims, provenance and no sensitive data. Prepare a 60-second fallback explanation if Fuzzball UI is unavailable. Final README contains: challenge track, original code built during event, team members, scientific use case, data sources, architecture, how to reproduce, actual measured benchmark, limitations and screenshots.

### Sunday 15:00–15:30 — submission safety buffer

Submit by organizer-stated deadline (may precede published event end), confirm repository access, demo link/asset integrity, screenshot backup, readme and run logs. No speculative last-minute model swap. **16:00 CT is the official advertised event end, not proof of the submission cutoff.**

---

## 15. Concrete GitHub issue backlog — cut and paste into Issues

**P0 issues must be completed in this order where dependencies exist. P1 means nice-to-have; Stretch may be excluded.**

| Issue | Owner | Priority | Specific action | Acceptance evidence / downstream dependency |
|---|---|---|---|---|
| FND-01 | P5 | P0 | Repo, collaborators, branch protection, README, `.gitignore`, contract skeleton | 5 collaborators can clone/push own branches; foundation PR merged |
| FND-02 | P5 | P0 | `AGENTS.md`, `CLAUDE.md`, PR template, CI | CI passes CPU tests and prevents fixture/secret tracking |
| FND-03 | P5 | P0 | JSON manifests, asset/tile/exposure/replay contract tests | `pytest tests/test_contracts.py tests/test_replay.py` |
| DAT-01 | P1 | P0 | UNOSAT three resource download+checksum+metadata | Real archives open, exact coverage/affected/frame layers separated |
| DAT-02 | P1 | P0 | Sen1Floods11 S2Hand/LabelHand/splits partial download | 13-band sample, 0/1/-1 labels and split list validated |
| DAT-03 | P1 | P0 | Prithvi repo config/checkpoint/sample with pinned revision | Hashes and model input tests; P2 can load files |
| DAT-04 | P1 | P0 | STAC catalog discovery, scene shortlist, choose real L1C | Actual IDs+metadata, or honest `no_usable_scene` |
| DAT-05 | P1 | P0 | Audited downloader with retry/checksum/raster-open | Mock failure test + real small file manifest |
| DAT-06 | P1 | P1 | Scene harmonization and cloud/invalid masks | Six-band aligned raster plus original timestamps |
| ML-01 | P2 | P0 | Sen1Floods11 dataset loader and split validator | Labels/radiometry/shape checked; no split overlap |
| ML-02 | P2 | P0 | 6-channel small U-Net + 2-step CPU training | Real forward/backward/checkpoint reload; loss finite |
| ML-03 | P2 | P0 | Upstream Prithvi sample inference smoke | Actual output or documented blocked dependency, no fabricated success |
| ML-04 | P2 | P0 | Deterministic tile manifest, inference worker, output schema | 1/2 partitions cover same tile set exactly |
| ML-05 | P2 | P0 | DDP script ready for torchrun | local CPU / 1-process smoke and correct DDP sampler logic |
| ML-06 | P2 | P1 | Water IoU/F1/precision/recall on compatible held-out labels | Confusion counts, valid pixels and split provenance |
| GIS-01 | P3 | P0 | UNOSAT geometries, AOI and EPSG:32645 area tests | No invented AOI/area, actual source CRS recorded |
| GIS-02 | P3 | P0 | Historical Jan 2026 OSM AOI extract and quality | Real buildings and highway geometries in metric CRS |
| GIS-03 | P3 | P0 | Stitch probability tiles, invalid mask and GeoTIFF | Valid raster alignment and complete tile coverage |
| GIS-04 | P3 | P0 | Exposure overlay, counted buildings, road lengths | Automated intersection unit tests + JSON null semantics |
| GIS-05 | P3 | P1 | UNOSAT analysis comparison and caveat documentation | No false 'flood accuracy' claims across classes |
| HPC-01 | P4 | P0 | 4.3.1 Fuzzball docs review + team endpoint questions | Platform runbook and blocker list |
| HPC-02 | P4 | P0 | Local worker CLI, output isolation, timing schema | CPU partition run logs and deterministic hashes |
| HPC-03 | P4 | P0 | Verified ARM64 container strategy/registry ingress | Real image manifest supports arm64; CUDA test to follow |
| HPC-04 | P4 | P0 | Team12 Fuzzball network, auth, node/volume smoke Sunday | Real CLI/UI job IDs and node IDs, no borrowed resources |
| HPC-05 | P4 | P0 | 1-node real GPU inference | Real GPU logs and georeferenced tile output |
| HPC-06 | P4 | P0 | 2-node matched inference and benchmark | Two distinct GPU nodes, exact tile union, measured CSV |
| HPC-07 | P4+P2 | Stretch | 2-node synchronized PyTorch DDP training | `WORLD_SIZE=2`, NCCL process group, real checkpoint/metrics |
| UI-01 | P5 | P0 | Streamlit map UI and provenance panels | Runs locally; real/fixture labels truthful |
| UI-02 | P5 | P0 | Historical cutoff filter and later reveal tests | Aug 28 UNOSAT excluded before cutoff; unknown availability excluded |
| UI-03 | P5 | P0 | Real artifact reader and backend-free integration | No hardcoded exposure counts or fake GPU numbers |
| UI-04 | P5 | P0 | CI, docs, five-minute demo and submission | Clean checkout+run test; judges can inspect source hashes |
| SCI-01 | P3/P5 | Stretch | HAND susceptibility scenario | Calibrated? **No**; labeled illustrative and isolated |
| SCI-02 | P5 | Stretch | Grounded English/Nepali source-backed summary | Numeric fidelity test; no invented statements |

### 15.1 Freeze gates / substitutions

| Deadline gate | Requirement | If missed |
|---|---|---|
| Saturday 06:00 | Real training sample, schemas, basic repo | Use official Sen1Floods11 sample; stop Rasuwa bulk downloads temporarily |
| Saturday noon | Source-to-model-to-mask thin path | Stop UI polish/HAND; repair model/dataset compatibility |
| Saturday 18:00 | Worker/manifest contracts frozen | Freeze scope; P4 only supports a stable worker script |
| Saturday 23:00 | GPU-ready artifact pack | Sunday use only the already tested sample and small model |
| Sunday 10:00 | Real GPU smoke | Request mentor help; preserve local demonstration |
| Sunday 11:30 | Matched two-node inference | Defer DDP; finish one-node real result and honest failure log |
| Sunday 12:45 | DDP proof if feasible | Do not chase DDP further; focus submission |
| Sunday 14:00 | Integration freeze | Essential hotfixes only |
| Sunday 15:00 | Submission ready | Submit immediately; avoid missing organizer cutoff |

---

## 16. Required tests and CI commands

`Makefile` proposed targets (P5 should implement, then all owners use):

```makefile
.PHONY: lint test contracts data-smoke ml-smoke gis-smoke app
lint:
	ruff check src scripts tests app
	ruff format --check src scripts tests app
test:
	pytest -q
contracts:
	pytest -q tests/test_contracts.py tests/test_replay.py
data-smoke:
	python scripts/audit_environment.py
ml-smoke:
	python scripts/train_ddp.py --config config/train_s2_unet.yaml --max-steps 2 --device cpu --output-dir outputs/smoke-cpu
gis-smoke:
	pytest -q tests/test_gis.py tests/test_tiles.py
app:
	python -m streamlit run app/Home.py
```

**Test matrix:**

| Category | Must-pass examples |
|---|---|
| Input validation | Bad/empty TIFF, missing CRS, wrong band count, incorrect dB/reflectance units, filename mismatch, bad label class, empty footprint |
| Time/provenance | UTC-aware timestamps, strict acquisition+availability gate, exclusion unknown time, UNOSAT only retrospective, future OSM snapshot rejected from strict cutoff |
| Tile system | Partition ID uniqueness, full coverage, deterministic ordering, overlap weights, no invalid-as-dry, monotonic window bounds |
| ML | Correct channel order, ignore-index metrics, finite loss/backward, checkpoint model hash, held-out split mapping |
| GIS | CRS meters, valid geometry, area/length, bridge dedup, missing vs zero, actual analysis coverage mask |
| HPC | Compare equal tile manifest hash, two distinct node IDs, reliable benchmark status, no fabricated null-to-zero |
| UI | Missing artifacts gracefully handled, source time displayed, fixture banner, post-event reveal gating |
| Security | No secrets in repo/URLs/logs; `.gitignore` protects caches and large assets |

**Document actual commands run and test counts**. A checklist item is not 'passed' until someone executes it and saves output.

---

## 17. Failure-mode playbook (most likely weekend blockers)

| Blocker | Detect quickly | Correct action | What to show honestly |
|---|---|---|---|
| No Fuzzball network | TCP 443/auth failure on authorized Wi-Fi | Ask CIQ for network/VPN and endpoints, don't bypass ACL | Local app and access-blocker log |
| Unknown CLI endpoints | Placeholder strings in panel | Obtain actual API+OIDC URLs from mentor | CLI installed, context unconfigured |
| No Team12 volume | Provisioner visible, volume missing | Create/attach according to quota; validate read/write on two workers | One-node local staging only until proven |
| ARM64 PyTorch/CUDA unavailable | CPU-only torch or `no kernel image` | Switch to verified aarch64 NVIDIA image, test tiny CUDA tensor | CPU smoke, CUDA integration incomplete |
| Prithvi 1.28 GB download slow | HF download fails/blocked | Use cached verified sample/model; train small U-Net | Small U-Net explicitly named, no Prithvi claim |
| Sentinel-2 data unavailable/cloudy | No scene/large invalid fraction | Use official training sample for GPU proof; show no eligible Rasuwa observation | No Rasuwa model prediction |
| UNOSAT downloads fail | Token expiry/ZIP error | Reopen stable page, record access status; no copied expiring URLs | Citation and context, no unverified geometry |
| Sen1Floods11 bucket mismatch | `gsutil` 404 | Inspect bucket listing, use confirmed v1.1 paths | Document pending download, not fake data |
| No DDP connectivity | `torchrun` rendezvous/NCCL timeout | Abandon after bounded troubleshooting; retain independent inference | Measured inference, DDP blocked |
| Incorrect model bands | Shape/range mismatch | Stop inference; fix preprocessing according to pinned model | Model not yet validated |
| No matching label semantics | Water map vs debris reference | Qualitative overlap + explicit difference; no IoU accuracy claim | Evidence-aware comparison |
| Latest OSM used accidentally | source date Nov 2026 | Label retrospective; use Jan 2026 for strict replay | Two clearly distinguished source dates |
| Inference very fast on one tile | Parallel overhead dominates | Use bigger real fixed batch of existing tiles, no artificial empty loops; measure compute & wall separately | Honest scaling overhead |
| Model quality weak in Himalaya | Cloud/shadow/debris systematic errors | Show uncertainty and missing coverage; fallback to source observations | No operational disaster prediction |
| Time runs out | Gates not met at planned stop | Freeze, preserve evidence and submit real partial product | What implemented/blocked |

---

## 18. Data integrity, safety, licence and reproducibility checklist

- [ ] Only documented public/authorized providers; no access-control bypass or unlicensed PlanetScope reuse.
- [ ] UNOSAT 28 Aug 2026 is only retrospective; protect its exact metadata/DOI and potential license/area inconsistencies.
- [ ] Source scripts have bounded retries, checksums, open tests and actual file-size audit.
- [ ] Historial OSM Jan 2026 vs current Nov 2026 are separate layers; OSM ODbL attribution and derivative obligations reviewed.
- [ ] Each Sentinel scene records acquisition and verified availability; if unknown, strict replay excludes it.
- [ ] Exactly six L1C optical channels for Prithvi (and separately trained U-Net), correct order and radiometric treatment; no S1-to-S2 crossfeeding.
- [ ] No accidental `nodata=0` when `0=no_water`.
- [ ] Invalid/cloud pixels excluded from binary-water metrics and exposure claims; coverage fraction reported.
- [ ] UNOSAT affected rock/mudflow extent not treated as perfect optical-water ground truth.
- [ ] Local metric CRS for areas and line lengths; all vector geometry validity tested.
- [ ] Model source, specific revision, weight hash, code commit and preprocessor versions recorded.
- [ ] Train/test leakage prevented; pretrained Prithvi's Sen1Floods11 use disclosed; labels never borrowed from a future event for pre-event claim.
- [ ] Team12-only Fuzzball allocation and actual GPU nodes demonstrated; two-node claim backed by distinct node logs.
- [ ] Same fixed dataset/tile manifest for scaling experiment; 1-node and 2-node result integrity comparison.
- [ ] Training: both ranks, NCCL process group, split sampler, global batch, real checkpoint and logged metrics, or explicitly blocked.
- [ ] No security secrets, private endpoints with credentials, tokens, user accounts or auth headers in public repo/screenshots.
- [ ] Versioned outputs immutable, failure statuses retained, sources cited.
- [ ] README/disclaimer states **research prototype, not operational early warning**.

---

## 19. Copy-and-paste Claude/Codex prompts for each person

Before using a prompt, that person clones the **real** shared repo, opens their branch, reads `docs/IMPLEMENTATION_RUNBOOK.md` and the current `docs/contracts.md`, and replaces branch/issue placeholders. Do not paste credentials or invented dataset paths.

### P1 prompt — data/evidence (Claude or Codex)

```text
You are P1, data and satellite evidence owner for PRAVAH Team 12. Read
AGENTS.md/CLAUDE.md, docs/IMPLEMENTATION_RUNBOOK.md sections 3, 7, 13,
and docs/contracts.md. Inspect repository and available data files FIRST.
Work ONLY in src/pravah/catalog, scripts/discover_*, scripts/download_*,
data/manifests and docs/data_sources.md on branch feat/p1-data-catalog.
Implement audited download + STAC discovery, UNOSAT analysis/affected/AOI
separation, official Sen1Floods11 S2Hand/LabelHand/splits intake, exact
13→6 optical-band metadata. Preserve real acquisition/availability time,
licensing, raster CRS, bands and file hashes. Never invent a source ID,
scene availability, successful download or link. Do not download huge
archives without a manifest and bounded concurrency. Write tests for
failures and bad rasters. Commit small coherent work, run tests, and
report: created paths, real files verified, actual commands, blockers,
and the P2/P3 handoff.
```

### P2 prompt — AI + two-node training ready

```text
You are P2, ML owner. Read shared runbook sections 2, 8, 11 and 13,
check the actual branch, official Prithvi model card/inference.py and
Sen1Floods11 files. Work ONLY in src/pravah/ml, scripts/prepare_chips.py,
scripts/infer_tiles.py, scripts/train_ddp.py, config/models,
config/train_s2_unet.yaml and tests/test_ml_* on branch feat/p2-ml.
Implement (A) a pinned Prithvi reference sample smoke using true
Sentinel-2 L1C B2 B3 B4 B8A B11 B12; (B) a separately trained small
6-channel U-Net on real Sen1Floods11 hand labels; (C) deterministic tiled
inference and a torchrun-compatible DDP trainer with DistributedSampler,
ignore_index -1 and rank-zero-only checkpoints. Start with a two-step
CPU forward/backward test; make no CUDA-success claim without real GPU
output. Protect held-out splits and distinguish published checkpoint
metrics from our measured metrics. Provide exact CLI, model metadata,
output contract, tests, limitations and P4/P3 handoff. Do not change
other owners' files without a reviewed contract change.
```

### P3 prompt — GIS and scientific QA

```text
You are P3, GIS/science QA owner. Read shared runbook sections 3, 9 and
13 and P1's actual verified AOI/UNOSAT files. Work ONLY in
src/pravah/gis, scripts/process_osm.py, scripts/exposure.py,
scripts/validate_reference.py and GIS tests on branch feat/p3-gis.
Implement valid geometry conversion, metric CRS EPSG:32645 where
appropriate, reproducible OSM historical snapshot extraction, careful
probability stitching, water-vs-nodata semantics, building intersections,
road overlap lengths and bridge de-duplication. Output precise JSON
with nulls for missing fields and source dates. Use UNOSAT's ANALYSIS
extent as coverage and affected mud/rockflow extent only for
retrospective cross-class comparison; never call it an independent
water mask without qualifications. Add known-shape unit tests and
handoff real georeferenced files and methodology to P5.
```

### P4 prompt — Fuzzball/GB10/benchmark

```text
You are P4, CIQ infrastructure and benchmark owner. Read shared runbook
sections 10, 11, 13, the real Team12 access-state notes, and public
Fuzzball docs. Work ONLY in infra, workflows, scripts/run_worker.py,
scripts/benchmark.py, benchmarks and docs/platform.md on branch
feat/p4-hpc. Prepare aarch64 GB10 CUDA container validation, a local
CPU-testable worker CLI and deterministic one-node/two-node benchmark.
Do NOT invent Fuzzball YAML, API/OIDC URLs, resource definitions, node
IDs, VAST claims, timing or CUDA results. Clearly mark templates
UNVERIFIED until tested against event Fuzzball v4.3.1. Sunday ask CIQ
for approved networking/auth, Team12 GPU allocation, shared volume and
valid examples; run smoke → 1GPU inference → 2 distinct-node inference;
only then try torchrun/NCCL DDP with P2. Store actual job IDs, logs,
node IDs, image digest, input/output hashes and measured timings.
Stop on unauthorized access or unknown versions. Report blockers
truthfully and pass real benchmark CSV to P5.
```

### P5 prompt — integration and final demo

```text
You are P5, integration lead and demo owner. Read the complete runbook
and inspect the repository. Work in app, src/pravah/contracts,
src/pravah/replay, .github, root README and docs/demo_script.md on
branch feat/p5-ui (coordinate initial foundation merge separately).
Implement canonical JSON contracts, strict historical cutoff requiring
both acquisition and verified availability before cutoff, a resilient
Streamlit map and provenance view, real artifact readers and a measured
HPC benchmark tab. Fixtures must be unmistakably labeled and never
become judge metrics. Merge bounded peer-reviewed PRs continuously,
keep CI green, preserve one-command local demo and source attributions.
At deadline freeze, prepare 5-minute evidence-based demonstration and
fallback from real saved outputs, with source dates, model identity,
science uncertainty, verified hardware proof and exact commands.
Never claim unrun work; report implemented, tested, partial and blocked.
```

---

## 20. Copyable demo claims and judge-facing language

**Good project description:**

> PRAVAH is a distributed scientific AI prototype for analysing satellite-observed flood-associated water and mapped infrastructure exposure. Using a historical Rasuwa/Nuwakot event, it combines date-aware satellite provenance, georeferenced model inference, source-aware GIS overlays and a reproducible multi-node GPU performance experiment. It separates retrospective evidence from what was available at a historical decision cutoff, and reports important limitations for debris-rich mountain hazards.

**Good results language:**

> On [real source scene ID], at [acquisition time], [real model/version] produced a model-derived water layer covering [actual valid AOI area]. Overlaying [dated OSM snapshot] identified [computed counts/lengths] **intersecting the inferred water layer**; these are exposure-screening estimates, not confirmed damage. On [two real Team 12 GB10 node IDs], matched workloads took [measured times] with [computed speedup]; logs and exact input manifests are available in the repository/run record.

**Never say (unless independently proven):** "We predicted the Aug 26 flood before it happened," "These bridges were destroyed," "X lives saved," "This simulation gives arrival time," "Our Nepal model has 95% accuracy," "Fuzzball sped this up 2×," "VAST storage worked," or "We fully trained a 300M model" absent the corresponding measured evidence.

---

## 21. Source notes, discrepancies and ongoing verification register

| Source / direct URL | Source-backed finding | What remains to be verified on Team 12 machines |
|---|---|---|
| [CIQ official hackathon](https://ciq.com/events/local-ai-hackathon-2026) | Nov 13–15, Fri 17:00–Sun 16:00 CT; 3–5 per team; code written after kickoff; two GB10 Lenovo PGX systems; four tracks | Exact judge submission time and allocation after Sunday 09:00 |
| [UNESCO UNOSAT case study](https://ihp-wins.unesco.org/en/dataset/nepal-satellite-detected-mudflow-and-rockflow-extent-nepal) | Rasuwa/Nuwakot Aug 26–27 event; Sentinel-2 Aug 27 + PlanetScope Aug 26 in reference interpretation; dataset published Aug 28; separate affected/analysis/frame files and DOI | Actual downloaded SHPs; CRS, recalculated areas and licence. Published summary vs detailed resource approximate areas differ |
| [WHO Nepal event page](https://www.who.int/nepal/emergencies/2026-rasuwa-flash-floods) | Contemporary official account of flood event/mechanism | Exact cited article dates/figures to display, if used |
| [Official Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) | 446 hand-labeled 512×512 S2/S1 chips, label meanings, S2 13-band mapping, GCS source | Public bucket reachability, subset names, actual split counts and dataset licence requirements |
| [Prithvi flood model card](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11) and [inference script](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/blob/main/inference.py) | S2 L1C, 6 bands, one-time-step, included sample/config/weights and 0-based indices `[1,2,3,8,11,12]` | Resolved checkpoint revision and actual checksum, usable local/GB10 Python stack |
| [Dhara model card](https://huggingface.co/adityabavadekar/dhara-flood-unet) | ONNX SAR 6-channel 2-pre/1-post and σ0 dB; non-operational educational model | Whether matched dates/normalization and ONNX CUDA on ARM64 are possible |
| [Copernicus STAC](https://documentation.dataspace.copernicus.eu/APIs/STAC.html) | Public STAC endpoint, Sentinel-2 L1C listed; discovery possible | Actual Rasuwa items, exact asset download auth/URLs and data latency |
| [Microsoft Planetary Computer docs](https://planetarycomputer.microsoft.com/docs/) | Public STAC and data (some throttled), convenient EO fallback | L2A vs L1C mismatch and provider access/rights |
| [Geofabrik Nepal](https://download.geofabrik.de/asia/nepal.html) | Archive includes **nepal-260101.osm.pbf** pre-event; current OSM newer | Download, file schema, Jan snapshot completeness and derivative ODbL compliance |
| [Copernicus DEM AWS](https://registry.opendata.aws/copernicus-dem/) | GLO-30 public DSM, nominal 30 m | Exact AOI tile access, datum/voids, hydrological appropriateness |
| [Fuzzball docs](https://docs.ciq.com/fuzzball) and [NCCL](https://docs.ciq.com/fuzzball/parallel-and-distributed/nccl) | Workflows, task arrays, generic multinode, NCCL via PyTorch `torchrun`; public latest docs 4.4 | Compatibility with **Team12's observed v4.3.1**, actual GPU/storage/network/auth and validated Fuzzfiles |
| [Fuzzball 4.3.1 notes](https://docs.ciq.com/fuzzball/release-notes/v4.3.1) | Version-specific release info and storage-related changes | Current event deployment and actual usable syntax |
| [NVIDIA DGX Spark multi-node NCCL](https://build.nvidia.com/spark/nccl) | Two-node Blackwell NCCL is technically possible on properly configured hardware | **Not** proof CIQ-managed Lenovo PGX cluster exposes root/SSH/NCCL or same interconnect to teams |
| [PyTorch DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) / [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html) | Distributed gradient synchronization, ranks, rendezvous and launch protocols | Actual Fuzzball 4.3.1 launcher and usable network |
| [IBM PEFT-GEOFM](https://github.com/IBM/peft-geofm) | Official geospatial foundation-model parameter-efficient fine-tuning examples, Sen1Floods11 listed | Compatible exact config, dependencies, and opportunity after MVP |
| [Codex AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md) / [Claude CLAUDE.md](https://support.claude.com/en/articles/14553240-give-claude-context-claude-md-and-better-prompts) | Supported repo-level instructions for different agents | Each person's tool access and permissions |

### Critical unresolved questions, prioritized

1. **Can Team 12 access authorized Fuzzball services at 09:00 Sunday, and what are the real endpoints/node IDs?** Currently unverified.
2. **Can the Team 12 cluster run a CUDA-capable aarch64 PyTorch image with compatible Fuzzball v4.3.1 multinode launch and persistent storage?** Untested.
3. **Which actual 2026-08-27 Sentinel-2 L1C item can be downloaded and processed over the target AOI?** UNOSAT cites an observation, but a specific STAC ID has not been audited.
4. **Does the official Sen1Floods11 subset download and match local package/tool versions?** Listed, not yet executed.
5. **Do UNOSAT SHPs download cleanly and what are their actual polygon areas?** Their published rounded descriptions disagree; calculate from files.
6. **Does the team have a legitimate path to upload data/model assets to shared volume before benchmarks?** Must be confirmed.
7. **What is the judge's required final submission format/time?** Official event end is not necessarily submission cutoff.

---

## 22. The very next 30 minutes — no dependencies on Fuzzball

1. **P5:** create the single real GitHub repo; invite four individual accounts; commit `AGENTS.md`, `CLAUDE.md`, this runbook, `.gitignore`, folders/contracts, CI skeleton. Agree on five IDs and issue ownership.
2. **P1:** download the UNOSAT three-resource archive, confirm real SHPs and the correct coverage geometry; start **only** Sen1Floods11 *hand-labeled* S2+mask+splits intake. Start STAC scene discovery in parallel, not a whole-country download.
3. **P2:** install/test a small local ML environment and load one real 13-band/label pair; implement `ignore_index=-1` dataset test; then run upstream Prithvi sample if practical.
4. **P3:** confirm projected AOI and OSM Jan 2026 extract method, implement intersect/length/count tests on **clearly labeled synthetic fixtures**, then replace fixtures with real GIS data.
5. **P4:** finish local worker/benchmark scripts and exact CIQ infrastructure question list; no private network probing. P5 should integrate all owner PRs as soon as tests pass.

**The one outcome all five should optimize before Sunday morning:** a clean GitHub `main` branch that can produce **one georeferenced model-derived water raster and documented exposure from true data on a laptop**, plus the *same exact tile-worker code and manifests* ready to run on two GB10s. Then Sunday compute is an extension of a proven pipeline rather than a last-minute rewrite.

---

*End of runbook. All performance measurements, run IDs, source scene IDs, user accounts, node allocations, downloaded file hashes and final scientific quantities must be filled in only from real execution and audited data. Maintain a dated changelog of what was implemented/tested/blocked throughout the hackathon.*
