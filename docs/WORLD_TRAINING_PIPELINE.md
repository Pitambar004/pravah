# World training pipeline: download → prepare → 1-node vs 2-node GB10 benchmark

**Status:** templates render and pass structure checks; the shell steps were exercised locally on synthetic
files. **Nothing here has run on Team 12 Fuzzball v4.3.1 yet.** Validate each rendered file with
`fuzzball workflow validate` (or the UI) and confirm the multinode block with a CIQ mentor before relying on it.

## What it trains on

| Dataset | Files | Size | Labels | Licence |
|---|---|---|---|---|
| WorldFloods v2 (revision `1f3faa29…`) | 2,547 | 102.0 GB | Curated from Copernicus EMS maps; not all hand-drawn | CC-BY-NC-4.0 (non-commercial) |
| Sen1Floods11 v1.1 | 21,409 | 36.3 GB | 446 hand-labelled + 4,384 automatic threshold labels | Unresolved; see `TRAINING_DATA_EXPANSION.md` |

The file list is pinned in `config/training/world_download_plan.json.gz`: public URLs, sizes and provider
checksums. Its SHA-256 is in the matching `.sha256` file, and the image refuses a plan that doesn't match.
Sentinel-2 L1C optical data only. Sentinel-1 files are kept but never mixed into this six-band model.

## Before you start

1. Build and push the image (ARM64, CUDA PyTorch for GB10) as described in `infra/docker/Dockerfile.api`. It
   now contains the download/prepare scripts and the pinned plan.
2. Run `workflows/fuzzball/00_gpu_smoke.yaml.tmpl` once and confirm `torch.cuda.is_available()` on a GB10.
3. Render all four workflows with one shared tag and identical training settings:

```sh
make world-pipeline API_IMAGE=docker://<registry>/pravah-api:<tag> EPOCHS=3 MAX_STEPS=0 GLOBAL_BATCH=8
# → build/fuzzball/world_<UTC tag>/{1_download_world,2_prepare_world,3a_train_1node,3b_train_2node}.yaml
```

## Run order

| # | Workflow | Hardware | Done when |
|---|---|---|---|
| 1 | `1_download_world.yaml` | 4 CPU, 8 GiB, ≤24 h | `audit/download_summary.json` shows no failures. Re-run to resume; files already verified are skipped. If some failed, re-render with `--set RETRY_FAILED=1` to repair only those. |
| 2 | `2_prepare_world.yaml` | 8 CPU, 16 GiB, ≤12 h | `manifests/optical_supervised.json` exists, the dry run prints its train/val/test counts, and sampled pairs load through `S2WaterDataset` |
| 3a | `3a_train_1node.yaml` | 1 × GB10 | `/data/outputs/training/world_1node_<tag>/training_summary.json` |
| 3b | `3b_train_2node.yaml` | 2 nodes × 1 GB10 | `/data/outputs/training/world_2node_<tag>/training_summary.json` |

All paths are under `/data/training/expanded_20261114/` (`WORLD_ROOT`), where an earlier intake already left
about a third of the WorldFloods files. Step 1 checks them instead of downloading them again. Budget: the plan
is ~138 GB, and the downloader refuses to start if plan + 80 GiB reserve exceeds the 500 GB volume budget.

## Reading the benchmark

Both training runs use the same manifest, global batch, epochs, step cap and seed. The dashboard's
**Training** page compares them only when every check passes: both completed, same global batch, same
manifest, and so on. Otherwise it shows the measured numbers side by side with the failing checks.

Evidence of a real 2-node run is **all** of the following:
- `training_summary.json` with `world_size: 2`
- two distinct hostnames
- `cuda_verified: true` on both ranks
- two distinct node IDs in the Fuzzball job logs

Report samples/second and wall time for both runs. Claim a speed-up only from these measured files.

Model quality is a separate question from speed. Score the chosen checkpoint on the hand-labelled
Sen1Floods11 test split and the Bolivia holdout, plus the WorldFloods test scenes, and compare it with
today's model (test IoU 0.819, Bolivia 0.780). Weak labels and WorldFloods do not guarantee better results.
None of this is flood forecasting: the model maps water in images that already exist.
