# Run the prepared U-Net on Fuzzball — 15 November 2026

For the working local dashboard with real inference, maps, exposure and Nepal environmental context, see [REAL_APP_HANDOFF.md](REAL_APP_HANDOFF.md). It is separate from the GPU batch below.

The launcher packages the completed local U-Net, its exact source code and one real six-band validation image. It uploads approximately 3.4 MB; **do not download or re-upload the full Fuzzball dataset**. Training mode reuses the original hand-labeled sources already stored on `Pravah-Dataset` and verifies their provider checksums before preparing remote paths.

## First: get one actual GPU inference result

On the authorized event network, from this Mac:

```bash
cd /Users/sachinshrestha/Projects/pravah
.venv/bin/python scripts/fuzzball_morning.py launch --mode inference
```

This checks the CLI/context, API connection and authentication, builds a fresh package, validates the generated workflow on the connected server, uploads it and submits one NVIDIA GPU job. It chooses the completed full-data checkpoint when available, otherwise the explicitly labeled completed pilot. It never chooses a checkpoint from a run still in progress. The job requires CUDA; it does not silently switch to CPU.

Inspect and collect the result:

```bash
.venv/bin/python scripts/fuzzball_morning.py status
.venv/bin/python scripts/fuzzball_morning.py logs
# After the workflow, including volume egress, finishes:
.venv/bin/python scripts/fuzzball_morning.py fetch
```

`logs` follows the job; Ctrl-C leaves the remote job running. `fetch` checks the downloaded archive checksum, extracts its actual GPU prediction and creates a comparison PNG from that prediction. The printed local results directory contains the mask, probability raster, scene metrics and actual device/environment record. It does not recompute the prediction locally.

The latest job ID and exact paths are in `outputs/fuzzball_morning/latest_launch.json`. Each run has a separate receipt and results directory. To inspect an earlier run, append `--receipt outputs/fuzzball_morning/<run_id>/receipt.json` to `status`, `logs` or `fetch`. An interrupted/uncertain submission is recorded; inspect the named workflow before retrying to avoid duplicate jobs.

## Optional: train a fresh model on the Fuzzball GPU

After the inference smoke test passes:

```bash
.venv/bin/python scripts/fuzzball_morning.py launch --mode train
```

This checks the bundled model first, verifies all 897 pinned hand-dataset assets on `/data/raw/sen1floods11/v1.1`, prepares new manifests within the run directory, then trains a fresh MiniUNet for 20 epochs. It preserves 251 training, 86 validation, 88 test images and 15 separate Bolivia holdouts. Six all-invalid source pairs are excluded. Checkpoint selection uses validation IoU; test and Bolivia are evaluated after selection. This is a **single-GPU** run, not distributed DDP or a scaling benchmark. It does not mix weak labels, SAR, terrain or rainfall into optical training.

Use the same `status`, `logs` and `fetch` commands. Training writes detailed progress to the persistent run's `results/training.log`; the followed job log shows the inference/preparation milestones and completion. The downloaded archive includes training checkpoints, configuration, metrics and provenance. Prepared VRTs inside remote training results reference the original Fuzzball rasters; they are not standalone local dataset copies.

## Storage and connection requirements

- Team 12 CLI **v4.3.1**, event endpoint `api.fb-hack.civilfritz.net`, authenticated group `team12@fb-hack.local`.
- Persistent volume `use: team12`, `name: Pravah-Dataset`; originals remain on this volume. Run outputs are `/data/pravah_ready/<run_id>/results`.
- One available NVIDIA GPU, 8 CPU cores and 16 GiB memory; image `nvcr.io/nvidia/pytorch:25.10-py3`, previously used on this deployment.
- Previously tested offline rasterio/affine wheels at `/data/training/expanded_20261114/environment/wheels`. Their `SHA256SUMS` is checked before installation. Missing cache/files stop the job with logs; nothing is silently downloaded.
- Object ingress and explicit result egress use the same unique prefix: `fb://group/pravah/morning/<run_id>/`. The upload has a three-day TTL. Fetch results promptly; the persistent copy also remains on the dataset volume.

To check connectivity without creating/submitting a workflow:

```bash
.venv/bin/python scripts/fuzzball_morning.py doctor
```

The event API is private. Ordinary internet access may not reach it; use the organizer-authorized network/access route and existing Fuzzball login. A fresh read-only validation attempt on 14 November CT timed out after 12 seconds. Receipt: `outputs/fuzzball_morning/connection_attempt.json`. **These new workflows have not yet passed server validation or executed on the GPU.** The launcher performs server validation before upload/submission tomorrow and stops if validation fails. Earlier successful workflows do not prove deployment of this new package.

## Local verification and scientific scope

`prepare --mode inference` or `prepare --mode train` builds packages offline without submitting them. The final package verification receipt is `outputs/fuzzball_morning/package_verification.json`; it records actual checks and their commands. Unit-test fixtures are synthetic and never enter real training or demonstration results.

Verified locally: the final 3,404,040-byte inference bundle was unpacked into a separate directory, its hashes passed, CPU inference completed, CUDA mode correctly refused to run on this Mac, all 897 resident local sources passed the same preparation function used by the remote worker, and a one-epoch portability check trained on two real training images with one real validation image without evaluating test data. Both generated job scripts passed `/bin/sh -n`. The tiny portability check is execution evidence, not model-quality evidence. The full Mac run's independent metrics are documented in the local handoff.

Exact final code checks:

```bash
.venv/bin/ruff check scripts/fuzzball_morning.py scripts/fuzzball_run.py \
  scripts/train_local.py tests/test_fuzzball_morning.py
# All checks passed!
.venv/bin/python -m pytest -q tests/test_fuzzball_morning.py \
  tests/test_local_training.py tests/test_local_result_export.py \
  tests/test_training_data_intake.py tests/test_training_pair_preparation.py \
  tests/test_user_context.py
# 32 passed in 0.63s
```

The bundled example is fixed official-validation scene `India_1018317`, selected before this run. Its label is used only for scoring and display, never as model input. One image's score is not aggregate test performance. Inputs are Sentinel-2 L1C `[B2,B3,B4,B8A,B11,B12]`, scaled `/10000` and clipped `[0,1]`. Masks preserve CRS/transform and encode `0=no water`, `1=model-inferred water`, `255=invalid source`. Probabilities use nodata `-9999`. Water includes permanent water. These are retrospective benchmark outputs, not validated Nepal flood forecasts.

Code is on feature branch `feat/local-demo-fallback`; existing P2 model/loader/DDP modules are unchanged. P2/P4 peer review is required before merge. No merge or remote submission was performed to prepare this handoff. See [local results and reproduction](LOCAL_DEMO_HANDOFF.md) and [dataset provenance](TRAINING_DATA_EXPANSION.md).
