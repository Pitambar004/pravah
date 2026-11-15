# Local model fallback — 14 November 2026 CT

The local demo does **not** require copying the full Fuzzball volume. It downloads only the optical Sen1Floods11 hand-labeled subset from the original provider: 897 assets, **1,021,199,879 bytes** (446 S2 images, 446 labels, four official split lists and metadata). The 138 GB expansion remains a separate Fuzzball task; current Fuzzball connectivity is blocked by an API connection timeout.

All runtime data, environments, weights and results are ignored by Git under `data/raw/` and `outputs/local_demo/`. Changes are separate local wrappers on `feat/local-demo-fallback`; the existing P2 model, dataset loader, DDP trainer and inference modules are unchanged. P2/P4 review is required before merge; no merge has occurred.

## Status and artifacts

**The full local run completed at 21:02 CT on 14 November.** All 897 source assets (1,021,199,879 bytes) passed verification. Preparation retained **251 train / 86 validation / 88 test** images, plus **15 Bolivia holdouts**, excluding six all-invalid pairs. Twenty epochs / 5,020 optimizer steps completed on MPS in **386.28 seconds including final evaluation**. Validation selected epoch 13. Official test water precision **0.8776**, recall **0.9244**, F1 **0.9004**, IoU **0.8188**, over 20,443,939 valid pixels. Separate Bolivia holdout F1 **0.8766**, IoU **0.7803**, over 2,849,799 valid pixels. These are retrospective benchmark results, not Nepal forecast validation or a distributed-compute benchmark.

Full checkpoint: `outputs/local_demo/unet_mps_20261115T021950Z/checkpoint_best.pt`, SHA256 `9ad57b8ac40c52e635591bf22407fb17747a7000c718726c01147b02385b4a67`. Exact metrics and actual environment are beside it in `final_evaluation.json` and `run_environment.json`. Offline test examples are in `outputs/local_demo/unet_demo/{India_1018327,Spain_5650136,USA_1049586}/comparison.png`, selected alphabetically within the declared events before inference. The full-data model is now the default selected by the [Fuzzball morning launcher](FUZZBALL_MORNING.md); no full dataset transfer is needed to run it there.

Defensible demo wording: “We trained a six-band optical U-Net on 251 Sen1Floods11 images. After validation-only checkpoint selection, it reached water IoU 0.819 on 88 official test images and 0.780 on 15 separate Bolivia holdouts. We export georeferenced water masks. This is retrospective water segmentation, not a validated Nepal flood forecast.”

The following pilot is retained as development history and is superseded by the completed full run above.

**A real pilot has completed:** ten epochs / 920 optimizer steps on MPS in 70.88 seconds, using 92 training images. Validation-only selection chose epoch 4: precision **0.7410**, recall **0.7486**, F1 **0.7448**, water IoU **0.5934**, aggregated over 36 validation images and 8,911,119 valid pixels. These are model-selection scores on a partial dataset, not independent test performance. The checkpoint is `outputs/local_demo/unet_pilot/checkpoint_best.pt` (SHA256 `72c2f3c7686db31bdf21a833a23ac9669914745beb3c85fc43c770e5ebdb4189`).

Three real validation examples were exported to `outputs/local_demo/pilot_demo/`: `Ghana_124834`, `India_1018317`, and `USA_1039203`. The first alphabetical available validation scene from each of these events was fixed before inference. India and USA comparison figures were visually inspected; all exports passed CRS/transform/nodata checks. Ghana contains neither labeled nor predicted water, so its water metrics are correctly undefined. The India scene's high score is an example result, not the aggregate pilot score. These older files remain available offline.

Read `outputs/local_demo/pipeline_status.json` for the actual full-run phase. The pipeline waits for the complete successful transfer audit, prepares georeferenced pairs, trains 20 epochs on MPS, selects the best validation-IoU checkpoint, evaluates the official test and separate Bolivia holdout, and exports fixed test examples. A running/downloaded status is not a completed training result.

- Transfer audit: `outputs/local_demo/intake/transfers.jsonl` and, when complete, `download_summary.json`.
- Full-data preparation: `outputs/local_demo/data/readiness.json`, `pairs.json`, `bolivia_holdout.json`, `raster_qa.json`.
- Training log: `outputs/local_demo/training.log`; epoch metrics and checkpoints are in the `checkpoint_dir` recorded in `pipeline_status.json`.
- Final metrics, when produced: `<checkpoint_dir>/final_evaluation.json`.
- Offline figures and georeferenced masks, when produced: `outputs/local_demo/unet_demo/<scene_id>/`.
- Separate early diagnostic pilot: `outputs/local_demo/unet_pilot/` and `pilot_training.log`; its snapshot contains **92 training, 36 validation and 43 unused test pairs**. It trains for ten epochs and does not evaluate test/Bolivia. The snapshot is biased by download completion, so its validation score is not a full-benchmark result.

The older Fuzzball checkpoint in `outputs/fuzzball/benchmark_A_30b0be9b/` was only a 30-step run and reported validation F1/IoU zero. It is not the newly trained local model.

## Reproduce on this Mac

The checked machine is an Apple M4 with 16 GiB unified memory. The project environment has Python 3.14.2, PyTorch 2.15.0 and working Metal/MPS. A real training-input timing probe on `Ghana_24858` measured five warmed-up steps averaging 0.05011 seconds each at batch size one, six channels, 512×512. This is a compatibility/timing probe, not a full-run benchmark; its model was discarded. Receipt: `outputs/local_demo/mac_real_batch_probe.json`.

The following full pipeline was launched and completed; no rerun is needed for tomorrow's inference demo:

```bash
caffeinate -i .venv/bin/python -u scripts/run_local_fallback.py --epochs 20 \
  > outputs/local_demo/pipeline.log 2>&1
```

`caffeinate` prevents idle sleep only while that command runs. Keep the Mac powered and its lid open. Read progress with:

```bash
cat outputs/local_demo/pipeline_status.json
tail -n 8 outputs/local_demo/training.log
```

The pipeline uses the existing local pinned transfer plan and verified source files. For a later independent run after preparation, choose a fresh output directory:

```bash
.venv/bin/python scripts/train_local.py \
  --pairs outputs/local_demo/data/pairs.json \
  --bolivia outputs/local_demo/data/bolivia_holdout.json \
  --output-dir outputs/local_demo/unet_repeat \
  --device mps --epochs 20 --batch-size 1
```

The early pilot was launched with:

```bash
.venv/bin/python -u scripts/train_local.py \
  --pairs outputs/local_demo/pilot_data/pairs.json \
  --output-dir outputs/local_demo/unet_pilot \
  --device mps --epochs 10 --batch-size 1 --validation-only \
  > outputs/local_demo/pilot_training.log 2>&1

.venv/bin/python scripts/render_local_results.py \
  --pairs outputs/local_demo/pilot_data/pairs.json \
  --checkpoint outputs/local_demo/unet_pilot/checkpoint_best.pt \
  --output-dir outputs/local_demo/pilot_demo --device mps \
  --split val --events Ghana India USA \
  --model-name 'PRAVAH MiniUNet - partial-data pilot'
```

## Scientific contract

This is the existing 489,154-parameter MiniUNet, freshly trained locally, separate from Prithvi. Inputs are Sentinel-2 L1C `[B2,B3,B4,B8A,B11,B12]`, scaled by the project loader `/10000` and clipped to `[0,1]`. Training uses AdamW, learning rate 0.001, horizontal/vertical flips, and weighted cross entropy ignoring `-1`; the water weight is computed only from training-label pixel counts. Source pixels are neither resampled nor recolored for model input.

Official within-event splits are retained; this does not establish independent event/country transfer. Bolivia is held out separately. All-invalid pairs are excluded, and prepared labels also mask invalid source pixels. Model selection uses validation water IoU only; test results do not select checkpoints. Per-file source hashes, exact script/model/loader hashes, the underlying repository commit, configuration, metrics and environment are saved with each run.

Exported masks preserve CRS and pixel transform. `model_water.tif` uses `0=no water`, `1=model-inferred water`, `255=source-invalid`; dry is not nodata. A separate `reference_valid_mask.tif` identifies pixels eligible for evaluation. Reference cloud/invalid masks affect scoring and comparison figures only, never the imagery-only model input. Unvalidated cloud-covered predictions are not independent flood observations. Water includes permanent water.

`comparison.png` shows actual source RGB, reference, prediction and errors. Its fixed RGB display stretch is DN/3000 and does not alter model preprocessing. Examples are selected alphabetically within declared events/splits before predictions are inspected. These are retrospective benchmark examples, not Nepal case-study validation or disaster forecasts. Missing verified availability times do not become cutoff-eligible observations.

Source/provenance and licensing ambiguity for Sen1Floods11 are documented in `TRAINING_DATA_EXPANSION.md`; no permissive redistribution license is inferred.

## Optional published Prithvi reference

The [official IBM/NASA flood fine-tune](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11) completed reference inference at 21:41 CT on 14 November, at revision `91ce9d38086a80b078a192b374df758b8855b732`. Its downloaded 1,276,843,350-byte checkpoint passed SHA256 `76eed77d8bd543ae441308b80e8408502243a871cb8a338ddc8220ed96dfc270`. The official sample/config/code/card files are pinned and verified. Three provider examples are in the official test split; selected examples do not constitute a new independent benchmark.

An isolated Python 3.12 environment under `outputs/local_demo/runtime/prithvi-venv` preserves the project environment. The adapter calls the unmodified upstream preprocessing and inference functions, disables the redundant backbone pre-download, and requires strict loading of the complete fine-tuned checkpoint. The export separates dry/water/nodata instead of the upstream writer's ambiguous 0/255 mask. `outputs/local_demo/prithvi_demo/summary.json` reports `passed`, including strict checkpoint loading. Selected-example F1/IoU: India_900498 **0.9579/0.9192**, Spain_7370579 **0.8410/0.7256**, USA_430764 **0.8059/0.6750**. These are not the same examples as the live U-Net service, so their scores are not a controlled model comparison.

The first attempt exposed a TerraTorch/Lightning callback configuration conflict. The adapter now permits the CLI-injected checkpoint callback during construction; it invokes inference directly and never trains. The successful retry is recorded in `prithvi_inference_retry.log`, with the previous failure retained separately. `prithvi_status.json` now points to the completed result and `runtime/prithvi_environment.txt` records the installed versions. Exact successful command:

```bash
PYTHONPATH=src outputs/local_demo/runtime/prithvi-venv/bin/python -u scripts/infer_prithvi_local.py --device cpu
# STRICT_CHECKPOINT_LOAD_PASSED
# PRITHVI_EXPORTED India_900498 ...
# PRITHVI_EXPORTED Spain_7370579 ...
# PRITHVI_EXPORTED USA_430764 ...
# PRITHVI_FINISHED
```

This is an upstream pretrained reference, not a model trained by PRAVAH, and remains separate from the interactive U-Net application.

## Checks

```bash
.venv/bin/ruff check scripts/prepare_local_hand.py scripts/train_local.py \
  scripts/render_local_results.py scripts/run_local_fallback.py \
  scripts/download_pinned_checkpoint.py scripts/infer_prithvi_local.py \
  scripts/run_prithvi_when_ready.py \
  tests/test_local_training.py tests/test_local_result_export.py
.venv/bin/python -m pytest -q tests/test_local_training.py \
  tests/test_local_result_export.py tests/test_training_data_intake.py \
  tests/test_training_pair_preparation.py
# All checks passed!
# 20 passed in 0.85s (including the validation-only pilot option).
```

Synthetic test fixtures remain in temporary test directories and never enter the training manifests or presentation evidence. The completed run logs and actual exported figures provide the additional real-data checks.
