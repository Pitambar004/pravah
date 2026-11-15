# Training data expansion — 14 November 2026

This is a P1 data-intake handoff for the existing six-band **small U-Net**. It does not train or validate Prithvi, establish flood-forecast skill, or change the Rasuwa replay cutoff rules. Work began on `feat/p1-training-data-expansion` and is preserved on `feat/local-demo-fallback`; P2/P4 review is required before merge or adopting the training configuration.

**Connectivity update, 14 November evening CT:** Fuzzball's API is timing out from this Mac. The remote states below are the last observed states, not confirmation that the full intake is still running or finished. The verified quick dataset remains the last successful readiness result. A separate local fallback is documented in `LOCAL_DEMO_HANDOFF.md`.

## Ready now — five-minute handoff

The urgent Sen1Floods11 expansion is **ready and checked** as of 2026-11-14 23:56:50 UTC. Its manifest is `/data/training/expanded_20261114/manifests/quick_sen1_with_weak.json`: **4,409 train, 86 validation, 88 test pairs**, with 15 Bolivia chips separately reserved. Training includes 251 hand-labeled and **4,158 additional automatically threshold-labeled** examples. Validation/test remain hand-labeled. Eight pairs were excluded for insufficient valid pixels after checking both imagery and labels; the whole Bolivia weak event (224 chips) is withheld.

All 21,409 selected Sen1Floods11 source files (36,306,011,419 bytes) are integrity-verified. Every included quick pair passed raster/grid/class/finite-pixel checks. Invalid image pixels are ignored in corrected labels where needed. The unchanged dataset loader and U-Net CPU forward pass passed on real weak data, and the existing training CLI passed its `--dry-run` (4,409/86/88). The quick manifest retains original 13-band imagery, which the existing loader selects into the required six channels. It is a separate Sen1 experiment; WorldFloods is not included yet.

Start training using the validated local workflow (3 epochs, no step cap):

```bash
fuzzball workflow start outputs/training_expansion_20261114/train_quick_sen1.yaml --name pravah-unet-sen1-expanded
```

The same workflow is stored inside Fuzzball at `/data/training/expanded_20261114/operations/train_quick_sen1.yaml`. Training has **not** been launched. Report: `audit/quick_training_readiness.json` on the volume; local copy: `outputs/training_expansion_20261114/quick_readiness_audit.json`. The queued urgent workflow `ec8a40fe-74d1-4f53-9507-5e479f6ac2b4` was canceled; the checks actually ran through `workflow exec` in the existing `3d7b51fc-cab0-43e4-9cdf-a7a8025ed6a4` container to avoid the queue. Its script is retained at `operations/quick_ready.py`.

The full WorldFloods transfer and combined preparation were configured to continue automatically under that existing workflow. Their current remote state cannot be checked during the connectivity outage. The full-library counts and readiness sections below remain unconfirmed until its final audit can be retrieved.

## Current status

The full intake was last observed running on Team 12 Fuzzball v4.3.1, workflow `3d7b51fc-cab0-43e4-9cdf-a7a8025ed6a4`, job `complete-intake`. Full download/preparation counts will be recorded after the final audit. The real-data pilot has already passed on Fuzzball; this remote intake did not launch model training.

Everything needed for these training datasets is being stored on persistent volume **Pravah-Dataset**:

```text
/data/raw/sen1floods11/v1.1/                  original provider files
/data/training/expanded_20261114/
  download_plan.json                       pinned assets, sizes, checksums
  worldfloods_v2/raw/{train,val,test}/       original WorldFloods resources
  prepared/                                six-band VRTs and binary label TIFFs
  manifests/                               separate supervised/weak/holdout lists
  code/                                    intake and existing U-Net source
  environment/wheels/                      tested rasterio/affine wheels + hashes
  operations/train_supervised.yaml         validated future GPU training workflow
  audit/                                   transfers, provenance and QA receipts
```

Raw training images are downloaded directly inside Fuzzball. Local `outputs/training_expansion_20261114/` contains only plans, workflow definitions and audit exports; it is ignored by Git. VRTs depend on the original rasters at their recorded absolute volume paths. Preserve both directories together.

## Project progress and scope decision

Read `README.md`, `IMPLEMENTATION_RUNBOOK.md`, `contracts.md` and `P5_INTEGRATION_HANDOFF.md`. The README still describes a starter; the UI handoff documents frontend work, not scientific model validation. The runbook's P1/P2 baseline-training path and its primary Prithvi inference path remain separate.

Read-only live inventory workflow `1e95e93e-d8dd-4018-9390-c3736176f469` found:

- All 446 official Sen1Floods11 hand-labeled S2/S1/label files already existed; the previous validated baseline manifest had 426 usable non-Bolivia pairs: 251 train, 86 validation, 89 test. Fifteen Bolivia chips are a separate holdout; this intake also reserves the bounding region of all 224 original Bolivia weak images.
- Nepal Sentinel-2 L1C imagery, Sentinel-1 imagery, terrain, hydrographic and weather resources already existed. These contextual inputs are not interchangeable with supervised optical water masks.
- The earlier small U-Net workflow `30b0be9b-3021-4138-972d-ac421698abf4` wrote `/data/outputs/benchmark/A`. That checkpoint was a 30-step run with reported validation F1/IoU of zero. Downloading more data does not by itself establish model quality; P2 still needs a proper training/evaluation run.
- Other Prithvi workflows exist in the cluster. This intake does not certify their model outputs, GIS exposure, replay eligibility or benchmark claims.

The expansion adds compatible global optical examples and the rest of Sen1Floods11's original resources. It keeps SAR inputs separate. RGB aerial imagery, building-damage datasets and unrelated weather/terrain grids are not silently mixed into the six-channel water-segmentation task.

## Provider inventory and rights

These are **planned pinned provider totals**, not a claim that every raster is usable:

| Source | Selected files | Bytes | Optical source pairs |
|---|---:|---:|---:|
| WorldFloods v2 | 2,547 | 102,001,424,857 | 509 scenes: 475 train, 16 validation, 18 test |
| Sen1Floods11 v1.1 | 21,409 | 36,306,011,419 | 446 hand-labeled plus 4,384 weak S2 images |

The combined provider inventory is 138,307,436,276 bytes (138.31 decimal GB), including already-present files. The initial plan estimated 136,556,754,554 new bytes. The 500,000,000,000-byte volume budget reserves 80 GiB for other work. The shared filesystem's reported free space is not treated as the team's quota.

- **WorldFloods:** [official hosted release](https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2), revision `1f3faa2989e69930ac31d5c0a79fd224461123f8`; CC-BY-NC-4.0, noncommercial use with attribution. Preserve the downloaded dataset card. Large files are verified against provider LFS SHA256; small repository files against Git blob SHA1, with an additional local SHA256 for every file. The [dataset paper](https://www.nature.com/articles/s41598-023-47595-7) confirms Sentinel-2 L1C and describes the curated/derived water references. Do not describe all labels as independent manual ground truth. Upstream `spaceml-org/ml4floods` code was checked at `192426c7286315b59a2f52c7a9cdebe589242de2`.
- **Sen1Floods11:** [official repository](https://github.com/cloudtostreet/Sen1Floods11) checked at `b9e460dabe70230111a3c3d36cd0c5ce03606ee9`; each GCS object is pinned by generation and verified by provider MD5 plus recorded SHA256. The official repository offers the data publicly for training/testing, but its STAC collections say `proprietary` and no explicit repository LICENSE was found. The audit records this ambiguity; no permissive redistribution license is inferred.
- The actual weak-label inventory contains 4,384 S2 images and 4,385 S2 index-label files, despite rounded/older prose counts. The unmatched label is reported and cannot become a training pair.
- Sen1Floods11 source names cover Bolivia, Colombia, Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri Lanka and the USA. Bolivia is withheld from training. Metadata country/date fields are retained only where the provider event name matches; the Mekong filename versus Cambodia metadata naming discrepancy is left explicit rather than guessed.
- S1 weak images, S1 Otsu labels, permanent-water resources and JRC references are retained as separate source resources. Their download does not make them compatible with the optical U-Net.

All assets are for retrospective training. Missing verified availability times remain null and are ineligible for earlier-cutoff replay. Sen1Floods11 acquisition dates retain the provider's date precision; no midnight timestamp is invented. WorldFloods country metadata is mostly absent, so country names are not guessed from filenames.

## Training contract and split protection

`scripts/download_training_prepare.py` checks CRS, dimensions, class values, band order and pixel validity before publishing pairs. Source imagery is unchanged. The output is a 512×512 six-band VRT in `[B2,B3,B4,B8A,B11,B12]` order plus an int16 label GeoTIFF, nodata `-1`. Edge images are zero-padded and their unobserved labels remain `-1`.

WorldFloods exports inspected on Fuzzball have 15 bands: the 13 Sentinel-2 bands followed by QA60 and cloud probability. One-based source indices `[2,3,4,9,12,13]` are selected. The two-band reference uses first-band `0/1/2 = invalid/clear/cloud` and second-band `0/1/2 = invalid/land/water`. Only clear land/water pixels become `0/1`; all other pixels become `-1`. Water includes permanent water. This is not a flood-damage class.

Sen1Floods11 hand/weak labels retain `-1/0/1`. A measured source-coordinate rounding difference of about `5.3e-12` image pixels is accepted; the maximum permitted displacement at any of the four raster corners is `1e-6` pixel. Larger grid mismatches are rejected. Nonfinite source pixels cause rejection because ignoring their labels alone would not prevent NaNs propagating through convolutions. No resampling occurs. CRS is preserved per scene, including geographic Sen1Floods11 and projected WorldFloods grids.

The existing baseline loader divides L1C values by 10,000 and clips to `[0,1]`. This is the existing small U-Net preprocessing, not a claim about Prithvi normalization. WorldFloods/weak tiles require at least 5% valid pixels in the 512×512 tile; hand-labeled tiles require at least one valid pixel to retain the official usable baseline.

Manifests:

- `sen1_hand_official.json`: usable hand-labeled examples with original non-Bolivia split assignments.
- `bolivia_holdout.json`: usable Bolivia examples, independent test only.
- `worldfloods_official.json`: usable WorldFloods tiles with their published scene split assignments.
- `optical_supervised.json`: recommended initial combined training input. Published assignments are retained; training tiles from held-out WorldFloods EMS events or intersecting any validation/test/Bolivia source bounding box are removed. Validation tiles from test EMS events or intersecting test/Bolivia boxes are also removed. The Bolivia protection includes the bounding region of all original hand and weak images.
- `weak_train_only.json`: automatically threshold-labeled optical pairs, training only. The entire Bolivia weak event and all protected validation/test geographies are excluded.
- `optical_with_weak.json`: combined supervised plus opt-in weak training pairs; validation/test remain supervised.

Metadata inspection found 11 WorldFloods EMS event IDs shared by the published training and validation sets: `EMSR260`, `EMSR271`, `EMSR274`, `EMSR279`, `EMSR280`, `EMSR293`, `EMSR312`, `EMSR329`, `EMSR358`, `EMSR427`, `EMSR442`. Their training scenes are excluded from the combined manifest. No EMS event IDs were shared with the published test set.

The bounding-box screen is conservative, time-independent and geographic. The official Sen1Floods11 hand protocol retains within-event splits, and cross-provider event identity is not fully resolved; complete event independence is not claimed. The official manifests preserve provider splits and may have geographic overlaps. Use the screened combined manifests for mixed-source experiments. Quarantined, insufficiently valid and unpaired sources remain in the audit rather than being mislabeled as usable examples.

## Validation and reproducibility

Actual Fuzzball sample QA:

- `efaf4f2f-ff98-4a45-b745-cad2a5dbbcb8`: inspected real raster bands, dtypes and label values using the cached PyTorch container plus rasterio 1.5.2.
- `01bdd261-a7b0-45da-a7b8-c6297daec6f3`: first preparation pilot found the source's tiny affine rounding difference and stopped.
- `f32d5ef3-af19-48e2-8cfb-9da3d10817fa`: measured the difference instead of assuming alignment.
- `05047aec-555a-433e-8ad2-f7eb527c44d6`: a fresh container installed rasterio 1.5.2 and affine 3.0.1 using only the cached, checksum-verified wheels (`--no-index --no-deps`) and loaded a real `[6,512,512]` sample. Actual environment: Python 3.12.3, PyTorch `2.9.0a0+145a3a7bda.nv25.10`, NumPy 2.1.0; CPU only.
- `6c29b981-88ce-47a9-80a3-359c26b537bd`: corrected pilot passed on three WorldFloods scenes and one Ghana weak pair; eight tiles were created. Existing `S2WaterDataset` and `MiniUNet` ran a CPU forward pass with input `[2,6,512,512]`, labels `[2,512,512]`, output `[2,2,512,512]` and finite tensors. No optimizer step or GPU-training claim was made.

The initial download run `76a6d159-4c2b-4ad8-bce3-04e7eb406b65` was deliberately stopped to increase transfer workers from three to eight and fix URL-encoding of provider filenames containing spaces. Its transfer audit is archived before resuming. Existing files are checksum-verified and reused; corrupt existing inputs are never silently replaced. Failed transfers do not publish partial destination files. Retries preserve the earlier audit and append repaired records.

Other setup failures retained in workflow history: an oversized inline asset plan exceeded Fuzzball's NATS message limit (resolved by object-cache ingress); Python 3.11 slim could not use rasterio 1.5.2, and rasterio 1.4.4 lacked a native library there (resolved using the previously verified PyTorch/Python 3.12 container). These failures are not successful deployment evidence.

Exact local checks:

```bash
.venv/bin/ruff check scripts/download_training_datasets.py scripts/download_training_prepare.py tests/test_training_data_intake.py tests/test_training_pair_preparation.py
# All checks passed!
.venv/bin/python -m pytest -q tests/test_training_data_intake.py tests/test_training_pair_preparation.py
# 14 passed in 0.21s```

Tests cover synthetic transfer integrity, path escape, truncated downloads, preserving corrupt originals, provider URL spaces, audited retry, class semantics, spatial overlap, held-out event protection, band selection/padding, realignment rejection and numerical grid rounding. Synthetic fixtures remain local temporary test files and never enter Fuzzball training datasets.

```bash
fuzzball workflow validate outputs/training_expansion_20261114/complete_intake.yaml
# "outputs/training_expansion_20261114/complete_intake.yaml" has been validated.
fuzzball workflow validate outputs/training_expansion_20261114/train_supervised.yaml
# "outputs/training_expansion_20261114/train_supervised.yaml" has been validated.
```

A source-only compatibility update was applied before full preparation began: grid displacement uses explicit affine arithmetic supported by affine 2.x and 3.x. The previous source and receipt are preserved under `audit/source_update_affine_compatibility/`; the updated file SHA256 is `ecc0d2b99ab6b926d100bd1da6cba46de3a4eba14a0d7fc48df3e71ee71144f7`.

A second update excluded the observed shared EMS events and widened the Bolivia protected region before preparation began. Its source SHA256 is `9c507bece47123cd8e628f506e579475c85aa7bafc4f1d5fb092593d36117aed`; the previous version is retained under `audit/source_update_event_protection/`.

The final source also rejects nonfinite imagery before publishing a pair. Its SHA256 is `d338dbdf376046886ad67e91b565f7f1eb59834bd88f54ff09a4846142bdfc9f`; previous source and receipt are preserved under `audit/source_update_finite_inputs/`.

The source bundle records local repository revision `9081f45bbfb298d30caac5afa1ff7d1d84c7a937` and individual file SHA256 values. New intake scripts are uncommitted additions, explicitly identified as such. Existing P2 model/loader/trainer modules are copied verbatim, not modified.

The full pipeline caches the tested aarch64/Python 3.12 rasterio wheel and affine wheel on the volume, records their hashes and the actual environment, verifies every published manifest path, exercises real batches for each source family, and runs the existing trainer's `--dry-run` against four manifests. Its final audit is exported to `fb://group/pravah/training/expanded_20261114/audit/`; that export excludes raw rasters and model checkpoints.

## P2/P4 next action

After `audit/final_validation.json` says `passed`, review the manifests and training template, then run the supervised baseline first. The prepared GPU template uses the successfully deployed Team 12 image/resource syntax, requests one GPU, runs 20 epochs without the earlier 30-step cap, and creates a fresh timestamped run directory. It has been schema-validated, **not executed** by this intake.

```bash
fuzzball workflow start outputs/training_expansion_20261114/train_supervised.yaml \
  --name pravah-unet-expanded-supervised
```

Record per-source validation metrics and class balance, inspect prediction examples, and evaluate the fixed test/Bolivia holdouts only after choosing the experiment. Compare weak-label training as a separate experiment. Keep all new results retrospective; training-data volume is not evidence of accuracy, Nepal transfer performance, disaster forecasting or distributed speedup.

P2/P4 peer review remains required before merge. No merge, push or teammate message is part of this intake.
