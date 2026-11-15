# Retrospective persistence evaluation — 15 November 2026

Implemented on `fix/p5-prediction-empty-state`: audited pre-event water-map export,
two local baseline prediction artifacts, real reference scoring, and explicit UI
labels. No trained flood-forecast model was created. Peer review requested before merge.

The exporter reads only the pre-event run. It preserves the raster grid, 0/1/255
classes and unknown coverage, checks the recorded raster SHA-256, and requires
known acquisition/publication times no later than the cutoff. It records the actual
export time rather than backdating issuance. Source imagery is optical Sentinel-2;
no SAR bands, discharge-to-depth conversion or post-event inputs are used.

## Exact execution commands

All four commands exited 0:

```bash
.venv/bin/python scripts/export_persistence_prediction.py --pre-run outputs/p3_prithvi_nepal_20260811_T45RUL --output outputs/pred_persistence_20260811_T45RUL --cutoff 2026-08-25T23:59:59Z
.venv/bin/python scripts/export_persistence_prediction.py --pre-run outputs/p3_prithvi_nepal_20260812_T45RUM --output outputs/pred_persistence_20260812_T45RUM --cutoff 2026-08-25T23:59:59Z
.venv/bin/python scripts/validate_prediction.py --prediction outputs/pred_persistence_20260811_T45RUL --reference outputs/p3_prithvi_nepal_20260827_T45RUL --baseline outputs/p3_prithvi_nepal_20260811_T45RUL --target new_water
.venv/bin/python scripts/validate_prediction.py --prediction outputs/pred_persistence_20260812_T45RUM --reference outputs/p3_prithvi_nepal_20260903_T45RUM --baseline outputs/p3_prithvi_nepal_20260812_T45RUM --target new_water
```

Both validators returned `status: scored`, `eligible: true`, `violations: []`.
Both warned `No event time given: lead time cannot be computed.` No documented
onset was supplied, so no early-warning claim is made.

| Prediction source | Later reference | Comparable pixels | Missed new-water pixels | IoU | Recall |
|---|---|---:|---:|---:|---:|
| Aug 11 T45RUL | Aug 27 T45RUL | 3,720,692 | 102,169 | 0 | 0 |
| Aug 12 T45RUM | Sep 3 T45RUM | 555,413 | 74,303 | 0 | 0 |

Both have zero hits and zero false alarms; false-alarm ratio is undefined.
The unchanged-water baseline cannot detect new inundation. These are separate
evaluations, not pooled event-peak coverage. Reference water is Prithvi-derived,
not field-verified truth. GeoTIFFs, agreement layers and full JSON results remain
in ignored `outputs/pred_persistence_*` directories, not Git.

## Validation

```text
$ .venv/bin/python -m pytest -q tests/test_persistence_export.py tests/test_prediction_validation.py tests/test_ui_apptest.py -k 'persistence or prediction or validation'
21 passed, 33 deselected in 1.92s
```

Unit fixtures remain temporary synthetic test data. Production exports reject
sources marked synthetic or lacking explicit non-synthetic provenance.

Streamlit AppTest also loaded the actual local outputs with
`PRAVAH_OUTPUTS_DIR=outputs` and selected each `pv_pred_run` in
`views/validation.py`. Both rendered without exceptions, showed
`Retrospective persistence baseline` and `Baseline evaluation`, and omitted
`No flood forecast supplied`. Existing Streamlit component deprecation warnings
were emitted. This checks local rendering, not remote deployment.

```text
$ .venv/bin/python -m ruff check scripts/export_persistence_prediction.py tests/test_persistence_export.py app/sections/validation.py
All checks passed!
$ git diff --check
(no output; exit 0)
```

## Remaining blockers

A learned future-inundation model, eligible historical predictor archive and
forecast training targets are not present in the inspected workspace. Current
river/rainfall outlook code is not an inundation model. The pre-event rasters were
segmented retrospectively; their provisional AOI lacks independently established
pre-event provenance. This baseline makes the evaluation pipeline usable but
does not resolve those scientific limitations or establish forecast skill.
