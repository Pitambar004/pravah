# Prediction contract: how to get a model's flood forecast scored

Use this when a forecast model (for example, one that uses rainfall or weather data) needs to be checked against
what satellites observed after the Rasuwa event. The dashboard page **Prediction vs Reality** reads the output of
`scripts/validate_prediction.py`. It never computes or estimates a score itself.

## 1. What the model must write

Write one folder per prediction under the outputs directory, for example `outputs/pred_rasuwa_<model>_<date>/`.
It holds:

| File | Required | Content |
|---|---|---|
| `predicted_flood.tif` | one of the two | uint8 GeoTIFF with a CRS: `0` = no flood, `1` = flood predicted, `255` = no prediction / outside the model domain |
| `predicted_flood_probability.tif` | one of the two | float GeoTIFF of probabilities in `0..1`. Set nodata or use NaN for "no prediction". Pass `--threshold` when you validate. |
| `prediction_metadata.json` | yes | see below |

The prediction can use any grid or CRS. The validator resamples it to each 10 m reference grid with
nearest-neighbour, which does not add detail. A coarse grid such as 1 km weather cells scores as blocks.

### `prediction_metadata.json`

```json
{
  "run_id": "pred_rasuwa_v1",
  "model_id": "team-flood-forecast v1 (git <sha>, checkpoint sha256 <...>)",
  "description": "One line: what the model predicts and at what resolution",
  "input_cutoff_utc": "2026-08-25T00:00:00Z",
  "issued_at_utc": "2026-08-25T00:30:00Z",
  "inputs": [
    {"name": "GPM IMERG Early 30-min rainfall", "type": "rainfall", "source": "<provider URL>",
     "acquired_at_utc": "2026-08-24T23:30:00Z", "available_at_utc": "2026-08-25T03:30:00Z"},
    {"name": "Sentinel-2 L1C pre-event", "type": "imagery", "scene_id": "S2A_MSIL1C_20260811T045241_...",
     "acquired_at_utc": "2026-08-11T04:52:41Z", "available_at_utc": "2026-08-11T08:30:07Z"}
  ]
}
```

The values above are only an example of the format. Record the real values for every input.

- **Every** input needs `acquired_at_utc` (or `valid_to_utc` for a forecast field) and `available_at_utc`. This is
  the time the data could actually be downloaded, for example a provider's publication time.
- `available_at_utc` is the field that most often leaks information. Near-real-time rainfall products are
  published hours after they are observed. If an availability time is unknown, the prediction is not eligible.
- Do not list post-event imagery, UNOSAT maps or reports as inputs, and do not use them in any other way.

## 2. Score it

```bash
python scripts/validate_prediction.py \
  --prediction outputs/pred_rasuwa_v1 \
  --reference outputs/p3_prithvi_nepal_20260827_T45RUL --baseline outputs/p3_prithvi_nepal_20260811_T45RUL \
  --reference outputs/p3_prithvi_nepal_20260903_T45RUM --baseline outputs/p3_prithvi_nepal_20260812_T45RUM \
  --target new_water --event-time <documented onset, ISO-8601 UTC>
```

- `--target new_water` asks whether the model predicted where water appeared that was not there before. This is
  the fair test of a flood forecast, because already-wet river pixels are excluded.
- `--target water_extent` scores all water, rivers included. That scoring is easier and inflated by permanent rivers.
- `--event-time` should be the documented onset. Do not leave the event day start as a placeholder.

The exit code tells you the outcome:

| Exit code | Meaning |
|---|---|
| `0` | Scored. |
| `2` | Bad input. |
| `3` | **Rejected**: an input or the reference breaks the cutoff, so no score is produced. |

## 3. What the scores mean and do not mean

- **CSI / IoU, POD (recall) and FAR** measure agreement with water that the Prithvi model mapped from cloud-free
  Sentinel-2 pixels. They do not measure agreement with field-verified ground truth.
- The **persistence baseline** is "the pre-event map is the forecast". A model shows skill only if it beats this
  baseline on the same pixels.
- One event is one sample. Good agreement at Rasuwa is evidence, but it does not prove the model works everywhere.
  Repeat the scoring on other events before claiming that it generalises.
- Rasuwa's Aug 26 event was largely a debris and mudflow. Sediment-laden flow can be spectrally unlike clear water,
  so optical water maps can miss it, which shows up as false alarms for the forecast.
