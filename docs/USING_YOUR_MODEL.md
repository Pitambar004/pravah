# Using your trained model

Your model is a small U-Net that finds water in Sentinel-2 images. These steps assume it is in
`outputs/models/<run>/` with a `checkpoint_best.pt`. It reads 6 bands (B2, B3, B4, B8A, B11, B12), or a 13-band
L1C image from which it picks those 6.

## 1. Put the scenes you want to map in the scenes folder

Copy each scene's GeoTIFF into `data/raw/sentinel2/`. For Rasuwa, copy the prepared images from the Fuzzball
volume at `/data/outputs/p3_audit_20261114T214402Z/s2_prepared_full/<scene>/s2_l1c_reflectance.tif`. Give each
copy a readable name such as `rasuwa_20260827_T45RUL.tif`.

Next to each image, copy the matching Prithvi run's `observed_metadata.json` under the same name, for example
`cp outputs/p3_prithvi_nepal_20260827_T45RUL/observed_metadata.json data/raw/sentinel2/rasuwa_20260827_T45RUL.json`.
This gives your run its satellite date, so the dashboard can place it before or after the event. Without it the
run is listed as "Undated".

After each run, `make model` also finds the buildings, roads and bridges inside your model's water map (from
`data/processed/osm/thematic/`). That is what the **Alerts** page uses.

## 2. Start the model service and the dashboard

Run each command in its own terminal:

```bash
make model        # loads the newest outputs/models/*/checkpoint_best.pt
make app          # dashboard
```

`make model` accepts these options:

| Option | Default | When to change it |
|---|---|---|
| `CHECKPOINT=...` | newest `outputs/models/*/checkpoint_best.pt` | to serve a different checkpoint |
| `INPUT_UNITS=` | `reflectance` | use `reflectance` for the Rasuwa prepared images (values about 0–1); use `dn` for raw L1C or Sen1Floods11 chips (values about 0–10000) |
| `DEVICE=` | `auto` | `mps` for the Mac GPU |

If you choose the wrong units, the run stops with an error instead of producing a wrong map.

## 3. Run it

1. Open **Live Prediction**.
2. Pick a scene and press **Run model**. A progress bar shows how many tiles are done.
3. When it finishes, the water map appears on the page. It is also saved in `outputs/<run_id>/` and shows on
   **Maps**.

## Without the dashboard

```bash
python -m pravah.inference --scene data/raw/sentinel2/rasuwa_20260827_T45RUL.tif \
  --checkpoint outputs/models/<run>/checkpoint_best.pt --output-dir outputs \
  --run-id unet_rasuwa_20260827_T45RUL --input-units reflectance --device mps \
  --source-metadata outputs/p3_prithvi_nepal_20260827_T45RUL/observed_metadata.json
```

`--source-metadata` copies the satellite acquisition dates from the matching Prithvi run, so **Maps** can place
your run before or after the event.

## Compare it with the Prithvi model on the same scene

```bash
python scripts/compare_models.py --run outputs/unet_rasuwa_20260827_T45RUL \
  --other outputs/p3_prithvi_nepal_20260827_T45RUL
```

The result is agreement between two models, not accuracy. There are no hand labels for Rasuwa, so neither map is
ground truth.

## On Fuzzball (GB10)

Use `workflows/fuzzball/06_inference_batch.yaml.tmpl` with `--set INPUT_UNITS=reflectance` and
`--set CHECKPOINT=/data/models/<run>/checkpoint_best.pt`. Copy the checkpoint to the volume first.
