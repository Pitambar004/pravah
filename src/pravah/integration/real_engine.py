"""Real-data inference adapter; no synthetic source or untrained-weight fallback."""

import base64
import hashlib
import io
import json
import runpy
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from PIL import Image
from shapely.geometry import box, mapping

from pravah.inference.engine import InferenceEngine


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def metric_crs(bounds):
    west, south, east, north = bounds
    zone = min(60, max(1, int(((west + east) / 2 + 180) // 6) + 1))
    return f"EPSG:{(32600 if (south + north) / 2 >= 0 else 32700) + zone}"


class RealDataEngine(InferenceEngine):
    def __init__(self, settings, project, config):
        self.project = Path(project)
        self.catalog = json.loads(Path(config["catalog"]).read_text())
        self.evaluation = json.loads(Path(config["evaluation"]).read_text())
        if sha(settings.checkpoint) != self.evaluation["checkpoint_sha256"]:
            raise ValueError("Checkpoint must match the completed training evaluation")
        super().__init__(
            settings.checkpoint,
            device=settings.device,
            require_cuda=settings.require_cuda,
            batch_size=settings.batch_size,
        )
        torch.set_num_threads(4)

        def finite_output(module, args, output):
            if not torch.isfinite(output).all().item():
                raise RuntimeError("Nonfinite model output; result rejected")

        self.model.register_forward_hook(finite_output)

    def predict(self, raster_path, output_dir, **kwargs):
        started = time.perf_counter()
        scene = Path(raster_path)
        record = self.catalog.get(scene.name)
        if record is None or sha(scene) != record["image_sha256"]:
            raise ValueError(
                "Scene must match a verified optical input in the scene catalog"
            )
        with rasterio.open(scene) as src:
            values = src.read()
            if (
                src.count != 6
                or not np.isfinite(values).all()
                or src.crs.to_string() != "EPSG:4326"
            ):
                raise ValueError(
                    "Staged benchmark requires finite six-band geographic pixels"
                )
            bounds = list(src.bounds)
        # Match the completed model's 512px chip evaluation. Labels are not inputs.
        kwargs.update(tile_size=512, overlap=0)
        result = super().predict(raster_path, output_dir, **kwargs)
        output = Path(output_dir)
        region = record["country"]
        result.metadata.update(
            {
                "is_synthetic": False,
                "scene_identity": record,
                "case_study": {
                    "id": record["scene_id"],
                    "region": region,
                    "corridor": "Sen1Floods11 official test example",
                    "event_date": record["acquired_date"],
                    "scope": "benchmark",
                },
                "acquired_date": record["acquired_date"],
                "available_at_utc": None,
                "analysis_mode": "retrospective benchmark",
                "training_evaluation": self.evaluation,
                "software_sha256": {
                    name: sha(self.project / name)
                    for name in [
                        "src/pravah/integration/real_engine.py",
                        "src/pravah/inference/engine.py",
                        "src/pravah/ml/unet.py",
                        "src/pravah/gis/exposure.py",
                        "scripts/exposure.py",
                    ]
                },
                "cloud_policy": "Image validity only; reference invalid/cloud labels affect scoring, not inference. Cloud predictions are unvalidated.",
            }
        )
        footprint = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": mapping(box(*bounds)),
                    "properties": {
                        "label": "Input image footprint",
                        "source": record["source_url"],
                        "scope": "Raster coverage, not an authoritative flood affected extent",
                    },
                }
            ],
        }
        save(output / "image_footprint.geojson", footprint)
        rgb = np.moveaxis(np.clip(values[[2, 1, 0]].astype(float) / 3000, 0, 1), 0, -1)
        stream = io.BytesIO()
        Image.fromarray((rgb * 255).astype("uint8")).save(stream, format="PNG")
        (output / "source_rgb.png").write_bytes(stream.getvalue())
        result.metadata["scene_preview"] = {
            "data_url": "data:image/png;base64,"
            + base64.b64encode(stream.getvalue()).decode(),
            "bounds": [[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
            "display_only": "Fixed B4/B3/B2 DN/3000 stretch; model pixels unchanged",
        }
        crs = metric_crs(bounds)
        exposure = runpy.run_path(str(self.project / "scripts/exposure.py"))
        with rasterio.open(output / "observed_water.tif") as src:
            predicted = src.read(1)
            water = exposure["_polygonize"](predicted == 1, src.transform, src.crs)
            coverage = exposure["_polygonize"](predicted != 255, src.transform, src.crs)
        osm_dir = self.project / "data/raw/app_osm" / record["scene_id"]
        receipt_file = osm_dir / "receipt.json"
        receipt = (
            json.loads(receipt_file.read_text()) if receipt_file.exists() else None
        )
        vectors = {k: None for k in ["buildings", "roads", "bridges"]}
        if receipt:
            for item in receipt["layers"].values():
                if sha(item["path"]) != item["sha256"]:
                    raise ValueError("OSM layer integrity failed")
            for key in vectors:
                item = receipt["layers"][key]
                vectors[key] = gpd.read_file(item["path"])
            rivers = json.loads(Path(receipt["layers"]["rivers"]["path"]).read_text())
            rivers["pravah_layer"] = {
                "source": "OpenStreetMap snapshot " + receipt["snapshot_timestamp"],
                "note": receipt["scope"],
                "filter": "OSM waterway ways",
            }
            save(output / "rivers.geojson", rivers)
            save(output / "osm_provenance.json", receipt)
        summary = exposure["summarize_exposure"](
            water,
            vectors["buildings"],
            vectors["roads"],
            vectors["bridges"],
            projected_crs=crs,
            coverage=coverage,
        )
        summary.update(
            {
                "source_type": "model_inferred_water",
                "mask_artifact": "observed_water.tif",
                "valid_pixel_count": int((predicted != 255).sum()),
                "osm_snapshot_timestamp": receipt["snapshot_timestamp"]
                if receipt
                else None,
                "context_scope": receipt["scope"]
                if receipt
                else "OSM context not downloaded",
                "data_attribution": {
                    "credit": receipt["license"],
                    "url": receipt["source_url"],
                }
                if receipt
                else None,
                "unavailable_fields": [] if receipt else list(vectors),
            }
        )
        save(output / "exposure_summary.json", summary)
        exposure["export_feature_layers"](water, vectors, crs, output)
        result.metadata.update(water_area_m2=summary["water_area_m2"], metric_crs=crs)
        label = Path(record["label_path"])
        if sha(label) != record["label_sha256"]:
            raise ValueError("Reference label integrity failed")
        renderer = runpy.run_path(str(self.project / "scripts/render_local_results.py"))
        with rasterio.open(output / "water_probability.tif") as src:
            probabilities = src.read(1)
        scored = renderer["export_prediction"](
            scene,
            label,
            np.where(predicted == 1, 1, 0).astype("uint8"),
            output / "comparison",
            {
                "name": "Trained PRAVAH U-Net · live inference",
                "scene_id": record["scene_id"],
                "split": record["split"],
                "checkpoint_sha256": self.info.checkpoint_sha256,
            },
            np.clip(probabilities, 0, 1),
        )
        result.metadata["reference_evaluation"] = scored["metrics"]
        timings = result.metadata["timings_seconds"]
        timings["inference_and_raster_export"] = timings["end_to_end"]
        timings["end_to_end"] = round(time.perf_counter() - started, 4)
        result.metadata["timing_scope"] = (
            "Input verification, inference, raster/vector exports, exposure and reference "
            "comparison; excludes queue wait, model startup and final metadata write"
        )
        save(output / "observed_metadata.json", result.metadata)
        result.files = sorted(p.name for p in output.iterdir() if p.is_file())
        return result
