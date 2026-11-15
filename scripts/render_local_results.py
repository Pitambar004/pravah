#!/usr/bin/env python3
"""Export actual model predictions as georeferenced rasters and offline figures."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw, ImageFont

from pravah.ml.dataset import S2_INDEXES
from pravah.ml.metrics import confusion, score
from pravah.ml.unet import MiniUNet


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size):
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def export_prediction(
    image_path, label_path, prediction, output_dir, identity, probability=None
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(image_path) as src:
        if src.count not in (6, 13) or src.crs is None:
            raise ValueError("Expected georeferenced 6/13-band optical raster")
        channels = src.read(
            [i + 1 for i in S2_INDEXES] if src.count == 13 else list(range(1, 7))
        )
        source_valid = np.isfinite(channels).all(axis=0) & (channels != 0).any(axis=0)
        source_valid &= (
            src.read_masks(
                [i + 1 for i in S2_INDEXES] if src.count == 13 else list(range(1, 7))
            )
            != 0
        ).all(axis=0)
        profile = {
            "driver": "GTiff",
            "width": src.width,
            "height": src.height,
            "count": 1,
            "crs": src.crs,
            "transform": src.transform,
            "compress": "deflate",
        }
        spatial = {
            "crs": str(src.crs),
            "transform": list(src.transform),
            "bounds": list(src.bounds),
        }
    with rasterio.open(label_path) as src:
        truth = src.read(1)
        if not np.isin(truth, (-1, 0, 1)).all():
            raise ValueError("Unsupported reference classes")
        if src.crs != profile["crs"] or not np.allclose(
            list(src.transform), list(profile["transform"]), rtol=0, atol=1e-10
        ):
            raise ValueError("Reference and image grids differ")
    if prediction.shape != truth.shape or not np.isin(prediction, (0, 1)).all():
        raise ValueError("Invalid prediction shape/classes")
    reference_valid = source_valid & (truth != -1)
    evaluation_truth = np.where(reference_valid, truth, -1)
    counts = confusion(evaluation_truth, prediction)
    metrics = score(counts)
    water = np.where(source_valid, prediction, 255).astype(np.uint8)
    with rasterio.open(
        output_dir / "model_water.tif", "w", **profile, dtype="uint8", nodata=255
    ) as dest:
        dest.write(water, 1)
    with rasterio.open(
        output_dir / "reference_valid_mask.tif",
        "w",
        **profile,
        dtype="uint8",
        nodata=None,
    ) as dest:
        dest.write(reference_valid.astype(np.uint8), 1)
    if probability is not None:
        if probability.shape != prediction.shape:
            raise ValueError("Probability and prediction shapes differ")
        if (
            not np.isfinite(probability).all()
            or not ((probability >= 0) & (probability <= 1)).all()
        ):
            raise ValueError("Invalid probability values")
        with rasterio.open(
            output_dir / "water_probability.tif",
            "w",
            **profile,
            dtype="float32",
            nodata=-9999,
        ) as dest:
            dest.write(np.where(source_valid, probability, -9999).astype(np.float32), 1)
    # Fixed display stretch only. This does not change model preprocessing.
    rgb = np.moveaxis(
        np.clip(channels[[2, 1, 0]].astype(np.float32) / 3000, 0, 1), 0, -1
    )
    rgb = (rgb * 255).astype(np.uint8)
    rgb[~source_valid] = [48, 53, 61]

    def overlay(mask):
        displayed = rgb.copy()
        displayed[mask & reference_valid] = (
            0.35 * displayed[mask & reference_valid] + 0.65 * np.array([0, 188, 230])
        ).astype(np.uint8)
        displayed[~reference_valid] = [65, 71, 80]
        return displayed

    error = np.full_like(rgb, [226, 232, 238])
    error[(prediction == 1) & (truth == 1)] = [24, 163, 112]
    error[(prediction == 1) & (truth == 0)] = [230, 102, 48]
    error[(prediction == 0) & (truth == 1)] = [171, 68, 174]
    error[~reference_valid] = [65, 71, 80]
    panels = [rgb, overlay(truth == 1), overlay(prediction == 1), error]
    labels = [
        "SENTINEL-2 RGB",
        "HAND-LABELED REFERENCE",
        "MODEL WATER PREDICTION",
        "PREDICTION / REFERENCE",
    ]
    canvas = Image.new("RGB", (1176, 1375), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (32, 25),
        "PRAVAH  |  Retrospective water segmentation",
        font=font(28),
        fill="#102239",
    )
    draw.text((32, 65), identity["name"], font=font(22), fill="#284b70")
    draw.text(
        (32, 99),
        identity["scene_id"]
        + "  |  "
        + identity.get("split", "official provider example"),
        font=font(18),
        fill="#42576b",
    )
    for i, (panel, title) in enumerate(zip(panels, labels, strict=True)):
        x, y = 32 + (i % 2) * 572, 145 + (i // 2) * 572
        draw.text((x, y), title, font=font(17), fill="#203a54")
        canvas.paste(Image.fromarray(panel).resize((540, 540)), (x, y + 27))
    text_metrics = " | ".join(
        f"{k.upper()} {v:.3f}" if v is not None else f"{k.upper()} unavailable"
        for k, v in metrics.items()
    )
    draw.text(
        (32, 1293), text_metrics + "  (this image only)", font=font(17), fill="#14334e"
    )
    draw.text(
        (32, 1321),
        "Green: correct water   Orange: false water   Purple: missed water   Gray: reference-invalid/cloud",
        font=font(15),
        fill="#42576b",
    )
    draw.text(
        (32, 1347),
        "Benchmark data: Sen1Floods11. Imagery-only model input. Not a forecast or a Nepal case-study result.",
        font=font(15),
        fill="#42576b",
    )
    canvas.save(output_dir / "comparison.png")
    report = {
        "identity": identity,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "source_image_sha256": file_hash(image_path),
        "source_label_sha256": file_hash(label_path),
        "counts": counts,
        "metrics": metrics,
        "spatial": spatial,
        "model_water_class_codes": {
            "0": "no_water",
            "1": "model_inferred_water",
            "255": "source_invalid",
        },
        "validity_policy": "GeoTIFF uses image validity; reference cloud/invalid mask used only for evaluation and comparison figure, not model inputs. Cloud screening outside reference coverage is not validated.",
        "display_stretch": "fixed RGB DN/3000 clipped [0,1]; display only",
        "claim_scope": "one benchmark scene, retrospective; not independent event-transfer or disaster-forecast validation",
    }
    (output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    with rasterio.open(output_dir / "model_water.tif") as checked:
        if (
            checked.crs != profile["crs"]
            or checked.transform != profile["transform"]
            or checked.nodata != 255
        ):
            raise RuntimeError("Exported geospatial contract failed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--events", nargs="+", default=["India", "Spain", "USA"])
    parser.add_argument("--model-name", default="Locally trained PRAVAH MiniUNet")
    args = parser.parse_args()
    records = json.loads(args.pairs.read_text())
    # Fix example selection from scene identifiers before viewing predictions.
    selected = []
    for event in args.events:
        candidates = sorted(
            [r for r in records if r["event_id"] == event and r["split"] == args.split],
            key=lambda r: r["scene_id"],
        )
        if candidates:
            selected.append(candidates[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not selected:
        raise ValueError("No examples found for the fixed display selection")
    (args.output_dir / "selection.json").write_text(
        json.dumps(
            {
                "rule": f"alphabetically first official {args.split} scene in each requested event, chosen before inference",
                "events": args.events,
                "scene_ids": [r["scene_id"] for r in selected],
            },
            indent=2,
        )
        + "\n"
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = MiniUNet().to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    reports = []
    for record in selected:
        with rasterio.open(record["image_path"]) as src:
            image = src.read().astype(np.float32)
        if image.shape[0] != 6:
            raise ValueError(
                "This local manifest must contain six-band prepared inputs"
            )
        x = torch.from_numpy(np.clip(image / 10000, 0, 1))[None].to(args.device)
        with torch.inference_mode():
            probability = model(x).softmax(1)[0, 1].cpu().numpy()
        identity = {
            "name": args.model_name,
            "scene_id": record["scene_id"],
            "split": "official validation split"
            if args.split == "val"
            else "official test split",
            "checkpoint_sha256": file_hash(args.checkpoint),
            "training_epoch": checkpoint["epoch"],
            "source_image_sha256": record["source_image_sha256"],
            "source_label_sha256": record["source_label_sha256"],
        }
        report = export_prediction(
            record["image_path"],
            record["label_path"],
            (probability > 0.5).astype(np.uint8),
            args.output_dir / record["scene_id"],
            identity,
            probability,
        )
        reports.append(report)
        print(
            "DEMO_EXPORTED",
            record["scene_id"],
            json.dumps(report["metrics"]),
            flush=True,
        )
    (args.output_dir / "demo_results.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
