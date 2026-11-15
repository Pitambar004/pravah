#!/usr/bin/env python3
"""Single-device Mac/CPU fallback using the existing PRAVAH MiniUNet.

Preserves the Fuzzball/DDP trainer. Selects checkpoints using validation only;
test and Bolivia holdouts are evaluated once after training completes.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pravah.ml.dataset import S2WaterDataset
from pravah.ml.metrics import confusion, score
from pravah.ml.unet import MiniUNet


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def choose_device(selection):
    if selection == "auto":
        selection = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    if selection == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Mac GPU requested but MPS is unavailable")
    if selection == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(selection)


def training_class_weights(records):
    training = [r for r in records if r["split"] == "train"]
    water = sum(r["water_pixels"] for r in training)
    valid = sum(r["valid_pixels"] for r in training)
    dry = valid - water
    if min(water, dry) <= 0:
        raise ValueError("Training set must contain both water and non-water labels")
    return [1.0, min(5.0, max(1.0, math.sqrt(dry / water)))], {
        "water": water,
        "non_water": dry,
        "valid": valid,
    }


def validate_split_paths(records):
    seen = set()
    for r in records:
        if r["image_path"] in seen:
            raise ValueError("Duplicate image or split leakage in manifest")
        seen.add(r["image_path"])
        if r["split"] not in ("train", "val", "test"):
            raise ValueError("Unexpected split")
        if not Path(r["image_path"]).is_file() or not Path(r["label_path"]).is_file():
            raise ValueError("Missing local raster")
    if not all(any(r["split"] == s for r in records) for s in ("train", "val", "test")):
        raise ValueError("Require separate train, validation and test subsets")


def evaluate(model, loader, device):
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "valid": 0}
    model.eval()
    with torch.inference_mode():
        for image, label in loader:
            logits = model(image.to(device))
            if not torch.isfinite(logits).all().item():
                raise RuntimeError("Nonfinite model output during evaluation")
            prediction = logits.argmax(1).cpu().numpy()
            batch = confusion(label.numpy(), prediction)
            for key in counts:
                counts[key] += batch[key]
    return {"counts": counts, "metrics": score(counts)}


def save_checkpoint(path, model, epoch, step, manifest_hash, validation, config):
    temporary = path.with_suffix(".pt.part")
    torch.save(
        {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "architecture": "MiniUNet",
            "epoch": epoch,
            "steps": step,
            "bands": ["B2", "B3", "B4", "B8A", "B11", "B12"],
            "source_data_pairs_sha256": manifest_hash,
            "validation": validation,
            "config": config,
        },
        temporary,
    )
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--bolivia", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="mps")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--validation-only",
        action="store_true",
        help="Diagnostic pilot: do not evaluate test or Bolivia",
    )
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        p.error("Invalid training settings")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("Use a new output directory to preserve previous results")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(args.pairs.read_text())
    validate_split_paths(records)
    weights, pixel_counts = training_class_weights(records)
    manifest_hash = sha256(args.pairs)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = choose_device(args.device)
    train = S2WaterDataset(str(args.pairs), "train")
    val = S2WaterDataset(str(args.pairs), "val")
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        val, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    model = MiniUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, device=device), ignore_index=-1
    )
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "loss": "weighted cross entropy, ignore_index=-1",
        "class_weights": weights,
        "weights_rule": "water weight sqrt(non_water/water), clamped to [1,5]; train pixels only",
        "augmentation": "independent horizontal and vertical flips with probability 0.5",
        "checkpoint_selection": "highest validation water IoU; first epoch wins ties",
        "threshold": "argmax over two logits (water probability > 0.5; ties non-water)",
        "training_pixels": pixel_counts,
        "validation_only_pilot": args.validation_only,
    }
    environment = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_manifest_sha256": manifest_hash,
        "training_script_sha256": sha256(__file__),
        "model_identity": "newly trained PRAVAH small U-Net, not Prithvi",
        "gpu_backend": "Metal/MPS" if device.type == "mps" else str(device),
    }
    project = Path(__file__).resolve().parents[1]
    bundle_manifest = project / "bundle_manifest.json"
    if bundle_manifest.is_file():
        commit = json.loads(bundle_manifest.read_text())["repository_commit"]
    else:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project, text=True
        ).strip()
    environment["repository_commit"] = commit
    environment["source_file_sha256"] = {
        name: sha256(project / name)
        for name in (
            "scripts/train_local.py",
            "src/pravah/ml/unet.py",
            "src/pravah/ml/dataset.py",
            "src/pravah/ml/metrics.py",
        )
    }
    environment["source_status"] = (
        "Local feature-branch additions may be uncommitted; per-file hashes identify executed code"
    )
    (args.output_dir / "repo_commit.txt").write_text(commit + "\n")
    write_json(args.output_dir / "config.json", config)
    write_json(args.output_dir / "run_environment.json", environment)
    write_json(args.output_dir / "data_splits.json", records)
    print(
        "TRAINING_STARTED",
        json.dumps(
            {
                "device": str(device),
                "train": len(train),
                "validation": len(val),
                "epochs": args.epochs,
                "class_weights": weights,
            }
        ),
        flush=True,
    )
    started, step, best_iou, best_epoch = time.perf_counter(), 0, -1.0, None
    with (args.output_dir / "train_metrics.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "epoch",
                "step",
                "mean_training_loss",
                "validation_precision",
                "validation_recall",
                "validation_f1",
                "validation_iou",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss, batches = 0.0, 0
            for image, label in loader:
                image, label = image.to(device), label.to(device)
                if not (label != -1).any().item():
                    raise ValueError("All-invalid training batch")
                for dimension in (2, 3):
                    if torch.rand(()).item() < 0.5:
                        image = image.flip(dimension)
                        label = label.flip(dimension - 1)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(image), label)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                value = float(loss.item())
                epoch_loss += value
                batches += 1
                step += 1
                if step % 50 == 0:
                    print("TRAINING_STEP", epoch, step, "loss", value, flush=True)
            validation = evaluate(model, val_loader, device)
            metrics = validation["metrics"]
            elapsed = time.perf_counter() - started
            row = {
                "epoch": epoch,
                "step": step,
                "mean_training_loss": epoch_loss / batches,
                **{"validation_" + k: v for k, v in metrics.items()},
                "elapsed_seconds": elapsed,
            }
            writer.writerow(row)
            output.flush()
            write_json(
                args.output_dir / "validation_latest.json",
                {"epoch": epoch, **validation},
            )
            save_checkpoint(
                args.output_dir / "checkpoint_last.pt",
                model,
                epoch,
                step,
                manifest_hash,
                validation,
                config,
            )
            iou = metrics["iou"]
            if iou is not None and iou > best_iou:
                best_iou, best_epoch = iou, epoch
                save_checkpoint(
                    args.output_dir / "checkpoint_best.pt",
                    model,
                    epoch,
                    step,
                    manifest_hash,
                    validation,
                    config,
                )
            print("EPOCH_COMPLETE", json.dumps(row), flush=True)
    saved = torch.load(
        args.output_dir / "checkpoint_best.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(saved["model"])
    final = {
        "status": "finished_validation_only" if args.validation_only else "finished",
        "selected_epoch": best_epoch,
        "selected_validation_iou": best_iou,
        "selected_validation": saved["validation"],
        "optimizer_steps": step,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_sha256": sha256(args.output_dir / "checkpoint_best.pt"),
        "evaluation_policy": "validation-only pilot; no test/Bolivia inference"
        if args.validation_only
        else "test evaluated once after validation-only selection; no tuning on test metrics",
    }
    if not args.validation_only:
        test_loader = DataLoader(
            S2WaterDataset(str(args.pairs), "test"),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        final["official_test"] = evaluate(model, test_loader, device)
    if args.bolivia and not args.validation_only:
        holdout = DataLoader(
            S2WaterDataset(str(args.bolivia), "test"),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        final["bolivia_holdout"] = evaluate(model, holdout, device)
    final["elapsed_seconds"] = time.perf_counter() - started
    final["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output_dir / "final_evaluation.json", final)
    print("TRAINING_FINISHED", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
