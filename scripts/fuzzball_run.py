#!/usr/bin/env python3
"""Portable single-GPU worker; CPU mode is only for local package verification."""

import argparse
import hashlib
import json
import os
import platform
import runpy
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_bundle(root):
    manifest = json.loads((root / "bundle_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
            raise ValueError("Bundle integrity failed: " + name)
    return manifest


def prepare_output(output):
    bootstrap_files = {
        "bootstrap_status.json",
        "wheel_integrity.log",
        "environment_install.log",
    }
    if output.exists() and any(p.name not in bootstrap_files for p in output.iterdir()):
        raise RuntimeError(
            "Use a fresh result directory; previous results are preserved"
        )
    output.mkdir(parents=True, exist_ok=True)


def prepare_remote_data(root, volume, output):
    intake = runpy.run_path(str(root / "scripts/download_training_datasets.py"))
    plan = json.loads((root / "assets/hand_download_plan.json").read_text())

    def check(asset):
        path = intake["target_path"](volume, asset["destination"])
        checksum = intake["check_file"](path, asset)
        return {
            **asset,
            "path": str(path),
            "sha256": checksum,
            "status": "verified_existing_transfer",
        }

    # Reuse the persistent files; no internet download and no weak labels mixed in.
    audit = output / "intake"
    audit.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        checked = list(pool.map(check, plan["assets"]))
    (audit / "transfers.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in checked)
    )
    write_json(
        audit / "download_summary.json",
        {
            "failed_assets": [],
            "verified_files": len(checked),
            "verified_bytes": sum(r["bytes"] for r in checked),
        },
    )
    command = [
        sys.executable,
        "scripts/prepare_local_hand.py",
        "--data-root",
        str(volume / "raw/sen1floods11/v1.1"),
        "--intake",
        str(audit),
        "--output",
        str(output / "data"),
    ]
    subprocess.run(command, cwd=root, check=True)
    return output / "data"


def prediction(root, checkpoint, output, device):
    import numpy as np
    import rasterio
    import torch

    from pravah.ml.metrics import confusion, score
    from pravah.ml.unet import MiniUNet

    output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = MiniUNet().to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    with rasterio.open(root / "assets/reference_image.tif") as src:
        image = src.read()
        if image.shape[0] != 6 or src.crs is None or not np.isfinite(image).all():
            raise ValueError("Invalid six-band optical reference")
        valid = (image != 0).any(axis=0) & (src.read_masks() != 0).all(axis=0)
        profile = src.profile.copy()
    with rasterio.open(root / "assets/reference_label.tif") as src:
        truth = src.read(1)
        if src.crs != profile["crs"] or src.transform != profile["transform"]:
            raise ValueError("Reference label grid mismatch")
    tensor = torch.from_numpy(np.clip(image.astype(np.float32) / 10000, 0, 1))[None].to(
        device
    )
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor)
        if not torch.isfinite(logits).all().item():
            raise ValueError("Nonfinite model logits")
        probability = logits.softmax(1)[0, 1].cpu().numpy()
    elapsed = time.perf_counter() - started
    classes = (probability > 0.5).astype(np.uint8)
    profile.update(count=1, compress="deflate")
    with rasterio.open(
        output / "model_water.tif", "w", **{**profile, "dtype": "uint8", "nodata": 255}
    ) as dst:
        dst.write(np.where(valid, classes, 255).astype(np.uint8), 1)
    with rasterio.open(
        output / "water_probability.tif",
        "w",
        **{**profile, "dtype": "float32", "nodata": -9999},
    ) as dst:
        dst.write(np.where(valid, probability, -9999).astype(np.float32), 1)
    counts = confusion(np.where(valid, truth, -1), classes)
    report = {
        "checkpoint_sha256": sha256(checkpoint),
        "device": str(device),
        "cuda_verified": device.type == "cuda",
        "inference_seconds_including_transfer": elapsed,
        "counts": counts,
        "metrics": score(counts),
        "scene": json.loads((root / "assets/reference_identity.json").read_text()),
        "claim_scope": "One fixed Sen1Floods11 validation example; not aggregate test or Nepal/forecast validation",
        "class_codes": {
            "0": "no_water",
            "1": "model_inferred_water",
            "255": "source_invalid",
        },
    }
    write_json(output / "inference.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["verify", "inference", "train"], default="inference"
    )
    parser.add_argument("--volume", type=Path, default=Path("/data"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    manifest = verify_bundle(root)
    if args.mode == "verify":
        print("BUNDLE_INTEGRITY_PASSED", flush=True)
        return
    output = args.output.resolve() if args.output else root / "results"
    prepare_output(output)
    write_json(output / "bundle_manifest.json", manifest)
    try:
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no silent CPU fallback")
        device = torch.device(args.device)
        torch.set_num_threads(4)
        environment = {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": str(device),
            "cuda_verified": device.type == "cuda",
            "mode": args.mode,
            "world_size": 1,
            "local_cpu_package_test": device.type == "cpu",
        }
        if device.type == "cuda":
            environment.update(
                gpu_name=torch.cuda.get_device_name(0), cuda_runtime=torch.version.cuda
            )
            actual = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,name,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            environment["nvidia_smi_gpu_identity"] = (
                actual.stdout.strip() if actual.returncode == 0 else None
            )
        write_json(output / "run_environment.json", environment)
        before = prediction(
            root, root / "assets/checkpoint.pt", output / "inference_existing", device
        )
        print("MODEL_INFERENCE_PASSED", json.dumps(before["metrics"]), flush=True)
        if args.mode == "train":
            prepared = prepare_remote_data(root, args.volume.resolve(), output)
            command = [
                sys.executable,
                "-u",
                "scripts/train_local.py",
                "--pairs",
                str(prepared / "pairs.json"),
                "--bolivia",
                str(prepared / "bolivia_holdout.json"),
                "--output-dir",
                str(output / "training"),
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                "1",
            ]
            with (output / "training.log").open("w") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            prediction(
                root,
                output / "training/checkpoint_best.pt",
                output / "inference_new",
                device,
            )
        write_json(
            output / "status.json",
            {
                "status": "passed",
                "mode": args.mode,
                "cuda_verified": device.type == "cuda",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception as exc:
        write_json(
            output / "status.json",
            {"status": "failed", "mode": args.mode, "error": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
