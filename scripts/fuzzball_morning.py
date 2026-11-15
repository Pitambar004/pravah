#!/usr/bin/env python3
"""Prepare, launch, inspect and fetch a portable Team 12 Fuzzball model run."""

import argparse
import hashlib
import json
import os
import re
import runpy
import shlex
import shutil
import socket
import subprocess
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "outputs/fuzzball_morning"
FILES = [
    "scripts/fuzzball_run.py",
    "scripts/train_local.py",
    "scripts/prepare_local_hand.py",
    "scripts/download_training_prepare.py",
    "scripts/download_training_datasets.py",
    "src/pravah/__init__.py",
    "src/pravah/ml/__init__.py",
    "src/pravah/ml/unet.py",
    "src/pravah/ml/dataset.py",
    "src/pravah/ml/metrics.py",
]
UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def cli_path():
    path = shutil.which("fuzzball") or str(Path.home() / "bin/fuzzball")
    if not Path(path).is_file():
        raise RuntimeError("Fuzzball CLI not found; install the Team 12 v4.3.1 CLI")
    return path


def call(arguments, timeout=30):
    result = subprocess.run(
        [cli_path(), *arguments],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def doctor():
    version = call(["version"])
    if "CLI: v4.3.1" not in version:
        raise RuntimeError(
            "Expected the Team 12 v4.3.1 CLI; validate a changed version before launch"
        )
    context = json.loads(call(["context", "show", "--output", "json"]))
    address = context["address"]
    parsed = urlsplit(address if "://" in address else "https://" + address)
    if (
        parsed.hostname != "api.fb-hack.civilfritz.net"
        or context.get("groupName") != "team12@fb-hack.local"
    ):
        raise RuntimeError(
            "Select the event context and log in to team12 before launching"
        )
    with socket.create_connection((parsed.hostname, parsed.port or 443), timeout=5):
        pass
    call(["workflow", "list", "--output", "json"], timeout=15)
    report = {
        "network_and_auth": "passed",
        "cli": "v4.3.1",
        "context": context["name"],
        "group": context["groupName"],
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save(BASE / "connection_check.json", report)
    print(json.dumps(report), flush=True)
    return report


def select_checkpoint():
    base = PROJECT / "outputs/local_demo"
    status = json.loads((base / "pipeline_status.json").read_text())
    full = Path(status["checkpoint_dir"])
    if status["phase"] == "complete" and (full / "final_evaluation.json").exists():
        return full / "checkpoint_best.pt", "completed_full_hand_dataset_run"
    pilot = base / "unet_pilot/checkpoint_best.pt"
    if not pilot.exists():
        raise RuntimeError("No completed local checkpoint is available")
    return pilot, "partial_data_pilot"


def workflow(run_id, mode, bundle_hash):
    remote_root = "/data/pravah_ready/" + run_id
    object_root = "fb://group/pravah/morning/" + run_id
    script = "\n".join(
        [
            "#!/bin/sh",
            "set -eu",
            "RUN_ROOT=" + shlex.quote(remote_root),
            'test ! -e "$RUN_ROOT" || { echo "Run directory already exists; preserve it and prepare a new run."; exit 1; }',
            'mkdir -p "$RUN_ROOT/results" /transfer/out',
            'printf \'%s\\n\' \'{"status":"bootstrap_started"}\' > "$RUN_ROOT/results/bootstrap_status.json"',
            "finish() {",
            "  code=$?",
            "  trap - EXIT",
            "  set +e",
            '  if [ ! -e "$RUN_ROOT/results/status.json" ]; then printf \'{"status":"failed","stage":"bootstrap","exit_code":%s}\\n\' "$code" > "$RUN_ROOT/results/status.json"; fi',
            '  tar -czf /transfer/out/results.tar.gz -C "$RUN_ROOT" results',
            "  (cd /transfer/out && sha256sum results.tar.gz > results.sha256)",
            '  exit "$code"',
            "}",
            "trap finish EXIT",
            "printf '%s  %s\\n' "
            + shlex.quote(bundle_hash)
            + " /transfer/in/bundle.tar.gz | sha256sum -c -",
            'tar -xzf /transfer/in/bundle.tar.gz --no-same-owner -C "$RUN_ROOT"',
            'cd "$RUN_ROOT"',
            "python scripts/fuzzball_run.py --mode verify",
            "WHEELS=/data/training/expanded_20261114/environment/wheels",
            '(cd "$WHEELS" && sha256sum -c SHA256SUMS) > results/wheel_integrity.log 2>&1',
            'python -m pip install --disable-pip-version-check --no-index --no-deps --find-links "$WHEELS" rasterio==1.5.2 affine==3.0.1 > results/environment_install.log 2>&1',
            'export PYTHONPATH="$RUN_ROOT/src"',
            "export GDAL_CACHEMAX=128",
            "python -u scripts/fuzzball_run.py --mode "
            + mode
            + " --device cuda --volume /data --epochs 20",
            "",
        ]
    )
    return {
        "version": "v4",
        "volumes": {
            "dataset": {"use": "team12", "name": "Pravah-Dataset"},
            "transfer": {
                "use": "ephemeral",
                "size": "1GiB",
                "ingress": [
                    {
                        "source": {"uri": object_root + "/bundle.tar.gz"},
                        "destination": {"uri": "file://in/bundle.tar.gz"},
                    }
                ],
                "egress": [
                    {
                        "source": {"uri": "file://out/" + name},
                        "destination": {"uri": object_root + "/" + name},
                    }
                    for name in ["results.tar.gz", "results.sha256"]
                ],
            },
        },
        "jobs": {
            "pravah-model": {
                "image": {"uri": "docker://nvcr.io/nvidia/pytorch:25.10-py3"},
                "resource": {
                    "cpu": {"cores": 8, "affinity": "NUMA"},
                    "memory": {"size": "16GiB"},
                    "devices": {"nvidia.com/gpu": 1},
                },
                "policy": {"timeout": {"execute": "12h" if mode == "train" else "20m"}},
                "mounts": {
                    "/data": {"volume": "dataset"},
                    "/transfer": {"volume": "transfer"},
                },
                "script": script,
            }
        },
    }


def prepare(mode):
    import rasterio

    checkpoint, identity = select_checkpoint()
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    directory = BASE / run_id
    bundle = directory / "bundle"
    (bundle / "assets").mkdir(parents=True)
    for name in FILES:
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, target)
    shutil.copy2(checkpoint, bundle / "assets/checkpoint.pt")
    evaluation = json.loads((checkpoint.parent / "final_evaluation.json").read_text())
    if evaluation["checkpoint_sha256"] != sha256(checkpoint):
        raise RuntimeError("Completed model checksum does not match its evaluation")
    model_run = bundle / "assets/model_run"
    model_run.mkdir()
    for name in [
        "final_evaluation.json",
        "config.json",
        "run_environment.json",
        "repo_commit.txt",
    ]:
        shutil.copy2(checkpoint.parent / name, model_run / name)
    shutil.copy2(
        PROJECT / "outputs/local_demo/hand_download_plan.json",
        bundle / "assets/hand_download_plan.json",
    )
    pairs_file = PROJECT / "outputs/local_demo/data/pairs.json"
    if not pairs_file.is_file():
        pairs_file = PROJECT / "outputs/local_demo/pilot_data/pairs.json"
    pairs = json.loads(pairs_file.read_text())
    record = next(
        r for r in pairs if r["scene_id"] == "India_1018317" and r["split"] == "val"
    )
    with rasterio.open(record["image_path"]) as src:
        pixels, profile = src.read(), src.profile.copy()
        profile.update(driver="GTiff", compress="deflate")
        with rasterio.open(
            bundle / "assets/reference_image.tif", "w", **profile
        ) as dst:
            dst.write(pixels)
            for i, name in enumerate(src.descriptions, 1):
                if name:
                    dst.set_band_description(i, name)
    shutil.copy2(record["label_path"], bundle / "assets/reference_label.tif")
    save(
        bundle / "assets/reference_identity.json",
        {
            "scene_id": record["scene_id"],
            "split": "official validation split",
            "source_image_sha256": record["source_image_sha256"],
            "source_label_sha256": record["source_label_sha256"],
            "selection": "fixed existing demonstration scene; label is only for scoring, never model input",
        },
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    manifest = {
        "repository_commit": commit,
        "source_status": "Uncommitted additions identified by individual hashes",
        "model_identity": identity,
        "model_source": str(checkpoint.relative_to(PROJECT)),
        "model_sha256": sha256(checkpoint),
        "bands": ["B2", "B3", "B4", "B8A", "B11", "B12"],
        "files": {
            str(p.relative_to(bundle)): sha256(p)
            for p in sorted(bundle.rglob("*"))
            if p.is_file()
        },
    }
    save(bundle / "bundle_manifest.json", manifest)
    archive = directory / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(bundle)), recursive=False)
    receipt = {
        "run_id": run_id,
        "mode": mode,
        "bundle": str(archive),
        "bundle_dir": str(bundle),
        "bundle_sha256": sha256(archive),
        "bundle_bytes": archive.stat().st_size,
        "model_identity": identity,
        "model_sha256": sha256(checkpoint),
        "persistent_results": "/data/pravah_ready/" + run_id + "/results",
        "object_prefix": "fb://group/pravah/morning/" + run_id,
        "workflow_file": str(directory / "workflow.yaml"),
        "state": "prepared_not_submitted",
        "server_validation": "not_yet_performed",
    }
    # JSON is valid YAML and avoids an additional local YAML dependency.
    save(
        Path(receipt["workflow_file"]), workflow(run_id, mode, receipt["bundle_sha256"])
    )
    save(directory / "receipt.json", receipt)
    save(BASE / "latest_prepared.json", receipt)
    print("PREPARED", json.dumps(receipt), flush=True)
    return receipt


def workflow_id(output):
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        for key in ["id", "ID", "workflow_id", "workflowId", "workflowID"]:
            if isinstance(data.get(key), str) and UUID.fullmatch(data[key]):
                return data[key]
    found = set(UUID.findall(output))
    if len(found) == 1:
        return found.pop()
    raise ValueError(
        "Submission response has no unambiguous workflow ID; inspect the saved response before retrying"
    )


def launch(mode):
    doctor()
    receipt = prepare(mode)
    directory = Path(receipt["bundle"]).parent
    validated = call(["workflow", "validate", receipt["workflow_file"]])
    (directory / "server_validation.txt").write_text(validated)
    receipt["server_validation"] = "passed_on_connected_server"
    save(directory / "receipt.json", receipt)
    call(
        [
            "object",
            "put",
            receipt["bundle"],
            receipt["object_prefix"] + "/bundle.tar.gz",
            "--ttl",
            "3d",
        ],
        timeout=180,
    )
    receipt["state"] = "submission_pending"
    save(directory / "receipt.json", receipt)
    save(BASE / "latest_launch.json", receipt)
    try:
        output = call(
            [
                "workflow",
                "start",
                receipt["workflow_file"],
                "--name",
                "pravah-" + mode + "-" + receipt["run_id"],
                "--output",
                "json",
            ],
            timeout=60,
        )
        (directory / "submission_response.txt").write_text(output)
        receipt.update(workflow_id=workflow_id(output), state="submitted")
    except Exception:
        receipt["state"] = (
            "submission_outcome_unknown_check_named_workflow_before_retry"
        )
        raise
    finally:
        save(directory / "receipt.json", receipt)
        save(BASE / "latest_launch.json", receipt)
    print("SUBMITTED", json.dumps(receipt), flush=True)


def safe_extract(archive, destination):
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()) or not (
                member.isfile() or member.isdir()
            ):
                raise ValueError("Unsafe result archive member")
        tar.extractall(destination, filter="data")


def fetch(receipt):
    folder = Path(receipt["bundle"]).parent / datetime.now(timezone.utc).strftime(
        "downloaded_%Y%m%dT%H%M%SZ"
    )
    folder.mkdir(parents=True, exist_ok=False)
    for name in ["results.sha256", "results.tar.gz"]:
        call(
            [
                "object",
                "get",
                receipt["object_prefix"] + "/" + name,
                str(folder / name),
            ],
            timeout=180,
        )
    expected = (folder / "results.sha256").read_text().split()[0]
    if sha256(folder / "results.tar.gz") != expected:
        raise ValueError("Downloaded results checksum mismatch")
    safe_extract(folder / "results.tar.gz", folder)
    status = json.loads((folder / "results/status.json").read_text())
    if status["status"] != "passed":
        raise RuntimeError("Remote run failed; inspect " + str(folder / "results"))
    import numpy as np
    import rasterio

    exporter = runpy.run_path(str(PROJECT / "scripts/render_local_results.py"))[
        "export_prediction"
    ]
    for directory in (folder / "results").glob("inference_*"):
        metadata = json.loads((directory / "inference.json").read_text())
        with rasterio.open(directory / "water_probability.tif") as src:
            probabilities = src.read(1)
        probabilities = np.where(probabilities >= 0, probabilities, 0)
        assets = Path(receipt["bundle_dir"]) / "assets"
        exporter(
            assets / "reference_image.tif",
            assets / "reference_label.tif",
            (probabilities > 0.5).astype(np.uint8),
            directory / "comparison",
            {
                "name": "PRAVAH MiniUNet - actual Fuzzball output",
                "scene_id": metadata["scene"]["scene_id"],
                "split": metadata["scene"]["split"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
            },
            probabilities,
        )
    print("RESULTS_VERIFIED", folder / "results", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["prepare", "doctor", "launch", "status", "logs", "fetch"]
    )
    parser.add_argument("--mode", choices=["inference", "train"], default="inference")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    os.chdir(PROJECT)
    if args.action == "prepare":
        prepare(args.mode)
    elif args.action == "doctor":
        doctor()
    elif args.action == "launch":
        launch(args.mode)
    else:
        receipt = json.loads((args.receipt or BASE / "latest_launch.json").read_text())
        if args.action == "fetch":
            fetch(receipt)
        elif args.action == "status":
            print(call(["workflow", "describe", receipt["workflow_id"]]))
        else:
            subprocess.run(
                [
                    cli_path(),
                    "workflow",
                    "log",
                    receipt["workflow_id"],
                    "pravah-model",
                    "--follow",
                ],
                check=True,
            )


if __name__ == "__main__":
    try:
        main()
    except (TimeoutError, socket.gaierror, subprocess.TimeoutExpired) as exc:
        raise SystemExit(
            "Connection or CLI timeout. Connect to the authorized event network and run 'doctor'. If submission was attempted, inspect latest_launch.json before retrying."
        ) from exc
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
