#!/usr/bin/env python3
"""Build the real-data transfer, training receipt and unverified hosting workflow."""

import argparse
import json
import runpy
import shlex
import shutil
import subprocess
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pravah.replay.experiment import sha256

PROJECT = Path(__file__).resolve().parents[1]


def save(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")


def relative_paths(value):
    if isinstance(value, dict):
        return {k: relative_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relative_paths(v) for v in value]
    if isinstance(value, str) and value.startswith(str(PROJECT) + "/"):
        return str(Path(value).relative_to(PROJECT))
    return value


def build(evidence, output, training=False):
    if not (evidence / "manifest.json").is_file():
        raise ValueError("Prepare the Nepal evidence before packaging")
    run_id = (
        "nepal_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    directory = output / run_id
    bundle = directory / "bundle"
    project = bundle / "project"
    project.mkdir(parents=True, exist_ok=False)
    for relative in ["src", "app", "scripts", ".streamlit"]:
        for source in (PROJECT / relative).rglob("*"):
            if relative == ".streamlit" and source.name != "config.toml":
                continue
            if source.is_symlink():
                raise ValueError("Source symlinks must be reviewed before packaging")
            if not source.is_file() or source.suffix not in [
                ".py",
                ".toml",
                ".css",
                ".sh",
            ]:
                continue
            if "__pycache__" in source.parts:
                continue
            target = project / source.relative_to(PROJECT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for name in ["pyproject.toml", "README.md", "AGENTS.md"]:
        shutil.copy2(PROJECT / name, project / name)
    for name in [
        "NEPAL_REPLAY_HANDOFF.md",
        "FUZZBALL_MORNING.md",
        "LOCAL_DEMO_HANDOFF.md",
        "USER_DATASETS_HANDOFF.md",
    ]:
        target = project / "docs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / "docs" / name, target)
    shutil.copytree(evidence, bundle / "evidence")
    shutil.copytree(
        PROJECT / "data/processed/live_app", project / "data/processed/live_app"
    )
    # OSM geometry and provenance only; original provider responses stay local.
    for source in (PROJECT / "data/raw/app_osm").rglob("*"):
        if source.is_file() and (
            source.suffix == ".geojson" or source.name == "receipt.json"
        ):
            target = project / source.relative_to(PROJECT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    rewritten = {}
    for path in (project / "data").rglob("*.json"):
        original = json.loads(path.read_text())
        adapted = relative_paths(original)
        if original != adapted:
            rewritten[str(path.relative_to(bundle))] = sha256(path)
            save(path, adapted)
    morning = runpy.run_path(str(PROJECT / "scripts/fuzzball_morning.py"))
    checkpoint, identity = morning["select_checkpoint"]()
    evaluation = checkpoint.parent / "final_evaluation.json"
    if json.loads(evaluation.read_text())["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError("Completed model and evaluation checksums disagree")
    (bundle / "assets").mkdir()
    shutil.copy2(checkpoint, bundle / "assets/checkpoint.pt")
    shutil.copy2(evaluation, bundle / "assets/final_evaluation.json")
    save(
        bundle / "runtime_config_template.json",
        {
            "catalog": "data/processed/live_app/scene_catalog.json",
            "scenes_dir": "data/processed/live_app/scenes",
            "checkpoint": "../assets/checkpoint.pt",
            "evaluation": "../assets/final_evaluation.json",
            "replay_manifest": "data/processed/live_app/replay_assets.json",
        },
    )
    manifest = {
        "run_id": run_id,
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip(),
        "source_identity": "Individual hashes include the current feature-branch working files",
        "model_identity": identity,
        "model_sha256": sha256(checkpoint),
        "scope": "Real benchmark water inference and Nepal evidence replay; downstream forecast missing",
        "path_only_rewrites_original_sha256": rewritten,
        "files": {
            str(p.relative_to(bundle)): sha256(p)
            for p in sorted(bundle.rglob("*"))
            if p.is_file()
        },
    }
    save(bundle / "bundle_manifest.json", manifest)
    archive = directory / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for source in sorted(bundle.rglob("*")):
            if source.is_file():
                tar.add(
                    source, arcname=str(source.relative_to(bundle)), recursive=False
                )
    remote = "/data/pravah_replay/" + run_id
    prefix = "fb://group/pravah/replay/" + run_id
    script = "\n".join(
        [
            "#!/bin/sh",
            "set -eu",
            "BUNDLE_ROOT=" + shlex.quote(remote),
            'test ! -e "$BUNDLE_ROOT"',
            "printf '%s  %s\\n' "
            + shlex.quote(sha256(archive))
            + " /transfer/in/bundle.tar.gz | sha256sum -c -",
            'mkdir -p "$BUNDLE_ROOT"',
            'tar -xzf /transfer/in/bundle.tar.gz --no-same-owner -C "$BUNDLE_ROOT"',
            'python "$BUNDLE_ROOT/project/scripts/run_nepal_bundle.py" --bundle "$BUNDLE_ROOT" --service verify',
            "",
        ]
    )
    staging = {
        "version": "v4",
        "volumes": {
            "dataset": {"use": "team12", "name": "Pravah-Dataset"},
            "transfer": {
                "use": "ephemeral",
                "size": "1GiB",
                "ingress": [
                    {
                        "source": {"uri": prefix + "/bundle.tar.gz"},
                        "destination": {"uri": "file://in/bundle.tar.gz"},
                    }
                ],
            },
        },
        "jobs": {
            "stage-replay": {
                "image": {"uri": "docker://python:3.11-bookworm"},
                "resource": {
                    "cpu": {"cores": 1, "affinity": "NUMA"},
                    "memory": {"size": "2GiB"},
                },
                "policy": {"timeout": {"execute": "20m"}},
                "mounts": {
                    "/data": {"volume": "dataset"},
                    "/transfer": {"volume": "transfer"},
                },
                "script": script,
            }
        },
    }
    save(directory / "stage.yaml", staging)
    import yaml

    template = yaml.safe_load(
        (PROJECT / "workflows/fuzzball/07_serve.yaml.tmpl").read_text()
    )
    for name, service in template["services"].items():
        service["image"]["uri"] = "{{SERVICE_IMAGE}}"
        service["command"] = [
            "python",
            "/opt/pravah/run_nepal_bundle.py",
            "--bundle",
            remote,
            "--service",
            "api" if name == "api" else "dashboard",
            "--device",
            "cuda",
        ]
        service["env"] = ["PRAVAH_ENABLE_TEST_FIXTURES=0"] + (
            ["PRAVAH_API_URL=http://api.service:8000"] if name == "dashboard" else []
        )
    (directory / "serve.yaml.tmpl").write_text(
        "# UNVERIFIED on Team 12 v4.3.1; stage the bundle before serving.\n"
        + yaml.safe_dump(template, sort_keys=False)
    )
    receipt = {
        "run_id": run_id,
        "bundle": str(archive),
        "bundle_dir": str(bundle),
        "bundle_sha256": sha256(archive),
        "bundle_bytes": archive.stat().st_size,
        "remote_bundle_root": remote,
        "object_prefix": prefix,
        "stage_workflow": str(directory / "stage.yaml"),
        "serve_template": str(directory / "serve.yaml.tmpl"),
        "server_validation": "not_performed",
        "remote_execution": "not_performed",
        "registry_image": None,
        "forecast_training": "blocked_missing_time_valid_inputs_and_forecast_targets",
    }
    if training:
        receipt["water_segmentation_training"] = morning["prepare"]("train")
    save(directory / "receipt.json", receipt)
    save(output / "latest_prepared.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence", type=Path, default=PROJECT / "outputs/nepal_replay/prepared"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "outputs/nepal_replay/packages"
    )
    parser.add_argument("--with-training", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.evidence, args.output, args.with_training), indent=2))


if __name__ == "__main__":
    main()
