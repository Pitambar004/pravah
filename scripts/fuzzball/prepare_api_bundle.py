"""Package the existing FastAPI service, checkpoint and offline ARM64 wheels for Fuzzball.

Explicit allowlist only: no credentials, private .env, training changes or broad repo archives.
Each file and the final archive are hashed. Source is the current working tree so fixes
can be tested on the server before committing; its commit and dirty state are recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_SETTINGS = {"PRAVAH_CDSE_SEARCH_URL", "PRAVAH_GCP_BUCKET_URL", "PRAVAH_LOCAL_SCHEME"}


def digest(path: Path) -> str:
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def prepare(checkpoint: Path, wheels: Path, out: Path, object_root: str,
            scene: Path | None = None) -> dict:
    if not re.fullmatch(r"fb://group/[A-Za-z0-9_./-]+", object_root) or ".." in object_root:
        raise ValueError("Use a private group Fuzzball object prefix")
    if not checkpoint.is_file() or not list(wheels.glob("*.whl")):
        raise ValueError("A checkpoint and downloaded ARM64 wheels are required")
    if any(p.name.startswith("torch-") for p in wheels.glob("*.whl")):
        raise ValueError("Do not replace the NVIDIA image's CUDA PyTorch")
    bundle_root = out / "bundle"
    bundle_root.mkdir(parents=True, exist_ok=False)
    files = [p for p in (ROOT / "src/pravah").rglob("*.py") if "__pycache__" not in p.parts]
    files += [ROOT / "scripts/fuzzball/start_api.sh", ROOT / "infra/api-requirements.txt"]
    for source in files:
        target = bundle_root / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # Only public provider settings, never the user's .env or authenticated URLs.
    settings = [line for line in (ROOT / ".env.example").read_text().splitlines()
                if line.split("=", 1)[0] in PUBLIC_SETTINGS]
    (bundle_root / ".env.example").write_text("\n".join(settings) + "\n")
    (bundle_root / "assets").mkdir()
    shutil.copy2(checkpoint, bundle_root / "assets/checkpoint.pt")
    shutil.copytree(wheels, bundle_root / "wheels", ignore=shutil.ignore_patterns("*.tmp", "*.part"))
    if scene:
        if not scene.with_suffix(".json").is_file():
            raise ValueError("Scene requires its provenance JSON sidecar")
        metadata = json.loads(scene.with_suffix(".json").read_text())
        if metadata.get("is_synthetic"):
            raise ValueError("Synthetic scenes cannot be deployed as observations")
        scene_dir = bundle_root / "assets/scenes"
        scene_dir.mkdir()
        shutil.copy2(scene, scene_dir / scene.name)
        shutil.copy2(scene.with_suffix(".json"), scene_dir / scene.with_suffix(".json").name)
    manifest = {str(p.relative_to(bundle_root)): digest(p) for p in sorted(bundle_root.rglob("*")) if p.is_file()}
    (bundle_root / "SHA256SUMS").write_text("".join(f"{sha}  {name}\n" for name, sha in manifest.items()))
    archive = out / "api-bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted(bundle_root.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=str(p.relative_to(bundle_root)), recursive=False)
    sha = digest(archive)
    uri = f"{object_root.rstrip('/')}/{sha}/api-bundle.tar.gz"
    startup = f'''#!/bin/sh
set -eu
echo '{sha}  /transfer/in/api-bundle.tar.gz' | sha256sum -c -
PRAVAH_CODE_ROOT=$(mktemp -d /tmp/pravah-api.XXXXXX)
export PRAVAH_CODE_ROOT
tar -xzf /transfer/in/api-bundle.tar.gz --no-same-owner -C "$PRAVAH_CODE_ROOT"
exec sh "$PRAVAH_CODE_ROOT/scripts/fuzzball/start_api.sh"
'''
    workflow = {
        "version": "v4",
        "volumes": {"dataset": {"use": "team12", "name": "Pravah-Dataset"},
                    "transfer": {"use": "ephemeral", "size": "2GiB", "ingress": [
                        {"source": {"uri": uri}, "destination": {"uri": "file://in/api-bundle.tar.gz"}}]}},
        "services": {"api": {
            "image": {"uri": "docker://nvcr.io/nvidia/pytorch:25.10-py3"},
            "mounts": {"/data": {"volume": "dataset"}, "/transfer": {"volume": "transfer"}},
            "env": ["PYTHONUNBUFFERED=1", "PRAVAH_REQUIRE_CUDA=1", "PRAVAH_DEVICE=cuda",
                    "PRAVAH_DATA_ROOT=/data", "PRAVAH_INPUT_UNITS=reflectance",
                    "PRAVAH_SCENES_DIR=/data/raw/sentinel2/api_reflectance",
                    "PRAVAH_OUTPUTS_DIR=/data/outputs/api_predictions", "PRAVAH_MAX_QUEUED_JOBS=4"],
            "script": startup,
            "network": {"ports": [{"name": "http", "port": 8000, "protocol": "tcp"}],
                        "endpoints": [{"name": "api", "port-name": "http", "protocol": "http",
                                       "type": "path", "scope": "group"}]},
            "readiness-probe": {"http-get": {"path": "/ready", "port": 8000},
                                "initial-delay-seconds": 15, "period-seconds": 10},
            "persist": True,
            "resource": {"cpu": {"cores": 4, "affinity": "NUMA"}, "memory": {"size": "16GiB"},
                         "devices": {"nvidia.com/gpu": 1}},
        }},
    }
    (out / "serve-api.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False))
    report = {"archive": str(archive), "bundle_sha256": sha, "object_uri": uri,
              "checkpoint_sha256": digest(checkpoint), "files": manifest,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
              "server_validation": "not yet run"}
    (out / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--wheels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--object-root", default="fb://group/pravah/api")
    p.add_argument("--scene", type=Path)
    a = p.parse_args()
    result = prepare(a.checkpoint, a.wheels, a.out, a.object_root, a.scene)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
