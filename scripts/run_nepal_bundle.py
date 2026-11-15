#!/usr/bin/env python3
"""Verify a portable bundle, then serve its API or dashboard on Fuzzball."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(bundle):
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    for relative, digest in manifest["files"].items():
        member = bundle / relative
        path = member.resolve()
        if (
            bundle.resolve() not in path.parents
            or not path.is_file()
            or member.is_symlink()
        ):
            raise ValueError("Invalid bundle member: " + relative)
        if sha256(path) != digest:
            raise ValueError("Bundle integrity mismatch: " + relative)
    return manifest


def configure(bundle, device):
    verify(bundle)
    project = bundle / "project"
    template = json.loads((bundle / "runtime_config_template.json").read_text())
    config = {k: str((project / v).resolve()) for k, v in template.items()}
    config["project_root"] = str(project)
    runtime = bundle / "runtime"
    runtime.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=runtime, delete=False) as stream:
        json.dump(config, stream)
        temporary = Path(stream.name)
    temporary.replace(runtime / "config.json")
    return {
        "PYTHONPATH": str(project / "src") + os.pathsep + str(project),
        "PRAVAH_REAL_APP_CONFIG": str(runtime / "config.json"),
        "PRAVAH_DATA_ROOT": str(project / "data"),
        "PRAVAH_SCENES_DIR": config["scenes_dir"],
        "PRAVAH_CHECKPOINT": config["checkpoint"],
        "PRAVAH_OUTPUTS_DIR": str(bundle / "runs"),
        "PRAVAH_REPLAY_ROOT": str(bundle / "evidence"),
        "PRAVAH_NEPAL_GEOGRAPHY": str(project / "data/processed/nepal_monitor"),
        "PRAVAH_LAKE_ROOT": str(project / "data/processed/nepal_lakes"),
        "PRAVAH_NATIONAL_ROOT": str(project / "data/processed/nepal_national"),
        "PRAVAH_IMAGERY_ROOT": str(project / "data/processed/nepal_imagery"),
        "PRAVAH_LIVE_ROOT": str(bundle / "live_cache"),
        "PRAVAH_REPLAY_MANIFEST": config["replay_manifest"],
        "PRAVAH_CONTEXT_DIR": str(project / "data/processed/live_app/context"),
        "PRAVAH_DEVICE": device,
        "PRAVAH_REQUIRE_CUDA": "1" if device == "cuda" else "0",
        "PRAVAH_BATCH_SIZE": "1",
        "PRAVAH_ENABLE_TEST_FIXTURES": "0",
        "PRAVAH_RUN_SOURCE": "local",
        "GDAL_CACHEMAX": "128",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--service", choices=["verify", "api", "dashboard"], default="verify"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    bundle = args.bundle.resolve()
    try:
        manifest = verify(bundle)
        if args.service == "verify":
            print(
                json.dumps(
                    {
                        "status": "verified",
                        "run_id": manifest["run_id"],
                        "file_count": len(manifest["files"]),
                    }
                )
            )
            return 0
        env = {**os.environ, **configure(bundle, args.device)}
        # The standalone verify path above needs only the standard library.
        # Application configuration is loaded only when starting a service.
        sys.path.insert(0, str(bundle / "project/src"))
        from pravah.configuration import configured_url

        env.setdefault("PRAVAH_API_URL", configured_url("PRAVAH_INTERNAL_API_URL"))
        env.setdefault("PRAVAH_NEPAL_API_URL", env["PRAVAH_API_URL"])
        if args.service == "api":
            command = [
                sys.executable,
                "-m",
                "uvicorn",
                "scripts.serve_nepal_replay:api",
                "--factory",
                "--host",
                "0.0.0.0",
                "--port",
                str(args.port or 8000),
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app/Home.py",
                "--server.address",
                "0.0.0.0",
                "--server.port",
                str(args.port or 8501),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ]
        return subprocess.call(command, cwd=bundle / "project", env=env)
    except (OSError, ValueError, KeyError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
