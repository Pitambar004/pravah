#!/usr/bin/env python3
"""Fuzzball/local API factory: real water segmentation plus audited Nepal replay.

  PRAVAH_REPLAY_ROOT=/data/pravah_replay/evidence \
  PRAVAH_REAL_APP_CONFIG=/data/pravah_replay/live/config.json \
  python -m uvicorn scripts.serve_nepal_replay:api --factory --host 0.0.0.0

Inference remains water segmentation. Missing forecast models remain missing.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from pravah.api.main import create_app
from pravah.api.settings import Settings
from pravah.replay.monitor_service import attach_monitor_routes
from pravah.replay.service import attach_replay_routes

PROJECT = Path(__file__).resolve().parents[1]


def api():
    settings = Settings.from_env()
    config_path = os.environ.get("PRAVAH_REAL_APP_CONFIG")
    if config_path:
        from pravah.integration.real_engine import RealDataEngine

        config = json.loads(Path(config_path).read_text())
        project = Path(config.get("project_root", PROJECT))
        app = create_app(settings, lambda s: RealDataEngine(s, project, config))
    else:
        app = create_app(settings)
    root = Path(
        os.environ.get("PRAVAH_REPLAY_ROOT", PROJECT / "outputs/nepal_replay/prepared")
    )
    attach_replay_routes(app, root, settings.api_token)
    return attach_monitor_routes(
        app,
        Path(os.environ.get("PRAVAH_LIVE_ROOT", PROJECT / "outputs/nepal_monitor/live")),
        Path(os.environ.get("PRAVAH_NEPAL_GEOGRAPHY", PROJECT / "data/processed/nepal_monitor")),
        settings.api_token,
        enabled=os.environ.get("PRAVAH_LIVE_ENABLED", "1") == "1",
        runs=settings.outputs_dir,
        lakes=Path(os.environ.get("PRAVAH_LAKE_ROOT", PROJECT / "data/processed/nepal_lakes")),
        national=Path(os.environ.get("PRAVAH_NATIONAL_ROOT", PROJECT / "data/processed/nepal_national")),
        imagery=Path(os.environ.get("PRAVAH_IMAGERY_ROOT", PROJECT / "data/processed/nepal_imagery")),
    )


def main():
    """Launch the prepared checkpoint, live monitor and dashboard together."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-port", type=int, default=int(os.environ.get("PRAVAH_API_PORT", "8006")))
    parser.add_argument("--ui-port", type=int, default=int(os.environ.get("PRAVAH_UI_PORT", "8506")))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    host = os.environ.get("PRAVAH_BIND_HOST", "127.0.0.1")
    scheme = os.environ.get("PRAVAH_LOCAL_SCHEME", "http")
    local_api = f"{scheme}://{host}:{args.api_port}"
    local_ui = f"{scheme}://{host}:{args.ui_port}"
    config = json.loads((PROJECT / "outputs/live_app/config.json").read_text())
    env = {**os.environ, "PYTHONPATH": str(PROJECT / "src") + os.pathsep + str(PROJECT),
           "PRAVAH_REAL_APP_CONFIG": str(PROJECT / "outputs/live_app/config.json"),
           "PRAVAH_CHECKPOINT": config["checkpoint"], "PRAVAH_SCENES_DIR": config["scenes_dir"],
           "PRAVAH_OUTPUTS_DIR": config["outputs_dir"], "PRAVAH_DATA_ROOT": str(PROJECT / "data"),
           "PRAVAH_CONTEXT_DIR": str(PROJECT / "data/processed/live_app/context"),
           "PRAVAH_REPLAY_MANIFEST": config["replay_manifest"],
           "PRAVAH_DEVICE": args.device, "PRAVAH_REQUIRE_CUDA": "1" if args.device == "cuda" else "0",
           "PRAVAH_BATCH_SIZE": "1", "PRAVAH_ENABLE_TEST_FIXTURES": "0",
           "PRAVAH_RUN_SOURCE": "local", "PRAVAH_API_URL": local_api, "PRAVAH_NEPAL_API_URL": local_api}
    out = PROJECT / "outputs/nepal_monitor"
    out.mkdir(parents=True, exist_ok=True)
    commands = [
        [sys.executable, "-m", "uvicorn", "scripts.serve_nepal_replay:api", "--factory", "--host", host, "--port", str(args.api_port)],
        [sys.executable, "-m", "streamlit", "run", "app/Home.py", "--server.address", host, "--server.port", str(args.ui_port), "--server.headless", "true", "--browser.gatherUsageStats", "false"],
    ]
    processes, streams = [], []
    try:
        for command, name in zip(commands, ["api.log", "dashboard.log"], strict=True):
            stream = (out / name).open("a")
            streams.append(stream)
            processes.append(subprocess.Popen(command, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT))
        print(f"Nepal dashboard: {local_ui} · API: {local_api}", flush=True)
        def stop(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        while all(p.poll() is None for p in processes):
            time.sleep(1)
        raise RuntimeError("A service exited. Inspect outputs/nepal_monitor/ logs.")
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
