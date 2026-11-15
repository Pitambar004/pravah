#!/usr/bin/env python3
"""Launch the real checkpoint API and fixture-free dashboard together."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def api():
    from pravah.api.main import create_app
    from pravah.api.settings import Settings
    from pravah.integration.real_engine import RealDataEngine

    config = json.loads((ROOT / "outputs/live_app/config.json").read_text())
    return create_app(
        Settings.from_env(), lambda settings: RealDataEngine(settings, ROOT, config)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=8501)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    os.chdir(ROOT)
    config = json.loads((ROOT / "outputs/live_app/config.json").read_text())
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT),
        "PRAVAH_DATA_ROOT": str(ROOT / "data"),
        "PRAVAH_SCENES_DIR": config["scenes_dir"],
        "PRAVAH_OUTPUTS_DIR": config["outputs_dir"],
        "PRAVAH_CHECKPOINT": config["checkpoint"],
        "PRAVAH_DEVICE": args.device,
        "PRAVAH_REQUIRE_CUDA": "1" if args.device == "cuda" else "0",
        "PRAVAH_BATCH_SIZE": "1",
        "PRAVAH_API_URL": f"http://127.0.0.1:{args.api_port}",
        "PRAVAH_ENABLE_TEST_FIXTURES": "0",
        "PRAVAH_RUN_SOURCE": "local",
        "PRAVAH_REPLAY_MANIFEST": config["replay_manifest"],
        "PRAVAH_CONTEXT_DIR": str(ROOT / "data/processed/live_app/context"),
        "GDAL_CACHEMAX": "128",
    }
    logdir = ROOT / "outputs/live_app"
    streams = [(logdir / n).open("a") for n in ["api.log", "dashboard.log"]]
    commands = [
        [
            sys.executable,
            "-m",
            "uvicorn",
            "scripts.serve_real_app:api",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.api_port),
        ],
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/Home.py",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(args.ui_port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
    ]
    processes = []
    try:
        for command, stream in zip(commands, streams, strict=True):
            processes.append(
                subprocess.Popen(
                    command, env=env, stdout=stream, stderr=subprocess.STDOUT
                )
            )
        print(
            f"Dashboard: http://127.0.0.1:{args.ui_port} · real model API: http://127.0.0.1:{args.api_port}",
            flush=True,
        )

        def stop(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)
        while all(p.poll() is None for p in processes):
            time.sleep(1)
        raise RuntimeError(
            "A service stopped; inspect outputs/live_app/api.log and dashboard.log"
        )
    except KeyboardInterrupt:
        pass
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
