#!/usr/bin/env python3
"""Wait for verified local hand data, then prepare, train and export the demo."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/local_demo"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--wait-hours", type=float, default=4)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    os.chdir(project)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("unet_mps_%Y%m%dT%H%M%SZ")
    run_dir = root / run_id
    state = {"run_id": run_id, "checkpoint_dir": str(run_dir)}

    def status(phase, **details):
        state.update(
            phase=phase,
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
            **details,
        )
        temporary = root / "pipeline_status.json.part"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(root / "pipeline_status.json")
        print(json.dumps(state), flush=True)

    def execute(phase, script, arguments):
        command = [
            sys.executable,
            "-u",
            str(project / "scripts" / script),
            *map(str, arguments),
        ]
        status(phase, command=command)
        with (root / (phase + ".log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        status("waiting_for_verified_data")
        deadline = time.monotonic() + args.wait_hours * 3600
        summary = root / "intake/download_summary.json"
        while not summary.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Data download has not completed; inspect hand_download.log"
                )
            time.sleep(5)
        receipt = json.loads(summary.read_text())
        if receipt["failed_assets"]:
            raise RuntimeError("Source downloads failed; repair before training")
        execute(
            "preparing",
            "prepare_local_hand.py",
            ["--intake", root / "intake", "--output", root / "data"],
        )
        execute(
            "training",
            "train_local.py",
            [
                "--pairs",
                root / "data/pairs.json",
                "--bolivia",
                root / "data/bolivia_holdout.json",
                "--output-dir",
                run_dir,
                "--device",
                "mps",
                "--epochs",
                args.epochs,
                "--batch-size",
                1,
            ],
        )
        execute(
            "rendering",
            "render_local_results.py",
            [
                "--pairs",
                root / "data/pairs.json",
                "--checkpoint",
                run_dir / "checkpoint_best.pt",
                "--output-dir",
                root / "unet_demo",
                "--device",
                "mps",
            ],
        )
        status(
            "complete",
            final_evaluation=str(run_dir / "final_evaluation.json"),
            demo_dir=str(root / "unet_demo"),
        )
    except Exception as exc:
        status("failed", failed_phase=state["phase"], error=str(exc))
        raise


if __name__ == "__main__":
    main()
