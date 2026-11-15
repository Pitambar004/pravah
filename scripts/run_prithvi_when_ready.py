#!/usr/bin/env python3
"""Run the optional pretrained sample after downloads and local U-Net finish."""

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[1]
    os.chdir(project)
    root = project / "outputs/local_demo"
    python = root / "runtime/prithvi-venv/bin/python"
    checkpoint = root / "models/prithvi/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=str(project / "src"),
        HF_HUB_OFFLINE="1",
        HF_HOME=str(root / "runtime/hf-cache"),
        NO_ALBUMENTATIONS_UPDATE="1",
        MPLCONFIGDIR=str(root / "runtime/matplotlib"),
    )

    def status(phase, **details):
        state = {
            "phase": phase,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        }
        temporary = root / "prithvi_status.json.part"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(root / "prithvi_status.json")
        print(json.dumps(state), flush=True)

    status(
        "waiting_for_baseline",
        reason="Prioritize hand-label dataset download; saved Prithvi ranges resume after baseline completion",
    )
    deadline = time.monotonic() + 4 * 3600
    while time.monotonic() < deadline:
        try:
            baseline = json.loads((root / "pipeline_status.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            baseline = {}
        if baseline.get("phase") == "complete" and not checkpoint.is_file():
            status("resuming_checkpoint_download")
            command = [
                str(project / ".venv/bin/python"),
                "-u",
                "scripts/download_pinned_checkpoint.py",
                "--plan",
                str(root / "prithvi_download_plan.json"),
                "--root",
                str(root / "models"),
            ]
            with (root / "prithvi_range_download.log").open("a") as stream:
                downloaded = subprocess.run(
                    command,
                    cwd=project,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if downloaded.returncode != 0:
                status(
                    "failed",
                    reason="Resumed checkpoint download failed; partial ranges and log preserved",
                )
                return
        if checkpoint.is_file() and baseline.get("phase") == "complete":
            probe = subprocess.run(
                [str(python), "-c", "import terratorch.cli_tools"],
                cwd=project,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            (root / "prithvi_import.log").write_text(probe.stdout)
            if probe.returncode != 0:
                status(
                    "failed",
                    reason="Isolated Prithvi dependency import failed",
                    log=str(root / "prithvi_import.log"),
                )
                return
            command = [
                str(python),
                "-u",
                "scripts/infer_prithvi_local.py",
                "--device",
                "cpu",
            ]
            status("running_reference_inference", command=command)
            with (root / "prithvi_inference.log").open("w") as stream:
                result = subprocess.run(
                    command,
                    cwd=project,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            status(
                "complete" if result.returncode == 0 else "failed",
                returncode=result.returncode,
                log=str(root / "prithvi_inference.log"),
            )
            if result.returncode == 0:
                freeze = subprocess.run(
                    [
                        str(python),
                        "-c",
                        "import importlib.metadata as m; print('\\n'.join(sorted(f'{d.metadata[\"Name\"]}=={d.version}' for d in m.distributions())))",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                (root / "runtime/prithvi_environment.txt").write_text(freeze.stdout)
            return
        if baseline.get("phase") == "failed":
            status(
                "blocked",
                reason="Full local baseline pipeline needs repair before this optional step",
            )
            return
        time.sleep(30)
    status(
        "blocked",
        reason="Prerequisite downloads or baseline did not finish within four hours",
    )


if __name__ == "__main__":
    main()
