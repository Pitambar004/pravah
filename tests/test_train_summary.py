"""CPU smoke test of scripts/train_ddp.py summary output. Chips are SYNTHETIC fixtures."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

pytest.importorskip("torch")
SCRIPT = Path(__file__).parent.parent / "scripts" / "train_ddp.py"


def _chip(d: Path, name: str, rng) -> dict:
    kw = {"driver": "GTiff", "height": 32, "width": 32, "crs": "EPSG:32645", "transform": from_origin(0, 0, 10, 10)}
    img, lab = d / f"{name}_img.tif", d / f"{name}_lab.tif"
    with rasterio.open(img, "w", count=6, dtype="uint16", **kw) as dst:
        dst.write(rng.integers(0, 5000, (6, 32, 32)).astype("uint16"))
    with rasterio.open(lab, "w", count=1, dtype="int16", **kw) as dst:
        dst.write(rng.integers(0, 2, (1, 32, 32)).astype("int16"))
    return {"image_path": str(img), "label_path": str(lab), "event_id": "fixture"}


def test_cpu_run_writes_training_summary(tmp_path):
    rng = np.random.default_rng(0)
    pairs = [{**_chip(tmp_path, f"t{i}", rng), "split": "train"} for i in range(4)]
    pairs.append({**_chip(tmp_path, "v0", rng), "split": "val"})
    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps(pairs))
    out = tmp_path / "run"
    cmd = [sys.executable, str(SCRIPT), "--pairs", str(manifest), "--output-dir", str(out), "--device", "cpu",
           "--epochs", "1", "--global-batch-size", "2", "--run-label", "fixture_cpu"]
    r = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    s = json.loads((out / "training_summary.json").read_text())
    assert s["status"] == "completed" and s["world_size"] == 1 and s["global_batch_size"] == 2
    assert s["optimizer_steps"] == 2 and s["samples_processed"] == 4
    assert s["cuda_verified"] is False and s["run_label"] == "fixture_cpu"
    assert s["final_validation"]["metrics"] is not None
    # Re-using the output dir must be refused so curves never mix two runs.
    r2 = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300)
    assert r2.returncode != 0 and "already holds a training run" in r2.stderr
