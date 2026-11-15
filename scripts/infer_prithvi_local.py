#!/usr/bin/env python3
"""Run the pinned upstream Prithvi sample with an explicit geospatial export."""

import argparse
import hashlib
import json
import runpy
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

REVISION = "91ce9d38086a80b078a192b374df758b8855b732"
CHECKPOINT_SHA256 = "76eed77d8bd543ae441308b80e8408502243a871cb8a338ddc8220ed96dfc270"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/local_demo"))
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    root = args.root.resolve()
    research = root / "prithvi_research"
    model_dir = root / "models/prithvi"
    output = root / "prithvi_demo"
    output.mkdir(parents=True, exist_ok=True)
    renderer = runpy.run_path(str(Path(__file__).with_name("render_local_results.py")))
    sha256 = renderer["file_hash"]
    info = json.loads((research / "model_info.json").read_text())
    if info["sha"] != REVISION:
        raise ValueError("Model repository revision differs from reviewed version")
    files = {f["rfilename"]: f for f in info["siblings"]}
    for name in ("config.yaml", "inference.py", "README.md", "requirements.txt"):
        content = (research / name).read_bytes()
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if blob != files[name]["blobId"]:
            raise ValueError("Upstream source checksum differs: " + name)
    checkpoint = model_dir / "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint checksum differs")
    upstream = runpy.run_path(str(research / "inference.py"))
    config = yaml.safe_load((research / "config.yaml").read_text())
    # The complete fine-tuned state is loaded strictly by LightningInferenceModel.
    # Avoid downloading an unused second backbone checkpoint during construction.
    config["model"]["init_args"]["model_args"]["backbone_pretrained"] = False
    config["trainer"].update(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        logger=False,
        callbacks=[],
        default_root_dir=str(output),
        # TerraTorch CLI inserts checkpoint callbacks during construction even
        # with callbacks=[]; inference calls the model directly and never fits.
        enable_checkpointing=True,
    )
    config["data"]["init_args"].update(
        data_root=str(Path("data/raw/sen1floods11/v1.1").resolve()),
        batch_size=1,
        num_workers=0,
    )
    config_path = output / "inference_config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    torch.set_num_threads(4)
    started = time.perf_counter()
    print("LOADING_PRITHVI", REVISION, "device", args.device, flush=True)
    loaded = upstream["LightningInferenceModel"].from_config(
        str(config_path), checkpoint
    )
    loaded.model.to(args.device).eval()

    def check_finite_output(module, inputs, result):
        if not torch.isfinite(result.output).all().item():
            raise RuntimeError("Nonfinite Prithvi logits")

    loaded.model.register_forward_hook(check_finite_output)
    print("STRICT_CHECKPOINT_LOAD_PASSED", flush=True)
    pairs = json.loads((root / "data/pairs.json").read_text())
    by_scene = {p["scene_id"]: p for p in pairs}
    reports = []
    for scene in ("India_900498", "Spain_7370579", "USA_430764"):
        record = by_scene[scene]
        if record["split"] != "test":
            raise ValueError("Provider example not in expected official test split")
        image = model_dir / "examples" / (scene + "_S2Hand.tif")
        expected = files["examples/" + image.name]["lfs"]["sha256"]
        if sha256(image) != expected:
            raise ValueError("Provider example checksum differs")
        data, temporal, location, _ = upstream["load_example"](
            [str(image)], indices=[1, 2, 3, 8, 11, 12]
        )
        if data.mean() > 1:
            data = (
                data / 10000
            )  # Exact upstream scale rule, followed by upstream standardization.
        prediction = upstream["run_model"](
            data, temporal, location, loaded.model, loaded.datamodule, 512
        )
        prediction = prediction.squeeze(0).numpy()
        if not np.isfinite(prediction).all() or not np.isin(prediction, (0, 1)).all():
            raise ValueError("Invalid upstream prediction")
        identity = {
            "name": "Published IBM/NASA Prithvi EO 2.0 300M flood fine-tune",
            "scene_id": scene,
            "split": "official provider example / official test split",
            "model_repository": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
            "model_revision": REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "attribution": "Pretrained by the upstream authors; not trained by PRAVAH",
            "evaluation_scope": "Selected upstream examples; not a new independent full-test benchmark or Nepal validation",
        }
        report = renderer["export_prediction"](
            image,
            record["label_path"],
            prediction.astype(np.uint8),
            output / scene,
            identity,
        )
        reports.append(report)
        print("PRITHVI_EXPORTED", scene, json.dumps(report["metrics"]), flush=True)
    summary = {
        "status": "passed",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "torch": str(torch.__version__),
        "elapsed_seconds": time.perf_counter() - started,
        "strict_checkpoint_load": True,
        "model_revision": REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "upstream_inference_sha256": sha256(research / "inference.py"),
        "original_config_sha256": sha256(research / "config.yaml"),
        "effective_config_sha256": sha256(config_path),
        "changes_from_upstream": [
            "disable unused backbone pre-download, then strictly load full fine-tuned checkpoint",
            "CPU float32, one device, logging disabled; CLI checkpoint callbacks allowed during construction, no training",
            "export 0=dry, 1=water, 255=source-invalid instead of upstream 0/255 with nodata=0",
        ],
        "normalization": "Unmodified upstream load_example and run_model: six bands /10000 when mean>1, then official datamodule means/stds",
        "examples": reports,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PRITHVI_FINISHED", flush=True)


if __name__ == "__main__":
    main()
