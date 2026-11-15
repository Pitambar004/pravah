"""CLI: ``python -m pravah.inference --scene S2.tif --checkpoint ckpt.pt --output-dir out/``.

Used by the batch Fuzzball workflow (``workflows/fuzzball/06_inference_batch.yaml.tmpl``)
and for local verification. Prints the run metadata JSON on success.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from pravah.inference.engine import CheckpointError, InferenceEngine

_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


def default_run_id(scene: Path) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{_SAFE.sub('_', scene.stem)[:60]}_{stamp}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True, help="Georeferenced 13-band S2 L1C or 6-band GeoTIFF")
    p.add_argument("--checkpoint", required=True, help="MiniUNet checkpoint written by scripts/train_ddp.py")
    p.add_argument("--output-dir", required=True, help="Parent directory; outputs go to <output-dir>/<run-id>/")
    p.add_argument("--run-id", default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--require-cuda", action="store_true", help="Fail instead of falling back to CPU")
    p.add_argument("--input-units", choices=["dn", "reflectance"], default="dn",
                   help="dn = Sen1Floods11-style digital numbers (/10000); reflectance = already-scaled TOA "
                        "reflectance such as the P3 s2_l1c_reflectance.tif")
    p.add_argument("--source-metadata", default=None,
                   help="observed_metadata.json of the same scene (e.g. the Prithvi run); copies scene_id and "
                        "acquisition/availability times so the dashboard can place this run before/after the event")
    args = p.parse_args(argv)

    scene = Path(args.scene)
    run_id = args.run_id or default_run_id(scene)
    if _SAFE.search(run_id):
        p.error("run-id may only contain letters, digits, '_' and '-'")
    try:
        engine = InferenceEngine(args.checkpoint, device=args.device, require_cuda=args.require_cuda,
                                 batch_size=args.batch_size)
    except (CheckpointError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"runtime": engine.runtime(), "model": engine.info.to_dict()}, indent=2), flush=True)
    extra = {}
    if args.source_metadata:
        src_md = json.loads(Path(args.source_metadata).read_text())
        extra = {k: src_md[k] for k in ("scene_id", "product_id", "acquired_at_utc", "available_at_utc",
                                         "availability_basis", "sensor", "processing_level") if k in src_md}
        extra["source_metadata_run_id"] = src_md.get("run_id")
    try:
        result = engine.predict(
            scene, Path(args.output_dir) / run_id, run_id=run_id, threshold=args.threshold,
            tile_size=args.tile_size, overlap=args.overlap, input_units=args.input_units, extra_metadata=extra,
            progress=lambda d, n: print(f"tiles {d}/{n}", flush=True) if d == n or d % 50 == 0 else None,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    summary = {k: v for k, v in result.metadata.items() if k not in ("model_info", "transform")}
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
