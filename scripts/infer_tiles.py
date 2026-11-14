#!/usr/bin/env python3
"""Run small U-Net inference on assigned REAL GeoTIFF windows, not Prithvi.

Each tile is saved as a .npz probability array + valid mask + metadata in
manifest. Workers write to different directory paths. Stitch separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch.nn import functional as F

from pravah.gis.tiles import partition
from pravah.ml.unet import MiniUNet
from pravah.ml.dataset import S2_INDEXES


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--partition-index", type=int, required=True)
    p.add_argument("--partition-count", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA-required run cannot silently fall back to CPU")
    use_gpu = torch.cuda.is_available() and args.device != "cpu"
    if args.device == "cuda" and not use_gpu:
        raise RuntimeError("Requested CUDA but no CUDA runtime exists")
    device = torch.device("cuda" if use_gpu else "cpu")
    manifest = json.loads(Path(args.manifest).read_text())
    tiles = partition(manifest["tiles"], args.partition_index, args.partition_count)
    checkpoint = Path(args.checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = MiniUNet().to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.perf_counter()
    with rasterio.open(manifest["raster"]) as src:
        if src.crs is None or src.count not in (6, 13):
            raise ValueError("Source must be georeferenced 6/13-band S2 raster")
        for tile in tiles:
            tile_start = time.perf_counter()
            w = Window(tile["col_off"], tile["row_off"], tile["width"], tile["height"])
            channels = src.read([i+1 for i in S2_INDEXES] if src.count == 13 else list(range(1, 7)), window=w)
            mask = src.dataset_mask(window=w) > 0
            x = np.clip(channels.astype(np.float32)/10000.0, 0, 1)
            # Network downscales 3 times; extend edge to next multiple of 8.
            h, wid = x.shape[-2:]
            pad_h, pad_w = (-h) % 8, (-wid) % 8
            if pad_h or pad_w:
                x = np.pad(x, ((0,0),(0,pad_h),(0,pad_w)), mode="constant")
            with torch.inference_mode():
                probs = F.softmax(model(torch.from_numpy(x[None]).to(device)), dim=1)[0, 1, :h, :wid]
                if use_gpu:
                    torch.cuda.synchronize()
                prob = probs.cpu().numpy().astype(np.float32)
            name = out / f'{tile["tile_id"]}.npz'
            np.savez_compressed(name, probability=prob, valid=mask.astype(np.uint8))
            results.append({"tile_id": tile["tile_id"], "path": str(name),
                            "sha256": sha256(name), "elapsed_seconds": time.perf_counter()-tile_start})
    summary = {"status": "finished", "cuda_verified": use_gpu,
               "gpu_name": torch.cuda.get_device_name(0) if use_gpu else None,
               "partition_index": args.partition_index, "partition_count": args.partition_count,
               "tile_manifest_sha256": manifest["tile_manifest_sha256"],
               "model_sha256": sha256(checkpoint), "tile_count": len(results),
               "inference_seconds": time.perf_counter()-t0, "tiles": results}
    (out / "worker_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "tiles"}, indent=2))


if __name__ == "__main__":
    main()
