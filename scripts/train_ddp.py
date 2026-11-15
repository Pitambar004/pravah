#!/usr/bin/env python3
"""Baseline Sen1Floods11 S2 U-Net training; CPU, 1 GPU, or real torchrun DDP.

This is a NEW, small model trained from scratch. It is NOT the IBM/NASA
pretrained Prithvi checkpoint. Input pairs manifest must use official splits.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from pravah.ml.dataset import S2WaterDataset
from pravah.ml.metrics import confusion, score
from pravah.ml.unet import MiniUNet


def resolve_device(selection: str, local_rank: int):
    if selection == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; will not claim GPU success")
    if selection == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "valid": 0}
    model.eval()
    with torch.inference_mode():
        for image, label in loader:
            predicted = model(image.to(device)).argmax(1).cpu().numpy()
            c = confusion(label.numpy(), predicted)
            for key in counts:
                counts[key] += c[key]
    return counts, score(counts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True, help="JSON list with real image, label, split, event_id")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2, help="Per-rank batch size")
    p.add_argument("--max-steps", type=int, default=0, help="Max optimizer steps per rank, 0=no cap")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--global-batch-size", type=int, default=0,
                   help="Fixed total batch across all ranks (overrides --batch-size = global / WORLD_SIZE). "
                        "Use the same value for 1-node and 2-node runs so the workload is identical.")
    p.add_argument("--run-label", default="", help="Human label shown in the dashboard, e.g. gb10_1node")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_steps < 0 or args.global_batch_size < 0:
        raise ValueError("Invalid training hyperparameters")
    manifest_path = Path(args.pairs)
    raw = manifest_path.read_bytes()
    records = json.loads(raw)
    if not records:
        raise ValueError("No training records")
    required = {"image_path", "label_path", "split", "event_id"}
    if any(not required <= r.keys() for r in records):
        raise ValueError("All pairs require image_path, label_path, split, event_id")
    if len({(r["image_path"], r["label_path"]) for r in records}) != len(records):
        raise ValueError("Duplicate training pairs")
    paths_per_split = {s: {r["image_path"] for r in records if r["split"] == s} for s in ("train", "val", "test")}
    if any(paths_per_split[a] & paths_per_split[b]
           for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Overlapping image paths across splits")
    if not paths_per_split["train"] or not paths_per_split["val"]:
        raise ValueError("Require train and validation split")
    print("Pairs:", {k: len(v) for k, v in paths_per_split.items()},
          "manifest sha256:", hashlib.sha256(raw).hexdigest())
    if args.dry_run:
        print("DRY RUN ONLY: no checkpoint created or GPU claim made")
        return

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.global_batch_size:
        if args.global_batch_size % world:
            raise ValueError(f"--global-batch-size {args.global_batch_size} not divisible by WORLD_SIZE {world}")
        args.batch_size = args.global_batch_size // world
    out = Path(args.output_dir)
    if (out / "train_metrics.csv").exists():
        # Appending to an old run would silently mix two experiments in one curve.
        raise FileExistsError(f"{out} already holds a training run; use a new --output-dir")
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    device = resolve_device(args.device, local_rank)
    distributed = world > 1
    if distributed:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo", init_method="env://")
    me = {"rank": rank, "local_rank": local_rank, "hostname": socket.gethostname(), "device": str(device),
          "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
          "cuda": device.type == "cuda"}
    ranks = [me]
    if distributed:
        ranks = [None] * world
        dist.all_gather_object(ranks, me)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed)  # same model initialization across ranks
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_data = S2WaterDataset(args.pairs, "train")
    sampler = DistributedSampler(train_data, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if distributed else None
    loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler,
                        shuffle=not distributed, num_workers=args.num_workers)
    base_model = MiniUNet().to(device)
    model = DDP(base_model, device_ids=[local_rank] if device.type == "cuda" else None) if distributed else base_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    config = {
        "run_label": args.run_label or out.name, "world_size": world, "ranks": ranks,
        "distinct_hostnames": len({r["hostname"] for r in ranks}),
        "rank0_device": str(device), "cuda_verified": all(r["cuda"] for r in ranks),
        "per_rank_batch_size": args.batch_size, "global_batch_size": args.batch_size * world,
        "epochs": args.epochs, "max_steps": args.max_steps, "lr": args.lr, "seed": args.seed,
        "train_samples": len(train_data), "torch_version": torch.__version__,
        "pairs_sha256": hashlib.sha256(raw).hexdigest(), "started_at_utc": started_at,
        "hostname_note": "Hostnames as seen inside the container; confirm physical node IDs in Fuzzball job logs.",
        "important": "newly trained baseline U-Net; NOT pretrained Prithvi",
    }
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "run_environment.json").write_text(json.dumps(config, indent=2) + "\n")
    if distributed:
        dist.barrier()
    step = 0
    samples_seen = 0
    train_seconds = 0.0  # data loading + forward/backward/step; excludes validation & checkpointing
    epochs_done = 0
    counts, metrics = None, None
    start = time.perf_counter()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        t_last = time.perf_counter()
        for image, label in loader:
            image, label = image.to(device), label.to(device)
            if not torch.any(label != -1):
                # All-invalid batches cannot teach the model; all ranks must have
                # consistent batches to avoid DDP hang. Reject this dataset upstream.
                raise ValueError("Encountered all-invalid batch")
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = criterion(logits, label)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            step += 1
            samples_seen += int(image.shape[0])
            now = time.perf_counter()
            train_seconds += now - t_last
            t_last = now
            if rank == 0:
                with (out / "train_metrics.csv").open("a", newline="") as file:
                    w = csv.writer(file)
                    if file.tell() == 0:
                        w.writerow(["epoch", "optimizer_step", "loss_rank0", "elapsed_sec", "world_size",
                                    "train_sec", "global_samples"])
                    w.writerow([epoch, step, float(loss.item()), time.perf_counter()-start, world,
                                round(train_seconds, 4), samples_seen * world])
            if args.max_steps and step >= args.max_steps:
                break
        epochs_done = epoch + 1
        if distributed:
            dist.barrier()
        if rank == 0:
            validation = S2WaterDataset(args.pairs, "val")
            eval_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
            counts, metrics = evaluate(base_model, eval_loader, device)
            print("epoch", epoch, "train_steps", step, "valid_metrics", metrics)
            (out / "validation_latest.json").write_text(json.dumps({"epoch": epoch, "counts": counts,
                                                                      "metrics": metrics}, indent=2) + "\n")
            torch.save({"model": base_model.state_dict(), "epoch": epoch, "steps": step,
                        "architecture": "MiniUNet", "bands": ["B2","B3","B4","B8A","B11","B12"],
                        "source_data_pairs_sha256": hashlib.sha256(raw).hexdigest()}, out / "checkpoint_last.pt")
        if distributed:
            dist.barrier()
        if args.max_steps and step >= args.max_steps:
            break
    total = torch.tensor([samples_seen, train_seconds], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(total[:1], op=dist.ReduceOp.SUM)  # samples summed over ranks
        dist.all_reduce(total[1:], op=dist.ReduceOp.MAX)  # slowest rank bounds throughput
    global_samples, slowest_train_s = int(total[0].item()), float(total[1].item())
    if rank == 0:
        wall = time.perf_counter() - start
        ckpt = out / "checkpoint_last.pt"
        summary = {
            **config, "schema_version": "pravah-train-summary-v1", "status": "completed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "epochs_completed": epochs_done, "optimizer_steps": step, "samples_processed": global_samples,
            "train_seconds": round(slowest_train_s, 4), "wall_seconds": round(wall, 4),
            "samples_per_second": round(global_samples / slowest_train_s, 4) if slowest_train_s else None,
            "steps_per_second": round(step / slowest_train_s, 4) if slowest_train_s else None,
            "final_validation": {"counts": counts, "metrics": metrics},
            "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest() if ckpt.exists() else None,
            "timing_note": "train_seconds = slowest rank's data+compute time, excluding validation/checkpoint.",
        }
        (out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("done", "steps", step, "elapsed_seconds", wall)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
