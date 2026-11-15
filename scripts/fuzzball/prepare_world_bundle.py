#!/usr/bin/env python3
"""Build the world-pipeline code bundle, render bundle-mode workflows and write exact CLI commands.

No Docker or registry needed: jobs run on the stock NVIDIA image and fetch this bundle from the
Fuzzball object store (same route as scripts/fuzzball_morning.py). The bundle is `git archive` of the
committed HEAD, so it is reproducible; uncommitted changes to bundled paths are refused.

    .venv/bin/python scripts/fuzzball/prepare_world_bundle.py [--epochs 3 --max-steps 0 --global-batch 8 --num-workers 4]

Writes build/fuzzball/world_<tag>/{bundle.tar.gz, 1_download_world.yaml, 2_prepare_world.yaml,
3a_train_1node.yaml, 3b_train_2node.yaml, COMMANDS.sh}. Not validated on Fuzzball v4.3.1 yet.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLED = ["pyproject.toml", "src", "scripts/train_ddp.py", "scripts/download_training_datasets.py",
           "scripts/download_training_prepare.py", "config/training"]
WORLD_ROOT = "/data/training/expanded_20261114"
MANIFEST = f"{WORLD_ROOT}/manifests/optical_supervised.json"
STOCK = "docker://nvcr.io/nvidia/pytorch:25.10-py3"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    p.add_argument("--epochs", default="3")
    p.add_argument("--max-steps", default="0")
    p.add_argument("--global-batch", default="8")
    p.add_argument("--num-workers", default="4", help="DataLoader workers per rank (job has 8 cores)")
    p.add_argument("--object-root", default="fb://group/pravah/world")
    p.add_argument("--allow-uncommitted", action="store_true",
                   help="Build from HEAD anyway; uncommitted changes listed below are NOT in the bundle")
    args = p.parse_args(argv)

    dirty = git("status", "--porcelain", "--", *BUNDLED).strip()
    if dirty and not args.allow_uncommitted:
        print("Commit these bundled paths first (the bundle is built from HEAD):\n" + dirty, file=sys.stderr)
        return 2
    if dirty:
        print("WARNING: bundle is HEAD only; these uncommitted changes are NOT included:\n" + dirty, file=sys.stderr)
    out = ROOT / "build/fuzzball" / f"world_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    bundle = out / "bundle.tar.gz"
    git("archive", "--format=tar.gz", "-o", str(bundle), "HEAD", *BUNDLED)
    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    commit = git("rev-parse", "--short", "HEAD").strip()
    prefix = f"{args.object_root}/{args.tag}"
    uri = f"{prefix}/bundle.tar.gz"

    renders = [
        ("03_download_world.yaml.tmpl", "1_download_world.yaml", [f"WORLD_ROOT={WORLD_ROOT}"], None),
        ("04_prepare_world.yaml.tmpl", "2_prepare_world.yaml",
         [f"WORLD_ROOT={WORLD_ROOT}", f"PAIRS={MANIFEST}", f"GLOBAL_BATCH={args.global_batch}"], None),
        ("05_train_1node.yaml.tmpl", "3a_train_1node.yaml", [], f"world_1node_{args.tag}"),
        ("05_train_2node_ddp.yaml.tmpl", "3b_train_2node.yaml", [], f"world_2node_{args.tag}"),
    ]
    for template, name, sets, run_id in renders:
        if run_id:
            sets = [f"PAIRS={MANIFEST}", f"RUN_ID={run_id}", f"GLOBAL_BATCH={args.global_batch}",
                    f"EPOCHS={args.epochs}", f"MAX_STEPS={args.max_steps}", f"NUM_WORKERS={args.num_workers}"]
        cmd = [sys.executable, str(ROOT / "scripts/fuzzball/render_workflow.py"), str(ROOT / "workflows/fuzzball" / template),
               "--set", f"API_IMAGE={STOCK}", "--bundle", uri, "--bundle-sha256", sha, "--results-prefix", prefix,
               "--out", str(out / name)]
        for s in sets:
            cmd += ["--set", s]
        subprocess.run(cmd, check=True)

    rel = out.relative_to(ROOT)
    t = args.tag
    wf = ("python3 -c \"import json,sys; d=json.load(sys.stdin); "
          "print(next(d[k] for k in ('id','ID','workflow_id','workflowId','workflowID') if k in d))\"")
    commands = f"""# PRAVAH world pipeline — tag {t}, commit {commit}, bundle sha256 {sha}
# Run from the repo root after: fuzzball context login --group team12
# Each 'start' prints a workflow ID into $WF; 'log --follow' streams until the job ends.

# 0. Check CLI and context (expect CLI v4.3.1 and group team12)
fuzzball version
fuzzball context show

# 1. Upload the code bundle (kept 3 days)
fuzzball object put {rel}/bundle.tar.gz {uri} --ttl 3d

# 2. Server-side validation of all four workflows (stop if any fails)
for f in {rel}/1_download_world.yaml {rel}/2_prepare_world.yaml {rel}/3a_train_1node.yaml {rel}/3b_train_2node.yaml; do fuzzball workflow validate "$f" || break; done

# 3. Download / resume WorldFloods v2 + Sen1Floods11 (CPU, up to 24 h)
WF=$(fuzzball workflow start {rel}/1_download_world.yaml --name pravah-world-download-{t} --output json | {wf}); echo "$WF"
fuzzball workflow log "$WF" download-world --follow
fuzzball workflow describe "$WF"

# 4. Prepare manifests + loader check (CPU, up to 12 h) — only after step 3 finished with no failed files
WF=$(fuzzball workflow start {rel}/2_prepare_world.yaml --name pravah-world-prepare-{t} --output json | {wf}); echo "$WF"
fuzzball workflow log "$WF" prepare-world --follow
fuzzball workflow describe "$WF"

# 5a. Benchmark A: 1 node x 1 GB10 — wait for it to finish before 5b
WF=$(fuzzball workflow start {rel}/3a_train_1node.yaml --name pravah-world-1node-{t} --output json | {wf}); echo "$WF"
fuzzball workflow log "$WF" train-1node --follow
fuzzball workflow describe "$WF"

# 5b. Benchmark B: 2 nodes x 1 GB10 (PyTorch DDP) — confirm the multinode block with a CIQ mentor first
WF=$(fuzzball workflow start {rel}/3b_train_2node.yaml --name pravah-world-2node-{t} --output json | {wf}); echo "$WF"
fuzzball workflow log "$WF" train-2node-ddp --follow
fuzzball workflow describe "$WF"

# 6. Bring both runs back for the dashboard Training page
mkdir -p outputs/training
fuzzball object get {prefix}/world_1node_{t}.tar.gz outputs/training/world_1node_{t}.tar.gz
fuzzball object get {prefix}/world_2node_{t}.tar.gz outputs/training/world_2node_{t}.tar.gz
tar -xzf outputs/training/world_1node_{t}.tar.gz -C outputs/training
tar -xzf outputs/training/world_2node_{t}.tar.gz -C outputs/training
"""
    (out / "COMMANDS.sh").write_text(commands)
    print(f"\nbundle {rel}/bundle.tar.gz  sha256 {sha}  commit {commit}\ncommands: {rel}/COMMANDS.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
