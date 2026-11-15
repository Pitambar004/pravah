#!/usr/bin/env python3
"""Build the Live Watch bundle, render the GB10 build + serve workflows and write the CLI commands.

    .venv/bin/python scripts/fuzzball/prepare_watch_bundle.py [--scope public] [--workers 5]

Bundle = committed HEAD (git archive of src, app, dashboard start script, .env.example, pyproject)
plus three small public-source asset folders that are not in Git (OSM Rasuwa geography — ODbL;
Glacial Lake Observatory inventory — CC BY 4.0; GloFAS station cells). Every asset file is listed
with its SHA-256 in WATCH_ASSETS.json inside the bundle. No credentials, .env, checkpoints or raw data.
Writes build/fuzzball/watch_<tag>/{bundle.tar.gz, 1_watch_build.yaml, 2_watch_serve.yaml, COMMANDS.sh}.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLED = ["pyproject.toml", ".env.example", "src", "app", "scripts/fuzzball/start_dashboard.sh"]
ASSETS = ["data/processed/nepal_monitor", "data/processed/nepal_lakes", "data/processed/forecast/glofas_cells.json"]
STOCK = "docker://nvcr.io/nvidia/pytorch:25.10-py3"
WATCH_ROOT = "/data/pravah_watch"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_bundle(out: Path) -> tuple[Path, str, dict]:
    head = subprocess.run(["git", "archive", "--format=tar", "HEAD", *BUNDLED], cwd=ROOT, check=True, capture_output=True).stdout
    files = []
    for a in ASSETS:
        p = ROOT / a
        files += sorted(x for x in (p.rglob("*") if p.is_dir() else [p]) if x.is_file() and not x.name.startswith("."))
    if not files:
        raise SystemExit("Missing data/processed assets (run the Nepal preparation first)")
    manifest = {"schema": "pravah-watch-assets-v1", "commit": git("rev-parse", "HEAD").strip(),
                "files": {str(f.relative_to(ROOT)): {"sha256": sha(f), "bytes": f.stat().st_size} for f in files}}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        with tarfile.open(fileobj=io.BytesIO(head)) as src:
            for m in src.getmembers():
                tar.addfile(m, src.extractfile(m) if m.isfile() else None)
        for f in files:
            tar.add(f, arcname=str(f.relative_to(ROOT)), recursive=False)
        data = json.dumps(manifest, indent=1).encode()
        info = tarfile.TarInfo("WATCH_ASSETS.json")
        info.size, info.mtime = len(data), int(datetime.now(timezone.utc).timestamp())
        tar.addfile(info, io.BytesIO(data))
    bundle = out / "bundle.tar.gz"
    bundle.write_bytes(gzip.compress(buf.getvalue(), mtime=0))
    return bundle, sha(bundle), manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    p.add_argument("--object-root", default="fb://group/pravah/watch")
    p.add_argument("--scope", default="public", choices=["user", "group", "organization", "public"])
    p.add_argument("--workers", default="5")
    p.add_argument("--members", default="5")
    p.add_argument("--epochs", default="40")
    p.add_argument("--s2-per-month", default="4")
    p.add_argument("--api-token", default=None, help="X-Pravah-Token for write routes (kept out of logs/Git)")
    p.add_argument("--allow-uncommitted", action="store_true")
    a = p.parse_args(argv)
    dirty = git("status", "--porcelain", "--", *BUNDLED, "workflows/fuzzball").strip()
    if dirty and not a.allow_uncommitted:
        print("Commit these bundled paths first (the bundle is built from HEAD):\n" + dirty, file=sys.stderr)
        return 2
    out = ROOT / "build/fuzzball" / f"watch_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    bundle, digest, manifest = build_bundle(out)
    prefix = f"{a.object_root}/{a.tag}"
    uri = f"{prefix}/bundle.tar.gz"
    renders = [("09_watch_build.yaml.tmpl", "1_watch_build.yaml",
                [f"WATCH_ROOT={WATCH_ROOT}", f"WATCH_WORKERS={a.workers}", f"EMULATOR_MEMBERS={a.members}",
                 f"EMULATOR_EPOCHS={a.epochs}", f"S2_PER_MONTH={a.s2_per_month}"], []),
               ("10_watch_serve.yaml.tmpl", "2_watch_serve.yaml",
                [f"WATCH_ROOT={WATCH_ROOT}", f"ENDPOINT_SCOPE={a.scope}"],
                [f"api:PRAVAH_API_TOKEN={a.api_token}"] if a.api_token else [])]
    for template, name, sets, envs in renders:
        cmd = [sys.executable, str(ROOT / "scripts/fuzzball/render_workflow.py"), str(ROOT / "workflows/fuzzball" / template),
               "--set", f"API_IMAGE={STOCK}", "--bundle", uri, "--bundle-sha256", digest, "--out", str(out / name)]
        for s in sets:
            cmd += ["--set", s]
        for e in envs:
            cmd += ["--env", e]
        subprocess.run(cmd, check=True)
    rel = out.relative_to(ROOT)
    t = a.tag
    wf = ("python3 -c \"import json,sys; d=json.load(sys.stdin); "
          "print(next(d[k] for k in ('id','ID','workflow_id','workflowId','workflowID') if k in d))\"")
    fb = "${FUZZBALL:-$HOME/bin/fuzzball}"
    (out / "COMMANDS.sh").write_text(f"""# PRAVAH Live Watch — tag {t}, commit {manifest['commit'][:10]}, bundle sha256 {digest}
# Run from the repo root after: {fb} context login
set -eu
FB={fb}
$FB version
# 1. Upload the code + asset bundle (kept 7 days)
$FB object put {rel}/bundle.tar.gz {uri} --ttl 7d
# 2. Server-side validation
$FB workflow validate {rel}/1_watch_build.yaml
$FB workflow validate {rel}/2_watch_serve.yaml
# 3. Build on BOTH GB10s in parallel (job a: history + 5-worker emulator ensemble; job b: DEM, river model, satellite, replay)
WF=$($FB workflow start {rel}/1_watch_build.yaml --name pravah-watch-build-{t} --output json | {wf}); echo "build workflow $WF"
echo "$WF" > {rel}/build_workflow_id
# follow with: $FB workflow log $WF watch-build-a --follow   /   $FB workflow log $WF watch-build-b --follow
# 4. After both jobs finish: start the 24/7 service (GPU API + dashboard on {a.scope} endpoints)
# WF2=$($FB workflow start {rel}/2_watch_serve.yaml --name pravah-watch-serve-{t} --output json | {wf}); echo "$WF2"
# $FB workflow describe "$WF2"      # lists the endpoint URLs to share
""")
    print(f"bundle {rel}/bundle.tar.gz sha256 {digest} ({bundle.stat().st_size / 1e6:.1f} MB, {len(manifest['files'])} asset files)")
    print(f"commands: {rel}/COMMANDS.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
