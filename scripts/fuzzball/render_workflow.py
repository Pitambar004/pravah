#!/usr/bin/env python3
"""Render a PRAVAH Fuzzball workflow template into a submittable Fuzzfile.

Templates in ``workflows/fuzzball/*.yaml.tmpl`` use ``{{TOKEN}}`` placeholders.
This script fills them, refuses to emit a file with any token left unfilled or
an invalid value, and runs structural checks (version, image URIs, GPU request,
volume mounts, endpoint scopes). Rendered output goes to ``build/fuzzball/``
(gitignored) because it may contain environment values.

Passing these checks is NOT proof the file is valid on Fuzzball v4.3.1 — run
``fuzzball workflow validate`` (or the UI validator) on the rendered file.

    python scripts/fuzzball/render_workflow.py workflows/fuzzball/07_serve.yaml.tmpl \
        --set API_IMAGE=docker://REGISTRY/pravah-api:TAG \
        --set DASHBOARD_IMAGE=docker://REGISTRY/pravah-dashboard:TAG \
        --set ENDPOINT_SCOPE=group
"""
from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

TOKEN = re.compile(r"\{\{\s*([A-Z0-9_]+)\s*\}\}")
DEFAULTS = {
    "VOLUME_PROVISIONER": "team12",  # config/fuzzball.yaml
    "VOLUME_NAME": "Pravah-Dataset",
    "CHECKPOINT": "/data/models/checkpoints/checkpoint_last.pt",
    "ENDPOINT_SCOPE": "group",
    # Training (05_train_*): identical defaults for 1-node and 2-node so the workloads match.
    "PAIRS": "/data/manifests/s2_pairs.json",
    "GLOBAL_BATCH": "8",
    "EPOCHS": "3",
    "MAX_STEPS": "0",
    "MASTER_PORT": "29500",
    # World training-data intake (03_download_world, 04_prepare_world)
    "WORLD_ROOT": "/data/training/expanded_20261114",
    "DOWNLOAD_WORKERS": "3",
    "RETRY_FAILED": "0",
    # Private GPU API (08_api_gpu)
    "MODEL_DIR": "/data/models/pravah/unet_mps_20261115T021950Z",
    "SCENES_DIR": "/data/outputs/p3_audit_20261114T214402Z/s2_prepared_full",
    # 06_inference_batch: "reflectance" for P3 s2_l1c_reflectance.tif, "dn" for raw L1C / Sen1Floods11 chips
    "INPUT_UNITS": "dn",
}
MULTINODE_IMPLEMENTATIONS = {"generic", "openmpi", "mpich"}
SCOPES = {"user", "group", "organization", "public"}
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
SAFE_SCENE = re.compile(r"^(?!/)(?!.*\.\.)[A-Za-z0-9_./-]+\.tiff?$")


class RenderError(ValueError):
    pass


def render(template: str, values: dict[str, str]) -> str:
    merged = {**DEFAULTS, **values}
    missing = sorted({m for m in TOKEN.findall(template) if m not in merged})
    if missing:
        raise RenderError(f"Missing values for: {', '.join(missing)} (use --set KEY=VALUE)")
    return TOKEN.sub(lambda m: merged[m.group(1)], template)


def _check_value(key: str, value: str) -> None:
    if key.endswith("_IMAGE") and not re.match(r"^docker://[^\s]+:[^\s/]+$|^docker://[^\s]+@sha256:[0-9a-f]{64}$", value):
        raise RenderError(f"{key} must be docker://<image>:<tag> or @sha256 digest (got {value!r})")
    if key == "ENDPOINT_SCOPE" and value not in SCOPES:
        raise RenderError(f"ENDPOINT_SCOPE must be one of {sorted(SCOPES)}")
    if key == "RUN_ID" and not SAFE_ID.match(value):
        raise RenderError("RUN_ID may only contain letters, digits, '_' and '-'")
    if key == "SCENE_ID" and not SAFE_SCENE.match(value):
        raise RenderError("SCENE_ID must be a relative .tif path under /data/raw/sentinel2 without '..'")
    if key == "CHECKPOINT_SHA256" and not SHA256.match(value):
        raise RenderError("CHECKPOINT_SHA256 must be 64 lowercase hex characters")
    if key in ("CHECKPOINT", "PAIRS", "WORLD_ROOT", "MODEL_DIR", "SCENES_DIR") and (not value.startswith("/data/") or ".." in value or " " in value):
        raise RenderError(f"{key} must live on the mounted volume (/data/...) without '..' or spaces")
    if key == "DOWNLOAD_WORKERS" and not (value.isdigit() and 1 <= int(value) <= 8):
        raise RenderError("DOWNLOAD_WORKERS must be 1-8")
    if key == "RETRY_FAILED" and value not in ("0", "1"):
        raise RenderError("RETRY_FAILED must be 0 or 1")
    if key == "INPUT_UNITS" and value not in ("dn", "reflectance"):
        raise RenderError("INPUT_UNITS must be dn or reflectance")
    if key in ("GLOBAL_BATCH", "EPOCHS", "MAX_STEPS", "MASTER_PORT"):
        if not value.isdigit():
            raise RenderError(f"{key} must be a non-negative integer")
        if key in ("GLOBAL_BATCH", "EPOCHS") and int(value) < 1:
            raise RenderError(f"{key} must be >= 1")
        if key == "GLOBAL_BATCH" and int(value) % 2:
            raise RenderError("GLOBAL_BATCH must be even so 1-node and 2-node runs share the same workload")
        if key == "MASTER_PORT" and not 1024 <= int(value) <= 65535:
            raise RenderError("MASTER_PORT must be 1024-65535")


def validate(doc: dict[str, Any]) -> list[str]:
    """Structural checks against the documented v4 Fuzzfile shape. Returns notes."""
    if doc.get("version") != "v4":
        raise RenderError("version must be v4")
    vols = doc.get("volumes") or {}
    units = {**(doc.get("jobs") or {}), **(doc.get("services") or {})}
    if not units:
        raise RenderError("workflow defines no jobs or services")
    notes = []
    for name, u in units.items():
        uri = ((u.get("image") or {}).get("uri")) or ""
        if not uri.startswith("docker://"):
            raise RenderError(f"{name}: image.uri must start with docker://")
        for path, vol in (u.get("mounts") or {}).items():
            vol = vol.get("volume") if isinstance(vol, dict) else vol  # 4.3.1 form: {volume: name}
            if not str(path).startswith("/") or vol not in vols:
                raise RenderError(f"{name}: mount {path} -> {vol} references an undefined volume")
        res = u.get("resource") or {}
        if "cores" not in (res.get("cpu") or {}) or "size" not in (res.get("memory") or {}):
            raise RenderError(f"{name}: resource.cpu.cores and resource.memory.size are required")
        gpus = (res.get("devices") or {}).get("nvidia.com/gpu")
        mn = u.get("multinode")
        if mn is not None:
            if not isinstance(mn.get("nodes"), int) or mn["nodes"] < 2:
                raise RenderError(f"{name}: multinode.nodes must be an integer >= 2")
            if mn.get("implementation") not in MULTINODE_IMPLEMENTATIONS:
                raise RenderError(f"{name}: multinode.implementation must be one of {sorted(MULTINODE_IMPLEMENTATIONS)}")
            notes.append(f"{name}: multinode nodes={mn['nodes']} implementation={mn['implementation']} "
                         f"(gpus per node={gpus or 0}) — confirm 4.3.1 syntax with CIQ")
        if not (u.get("command") or u.get("script")):
            raise RenderError(f"{name}: needs command or script")
        notes.append(f"{name}: image={uri} gpus={gpus or 0}")
        net = u.get("network") or {}
        ports = {p["name"] for p in net.get("ports") or []}
        for ep in net.get("endpoints") or []:
            if ep.get("port-name") not in ports:
                raise RenderError(f"{name}: endpoint {ep.get('name')} references unknown port")
            if ep.get("scope") not in SCOPES:
                raise RenderError(f"{name}: endpoint scope must be one of {sorted(SCOPES)}")
            if ep["scope"] == "public":
                notes.append(f"WARNING {name}/{ep['name']}: public endpoint — anyone with the URL can reach it")
        for dep in u.get("depends-on") or []:
            if dep.get("name") not in units:
                raise RenderError(f"{name}: depends-on unknown {dep.get('name')}")
    return notes


def apply_extras(doc: dict[str, Any], image_secret: str | None, env: dict[str, list[str]]) -> None:
    units = {**(doc.get("jobs") or {}), **(doc.get("services") or {})}
    for name, u in units.items():
        if image_secret:
            u["image"]["secret"] = image_secret
        extra = env.get(name, []) + env.get("*", [])
        if extra:
            keys = {e.split("=", 1)[0] for e in extra}
            u["env"] = [e for e in u.get("env") or [] if e.split("=", 1)[0] not in keys] + extra



# --- Bundle mode: stock NVIDIA image + code bundle from the Fuzzball object store (no registry) -------
BUNDLE_IMAGE = "docker://nvcr.io/nvidia/pytorch:25.10-py3"  # recorded as torch 2.9.0a0+nv25.10 on Team 12 GB10
WHEELS = "/data/training/expanded_20261114/environment/wheels"  # checksum-listed rasterio/affine wheels
INSTALL = (f'(cd {WHEELS} && sha256sum -c --quiet SHA256SUMS) && python -m pip install --disable-pip-version-check '
           f'--no-index --no-deps --find-links {WHEELS} rasterio==1.5.2 affine==3.0.1 >/dev/null')
SHA256 = re.compile(r"^[0-9a-f]{64}$")
FB_URI = re.compile(r"^fb://[A-Za-z0-9_./-]+$")


def apply_bundle(doc: dict[str, Any], uri: str, sha: str, results: str | None, run_id: str | None,
                 extra_ingress: list[str] | None = None) -> None:
    """Rewrite every job to run the repo code from a checksum-verified bundle on the stock image."""
    if not FB_URI.match(uri) or not uri.endswith(".tar.gz"):
        raise RenderError("--bundle must be an fb://... .tar.gz object URI")
    if not SHA256.match(sha):
        raise RenderError("--bundle-sha256 must be 64 lowercase hex characters")
    if results is not None and not FB_URI.match(results):
        raise RenderError("--results-prefix must be an fb://... URI")
    code = f"/data/pravah_code/{sha[:16]}"
    transfer = {"use": "ephemeral", "size": "2GiB",
                "ingress": [{"source": {"uri": uri}, "destination": {"uri": "file://in/bundle.tar.gz"}}]}
    for item in extra_ingress or []:  # fb://...=name  ->  /transfer/in/name
        src, _, name = item.partition("=")
        if not FB_URI.match(src) or not re.match(r"^[A-Za-z0-9_.-]+$", name or ""):
            raise RenderError("--ingress must be fb://...=<file name>")
        transfer["ingress"].append({"source": {"uri": src}, "destination": {"uri": f"file://in/{name}"}})
    boot = "\n".join([
        "#!/bin/sh", "set -eu", f"CODE={code}",
        'if [ "$(cat "$CODE/.bundle_sha256" 2>/dev/null)" != "' + sha + '" ]; then',
        f'  echo "{sha}  /transfer/in/bundle.tar.gz" | sha256sum -c -',
        '  TMP="$CODE.tmp.$$"; rm -rf "$TMP"; mkdir -p "$TMP"',
        '  tar -xzf /transfer/in/bundle.tar.gz --no-same-owner -C "$TMP"',
        f'  echo {sha} > "$TMP/.bundle_sha256"',
        '  mkdir -p "$(dirname "$CODE")"; rm -rf "$CODE"; mv "$TMP" "$CODE"',
        "fi",
        'export PYTHONPATH="$CODE/src"', ""])
    units = {**(doc.get("jobs") or {}), **(doc.get("services") or {})}
    for u in units.values():
        u["image"] = {"uri": BUNDLE_IMAGE}
        u.setdefault("mounts", {})["/transfer"] = {"volume": "transfer"}
        if "command" in u:
            body = " ".join(shlex.quote(str(c)) for c in u.pop("command"))
        else:
            body = u.get("script", "")
            body = body.split("\n", 1)[1] if body.startswith("#!") else body
            body = body.replace("set -eu\n", "", 1)
        body = body.replace("/opt/pravah", "$CODE")
        needs_raster = "train_ddp.py" in body or "download_training_prepare.py" in body
        if "exec_all" in body:  # every node needs the wheels and the code path
            body = body.replace('exec_all "torchrun', f'exec_all "{INSTALL} && PYTHONPATH=$CODE/src torchrun')
        elif needs_raster:
            body = INSTALL + "\n" + body
        tail = ""
        if results and run_id and f"/data/outputs/training/{run_id}" in body:
            transfer.setdefault("egress", []).append(
                {"source": {"uri": f"file://out/{run_id}.tar.gz"}, "destination": {"uri": f"{results}/{run_id}.tar.gz"}})
            tail = ("\nfinish() { code=$?; trap - EXIT; mkdir -p /transfer/out; "
                    f"tar -czf /transfer/out/{run_id}.tar.gz -C /data/outputs/training {run_id} || true; exit $code; }}")
            boot_job = boot + tail.lstrip("\n") + "\ntrap finish EXIT\n"
        else:
            boot_job = boot
        u["script"] = boot_job + body.strip() + "\n"
    doc.setdefault("volumes", {})["transfer"] = transfer


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("template", type=Path)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--env", action="append", default=[], metavar="UNIT:KEY=VALUE",
                   help="Extra env for a job/service ('*' = all), e.g. api:PRAVAH_API_TOKEN=...")
    p.add_argument("--image-secret", default=None, help="Fuzzball secret ref for a private registry")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--bundle", default=None, help="fb://.../bundle.tar.gz: run on the stock NVIDIA image from this code bundle")
    p.add_argument("--bundle-sha256", default=None, help="SHA-256 of the bundle (required with --bundle)")
    p.add_argument("--results-prefix", default=None, help="fb://... prefix that training jobs egress their run folder to")
    p.add_argument("--ingress", action="append", default=[], metavar="fb://...=NAME",
                   help="Extra object staged to /transfer/in/NAME in bundle mode (e.g. a checkpoint)")
    args = p.parse_args(argv)
    try:
        values = dict(kv.split("=", 1) for kv in args.set)
        for k, v in {**DEFAULTS, **values}.items():
            _check_value(k, v)
        env: dict[str, list[str]] = {}
        for item in args.env:
            unit, kv = item.split(":", 1)
            if "=" not in kv:
                raise RenderError(f"--env {item!r} must be UNIT:KEY=VALUE")
            env.setdefault(unit, []).append(kv)
        doc = yaml.safe_load(render(args.template.read_text(), values))
        apply_extras(doc, args.image_secret, env)
        if args.bundle:
            if not args.bundle_sha256:
                raise RenderError("--bundle requires --bundle-sha256")
            apply_bundle(doc, args.bundle, args.bundle_sha256, args.results_prefix, values.get("RUN_ID"),
                         args.ingress)
        notes = validate(doc)
    except (RenderError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    out = args.out or Path("build/fuzzball") / args.template.name.removesuffix(".tmpl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Rendered from {args.template} — do not commit (may contain env values)\n"
                   + yaml.safe_dump(doc, sort_keys=False, width=120))
    print(f"wrote {out}")
    for n in notes:
        print(" ", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
