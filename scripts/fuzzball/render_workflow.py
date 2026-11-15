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
}
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
    if key == "CHECKPOINT" and not value.startswith("/data/"):
        raise RenderError("CHECKPOINT must live on the mounted volume (/data/...)")


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("template", type=Path)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--env", action="append", default=[], metavar="UNIT:KEY=VALUE",
                   help="Extra env for a job/service ('*' = all), e.g. api:PRAVAH_API_TOKEN=...")
    p.add_argument("--image-secret", default=None, help="Fuzzball secret ref for a private registry")
    p.add_argument("--out", type=Path, default=None)
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
