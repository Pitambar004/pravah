#!/usr/bin/env python3
"""Gate C GPU smoke test: prove CUDA works *inside* the allocated container.

Prints and (optionally) saves a JSON record of what was actually observed:
architecture, ``nvidia-smi`` output, torch build, device name and a real CUDA
matmul. Exits non-zero if CUDA is unusable, so a CPU fallback can never be
mistaken for a GPU run. Nothing here is estimated.

    python scripts/fuzzball/gpu_smoke.py --out /data/outputs/evaluation
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None, help="Directory to write gpu_smoke_<host>_<utc>.json")
    args = p.parse_args()
    rec: dict = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }
    smi = shutil.which("nvidia-smi")
    if smi:
        r = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30, check=False)
        rec["nvidia_smi"] = (r.stdout or r.stderr).strip()
    else:
        rec["nvidia_smi"] = None
    try:
        import torch
    except ImportError as exc:
        rec["error"] = f"torch import failed: {exc}"
        print(json.dumps(rec, indent=2))
        return 2
    rec.update(torch_version=torch.__version__, torch_cuda_build=torch.version.cuda,
               cuda_available=torch.cuda.is_available())
    ok = False
    if rec["cuda_available"]:
        rec["device_count"] = torch.cuda.device_count()
        rec["device_name"] = torch.cuda.get_device_name(0)
        rec["capability"] = list(torch.cuda.get_device_capability(0))
        a = torch.randn(2048, 2048, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        b = a @ a
        torch.cuda.synchronize()
        rec["matmul_2048_seconds"] = round(time.perf_counter() - t0, 6)
        rec["matmul_finite"] = bool(torch.isfinite(b).all().item())
        ok = rec["matmul_finite"]
    rec["cuda_verified"] = ok
    text = json.dumps(rec, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stamp = rec["time_utc"].replace(":", "").replace("-", "")
        (out / f"gpu_smoke_{rec['hostname']}_{stamp}.json").write_text(text + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
