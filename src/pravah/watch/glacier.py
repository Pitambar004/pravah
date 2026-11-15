"""Glacier surface displacement by batched FFT normalized cross-correlation (GPU, PyTorch).

Two co-registered Sentinel-2 NIR (B08, 10 m) images on the same grid are cut into
overlapping chips; each chip pair's phase-correlation peak gives the sub-pixel offset.
Speed = offset × pixel size / days. Only chips that are mostly valid snow/ice in both
dates, with a clear correlation peak, are kept. Results describe apparent surface motion
(including residual co-registration error, typically a fraction of a pixel), not ice flux.
"""
from __future__ import annotations

import numpy as np


def _device(name: str | None = None):
    import torch
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def track(a: np.ndarray, b: np.ndarray, usable: np.ndarray, chip: int = 64, step: int = 32,
          min_usable: float = 0.7, min_snr: float = 6.0, device: str | None = None, batch: int = 4096) -> dict:
    """Return per-chip row/col centres, (dy, dx) pixel offsets b relative to a, SNR and keep mask."""
    import torch
    dev = _device(device)
    H, W = a.shape
    rows = np.arange(0, H - chip + 1, step)
    cols = np.arange(0, W - chip + 1, step)
    if len(rows) == 0 or len(cols) == 0:
        return {"row": np.array([]), "col": np.array([]), "dy": np.array([]), "dx": np.array([]),
                "snr": np.array([]), "keep": np.zeros(0, bool)}
    ta = torch.as_tensor(np.nan_to_num(a, nan=0.0), dtype=torch.float32, device=dev)
    tb = torch.as_tensor(np.nan_to_num(b, nan=0.0), dtype=torch.float32, device=dev)
    tu = torch.as_tensor(usable.astype("float32"), device=dev)
    ca = ta.unfold(0, chip, step).unfold(1, chip, step).reshape(-1, chip, chip)
    cb = tb.unfold(0, chip, step).unfold(1, chip, step).reshape(-1, chip, chip)
    cu = tu.unfold(0, chip, step).unfold(1, chip, step).reshape(-1, chip, chip).mean((1, 2))
    win = torch.hann_window(chip, device=dev)
    win = win[:, None] * win[None, :]
    dys, dxs, snrs = [], [], []
    for s in range(0, ca.shape[0], batch):
        x, y = ca[s:s + batch], cb[s:s + batch]
        x = (x - x.mean((1, 2), keepdim=True)) / (x.std((1, 2), keepdim=True) + 1e-6) * win
        y = (y - y.mean((1, 2), keepdim=True)) / (y.std((1, 2), keepdim=True) + 1e-6) * win
        R = torch.fft.fft2(y) * torch.conj(torch.fft.fft2(x))
        R = R / (R.abs() + 1e-9)
        r = torch.fft.fftshift(torch.fft.ifft2(R).real, dim=(1, 2))
        flat = r.reshape(r.shape[0], -1)
        peak, idx = flat.max(1)
        py, px = idx // chip, idx % chip
        # parabolic sub-pixel refinement on the 3-point neighbourhood
        def refine(p, axis, r=r, px=px, py=py):
            lo = (p - 1).clamp(0, chip - 1)
            hi = (p + 1).clamp(0, chip - 1)
            n = torch.arange(r.shape[0], device=dev)
            if axis == 0:
                c0, cm, cp = r[n, p, px], r[n, lo, px], r[n, hi, px]
            else:
                c0, cm, cp = r[n, py, p], r[n, py, lo], r[n, py, hi]
            den = cm - 2 * c0 + cp
            return p.float() + torch.where(den.abs() > 1e-9, 0.5 * (cm - cp) / den, torch.zeros_like(den))
        dys.append(refine(py, 0) - chip // 2)
        dxs.append(refine(px, 1) - chip // 2)
        snrs.append(peak / (flat.abs().mean(1) + 1e-9))
    dy = torch.cat(dys).cpu().numpy()
    dx = torch.cat(dxs).cpu().numpy()
    snr = torch.cat(snrs).cpu().numpy()
    usable_frac = cu.cpu().numpy()
    rr, cc = np.meshgrid(rows + chip // 2, cols + chip // 2, indexing="ij")
    keep = (usable_frac >= min_usable) & (snr >= min_snr) & (np.hypot(dy, dx) < chip / 4)
    return {"row": rr.ravel(), "col": cc.ravel(), "dy": dy, "dx": dx, "snr": snr, "keep": keep,
            "device": str(dev)}


def zone_speeds(result: dict, zones: np.ndarray, zone_ids: list[str], pixel_m: float, days: float,
                min_vectors: int = 5) -> list[dict]:
    """Median speed (m/day) of kept vectors per zone (zones raster indexed by chip centre)."""
    out = []
    if days <= 0:
        raise ValueError("Image pair must be in time order")
    speed = np.hypot(result["dy"], result["dx"]) * pixel_m / days
    zone_at = zones[result["row"].astype(int), result["col"].astype(int)] if len(result["row"]) else np.array([], int)
    for i, zid in enumerate(zone_ids, start=1):
        sel = result["keep"] & (zone_at == i)
        n = int(sel.sum())
        out.append({"zone_id": zid, "median_speed_m_per_day": float(np.median(speed[sel])) if n >= min_vectors else None,
                    "p90_speed_m_per_day": float(np.percentile(speed[sel], 90)) if n >= min_vectors else None,
                    "vectors": n})
    return out
