"""MiniUNet inference over a whole georeferenced Sentinel-2 L1C raster.

Contract (matches ``docs/contracts.md`` and ``scripts/stitch.py``):

* Input: georeferenced 13-band S2 L1C (bands selected by ``S2_INDEXES``) or a
  6-band raster already ordered ``[B2, B3, B4, B8A, B11, B12]``. With
  ``input_units="dn"`` (default, Sen1Floods11 style) values are scaled ``/10000``;
  with ``input_units="reflectance"`` (already-scaled TOA reflectance, e.g. the P3
  ``s2_l1c_reflectance.tif``) they are used as-is. Both are clipped to ``[0, 1]`` —
  the project U-Net baseline scaling, *not* Prithvi preprocessing. A raster whose
  value range contradicts the declared units is refused. Sentinel-1 SAR is rejected by band count.
* Output, in the source CRS and pixel grid:
  ``observed_water.tif`` (uint8: 0 no water, 1 model-inferred water, 255 invalid),
  ``invalid_mask.tif`` (uint8: 1 invalid), ``water_probability.tif``
  (float32, nodata -9999), ``observed_metadata.json`` and, for the dashboard map,
  ``observed_water.geojson`` (EPSG:4326 polygons of class 1).

All timings recorded here are measured with ``time.perf_counter`` in this
process; nothing is estimated. ``cuda_verified`` is only true when the model
actually ran on a CUDA device.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import socket
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from rasterio.warp import transform_geom
from rasterio.windows import Window
from shapely.geometry import shape
from torch.nn import functional as F

from pravah.gis.tiles import fingerprint, make_tiles
from pravah.ml.dataset import S2_INDEXES
from pravah.ml.unet import MiniUNet

BANDS = ["B2", "B3", "B4", "B8A", "B11", "B12"]
CLASS_CODES = {"0": "no_water", "1": "model_inferred_water", "255": "invalid"}
SCIENCE_NOTE = (
    "Model-derived optical water from a small baseline U-Net; not a flood forecast, "
    "not water depth, not debris-flow or structural damage."
)
PROB_NODATA = -9999.0

ProgressFn = Callable[[int, int], None]


class CheckpointError(RuntimeError):
    """The checkpoint is missing, unreadable, or not a compatible MiniUNet."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_device(preference: str = "auto", require_cuda: bool = False) -> torch.device:
    """Pick a device without ever silently claiming GPU use.

    ``require_cuda`` (set on Fuzzball GPU services) makes a missing CUDA runtime
    a hard error instead of a quiet CPU fallback.
    """
    if preference not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("device must be auto, cpu, cuda or mps")
    if preference == "mps":  # Apple-silicon laptop GPU; never reported as CUDA
        if require_cuda:
            raise RuntimeError("require_cuda conflicts with device=mps")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
        return torch.device("mps")
    cuda = torch.cuda.is_available()
    if (require_cuda or preference == "cuda") and not cuda:
        raise RuntimeError("CUDA required but torch.cuda.is_available() is False; refusing CPU fallback")
    if preference == "cpu":
        if require_cuda:
            raise RuntimeError("require_cuda conflicts with device=cpu")
        return torch.device("cpu")
    return torch.device("cuda" if cuda else "cpu")


@dataclass
class ModelInfo:
    checkpoint_path: str
    checkpoint_sha256: str
    architecture: str
    bands: list[str]
    epoch: int | None = None
    steps: int | None = None
    source_data_pairs_sha256: str | None = None
    parameter_count: int = 0
    loaded_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_checkpoint(path: Path | str, device: torch.device) -> tuple[MiniUNet, ModelInfo]:
    """Load a ``train_ddp.py`` checkpoint strictly; reject anything incompatible."""
    p = Path(path)
    if not p.is_file():
        raise CheckpointError(f"Checkpoint not found: {p}")
    if p.stat().st_size == 0:
        raise CheckpointError(f"Checkpoint is empty: {p}")
    try:
        saved = torch.load(p, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises several unrelated types here
        raise CheckpointError(f"Cannot read checkpoint {p.name}: {exc}") from exc
    if not isinstance(saved, dict) or "model" not in saved:
        raise CheckpointError("Checkpoint must be a dict with a 'model' state_dict (train_ddp.py format)")
    arch = saved.get("architecture", "MiniUNet")
    if arch != "MiniUNet":
        raise CheckpointError(f"Unsupported architecture {arch!r}; this service only serves MiniUNet")
    bands = saved.get("bands", BANDS)
    if list(bands) != BANDS:
        raise CheckpointError(f"Checkpoint band order {bands} != expected {BANDS}")
    model = MiniUNet()
    try:
        model.load_state_dict(saved["model"], strict=True)
    except RuntimeError as exc:
        raise CheckpointError(f"State dict does not match MiniUNet: {exc}") from exc
    model.to(device).eval()
    info = ModelInfo(
        checkpoint_path=str(p.resolve()),
        checkpoint_sha256=sha256_file(p),
        architecture="MiniUNet",
        bands=list(BANDS),
        epoch=saved.get("epoch"),
        steps=saved.get("steps"),
        source_data_pairs_sha256=saved.get("source_data_pairs_sha256"),
        parameter_count=sum(t.numel() for t in model.parameters()),
        loaded_at_utc=_utcnow(),
    )
    return model, info


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def describe_raster(path: Path) -> dict[str, Any]:
    """Cheap header read used for scene listings and input validation."""
    with rasterio.open(path) as src:
        return {
            "width": src.width,
            "height": src.height,
            "band_count": src.count,
            "dtype": src.dtypes[0],
            "crs": src.crs.to_string() if src.crs else None,
            "bounds": list(src.bounds),
            "pixel_size": [abs(src.transform.a), abs(src.transform.e)],
            "model_compatible": src.crs is not None and src.count in (6, 13),
        }


def _pixel_area_m2(crs, transform) -> float | None:
    """Pixel area in m² for projected metre CRSs; None otherwise (never guessed)."""
    if crs is None or not crs.is_projected:
        return None
    try:
        factor = crs.linear_units_factor[1]
    except Exception:  # noqa: BLE001 - pyproj raises several types for unusual CRSs
        return None
    return abs(transform.a * transform.e) * factor * factor


INPUT_UNITS = ("dn", "reflectance")


def check_input_units(sample: np.ndarray, units: str) -> None:
    """Refuse obviously mismatched scaling (e.g. reflectance fed as DN -> near-black input)."""
    vals = sample[np.isfinite(sample) & (sample > 0)]
    if vals.size < 100:
        return
    p99 = float(np.percentile(vals, 99))
    if units == "dn" and p99 <= 2.0:
        raise ValueError(f"Input looks like reflectance (99th percentile {p99:.3f}); use input_units='reflectance'")
    if units == "reflectance" and p99 > 10.0:
        raise ValueError(f"Input looks like DN (99th percentile {p99:.0f}); use input_units='dn'")


@dataclass
class PredictionResult:
    run_id: str
    output_dir: str
    metadata: dict[str, Any]
    files: list[str] = field(default_factory=list)


class InferenceEngine:
    """Holds one loaded MiniUNet on one device; ``predict`` is not re-entrant.

    The API runs a single worker thread per GPU, so calls are serialised.
    """

    def __init__(self, checkpoint: Path | str, device: str = "auto", require_cuda: bool = False,
                 batch_size: int = 8):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.device = resolve_device(device, require_cuda)
        self.model, self.info = load_checkpoint(checkpoint, self.device)
        self.batch_size = batch_size

    # ------------------------------------------------------------------ runtime
    def runtime(self) -> dict[str, Any]:
        on_gpu = self.device.type == "cuda"
        return {
            "device": str(self.device),
            "cuda_verified": on_gpu,
            "gpu_name": torch.cuda.get_device_name(self.device) if on_gpu else None,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
        }

    # ------------------------------------------------------------------ predict
    def predict(
        self,
        raster_path: Path | str,
        output_dir: Path | str,
        *,
        run_id: str,
        threshold: float = 0.5,
        tile_size: int = 256,
        overlap: int = 32,
        max_polygons: int = 5000,
        progress: ProgressFn | None = None,
        extra_metadata: dict[str, Any] | None = None,
        input_units: str = "dn",
    ) -> PredictionResult:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be within [0, 1]")
        if input_units not in INPUT_UNITS:
            raise ValueError(f"input_units must be one of {INPUT_UNITS}")
        divisor = 10000.0 if input_units == "dn" else 1.0
        src_path = Path(raster_path)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        t_total = time.perf_counter()
        compute_s = 0.0
        on_gpu = self.device.type == "cuda"

        with rasterio.open(src_path) as src:
            if src.crs is None:
                raise ValueError("Input raster has no CRS; refusing to produce ungeoreferenced output")
            if src.count == 13:
                band_idx = [i + 1 for i in S2_INDEXES]
            elif src.count == 6:
                band_idx = list(range(1, 7))
            else:
                raise ValueError(f"Expected 13-band S2 L1C or 6-band [B2,B3,B4,B8A,B11,B12]; got {src.count} bands")
            h, w = src.height, src.width
            tiles = make_tiles(w, h, tile_size, overlap)
            tile_doc = {"raster": str(src_path.resolve()), "width": w, "height": h,
                        "tile_size": tile_size, "overlap": overlap, "tiles": tiles}
            weighted = np.zeros((h, w), dtype=np.float32)
            counts = np.zeros((h, w), dtype=np.float32)
            pad_to = int(math.ceil(tile_size / 8) * 8)
            profile = src.profile.copy()
            crs, transform = src.crs, src.transform
            # Units check on a central window before any compute
            cw = Window(max(0, w // 2 - 256), max(0, h // 2 - 256), min(512, w), min(512, h))
            check_input_units(src.read(band_idx, window=cw, masked=True).astype(np.float32).filled(np.nan),
                              input_units)

            done = 0
            for start in range(0, len(tiles), self.batch_size):
                batch = tiles[start:start + self.batch_size]
                arrays, masks = [], []
                for t in batch:
                    win = Window(t["col_off"], t["row_off"], t["width"], t["height"])
                    x = np.clip(src.read(band_idx, window=win).astype(np.float32) / divisor, 0.0, 1.0)
                    x = np.nan_to_num(x, nan=0.0)
                    masks.append(src.dataset_mask(window=win) > 0)
                    xp = np.zeros((6, pad_to, pad_to), dtype=np.float32)
                    xp[:, : t["height"], : t["width"]] = x
                    arrays.append(xp)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    xb = torch.from_numpy(np.stack(arrays)).to(self.device, non_blocking=True)
                    probs = F.softmax(self.model(xb), dim=1)[:, 1]
                    if on_gpu:
                        torch.cuda.synchronize(self.device)
                    elif self.device.type == "mps":
                        torch.mps.synchronize()
                    probs = probs.float().cpu().numpy()
                compute_s += time.perf_counter() - t0
                for t, prob, valid in zip(batch, probs, masks):
                    ys = slice(t["row_off"], t["row_off"] + t["height"])
                    xs = slice(t["col_off"], t["col_off"] + t["width"])
                    p = prob[: t["height"], : t["width"]]
                    weighted[ys, xs] += np.where(valid, p, 0.0)
                    counts[ys, xs] += valid
                done += len(batch)
                if progress:
                    progress(done, len(tiles))

        valid = counts > 0
        pwater = np.full((h, w), np.nan, dtype=np.float32)
        pwater[valid] = weighted[valid] / counts[valid]
        cls = np.full((h, w), 255, dtype=np.uint8)
        cls[valid] = (pwater[valid] >= threshold).astype(np.uint8)
        invalid = (~valid).astype(np.uint8)

        base = profile
        for k in ("blockxsize", "blockysize", "photometric", "interleave"):
            base.pop(k, None)
        base.update(count=1, compress="deflate", tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
        if w < 256 or h < 256:  # tiled GeoTIFF blocks must not exceed tiny rasters awkwardly
            base.update(tiled=False)
            base.pop("blockxsize", None)
            base.pop("blockysize", None)
        with rasterio.open(out / "observed_water.tif", "w", **{**base, "dtype": "uint8", "nodata": 255}) as dst:
            dst.write(cls, 1)
        with rasterio.open(out / "invalid_mask.tif", "w", **{**base, "dtype": "uint8", "nodata": None}) as dst:
            dst.write(invalid, 1)
        with rasterio.open(out / "water_probability.tif", "w", **{**base, "dtype": "float32",
                                                                  "nodata": PROB_NODATA}) as dst:
            dst.write(np.where(valid, pwater, PROB_NODATA).astype(np.float32), 1)

        geojson, n_total = water_polygons(cls, transform, crs, max_polygons=max_polygons)
        (out / "observed_water.geojson").write_text(json.dumps(geojson) + "\n")

        water_px = int((cls == 1).sum())
        px_area = _pixel_area_m2(crs, transform)
        metadata: dict[str, Any] = {
            "run_id": run_id,
            "engine": "pravah.inference",
            "model": "MiniUNet baseline (Team 12, not Prithvi)",
            "model_info": self.info.to_dict(),
            "runtime": self.runtime(),
            "source_raster": str(src_path.resolve()),
            "source_raster_sha256": sha256_file(src_path),
            "crs": crs.to_string(),
            "transform": list(transform)[:6],
            "raster_width": w,
            "raster_height": h,
            "bands_used": BANDS,
            "input_units": input_units,
            "input_scaling": ("DN / 10000" if input_units == "dn" else "reflectance as-is")
            + ", clipped to [0, 1] (baseline U-Net scaling)",
            "threshold": threshold,
            "tile_size": tile_size,
            "overlap": overlap,
            "tile_count": len(tiles),
            "tile_manifest_sha256": fingerprint(tile_doc),
            "batch_size": self.batch_size,
            "valid_pixel_count": int(valid.sum()),
            "invalid_pixel_count": int(invalid.sum()),
            "water_pixel_count": water_px,
            "water_area_m2": water_px * px_area if px_area is not None else None,
            "water_area_note": None if px_area is not None else "CRS not projected in metres; area not computed",
            "water_polygon_count": n_total,
            "water_polygons_written": len(geojson["features"]),
            "class_codes": CLASS_CODES,
            "timings_seconds": {
                "model_compute": round(compute_s, 4),
                "end_to_end": round(time.perf_counter() - t_total, 4),
            },
            "created_at_utc": _utcnow(),
            "science_note": SCIENCE_NOTE,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        (out / "observed_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
        files = sorted(p.name for p in out.iterdir() if p.is_file())
        return PredictionResult(run_id=run_id, output_dir=str(out), metadata=metadata, files=files)


def water_polygons(cls: np.ndarray, transform, crs, max_polygons: int = 5000) -> tuple[dict[str, Any], int]:
    """Vectorise class-1 pixels to EPSG:4326 GeoJSON for the dashboard map.

    Polygons are ranked by pixel count; at most ``max_polygons`` are written and
    the total is returned so truncation is never hidden.
    """
    water = (cls == 1).astype(np.uint8)
    px_area = _pixel_area_m2(crs, transform)
    feats = []
    for geom, value in shapes(water, mask=water.astype(bool), transform=transform, connectivity=8):
        if value != 1:
            continue
        # area in source CRS units (holes subtracted); only reported when the CRS is metric
        area = shape(geom).area * (crs.linear_units_factor[1] ** 2 if px_area is not None else 1.0)
        feats.append((area, geom))
    total = len(feats)
    feats.sort(key=lambda f: f[0], reverse=True)
    features = []
    for i, (area, geom) in enumerate(feats[:max_polygons]):
        g = transform_geom(crs, "EPSG:4326", geom, precision=6)
        features.append({
            "type": "Feature",
            "geometry": g,
            "properties": {
                "id": i,
                "class": "model_inferred_water",
                "area_m2": round(area, 1) if px_area is not None else None,
                "label": "Model-inferred water",
            },
        })
    return {"type": "FeatureCollection", "features": features}, total
