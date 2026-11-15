"""Real provider-image pairs; preview differences are visual review evidence."""

import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def visual_difference(before_bytes, current_bytes):
    a, b = [np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=float) for data in [before_bytes, current_bytes]]
    if a.shape != b.shape:
        raise ValueError("Provider previews have different pixel grids; visual differencing withheld")
    common = (a.max(2) > 8) & (b.max(2) > 8)
    delta = np.abs(a-b).mean(2)/255
    changed = common & (delta >= .18)
    overlay = b.astype("uint8").copy()
    overlay[changed] = (.55*overlay[changed] + .45*np.array([255, 65, 80])).astype("uint8")
    stream = io.BytesIO(); Image.fromarray(overlay).save(stream, format="PNG")
    fraction = float(changed.sum()/common.sum()) if common.any() else None
    return {"changed_preview_fraction": fraction, "common_preview_pixels": int(common.sum()),
            "review_recommended": fraction is None or fraction >= .1,
            "method": "Mean RGB difference ≥0.18 in nonblack provider-preview pixels; analyst review threshold ≥10%",
            "measurement_scope": "Visual appearance only; clouds, snow, shadows and alignment can cause differences",
            "water_area_change_m2": None, "water_level_m": None, "depth_m": None,
            "glacier_displacement_m": None, "forecast_accuracy": None}, stream.getvalue()


def make_pair(root, before, current, label):
    if before["acquired_at_utc"] >= current["acquired_at_utc"]:
        raise ValueError("Image acquisitions must be ordered")
    a, b = [root / row["preview_file"] for row in [before, current]]
    summary, overlay = visual_difference(a.read_bytes(), b.read_bytes())
    digest = hashlib.sha256((before["preview_sha256"]+current["preview_sha256"]).encode()).hexdigest()
    path = root / "previews" / (digest + "_change.png")
    path.write_bytes(overlay)
    cloud = max(before.get("cloud_cover_percent") or 0, current.get("cloud_cover_percent") or 0)
    summary.update(cloud_limited=cloud > 20, maximum_scene_cloud_percent=cloud)
    if cloud > 20:
        summary["review_recommended"] = True
    return {"id": digest[:24], "label": label, "kind": "provider_preview_comparison",
            "before": before, "current": current, "summary": summary,
            "overlay_file": str(path.relative_to(root)), "overlay_sha256": hashlib.sha256(overlay).hexdigest(),
            "overlay_data_url": "data:image/png;base64,"+base64.b64encode(overlay).decode(),
            "model_input_available": False,
            "historical_eligibility": False,
            "availability_note": "Downloaded now; historical availability is unverified"}


def load_image_pairs(root: Path):
    doc = json.loads((root / "pairs.json").read_text())
    for pair in doc["pairs"].values():
        for row in [pair["before"], pair["current"]]:
            path = root / row["preview_file"]
            if root.resolve() not in path.resolve().parents or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != row["preview_sha256"]:
                raise ValueError("Image-pair source integrity mismatch")
        path = root / pair["overlay_file"]
        if root.resolve() not in path.resolve().parents or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != pair["overlay_sha256"]:
            raise ValueError("Image comparison integrity mismatch")
    return doc
