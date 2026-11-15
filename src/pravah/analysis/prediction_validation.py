"""Score a flood *prediction* against an independent post-event satellite observation.

This is the module that turns "the model predicted X" into evidence. It answers
three questions, in order, and refuses to answer the later ones if an earlier one fails:

1. **Was it a genuine prediction?** (temporal integrity) Every model input must have
   been acquired *and* available at or before the prediction's ``input_cutoff_utc``,
   and that cutoff must be strictly earlier than the reference observation it is
   scored against. An input with an unknown availability time is not eligible.
   A cutoff after the event start is allowed but labelled ``post_event_nowcast``,
   never ``early_warning``.
2. **How well does it match what was observed?** Pixel confusion on the reference
   grid, pixels invalid in either raster excluded (``not_comparable``, never "dry").
   Reported with both ML names (precision/recall/F1/IoU) and the flood-forecast
   verification names (POD, FAR, CSI).
3. **Is it better than doing nothing clever?** The same pixels are scored for a
   naive *persistence* baseline (the pre-event water map). A model only shows skill
   if it beats that baseline on identical pixels.

Raster contract (all inputs): uint8, 0 = no water/flood, 1 = water/flood, 255 = invalid.
A float probability prediction is thresholded first (``>= threshold`` -> 1).

Targets:
    ``water_extent`` - truth is all water in the reference (rivers included).
    ``new_water``    - truth is water in the reference that was NOT water in the
                       pre-event baseline; pixels already wet before are excluded.
                       This is the stricter "did it predict the flood" test.

Agreement codes (agreement.tif): 0 correct negative, 1 hit, 2 false alarm,
3 miss, 255 not comparable.

The reference is a *model-derived* optical observation unless stated otherwise in
its metadata; scores are agreement with that observation, not with ground truth.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from pravah.inference.engine import _pixel_area_m2, water_polygons
from pravah.ml.metrics import confusion, score

SCHEMA = "pravah-prediction-validation-v1"
TARGETS = ("water_extent", "new_water")
INVALID = 255
AGREEMENT_CODES = {"0": "correct_negative", "1": "hit", "2": "false_alarm", "3": "miss", "255": "not_comparable"}
PREDICTION_FILES = ("predicted_flood.tif", "predicted_flood_probability.tif")


def parse_utc(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def check_integrity(pred_meta: dict[str, Any], ref_metas: list[dict[str, Any]],
                    event_time_utc: str | None = None) -> dict[str, Any]:
    """Temporal-leakage and provenance checks. ``eligible`` False means: do not score."""
    violations: list[str] = []
    warnings: list[str] = []
    cutoff = parse_utc(pred_meta.get("input_cutoff_utc"))
    if cutoff is None:
        violations.append("prediction_metadata.input_cutoff_utc missing or not ISO-8601")
    inputs = pred_meta.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        violations.append("prediction_metadata.inputs must list every model input with acquired/available times")
        inputs = []
    input_ids = set()
    for i, item in enumerate(inputs):
        name = item.get("name") or item.get("id") or f"inputs[{i}]"
        input_ids.update(str(item.get(k)) for k in ("id", "scene_id", "product_id") if item.get(k))
        acq = parse_utc(item.get("acquired_at_utc") or item.get("valid_to_utc"))
        avail = parse_utc(item.get("available_at_utc"))
        if acq is None:
            violations.append(f"{name}: acquired_at_utc (or valid_to_utc) missing")
        if avail is None:
            violations.append(f"{name}: available_at_utc missing — unknown availability is not cutoff-eligible")
        if cutoff and acq and acq > cutoff:
            violations.append(f"{name}: acquired {acq.isoformat()} after cutoff {cutoff.isoformat()}")
        if cutoff and avail and avail > cutoff:
            violations.append(f"{name}: available {avail.isoformat()} after cutoff {cutoff.isoformat()}")
    issued = parse_utc(pred_meta.get("issued_at_utc"))
    if cutoff and issued and issued < cutoff:
        violations.append("issued_at_utc is earlier than input_cutoff_utc")
    if pred_meta.get("is_synthetic"):
        violations.append("prediction is marked is_synthetic; fixtures are never scored as evidence")

    ref_times = []
    for ref in ref_metas:
        rid = ref.get("run_id") or "reference"
        if ref.get("is_synthetic"):
            violations.append(f"{rid}: reference is marked is_synthetic")
        r_acq = parse_utc(ref.get("acquired_at_utc"))
        if r_acq is None:
            violations.append(f"{rid}: reference acquired_at_utc missing")
            continue
        ref_times.append(r_acq)
        if cutoff and r_acq <= cutoff:
            violations.append(f"{rid}: reference acquired {r_acq.isoformat()} is not after the input cutoff "
                              "(the model could have seen it)")
        for k in ("scene_id", "product_id"):
            if ref.get(k) and str(ref[k]) in input_ids:
                violations.append(f"{rid}: reference {k} {ref[k]} is also a model input")

    event = parse_utc(event_time_utc)
    lead_h = None
    claim = "unknown"
    if cutoff and event:
        lead_h = round((event - cutoff).total_seconds() / 3600, 2)
        if cutoff <= event:
            claim = "early_warning"
        else:
            claim = "post_event_nowcast"
            warnings.append("Input cutoff is after the event start: this is a nowcast/mapping test, "
                            "not an early warning.")
    elif event is None:
        warnings.append("No event time given: lead time cannot be computed.")
    return {
        "eligible": not violations,
        "violations": violations,
        "warnings": warnings,
        "input_cutoff_utc": cutoff.isoformat() if cutoff else None,
        "issued_at_utc": issued.isoformat() if issued else None,
        "event_time_utc": event.isoformat() if event else None,
        "reference_acquired_utc": sorted(t.isoformat() for t in ref_times),
        "lead_time_hours_before_event": lead_h,
        "claim_type": claim,
        "inputs_checked": len(inputs),
    }


def read_prediction(pred_dir: Path, threshold: float | None = None):
    """Return (class_array uint8 0/1/255, profile) on the prediction's own grid."""
    cls_path, prob_path = (pred_dir / f for f in PREDICTION_FILES)
    if cls_path.is_file():
        with rasterio.open(cls_path) as src:
            arr, prof = src.read(1), src.profile.copy()
        if not np.isin(arr, (0, 1, INVALID)).all():
            raise ValueError("predicted_flood.tif has values outside {0, 1, 255}")
        return arr.astype(np.uint8), prof
    if prob_path.is_file():
        if threshold is None:
            raise ValueError("probability prediction requires --threshold")
        with rasterio.open(prob_path) as src:
            p, prof = src.read(1, masked=True), src.profile.copy()
        data = np.asarray(p.filled(np.nan), dtype=np.float64)
        bad = ~np.isfinite(data) | (data < 0) | (data > 1)
        out = np.where(data >= threshold, 1, 0).astype(np.uint8)
        out[bad] = INVALID
        return out, prof
    raise FileNotFoundError(f"{pred_dir}: needs {PREDICTION_FILES[0]} or {PREDICTION_FILES[1]}")


def to_grid(arr: np.ndarray, src_prof: dict, dst_shape, dst_transform, dst_crs) -> np.ndarray:
    """Nearest-neighbour resample of a 0/1/255 class array to a reference grid."""
    if src_prof.get("crs") is None:
        raise ValueError("prediction raster has no CRS")
    if (src_prof["crs"] == dst_crs and src_prof["transform"] == dst_transform
            and arr.shape == tuple(dst_shape)):
        return arr
    out = np.full(dst_shape, INVALID, dtype=np.uint8)
    reproject(arr, out, src_transform=src_prof["transform"], src_crs=src_prof["crs"], src_nodata=INVALID,
              dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=INVALID, resampling=Resampling.nearest)
    return out


def _verification(counts: dict[str, int]) -> dict[str, Any]:
    m = score(counts)
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    n = tp + fp + fn + tn
    return {
        **m,
        "pod": m["recall"],  # probability of detection
        "far": (fp / (tp + fp)) if tp + fp else None,  # false alarm ratio
        "csi": m["iou"],  # critical success index (threat score)
        "accuracy": ((tp + tn) / n) if n else None,
        "truth_fraction": ((tp + fn) / n) if n else None,
    }


def score_against_reference(pred: np.ndarray, ref: np.ndarray, baseline: np.ndarray | None = None,
                            target: str = "water_extent") -> tuple[dict[str, Any], np.ndarray]:
    """Return (scores, agreement codes) for arrays already on one grid."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    arrays = {"prediction": pred, "reference": ref, **({"baseline": baseline} if baseline is not None else {})}
    for name, a in arrays.items():
        if a.shape != ref.shape:
            raise ValueError(f"{name} grid differs from reference grid")
        if not np.isin(a, (0, 1, INVALID)).all():
            raise ValueError(f"{name} has values outside {{0, 1, 255}}")
    if target == "new_water" and baseline is None:
        raise ValueError("target new_water requires a pre-event baseline")
    valid = (pred != INVALID) & (ref != INVALID)
    if baseline is not None:
        valid &= baseline != INVALID
    if target == "new_water":
        valid &= baseline == 0  # already-wet pixels cannot show a *new* flood
    truth = np.where(valid, ref.astype(np.int8), np.int8(-1))
    model = np.where(valid, pred, 0).astype(np.int8)
    counts = confusion(truth, model)
    result: dict[str, Any] = {"target": target, "comparable_pixels": int(valid.sum()),
                              "counts": counts, "metrics": _verification(counts)}
    if baseline is not None:
        # Persistence baseline: "the pre-event map is the forecast". For new_water this
        # predicts no new flooding anywhere.
        base_pred = np.where(valid, baseline if target == "water_extent" else 0, 0).astype(np.int8)
        b_counts = confusion(truth, base_pred)
        b = _verification(b_counts)
        result["persistence_baseline"] = {"counts": b_counts, "metrics": b}
        mi, bi = result["metrics"]["iou"], b["iou"]
        result["skill_vs_persistence"] = {
            "iou_gain": (mi or 0.0) - (bi or 0.0) if mi is not None else None,
            "beats_baseline": bool(mi is not None and mi > (bi or 0.0)),
        }
    agreement = np.full(ref.shape, INVALID, dtype=np.uint8)
    agreement[valid & (model == 0) & (truth == 0)] = 0
    agreement[valid & (model == 1) & (truth == 1)] = 1
    agreement[valid & (model == 1) & (truth == 0)] = 2
    agreement[valid & (model == 0) & (truth == 1)] = 3
    return result, agreement


def _read_class(path: Path):
    with rasterio.open(path) as src:
        return src.read(1), src.profile.copy()


def _sum_counts(items: list[dict[str, int]]) -> dict[str, int]:
    return {k: sum(c[k] for c in items) for k in ("tp", "tn", "fp", "fn", "valid")}


def validate_prediction_run(pred_dir: Path, ref_dirs: list[Path], baseline_dirs: list[Path] | None = None,
                            target: str = "water_extent", event_time_utc: str | None = None,
                            threshold: float | None = None, max_polygons: int = 5000) -> dict[str, Any]:
    """Score one prediction folder against one or more reference run folders.

    ``baseline_dirs`` (optional, same length/order as ``ref_dirs``) are pre-event runs on
    the same grids. Writes ``prediction_validation.json`` (combined) into ``pred_dir`` and
    per-reference outputs into ``pred_dir/validation/<ref_run_id>/``.
    """
    pred_dir = Path(pred_dir)
    ref_dirs = [Path(r) for r in ref_dirs]
    baseline_dirs = [Path(b) for b in baseline_dirs] if baseline_dirs else []
    if baseline_dirs and len(baseline_dirs) != len(ref_dirs):
        raise ValueError("--baseline must be given once per --reference, in the same order")
    pred_meta = json.loads((pred_dir / "prediction_metadata.json").read_text())
    ref_metas = [json.loads((r / "observed_metadata.json").read_text()) for r in ref_dirs]
    for r, m in zip(ref_dirs, ref_metas, strict=True):
        m.setdefault("run_id", r.name)
    integrity = check_integrity(pred_meta, ref_metas, event_time_utc)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA,
        "prediction_run_id": pred_meta.get("run_id", pred_dir.name),
        "model_id": pred_meta.get("model_id"),
        "target": target,
        "threshold": threshold,
        "integrity": integrity,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "references": [],
        "agreement_codes": AGREEMENT_CODES,
        "is_synthetic": bool(pred_meta.get("is_synthetic")),
        "interpretation_limits": [
            ("Scores measure agreement with the reference observation, which is itself model-derived "
             "optical water (not field-verified ground truth) unless its metadata says otherwise."),
            "Only pixels valid in every compared raster are scored; cloud/nodata are not_comparable.",
            "Coarse predictions are nearest-resampled to the 10 m reference grid; this does not add detail.",
            "A single event is one sample: good agreement here does not establish skill on other events.",
        ],
    }
    if not integrity["eligible"]:
        summary["status"] = "rejected_integrity"
        (pred_dir / "prediction_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    pred_arr, pred_prof = read_prediction(pred_dir, threshold)
    all_counts, base_counts, pred_polys_done = [], [], False
    for i, (rdir, rmeta) in enumerate(zip(ref_dirs, ref_metas, strict=True)):
        ref, rprof = _read_class(rdir / "observed_water.tif")
        crs, tr = rprof["crs"], rprof["transform"]
        pred = to_grid(pred_arr, pred_prof, ref.shape, tr, crs)
        base = None
        if baseline_dirs:
            base, bprof = _read_class(baseline_dirs[i] / "observed_water.tif")
            if bprof["crs"] != crs or bprof["transform"] != tr or base.shape != ref.shape:
                raise ValueError(f"baseline {baseline_dirs[i].name} grid differs from reference {rdir.name}")
        res, agreement = score_against_reference(pred, ref, base, target)
        out = pred_dir / "validation" / rmeta["run_id"]
        out.mkdir(parents=True, exist_ok=True)
        prof = {**rprof, "dtype": "uint8", "nodata": INVALID, "count": 1, "compress": "deflate"}
        with rasterio.open(out / "agreement.tif", "w", **prof) as dst:
            dst.write(agreement, 1)
        px = _pixel_area_m2(crs, tr)
        polys = {}
        for code, name in ((1, "hit"), (2, "false_alarm"), (3, "miss")):
            gj, n = water_polygons((agreement == code).astype(np.uint8), tr, crs, max_polygons=max_polygons)
            for f in gj["features"]:
                f["properties"].update({"class": name, "label": name.replace("_", " ")})
            (out / f"agreement_{name}.geojson").write_text(json.dumps(gj) + "\n")
            polys[name] = {"total": n, "written": len(gj["features"])}
        if not pred_polys_done and not (pred_dir / "predicted_flood.geojson").exists():
            gj, n = water_polygons(pred_arr, pred_prof["transform"], pred_prof["crs"], max_polygons=max_polygons)
            for f in gj["features"]:
                f["properties"].update({"class": "predicted_flood", "label": "Predicted flood"})
            (pred_dir / "predicted_flood.geojson").write_text(json.dumps(gj) + "\n")
            pred_polys_done = True
        res.update({
            "reference_run_id": rmeta["run_id"],
            "reference_scene_id": rmeta.get("scene_id"),
            "reference_acquired_at_utc": rmeta.get("acquired_at_utc"),
            "reference_model": rmeta.get("model_id") or rmeta.get("model"),
            "baseline_run_id": baseline_dirs[i].name if baseline_dirs else None,
            "crs": crs.to_string() if crs else None,
            "pixel_area_m2": px,
            "area_km2": {k: (round(v * px / 1e6, 4) if px else None)
                         for k, v in (("hit", res["counts"]["tp"]), ("false_alarm", res["counts"]["fp"]),
                                      ("miss", res["counts"]["fn"]), ("correct_negative", res["counts"]["tn"]))},
            "polygons": polys,
        })
        (out / "prediction_validation.json").write_text(json.dumps(res, indent=2) + "\n")
        summary["references"].append(res)
        all_counts.append(res["counts"])
        if "persistence_baseline" in res:
            base_counts.append(res["persistence_baseline"]["counts"])

    combined = _sum_counts(all_counts)
    summary["combined"] = {"counts": combined, "metrics": _verification(combined)}
    if base_counts and len(base_counts) == len(all_counts):
        bc = _sum_counts(base_counts)
        bm = _verification(bc)
        summary["combined"]["persistence_baseline"] = {"counts": bc, "metrics": bm}
        mi, bi = summary["combined"]["metrics"]["iou"], bm["iou"]
        summary["combined"]["skill_vs_persistence"] = {
            "iou_gain": (mi or 0.0) - (bi or 0.0) if mi is not None else None,
            "beats_baseline": bool(mi is not None and mi > (bi or 0.0)),
        }
    summary["status"] = "scored" if combined["valid"] else "no_comparable_pixels"
    (pred_dir / "prediction_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
