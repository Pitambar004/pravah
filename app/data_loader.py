"""Artifact loading and validation layer for PRAVAH.

Safely reads output artifacts from outputs/<run_id>/ complying with docs/contracts.md.
Never crashes on missing, empty, or malformed files.
"""
import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.fixtures.synthetic_sample import (
    SYNTHETIC_AOI_GEOJSON,
    SYNTHETIC_BENCHMARK,
    SYNTHETIC_EXPOSURE_SUMMARY,
    SYNTHETIC_OBSERVED_METADATA,
    SYNTHETIC_REPLAY_ASSETS,
    SYNTHETIC_UNOSAT_GEOJSON,
    SYNTHETIC_WATER_GEOJSON,
)


@dataclass
class RunArtifacts:
    """Loaded artifacts for a specific inference / exposure run."""
    run_id: str
    folder: Path
    is_synthetic: bool = False
    
    # Metadata & summaries
    observed_metadata: dict[str, Any] | None = None
    exposure_summary: dict[str, Any] | None = None
    benchmark: dict[str, Any] | None = None
    
    # Paths to raster files
    observed_water_tif: Path | None = None
    invalid_mask_tif: Path | None = None
    water_probability_tif: Path | None = None
    
    # Vector data (GeoJSON / GeoDataFrame)
    water_geojson: dict[str, Any] | None = None
    aoi_geojson: dict[str, Any] | None = None
    unosat_geojson: dict[str, Any] | None = None
    
    # Set when artifacts were fetched over HTTP from the inference API (e.g. on Fuzzball)
    remote_source: str | None = None

    # Optional pre/post change analysis (scripts/compare_observations.py) in a post-event run
    change_summary: dict[str, Any] | None = None
    change_new_water_geojson: dict[str, Any] | None = None
    change_lost_water_geojson: dict[str, Any] | None = None

    # Optional per-category exposure features (scripts/exposure.py --features-dir)
    exposure_features: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Study-area context layers shared by all runs (scripts/build_context_layers.py)
    context_layers: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Flood *prediction* run (docs/PREDICTION_CONTRACT.md) and its scoring
    # against post-event observations (scripts/validate_prediction.py)
    prediction_metadata: dict[str, Any] | None = None
    prediction_validation: dict[str, Any] | None = None
    predicted_flood_geojson: dict[str, Any] | None = None
    # {reference_run_id: {"hit"|"false_alarm"|"miss": FeatureCollection}}
    agreement_layers: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    # Status flags & diagnostics
    missing_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def has_any_artifacts(self) -> bool:
        return bool(
            self.observed_metadata
            or self.exposure_summary
            or self.benchmark
            or self.observed_water_tif
            or self.water_geojson
            or self.prediction_metadata
        )

    @property
    def is_prediction(self) -> bool:
        return bool(self.prediction_metadata)

    @property
    def acquired_at(self) -> str | None:
        return (self.observed_metadata or {}).get("acquired_at_utc")


EXPOSURE_FEATURE_CATEGORIES = ("buildings", "roads", "bridges")
CONTEXT_LAYER_FILES = ("rivers", "admin_boundaries")

RUN_MARKER_FILES = (
    "observed_metadata.json", "exposure_summary.json", "benchmark.json", "observed_water.tif", "job.json",
    "prediction_metadata.json",
)
AGREEMENT_CLASSES = ("hit", "false_alarm", "miss")


def run_has_artifacts(run_folder: Path | str) -> bool:
    """True when a run folder contains at least one known run artifact."""
    p = Path(run_folder)
    return p.is_dir() and any((p / f).exists() for f in RUN_MARKER_FILES)


def discover_available_runs(outputs_dir: Path | str = "outputs") -> list[str]:
    """Find all run subdirectories in outputs/."""
    p = Path(outputs_dir)
    if not p.exists() or not p.is_dir():
        return []
    runs = []
    for item in p.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            runs.append(item.name)
    runs.sort(reverse=True)
    return runs


def safe_read_json(file_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Safely load JSON without throwing unhandled exceptions."""
    if not file_path.exists():
        return None, None
    try:
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            return None, f"File is empty: {file_path.name}"
        return json.loads(content), None
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError) as e:
        return None, f"JSON parse error in {file_path.name}: {e!s}"


def safe_read_geojson(file_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Safely load a GeoJSON file into a dict representation."""
    if not file_path.exists():
        return None, None
    try:
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            return None, f"GeoJSON file is empty: {file_path.name}"
        data = json.loads(content)
        if not isinstance(data, dict) or data.get("type") not in ("FeatureCollection", "Feature", "Polygon", "MultiPolygon"):
            return None, f"Invalid GeoJSON structure in {file_path.name}"
        return data, None
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError) as e:
        return None, f"Failed to read GeoJSON {file_path.name}: {e!s}"


def load_run_artifacts(
    run_folder: Path | str,
    allow_synthetic_fallback: bool = False
) -> RunArtifacts:
    """Load artifacts from a target run directory.
    
    If allow_synthetic_fallback is True AND the folder has no real artifacts,
    returns the explicitly labeled synthetic test fixtures for UI verification.
    """
    path = Path(run_folder)
    artifacts = RunArtifacts(run_id=path.name, folder=path)
    
    # Try reading real artifacts
    if path.exists() and path.is_dir():
        # 1. observed_metadata.json
        meta_file = path / "observed_metadata.json"
        meta, err = safe_read_json(meta_file)
        if meta:
            artifacts.observed_metadata = meta
        elif err:
            artifacts.errors.append(err)
        else:
            artifacts.missing_files.append("observed_metadata.json")

        # 2. exposure_summary.json
        exp_file = path / "exposure_summary.json"
        exp, err = safe_read_json(exp_file)
        if exp:
            artifacts.exposure_summary = exp
        elif err:
            artifacts.errors.append(err)
        else:
            artifacts.missing_files.append("exposure_summary.json")

        # 3. benchmark.json
        bench_file = path / "benchmark.json"
        bench, err = safe_read_json(bench_file)
        if bench:
            artifacts.benchmark = bench
        elif err:
            artifacts.errors.append(err)
        else:
            artifacts.missing_files.append("benchmark.json")

        # 4. GeoTIFFs
        for tif_name, attr in [
            ("observed_water.tif", "observed_water_tif"),
            ("invalid_mask.tif", "invalid_mask_tif"),
            ("water_probability.tif", "water_probability_tif"),
        ]:
            tif_path = path / tif_name
            if tif_path.exists() and tif_path.stat().st_size > 0:
                setattr(artifacts, attr, tif_path)
            else:
                artifacts.missing_files.append(tif_name)

        # 5. GeoJSON vectors
        for geo_candidate in ["exposure.geojson", "polygons.geojson", "observed_water.geojson"]:
            geo_path = path / geo_candidate
            if geo_path.exists():
                geo_data, err = safe_read_geojson(geo_path)
                if geo_data:
                    artifacts.water_geojson = geo_data
                    break
                elif err:
                    artifacts.errors.append(err)

        # 6. Optional analyses (absent files are simply not shown; never synthesised)
        change, err = safe_read_json(path / "change_summary.json")
        if change:
            artifacts.change_summary = change
            for name, attr in (("change_new_water.geojson", "change_new_water_geojson"),
                               ("change_lost_water.geojson", "change_lost_water_geojson")):
                geo, gerr = safe_read_geojson(path / name)
                if geo:
                    setattr(artifacts, attr, geo)
                elif gerr:
                    artifacts.errors.append(gerr)
        elif err:
            artifacts.errors.append(err)
        # 7. Prediction run + validation (optional; never synthesised)
        for name, attr, reader in (("prediction_metadata.json", "prediction_metadata", safe_read_json),
                                   ("prediction_validation.json", "prediction_validation", safe_read_json),
                                   ("predicted_flood.geojson", "predicted_flood_geojson", safe_read_geojson)):
            val, verr = reader(path / name)
            if val:
                setattr(artifacts, attr, val)
            elif verr:
                artifacts.errors.append(verr)
        vdir = path / "validation"
        if vdir.is_dir():
            for ref_dir in sorted(d for d in vdir.iterdir() if d.is_dir()):
                for cls in AGREEMENT_CLASSES:
                    geo, gerr = safe_read_geojson(ref_dir / f"agreement_{cls}.geojson")
                    if geo:
                        artifacts.agreement_layers.setdefault(ref_dir.name, {})[cls] = geo
                    elif gerr:
                        artifacts.errors.append(gerr)
        for category in EXPOSURE_FEATURE_CATEGORIES:
            geo, gerr = safe_read_geojson(path / f"exposure_{category}.geojson")
            if geo:
                artifacts.exposure_features[category] = geo
            elif gerr:
                artifacts.errors.append(gerr)

    # Study-area context layers (rivers, admin boundaries); cleared below in fixture mode
    context_dir = Path(os.environ.get("PRAVAH_CONTEXT_DIR", "data/processed/context"))
    for name in CONTEXT_LAYER_FILES:
        local_context = path / f"{name}.geojson"
        geo, gerr = safe_read_geojson(local_context if local_context.exists() else context_dir / f"{name}.geojson")
        if geo:
            artifacts.context_layers[name] = geo
        elif gerr:
            artifacts.errors.append(gerr)

    # Check for core AOI in config/
    aoi_candidates = [
        path / "image_footprint.geojson",
        Path("config/aois/aoi_core.geojson"),
        Path("config/aois/analysis_coverage.geojson"),
        Path("config/aoi_core.geojson")
    ]
    for cand in aoi_candidates:
        if cand.exists():
            aoi_data, _ = safe_read_geojson(cand)
            if aoi_data:
                artifacts.aoi_geojson = aoi_data
                break

    # If no real artifacts exist AND synthetic fallback is requested
    if not artifacts.has_any_artifacts and allow_synthetic_fallback:
        artifacts.is_synthetic = True
        artifacts.run_id = "synthetic_fixture_demo"
        artifacts.context_layers = {}  # never mix real context into fixture mode
        artifacts.observed_metadata = SYNTHETIC_OBSERVED_METADATA
        artifacts.exposure_summary = SYNTHETIC_EXPOSURE_SUMMARY
        artifacts.benchmark = SYNTHETIC_BENCHMARK
        artifacts.water_geojson = SYNTHETIC_WATER_GEOJSON
        artifacts.aoi_geojson = SYNTHETIC_AOI_GEOJSON
        artifacts.unosat_geojson = SYNTHETIC_UNOSAT_GEOJSON

    return artifacts


def load_replay_catalog(manifest_path: Path | str | None = None, allow_synthetic: bool = True) -> list[dict[str, Any]]:
    """Load satellite and dataset assets for Historical Replay audit."""
    items, _ = load_replay_catalog_with_source(manifest_path, allow_synthetic=allow_synthetic)
    return items


def load_replay_catalog_with_source(
    manifest_path: Path | str | None = None, allow_synthetic: bool = False
) -> tuple[list[dict[str, Any]], str | None]:
    """Load the replay catalog and report where it came from.

    Returns ``(items, source)`` where ``source`` is the manifest path, the string
    ``"synthetic_fixture"``, or ``None`` when nothing is available. Synthetic assets
    are only returned when ``allow_synthetic`` is True, so the UI can never show
    fixture scenes as real observations without labelling them.
    """
    paths_to_check = []
    if os.environ.get("PRAVAH_REPLAY_MANIFEST"):
        paths_to_check.append(Path(os.environ["PRAVAH_REPLAY_MANIFEST"]))
    if manifest_path:
        paths_to_check.append(Path(manifest_path))
    paths_to_check.extend([
        Path("data/manifests/assets.json"),
        Path("data/manifests/scenes.json"),
        Path("data/assets.json"),
    ])
    
    for p in paths_to_check:
        if p.exists():
            data, _ = safe_read_json(p)
            if isinstance(data, list) and data:
                return data, str(p)
            elif isinstance(data, dict) and "features" in data:
                # STAC FeatureCollection format
                items = []
                for f in data["features"]:
                    props = f.get("properties", {})
                    items.append({
                        "asset_id": f.get("id"),
                        "platform": props.get("platform", "Sentinel-2"),
                        "product_type": props.get("constellation", "optical"),
                        "acquired_at_utc": props.get("datetime") or props.get("start_datetime"),
                        "available_at_utc": props.get("verified_available_at_utc"),
                        "cloud_cover_percent": props.get("eo:cloud_cover"),
                        "provider": "Copernicus STAC",
                        "description": f.get("id")
                    })
                return items, str(p)

    if allow_synthetic:
        return SYNTHETIC_REPLAY_ASSETS, "synthetic_fixture"
    return [], None


def load_all_runs(outputs_dir: Path | str = "outputs") -> list[RunArtifacts]:
    """Every run folder that holds real artifacts, oldest acquisition first (predictions last)."""
    root = Path(outputs_dir)
    runs = [load_run_artifacts(root / r) for r in discover_available_runs(root) if run_has_artifacts(root / r)]
    return sorted(runs, key=lambda a: (a.is_prediction, a.acquired_at or "9999", a.run_id))


# ---------------------------------------------------------------- training runs
TRAINING_FILES = ("run_environment.json", "training_summary.json", "train_metrics.csv", "final_evaluation.json")
# Column aliases: scripts/train_ddp.py (per step) vs the local trainer (per epoch)
_METRIC_ALIASES = {"step": "optimizer_step", "mean_training_loss": "loss_rank0", "elapsed_seconds": "train_sec"}


def _read_metrics_csv(path: Path) -> tuple[list[dict[str, float]], str | None]:
    if not path.is_file():
        return [], None
    rows: list[dict[str, float]] = []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    rows.append({_METRIC_ALIASES.get(k, k): float(v) for k, v in row.items() if v not in (None, "")})
                except ValueError:
                    continue  # partially written last line while training is running
    except OSError as exc:
        return [], f"{path.name}: {exc}"
    return rows, None


def load_training_run(folder: Path | str) -> dict[str, Any]:
    """One scripts/train_ddp.py output folder. Status is derived only from files present."""
    d = Path(folder)
    env, e1 = safe_read_json(d / "run_environment.json")
    summary, e2 = safe_read_json(d / "training_summary.json")
    val, e3 = safe_read_json(d / "validation_latest.json")
    metrics, e4 = _read_metrics_csv(d / "train_metrics.csv")
    evaluation, e5 = safe_read_json(d / "final_evaluation.json")
    config, _ = safe_read_json(d / "config.json")
    info = summary or env or {}
    # "finished", "finished_validation_only", ... (the local trainer records how far evaluation went)
    finished = summary or str((evaluation or {}).get("status") or "").startswith("finished")
    status = "completed" if finished else ("running" if env or metrics else "empty")
    return {
        "id": d.name,
        "folder": str(d),
        "label": info.get("run_label") or d.name,
        "status": status,
        "environment": env,
        "summary": summary,
        "validation": val,
        "evaluation": evaluation,  # held-out test scores written once after model selection
        "config": config,
        "metrics": metrics,
        "mtime": max((p.stat().st_mtime for p in d.iterdir()), default=0.0) if d.is_dir() else 0.0,
        "errors": [e for e in (e1, e2, e3, e4, e5) if e],
    }


def load_training_runs(*training_dirs: Path | str) -> list[dict[str, Any]]:
    runs = []
    for root in map(Path, training_dirs):
        if root.is_dir():
            runs += [load_training_run(d) for d in root.iterdir()
                     if d.is_dir() and not d.name.startswith(".") and any((d / f).exists() for f in TRAINING_FILES)]
    return sorted(runs, key=lambda r: r["mtime"], reverse=True)
