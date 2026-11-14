"""Artifact loading and validation layer for PRAVAH.

Safely reads output artifacts from outputs/<run_id>/ complying with docs/contracts.md.
Never crashes on missing, empty, or malformed files.
"""
import json
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
        )


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

    # Check for core AOI in config/
    aoi_candidates = [
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
        artifacts.observed_metadata = SYNTHETIC_OBSERVED_METADATA
        artifacts.exposure_summary = SYNTHETIC_EXPOSURE_SUMMARY
        artifacts.benchmark = SYNTHETIC_BENCHMARK
        artifacts.water_geojson = SYNTHETIC_WATER_GEOJSON
        artifacts.aoi_geojson = SYNTHETIC_AOI_GEOJSON
        artifacts.unosat_geojson = SYNTHETIC_UNOSAT_GEOJSON

    return artifacts


def load_replay_catalog(manifest_path: Path | str | None = None, allow_synthetic: bool = True) -> list[dict[str, Any]]:
    """Load satellite and dataset assets for Historical Replay audit."""
    paths_to_check = []
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
                return data
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
                        "available_at_utc": props.get("published") or props.get("created"),
                        "cloud_cover_percent": props.get("eo:cloud_cover"),
                        "provider": "Copernicus STAC",
                        "description": f.get("id")
                    })
                return items

    if allow_synthetic:
        return SYNTHETIC_REPLAY_ASSETS
    return []
