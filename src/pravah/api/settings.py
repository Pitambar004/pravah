"""Environment-driven settings for the PRAVAH inference API.

Defaults follow ``config/fuzzball.yaml`` / ``config/dataset.yaml`` (volume
``Pravah-Dataset`` mounted at ``/data``). Every path can be overridden so the
same code runs locally against ``./data``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


def _bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    data_root: Path
    scenes_dir: Path
    outputs_dir: Path
    checkpoint: Path
    device: str
    require_cuda: bool
    batch_size: int
    api_token: str | None
    max_queued_jobs: int
    # URL prefix a Fuzzball path endpoint may forward (``/endpoints/<id>``); stripped before routing.
    root_path: str = ""
    # "dn" (Sen1Floods11 / raw L1C) or "reflectance" (P3 s2_l1c_reflectance.tif); see engine docs.
    input_units: str = "dn"
    # OSM thematic layers (buildings/roads/bridges .gpkg) for automatic exposure after each run; None = skip
    osm_dir: Path | None = None
    osm_date: str = "unknown"

    @classmethod
    def from_env(cls) -> Settings:
        root = Path(os.environ.get("PRAVAH_DATA_ROOT", "/data"))
        token = os.environ.get("PRAVAH_API_TOKEN") or None
        return cls(
            data_root=root,
            scenes_dir=Path(os.environ.get("PRAVAH_SCENES_DIR", root / "raw" / "sentinel2")),
            outputs_dir=Path(os.environ.get("PRAVAH_OUTPUTS_DIR", root / "outputs" / "predictions")),
            checkpoint=Path(os.environ.get("PRAVAH_CHECKPOINT",
                                           root / "models" / "checkpoints" / "checkpoint_last.pt")),
            device=os.environ.get("PRAVAH_DEVICE", "auto"),
            require_cuda=_bool("PRAVAH_REQUIRE_CUDA"),
            batch_size=int(os.environ.get("PRAVAH_BATCH_SIZE", "8")),
            api_token=token,
            max_queued_jobs=int(os.environ.get("PRAVAH_MAX_QUEUED_JOBS", "8")),
            root_path=_root_path(),
            input_units=os.environ.get("PRAVAH_INPUT_UNITS", "dn"),
            osm_dir=Path(os.environ["PRAVAH_OSM_DIR"]) if os.environ.get("PRAVAH_OSM_DIR") else None,
            osm_date=os.environ.get("PRAVAH_OSM_DATE", "unknown"),
        )


def _root_path() -> str:
    """PRAVAH_API_ROOT_PATH, else the path of Fuzzball's injected FB_ENDPOINT_URL_API, else ''."""
    raw = os.environ.get("PRAVAH_API_ROOT_PATH")
    if raw is None and os.environ.get("FB_ENDPOINT_URL_API"):
        raw = urlparse(os.environ["FB_ENDPOINT_URL_API"]).path
    raw = (raw or "").strip("/")
    return f"/{raw}" if raw else ""
