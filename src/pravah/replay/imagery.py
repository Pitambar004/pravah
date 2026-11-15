"""Latest available Rasuwa provider preview, retaining its true acquisition age."""

import base64
import hashlib
import io
import json
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

import httpx
from PIL import Image

from pravah.configuration import configured_url


def retain_preview(previous, current):
    """A failed newer preview never changes the displayed image's acquisition."""
    if current.get("preview_data_url") or not (previous or {}).get("preview_data_url"):
        return current
    return {**previous, "status": "preview_retained", "reason": current.get("reason", "Prior preview retained"),
            "last_catalog_check_at_utc": current.get("retrieved_at_utc"),
            "latest_catalog_scene_id": current.get("scene_id"),
            "latest_catalog_acquired_at_utc": current.get("acquired_at_utc")}


def latest_preview(root: Path, now):
    endpoint = configured_url("PRAVAH_CDSE_SEARCH_URL")
    query = {"collections": ["sentinel-2-l1c"], "bbox": [85.25, 28.05, 85.65, 28.5],
             "datetime": (now - timedelta(days=30)).isoformat() + "/" + now.isoformat(),
             "limit": 3, "sortby": [{"field": "datetime", "direction": "desc"}]}
    with httpx.Client(timeout=12, follow_redirects=True) as client:
        with client.stream("POST", endpoint, json=query) as r:
            r.raise_for_status()
            blocks, size = [], 0
            for block in r.iter_bytes():
                size += len(block)
                if size > 1_000_000:
                    raise ValueError("Rasuwa catalog exceeds 1 MB budget")
                blocks.append(block)
        features = json.loads(b"".join(blocks))["features"]
        if not features:
            return {"status": "unavailable", "records": [], "reason": "No Rasuwa L1C catalog scene in the last 30 days"}
        return fetch_preview(features[0], root, now, client)


def fetch_preview(item, root, now, client):
    timestamp = item["properties"].get("datetime") or item["properties"].get("start_datetime")
    doc = {"status": "catalog_only", "records": [{"id": item["id"], "observed_at_utc": timestamp}],
           "scene_id": item["id"], "acquired_at_utc": timestamp,
           "cloud_cover_percent": item["properties"].get("eo:cloud_cover"),
           "collection": item["collection"], "source_uri": configured_url("PRAVAH_CDSE_SEARCH_URL"), "retrieved_at_utc": now.isoformat(),
           "preview_data_url": None, "model_input_available": False,
           "semantics": "Provider downsampled visual preview; not model-ready bands or a water/deformation measurement"}
    asset = item.get("assets", {}).get("thumbnail", {})
    href = asset.get("href", "")
    parsed = urlparse(href)
    if parsed.scheme != "https" or parsed.hostname not in {"datahub.creodias.eu", "catalogue.dataspace.copernicus.eu"} or parsed.query:
        return {**doc, "reason": "No approved stable public thumbnail; model pixels require authenticated acquisition"}
    folder = root / "previews"; folder.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(item["id"].encode()).hexdigest()
    path = folder / (key + ".jpg")
    try:
        if not path.exists():
            with client.stream("GET", href) as response:
                response.raise_for_status()
                blocks, size = [], 0
                for block in response.iter_bytes():
                    size += len(block)
                    if size > 3_000_000:
                        raise ValueError("Provider preview exceeds 3 MB budget")
                    blocks.append(block)
            content = b"".join(blocks)
            with Image.open(io.BytesIO(content)) as im:
                im.verify()
            path.write_bytes(content)
        content = path.read_bytes()
        return {**doc, "status": "preview_available", "preview_sha256": hashlib.sha256(content).hexdigest(),
                "preview_source_uri": href, "preview_file": str(path.relative_to(root)),
                "preview_data_url": "data:image/jpeg;base64," + base64.b64encode(content).decode()}
    except (httpx.HTTPError, OSError, ValueError):
        return {**doc, "reason": "Provider preview unavailable; acquisition metadata remains visible"}
