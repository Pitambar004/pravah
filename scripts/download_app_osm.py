#!/usr/bin/env python3
"""Download a bounded, timestamped OSM extract for each staged real scene."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import osm2geojson
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://overpass-api.de/api/interpreter"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False) + "\n")


def convert(data):
    if data.get("remark"):
        raise ValueError(
            "Overpass returned an incomplete/error response: " + data["remark"]
        )
    converted = osm2geojson.json2geojson(data)
    result = {name: [] for name in ["buildings", "roads", "bridges", "rivers"]}
    for feature in converted["features"]:
        props = feature.get("properties", {})
        tags = props.get("tags", {})
        geom = feature.get("geometry")
        if not geom or shape(geom).is_empty:
            continue
        properties = {
            k: tags[k]
            for k in ["name", "building", "highway", "bridge", "waterway"]
            if k in tags
        }
        properties["osm_id"] = f"{props.get('type')}/{props.get('id')}"
        record = {"type": "Feature", "geometry": geom, "properties": properties}
        if tags.get("building") not in (None, "no") and geom["type"] in (
            "Polygon",
            "MultiPolygon",
        ):
            result["buildings"].append(record)
        if tags.get("highway") and geom["type"] in ("LineString", "MultiLineString"):
            result["roads"].append(record)
        if tags.get("bridge") not in (None, "no"):
            result["bridges"].append(record)
        if tags.get("waterway") and geom["type"] in ("LineString", "MultiLineString"):
            result["rivers"].append(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene")
    args = parser.parse_args()
    catalog = json.loads(
        (ROOT / "data/processed/live_app/scene_catalog.json").read_text()
    )
    for item in catalog.values():
        if args.scene and item["scene_id"] != args.scene:
            continue
        out = ROOT / "data/raw/app_osm" / item["scene_id"]
        raw = out / "overpass.json"
        west, south, east, north = item["bounds_wgs84"]
        if (east - west) * (north - south) > 0.01:
            raise ValueError(
                "Use a bounded scene extract, not a country-wide public-server query"
            )
        bbox = f"{south},{west},{north},{east}"
        query = f"[out:json][timeout:45];(way[building]({bbox});relation[building][type=multipolygon]({bbox});way[highway]({bbox});way[bridge]({bbox});way[waterway]({bbox}););out body;>;out skel qt;"
        if raw.exists():
            data = json.loads(raw.read_text())
        else:
            response = httpx.post(
                ENDPOINT,
                data={"data": query},
                timeout=60,
                headers={
                    "User-Agent": "PRAVAH-research/0.1 (bounded public-data extract)"
                },
            )
            response.raise_for_status()
            data = response.json()
            if data.get("remark"):
                raise RuntimeError(data["remark"])
            save(raw, data)
        layers = convert(data)
        stamp = data.get("osm3s", {}).get("timestamp_osm_base")
        if not stamp:
            raise ValueError("OSM snapshot timestamp missing")
        files = {}
        for category, features in layers.items():
            path = out / (category + ".geojson")
            save(path, {"type": "FeatureCollection", "features": features})
            files[category] = {
                "path": str(path),
                "features": len(features),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        receipt = {
            "provider": ENDPOINT,
            "query": query,
            "bounds_wgs84": item["bounds_wgs84"],
            "snapshot_timestamp": stamp,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "layers": files,
            "license": "OpenStreetMap contributors, ODbL 1.0",
            "source_url": "https://www.openstreetmap.org/copyright",
            "scope": "Current mapped infrastructure over historical benchmark imagery; not historical event exposure or damage",
            "strict_replay_eligible": False,
        }
        save(out / "receipt.json", receipt)
        print(
            item["scene_id"], {k: len(v) for k, v in layers.items()}, stamp, flush=True
        )


if __name__ == "__main__":
    main()
