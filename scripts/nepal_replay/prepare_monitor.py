#!/usr/bin/env python3
"""Prepare real Nepal mapped context and public live-source cache for transfer.

Download a bounded (<450 MB) current Geofabrik Nepal PBF. Select actual
Bhote Koshi/Trishuli/Narayani lines and mapped assets within 350 m in
EPSG:32645. This is current OSM context, never historical disaster truth.
"""

import argparse
import hashlib
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import httpx
import pandas as pd
import pyogrio
from shapely.geometry import box

from pravah.configuration import configured_url
from pravah.replay.experiment import sha256
from pravah.replay.live import LiveMonitor

PROJECT = Path(__file__).resolve().parents[2]
BBOX = (83.85, 27.30, 85.70, 28.50)
RIVER_NAMES = re.compile(r"trish?ul|trisuli|bhote|bhotek|narayani|त्रिश|त्रिस|भोटे|नारायण", re.IGNORECASE)


def save(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")


def acquire(root):
    root.mkdir(parents=True, exist_ok=True)
    target = root / "nepal.osm.pbf"
    receipt = root / "receipt.json"
    if target.exists() and receipt.exists():
        doc = json.loads(receipt.read_text())
        if sha256(target) != doc["sha256"]:
            raise ValueError("Cached PBF integrity mismatch")
        return target, doc
    endpoint = configured_url("PRAVAH_GEOFABRIK_NEPAL_URL")
    temporary = target.with_suffix(".pbf.part")
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        head = client.head(endpoint)
        head.raise_for_status()
        resolved = str(head.url)
        size = int(head.headers["content-length"])
        if not 0 < size <= 450_000_000:
            raise ValueError("Nepal PBF exceeds the 450 MB acquisition budget")
        start = temporary.stat().st_size if temporary.exists() else 0
        if start > size:
            raise ValueError("Partial download is larger than provider file")
        ranges = [(a, min(a + 16_000_000, size) - 1) for a in range(start, size, 16_000_000)]
        def part(bounds):
            a, b = bounds
            path = root / f"chunk_{a}_{b}.part"
            expected = b - a + 1
            if path.exists() and path.stat().st_size == expected:
                return path
            with httpx.Client(timeout=30) as worker, worker.stream("GET", resolved, headers={"Range": f"bytes={a}-{b}"}) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get("content-range") != f"bytes {a}-{b}/{size}":
                    raise ValueError("Provider did not honor the requested byte range")
                count = 0
                with path.open("wb") as output:
                    for block in response.iter_bytes(1024 * 1024):
                        count += len(block)
                        if count > expected:
                            raise ValueError("Provider range exceeds its declared length")
                        output.write(block)
            if count != expected:
                raise ValueError("Partial range length mismatch")
            return path
        print(f"Downloading {size} bytes from {resolved}; verified range support required", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            parts = list(pool.map(part, ranges))
        assembled = root / "nepal.assembled.part"
        with assembled.open("wb") as output:
            if temporary.exists():
                with temporary.open("rb") as source:
                    shutil.copyfileobj(source, output)
            for path in parts:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, output)
        if assembled.stat().st_size != size:
            raise ValueError("Assembled PBF length mismatch")
        md5 = hashlib.md5(usedforsecurity=False)
        with assembled.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                md5.update(block)
        check = client.get(resolved + ".md5")
        check.raise_for_status()
        if md5.hexdigest() != check.text.split()[0]:
            raise ValueError("Provider PBF MD5 mismatch")
    assembled.replace(target)
    temporary.unlink(missing_ok=True)
    for path in parts:
        path.unlink()
    doc = {"source_uri": resolved, "stable_source": endpoint, "sha256": sha256(target),
           "provider_md5_verified": True, "bytes": size,
           "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
           "license": "OpenStreetMap contributors, ODbL 1.0",
           "attribution_uri": configured_url("PRAVAH_OSM_ATTRIBUTION_URL"),
           "historical_eligibility": False}
    save(receipt, doc)
    return target, doc


def tags(value):
    return dict(re.findall(r'"([^"]+)"=>"([^"]*)"', value if isinstance(value, str) else ""))


def export(frame, root, name):
    frame = frame.copy().to_crs("EPSG:4326")
    records = []
    for _, row in frame.iterrows():
        extra = tags(row.get("other_tags"))
        props = {k: str(row[k]) for k in ["name", "highway", "waterway", "building", "natural", "landuse", "amenity", "power", "place"]
                 if k in row and row[k] is not None and not pd.isna(row[k])}
        props.update({k: extra[k] for k in ["name:en", "name:ne", "bridge", "water", "amenity", "power", "place"] if k in extra})
        identity = row.get("osm_id")
        way_id = row.get("osm_way_id")
        source_type = "node" if row.geometry.geom_type == "Point" else "relation" if row.geometry.geom_type in {"Polygon", "MultiPolygon"} else "way"
        props["osm_id"] = "way/" + str(way_id) if pd.notna(way_id) else source_type + "/" + str(identity)
        records.append({"type": "Feature", "geometry": row.geometry.__geo_interface__, "properties": props})
    path = root / (name + ".geojson")
    save(path, {"type": "FeatureCollection", "is_synthetic": False, "features": records})
    return {"file": path.name, "sha256": sha256(path), "features": len(records), "crs": "EPSG:4326"}


def prepare_geography(pbf, provenance, root):
    if (root / "manifest.json").exists():
        raise ValueError("Geography already exists; use a new --geography root")
    print("Reading bounded OSM river and road lines", flush=True)
    lines = pyogrio.read_dataframe(pbf, layer="lines", bbox=BBOX)
    if lines.crs is None:
        raise ValueError("OSM line CRS missing")
    labels = lines["name"].fillna("") + " " + lines["other_tags"].fillna("")
    rivers = lines[lines["waterway"].eq("river") & labels.map(lambda x: bool(RIVER_NAMES.search(x)))].copy()
    if rivers.empty:
        raise ValueError("Named corridor rivers were not found")
    boundary = box(*BBOX)
    rivers.geometry = rivers.geometry.intersection(boundary)
    rivers = rivers[~rivers.geometry.is_empty & rivers.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    river_metric = rivers.to_crs("EPSG:32645").geometry.union_all()
    corridor = river_metric.buffer(350)
    lines = lines[lines.to_crs("EPSG:32645").intersects(corridor)]
    roads = lines[lines["highway"].notna()].copy()
    bridges = lines[lines["other_tags"].map(lambda x: tags(x).get("bridge") not in {None, "no"})].copy()
    print("Reading mapped building/water polygons", flush=True)
    polygons = pyogrio.read_dataframe(pbf, layer="multipolygons", bbox=BBOX)
    polygons = polygons[polygons.to_crs("EPSG:32645").intersects(corridor)]
    buildings = polygons[polygons["building"].notna() & ~polygons["building"].eq("no")].copy()
    riverbanks = polygons[polygons["other_tags"].map(lambda x: tags(x).get("water") == "river")
                          | polygons["other_tags"].map(lambda x: tags(x).get("waterway") == "riverbank")].copy()
    print("Reading mapped facilities and settlements", flush=True)
    points = pyogrio.read_dataframe(pbf, layer="points", bbox=BBOX)
    points = points[points.to_crs("EPSG:32645").intersects(corridor)]
    settlement_points = points[points["place"].isin(["city", "town", "village", "hamlet", "isolated_dwelling"])].copy()
    def facility(row):
        t = tags(row.get("other_tags"))
        return row.get("amenity") in {"hospital", "clinic", "school", "police", "fire_station"} or t.get("amenity") in {"hospital", "clinic", "school", "police", "fire_station"} or t.get("power") in {"plant", "substation", "generator"}
    facility_frames = []
    for f in [points, polygons]:
        selected = f[f.apply(facility, axis=1)] if not f.empty else f
        facility_frames.append(selected)
    facilities = gpd.GeoDataFrame(pd.concat(facility_frames, ignore_index=True), crs=points.crs)
    root.mkdir(parents=True, exist_ok=True)
    layers = {name: export(frame, root, name) for name, frame in {
        "rivers": rivers, "riverbanks": riverbanks, "roads": roads, "bridges": bridges,
        "buildings": buildings, "facilities": facilities, "settlements": settlement_points,
    }.items()}
    save(root / "manifest.json", {"schema_version": 1, "is_synthetic": False,
         "retrieved_at_utc": provenance["retrieved_at_utc"], "source": provenance,
         "bounds_wgs84": BBOX, "layers": layers, "historical_eligibility": False,
         "asset_selection_distance_m": 350, "metric_crs": "EPSG:32645",
         "river_length_m": float(river_metric.length),
         "river_names": sorted(str(x) for x in rivers["name"].dropna().unique()),
         "semantics": "Named OSM river lines; no added connectors or inferred widths. Mapping gaps may exist. Upstream Tibet coverage is not provided by the Nepal extract.",
         "coverage": "Mapped assets intersecting a 350 m corridor within the selection rectangle. Incomplete OSM coverage; not a complete household or infrastructure inventory."})
    print(json.dumps(layers, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["geography", "live", "all"])
    parser.add_argument("--geography", type=Path, default=PROJECT / "data/processed/nepal_monitor")
    parser.add_argument("--raw", type=Path, default=PROJECT / "data/raw/nepal_monitor")
    parser.add_argument("--live", type=Path, default=PROJECT / "outputs/nepal_monitor/live")
    args = parser.parse_args()
    if args.command in {"geography", "all"}:
        pbf, receipt = acquire(args.raw)
        prepare_geography(pbf, receipt, args.geography)
    if args.command in {"live", "all"}:
        monitor = LiveMonitor(args.live)
        monitor.refresh(force=True)
        print(json.dumps({k: {"status": v["status"], "records": len(v["records"])} for k, v in monitor.snapshot()["sources"].items()}, indent=2))


if __name__ == "__main__":
    main()
