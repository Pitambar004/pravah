#!/usr/bin/env python3
"""Acquire and clip the real GLO inventory; preserve later ICIMOD findings.

The 2017–2024 merged inventory is baseline context, not a 2026 daily lake
measurement. SAR/optical sources remain separate. No water depth is inferred.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import httpx

from pravah.configuration import configured_url
from pravah.replay.experiment import sha256

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = configured_url("PRAVAH_GLO_URL")
ICIMOD = configured_url("PRAVAH_ICIMOD_FINDINGS_URL")


def save(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")


def prepare(raw: Path, output: Path):
    if (output / "manifest.json").exists():
        raise ValueError("Lake context already exists; use a new output directory")
    raw.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with httpx.Client(timeout=25, follow_redirects=True) as client:
        response = client.get(configured_url("PRAVAH_GLO_API_URL"))
        response.raise_for_status()
        metadata = response.json()
        license_id = metadata["metadata"]["license"]["id"]
        if license_id != "cc-by-4.0":
            raise ValueError("Inventory license changed; review before redistribution")
        layers = {}
        for sensor in ["S1", "S2"]:
            name = f"{sensor}_20172024_TB_GLOID_UniqueLakes_v1_0.gpkg"
            item = next(f for f in metadata["files"] if f["key"] == name)
            path = raw / name
            if not path.exists():
                with client.stream("GET", item["links"]["self"]) as r:
                    r.raise_for_status()
                    count = 0
                    with path.open("wb") as stream:
                        for block in r.iter_bytes():
                            count += len(block)
                            if count > 10_000_000:
                                raise ValueError("Lake inventory exceeds acquisition limit")
                            stream.write(block)
            if path.stat().st_size != item["size"] or hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest() != item["checksum"].split(":")[1]:
                raise ValueError("Lake inventory provider checksum mismatch")
            frame = gpd.read_file(path)
            original_crs = frame.crs.to_string()
            frame = frame.to_crs("EPSG:4326")
            frame = frame.cx[85.1:85.8, 28.1:28.7]
            frame = frame[frame["BASIN"].eq("Gandaki")].copy()
            for column in ["START_DATE", "END_DATE"]:
                if column in frame:
                    frame[column] = frame[column].astype(str)
            target = output / (sensor.lower() + "_lake_inventory.geojson")
            target.write_text(frame.to_json(drop_id=True) + "\n")
            layers[sensor.lower()] = {"file": target.name, "sha256": sha256(target), "features": len(frame),
                                      "source_file": name, "source_sha256": sha256(path), "provider_md5": item["checksum"],
                                      "source_crs": original_crs, "export_crs": "EPSG:4326", "sensor": "Sentinel-1 SAR" if sensor == "S1" else "Sentinel-2 optical"}
            print(sensor, len(frame), "regional lakes; source/geometry verified", flush=True)
        item = next(f for f in metadata["files"] if f["key"] == "README.md")
        response = client.get(item["links"]["self"])
        response.raise_for_status()
        if hashlib.md5(response.content, usedforsecurity=False).hexdigest() != item["checksum"].split(":")[1]:
            raise ValueError("Provider README checksum mismatch")
        (output / "SOURCE_README.md").write_bytes(response.content)
        # Preserve actual wire bytes; some cached pages have an incorrect
        # Content-Encoding header and cannot be decoded by httpx automatically.
        page_status = "downloaded"
        try:
            with client.stream("GET", ICIMOD, headers={"Accept-Encoding": "identity"}) as reference:
                reference.raise_for_status()
                blocks, size = [], 0
                for block in reference.iter_raw():
                    size += len(block)
                    if size > 4_000_000:
                        raise ValueError("ICIMOD reference exceeds response budget")
                    blocks.append(block)
                reference_file = raw / "icimod_press_reference.html"
                reference_file.write_bytes(b"".join(blocks))
        except httpx.HTTPError:
            page_status = "direct_page_request_failed"
            response = client.get(configured_url("PRAVAH_ICIMOD_API_URL"))
            response.raise_for_status()
            response.json()  # Verify that the fallback is actual report metadata.
            reference_file = raw / "icimod_report_metadata.json"
            reference_file.write_bytes(response.content)
    later = {
        "source_uri": ICIMOD, "report_uri": configured_url("PRAVAH_ICIMOD_REPORT_URL"),
        "published_date": "2026-10-13", "retrieved_at_utc": stamp,
        "reference_receipt_sha256": sha256(reference_file), "direct_page_status": page_status,
        "receipt_kind": "press_page" if page_status == "downloaded" else "report_metadata",
        "role": "evaluation_reference",
        "findings": "ICIMOD later reported pre-collapse Sentinel-1 deformation. Its synthesis identifies rock/ice collapse rather than a conventional GLOF; no single trigger is confirmed.",
        "limits": "Later synthesis, not a daily displacement series available at the replay cutoff. Do not interpolate the total, infer an earthquake trigger, or claim exact collapse prediction."}
    save(output / "icimod_later_reference.json", later)
    manifest = {"schema_version": 1, "is_synthetic": False, "source_uri": SOURCE,
                "doi": "10.5281/zenodo.17802334", "license": "CC BY 4.0",
                "citation": "Rawlins, Watson, Bhambri, Khadka and Chand (2025), Glacial Lake Observatory, Zenodo",
                "publication_date": metadata["metadata"]["publication_date"], "retrieved_at_utc": stamp,
                "historical_eligibility": False, "layers": layers, "bounds_wgs84": [85.1, 28.1, 85.8, 28.7],
                "time_scope": "Merged maximum unique-lake outlines over annual 2017–2024 mosaics",
                "area_units": "km²", "expansion_rate_units": "km²/year, 2017–2024",
                "elevation_semantics": "ALOS 30 m DSM summary, not a 2026 measured water level or water depth",
                "limits": "Inventory baselines are not 2026 lake filling observations or verified flood-source lakes. Hydrological connection to the disaster is not established for individual lakes.",
                "references": {"icimod": {"file": "icimod_later_reference.json", "sha256": sha256(output / "icimod_later_reference.json")}}}
    save(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=PROJECT / "outputs/nepal_replay/lakes")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/processed/nepal_lakes")
    args = parser.parse_args()
    prepare(args.raw, args.output)


if __name__ == "__main__":
    main()
