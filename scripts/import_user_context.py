#!/usr/bin/env python3
"""Import local context archives without treating terrain/weather as S2 labels."""

import argparse
import csv
import hashlib
import json
import math
import os
import stat
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds, transform


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    os.replace(part, path)


def safe_member(info):
    name = PurePosixPath(info.filename)
    if name.is_absolute() or ".." in name.parts or "\\" in info.filename:
        raise ValueError("Archive member has unsafe path")
    if stat.S_ISLNK(info.external_attr >> 16):
        raise ValueError("Archive symlinks are not permitted")
    if info.is_dir() or any(p == "__MACOSX" or p.startswith(".") for p in name.parts):
        return "metadata"
    if name.suffix == ".part":
        return "incomplete"
    return "data"


def expected_checksums(archive):
    result = {}
    for info in archive.infolist():
        if (
            safe_member(info) != "data"
            or PurePosixPath(info.filename).name != "manifest.json"
        ):
            continue
        manifest = json.loads(archive.read(info))
        base = PurePosixPath(info.filename).parent
        for record in manifest.get("success", []):
            for item in [record, *record.get("generated_files", [])]:
                if item.get("sha256"):
                    name = str(base / item["path"])
                    result[name] = {
                        "sha256": item["sha256"],
                        "bytes": item["bytes"],
                        "packaged_status": record["status"],
                    }
    return result


def extract_archive(path, root):
    records, skipped = [], []
    archive_hash = sha256(path)
    with zipfile.ZipFile(path) as archive:
        expected = expected_checksums(archive)
        seen = set()
        for info in archive.infolist():
            disposition = safe_member(info)
            if disposition != "data":
                if disposition == "incomplete":
                    skipped.append(
                        {
                            "member": info.filename,
                            "bytes": info.file_size,
                            "reason": "incomplete_download",
                        }
                    )
                continue
            if info.filename in seen:
                raise ValueError("Duplicate archive member")
            seen.add(info.filename)
            target = root / info.filename
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError("Destination escapes import root")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                actual = sha256(target)
                # Check original archive content too, even for sources without a manifest.
                digest = hashlib.sha256()
                with archive.open(info) as source:
                    for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                        digest.update(chunk)
                if actual != digest.hexdigest():
                    raise ValueError(
                        "Existing destination differs from original archive"
                    )
            else:
                part = target.with_suffix(target.suffix + ".part")
                with archive.open(info) as source, part.open("wb") as dest:
                    for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                        dest.write(chunk)  # ZipFile checks CRC at EOF.
                actual = sha256(part)
                pinned = expected.get(info.filename)
                if pinned and (
                    actual != pinned["sha256"] or part.stat().st_size != pinned["bytes"]
                ):
                    raise ValueError("Archive data differs from its supplied manifest")
                os.replace(part, target)
            pinned = expected.get(info.filename)
            if pinned and (
                actual != pinned["sha256"] or target.stat().st_size != pinned["bytes"]
            ):
                raise ValueError("Existing data differs from supplied manifest")
            records.append(
                {
                    "archive": path.name,
                    "member": info.filename,
                    "path": str(target),
                    "bytes": target.stat().st_size,
                    "sha256": actual,
                    "archive_crc_checked": True,
                    "matches_packaged_manifest": True if pinned else None,
                    "packaged_status": pinned.get("packaged_status")
                    if pinned
                    else None,
                    "provider_response_independently_verified": False,
                    "strict_replay_eligible": False,
                    "verified_available_at_utc": None,
                }
            )
    return {
        "archive_path": str(path),
        "archive_sha256": archive_hash,
        "files": records,
        "excluded_partial_files": skipped,
    }


def clean_number(value, fill_value=None):
    if value is None or value == "":
        return ""
    number = float(value)
    return "" if not math.isfinite(number) or number == fill_value else number


def utc_timestamp(local, zone):
    value = datetime.fromisoformat(local)
    if value.tzinfo is not None:
        raise ValueError("Expected naive provider local time plus explicit timezone")
    return value.replace(tzinfo=ZoneInfo(zone)).astimezone(timezone.utc).isoformat()


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def weather_tables(root, output):
    records, locations = [], {}
    main = root / "nepal_pre_aug_2026"
    for source_root in (main, main / "era5_supplement"):
        tasks = json.loads((source_root / "plan.json").read_text())["tasks"]
        grouped = defaultdict(list)
        for task in tasks:
            if task["provider"] in ("open_meteo", "nasa_power"):
                location = PurePosixPath(task["relpath"]).parts[2]
                model = task["params"].get("models", "NASA_POWER")
                grouped[(task["provider"], model, location)].append(task)
        for (provider, model, location), group in sorted(grouped.items()):
            rows, sources, units, keys = [], [], None, None
            query_coordinates = None
            for task in sorted(group, key=lambda t: t["start"]):
                source = source_root / task["relpath"]
                data = json.loads(source.read_text())
                sources.append({"path": str(source), "sha256": sha256(source)})
                params = task["params"]
                lat = params.get("latitude")
                lon = params.get("longitude")
                if lat is None or lon is None:
                    raise ValueError("Weather request coordinates missing")
                query_coordinates = [lon, lat]
                if location in locations and locations[location] != query_coordinates:
                    raise ValueError(
                        "Location name resolves to different requested points"
                    )
                locations[location] = query_coordinates
                if provider == "open_meteo":
                    values = data["hourly"]
                    current_keys = [k for k in values if k != "time"]
                    current_units = {k: data["hourly_units"][k] for k in current_keys}
                    for i, timestamp in enumerate(values["time"]):
                        row = {
                            "timestamp_utc": utc_timestamp(timestamp, data["timezone"]),
                            "timestamp_npt": timestamp,
                            "grid_longitude": data["longitude"],
                            "grid_latitude": data["latitude"],
                        }
                        for key in current_keys:
                            if len(values[key]) != len(values["time"]):
                                raise ValueError("Weather time/value lengths differ")
                            row[key] = clean_number(values[key][i])
                        rows.append(row)
                    time_key = "timestamp_utc"
                    prefix = [
                        time_key,
                        "timestamp_npt",
                        "grid_longitude",
                        "grid_latitude",
                    ]
                    time_basis = "hourly Asia/Kathmandu converted to UTC; original local timestamp retained"
                else:
                    if data["header"]["time_standard"] != "UTC":
                        raise ValueError("NASA daily time convention is not UTC")
                    values = data["properties"]["parameter"]
                    current_keys = list(values)
                    current_units = {
                        k: data["parameters"][k]["units"] for k in current_keys
                    }
                    dates = list(values[current_keys[0]])
                    if any(list(values[k]) != dates for k in current_keys):
                        raise ValueError("NASA parameter dates differ")
                    for date in dates:
                        rows.append(
                            {
                                "date_utc": datetime.strptime(date, "%Y%m%d")
                                .date()
                                .isoformat(),
                                **{
                                    k: clean_number(
                                        values[k][date], data["header"]["fill_value"]
                                    )
                                    for k in current_keys
                                },
                            }
                        )
                    time_key, prefix, time_basis = (
                        "date_utc",
                        ["date_utc"],
                        "UTC calendar date; not hourly measurements",
                    )
                if keys is not None and (
                    current_keys != keys or current_units != units
                ):
                    raise ValueError(
                        "Weather schema or units changed across monthly files"
                    )
                keys, units = current_keys, current_units
            rows.sort(key=lambda r: r[time_key])
            if len({r[time_key] for r in rows}) != len(rows):
                raise ValueError("Duplicate weather timestamp")
            output_path = output / "weather" / model / (location + ".csv")
            write_csv(output_path, rows, prefix + keys)
            record = {
                "path": str(output_path),
                "sha256": sha256(output_path),
                "provider": provider,
                "model": model,
                "location": location,
                "requested_lon_lat": query_coordinates,
                "rows": len(rows),
                "first_time": rows[0][time_key],
                "last_time": rows[-1][time_key],
                "units": units,
                "time_basis": time_basis,
                "missing_values": {k: sum(r[k] == "" for r in rows) for k in keys},
                "sources": sources,
                "role": "retrospective_context",
                "strict_replay_eligible": False,
                "verified_available_at_utc": None,
                "u_net_training_compatible": False,
            }
            write_json(output_path.with_suffix(".metadata.json"), record)
            records.append(record)
    return records, locations


def rainfall_crops(root, output, locations):
    records, samples = [], []
    for path in sorted(
        (root / "nepal_maps_and_events/raw/chirps_global/2026").glob("*.tif")
    ):
        date = "-".join(path.stem.split(".")[-3:])
        datetime.strptime(date, "%Y-%m-%d")
        with rasterio.open(path) as src:
            if src.count != 1 or src.crs.to_epsg() != 4326:
                raise ValueError("Unexpected CHIRPS layout")
            window = from_bounds(80, 26, 89, 31, transform=src.transform)
            window = Window(
                round(window.col_off),
                round(window.row_off),
                round(window.width),
                round(window.height),
            )
            pixels = src.read(1, window=window)
            good = np.isfinite(pixels) & (pixels >= 0)
            values = np.where(good, pixels, -9999).astype(np.float32)
            profile = {
                "driver": "GTiff",
                "count": 1,
                "width": values.shape[1],
                "height": values.shape[0],
                "crs": src.crs,
                "transform": transform(window, src.transform),
                "dtype": "float32",
                "nodata": -9999,
                "compress": "deflate",
            }
            target = output / "chirps_nepal_context" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(target, "w", **profile) as dest:
                dest.write(values, 1)
                dest.set_band_description(
                    1, "CHIRPS v3 rnl daily precipitation, mm/day"
                )
                dest.update_tags(
                    role="retrospective_context",
                    strict_replay_eligible="false",
                    date_utc=date,
                    bbox_scope="rectangular context window, not Nepal boundary or flood footprint",
                )
            for location, (lon, lat) in sorted(locations.items()):
                row, column = src.index(lon, lat)
                value = float(src.read(1, window=Window(column, row, 1, 1))[0, 0])
                samples.append(
                    {
                        "date_utc": date,
                        "location": location,
                        "requested_longitude": lon,
                        "requested_latitude": lat,
                        "cell_longitude": src.xy(row, column)[0],
                        "cell_latitude": src.xy(row, column)[1],
                        "precipitation_mm_per_day": value
                        if math.isfinite(value) and value >= 0
                        else "",
                    }
                )
        with rasterio.open(target) as checked:
            if (
                not np.array_equal(checked.read(1), values)
                or checked.transform != profile["transform"]
            ):
                raise RuntimeError("Crop changed original grid or values")
        records.append(
            {
                "path": str(target),
                "source": str(path),
                "source_sha256": sha256(path),
                "sha256": sha256(target),
                "date_utc": date,
                "shape": list(values.shape),
                "valid_pixels": int(good.sum()),
                "source_nodata_tag": None,
                "output_nodata": -9999,
                "units": "mm/day",
                "grid": "EPSG:4326, original 0.05 degree grid; no resampling",
                "role": "retrospective_context",
                "strict_replay_eligible": False,
                "u_net_training_compatible": False,
            }
        )
    if samples:
        write_csv(output / "chirps_point_series.csv", samples, list(samples[0]))
    return records, len(samples)


def terrain_inventory(root, output, locations):
    records, coverage = [], {}
    for path in sorted((root / "terrain").glob("*.tif")):
        with rasterio.open(path) as src:
            if src.count != 1 or src.crs is None:
                raise ValueError("Unexpected terrain raster layout")
            values = src.read(
                1, masked=True
            )  # Full raster decoding, including every block.
            finite = np.isfinite(values.compressed())
            if not finite.all():
                raise ValueError("Nonfinite terrain pixels")
            record = {
                "path": str(path),
                "sha256": sha256(path),
                "shape": list(src.shape),
                "crs": str(src.crs),
                "resolution_degrees": list(src.res),
                "nodata": src.nodata,
                "bounds": list(src.bounds),
                "min": float(values.min()),
                "max": float(values.max()),
                "full_raster_read": True,
                "units": "metres for the named Copernicus DSM product; source TIFF has no units tag",
                "role": "terrain_context",
                "strict_replay_eligible": False,
                "u_net_training_compatible": False,
            }
            records.append(record)
            for location, (lon, lat) in locations.items():
                if (
                    src.bounds.left <= lon < src.bounds.right
                    and src.bounds.bottom < lat <= src.bounds.top
                ):
                    value = float(next(src.sample([(lon, lat)]))[0])
                    coverage[location] = {
                        "source": str(path),
                        "surface_height_m": value,
                    }
    coverage = {
        name: {
            "requested_lon_lat": point,
            "complete_tile_available": name in coverage,
            **coverage.get(name, {}),
        }
        for name, point in sorted(locations.items())
    }
    write_json(output / "terrain_point_coverage.json", coverage)
    return records, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", nargs="+", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/user_supplied"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/user_context")
    )
    parser.add_argument(
        "--audit", type=Path, default=Path("outputs/user_dataset_intake_20261114")
    )
    args = parser.parse_args()
    root, output, audit = (
        args.raw_root.resolve(),
        args.output.resolve(),
        args.audit.resolve(),
    )
    imports = []
    for path in args.archives:
        print("IMPORTING", path.name, flush=True)
        result = extract_archive(path.resolve(), root)
        imports.append(result)
        write_json(audit / (path.stem + ".intake.json"), result)
        print(
            "IMPORTED",
            path.name,
            len(result["files"]),
            "excluded partial",
            len(result["excluded_partial_files"]),
            flush=True,
        )
    print("PREPARING_WEATHER", flush=True)
    weather, locations = weather_tables(root, output)
    print("PREPARING_RAINFALL", flush=True)
    rainfall, sample_count = rainfall_crops(root, output, locations)
    print("CHECKING_TERRAIN", flush=True)
    terrain, terrain_coverage = terrain_inventory(root, output, locations)
    imported = [r for item in imports for r in item["files"]]
    hash_counts = Counter(r["sha256"] for r in imported)
    report = {
        "status": "context_prepared",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "archives": [
            {k: v for k, v in item.items() if k != "files"} for item in imports
        ],
        "imported_files": len(imported),
        "imported_bytes": sum(r["bytes"] for r in imported),
        "matches_supplied_manifest": sum(
            r["matches_packaged_manifest"] is True for r in imported
        ),
        "duplicate_content_files_beyond_first": sum(
            n - 1 for n in hash_counts.values()
        ),
        "weather_tables": len(weather),
        "weather_rows": sum(r["rows"] for r in weather),
        "weather_models": dict(Counter(r["model"] for r in weather)),
        "locations": locations,
        "rainfall_crops": len(rainfall),
        "rainfall_point_rows": sample_count,
        "rainfall_dates": [
            min(r["date_utc"] for r in rainfall),
            max(r["date_utc"] for r in rainfall),
        ],
        "complete_terrain_tiles": len(terrain),
        "rasuwa_terrain": terrain_coverage.get("rasuwa"),
        "new_u_net_training_pairs": 0,
        "strict_replay_eligible": False,
        "limitations": [
            "No labeled six-band optical training pairs in these archives",
            "Sentinel and GPM JSON files are partial catalogs, not image/granule downloads",
            "Missing historical availability; archive names are not evidence of cutoff eligibility",
            "Six partial terrain downloads remain only in the original zip",
            "Rainfall/weather periods end in July; no August 26 event-period rainfall in this package",
            "CHIRPS rnl uses ERA5 daily timing; agreement is not independent validation",
            "Packaged checksums verify internal integrity, not independent provider authenticity",
        ],
    }
    write_json(
        output / "context_catalog.json",
        {
            "weather": weather,
            "rainfall": rainfall,
            "terrain": terrain,
            "source_catalogs": [r for r in imported if "/catalogue/" in r["member"]],
            "role": "retrospective_context",
            "u_net_training_compatible": False,
        },
    )
    write_json(audit / "summary.json", report)
    print("CONTEXT_READY", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
