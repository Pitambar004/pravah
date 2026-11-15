#!/usr/bin/env python3
"""Prepare verified files and cutoff snapshots for the Nepal research replay.

All downloaded historical responses are current versions: historical public
availability remains unknown. They are never silently admitted before a cutoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from pravah.replay.experiment import audit, freeze, register_evaluation, sha256
from pravah.configuration import configured_url

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "outputs/nepal_replay/prepared"
EVENT_ID = "rasuwa-2026-08-26"
DEFAULT_CUTOFF = "2026-08-26T08:30:00+05:45"
REPORT = configured_url("PRAVAH_DHM_EVENT_REPORT_URL")
SOURCES = [
    (
        "usgs_us7000tbwb",
        "https://earthquake.usgs.gov/earthquakes/feed/v1.0/detail/us7000tbwb.geojson",
        "json",
        "evaluation_reference",
    ),
    ("dhm_2026_technical_report", REPORT, "pdf", "evaluation_reference"),
    (
        "ndrrma_2025_event",
        configured_url("PRAVAH_NDRRMA_2025_REPORT_URL"),
        "pdf",
        "context_only",
    ),
]


def save(path: Path, document: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def record(
    root: Path, path: Path, identity: str, source: str, role: str, **extra
) -> dict:
    return {
        "asset_id": identity,
        "path": str(path.relative_to(root)),
        "sha256": sha256(path),
        "file_bytes": path.stat().st_size,
        "source_uri": source,
        "role": role,
        "download_status": "verified",
        "acquired_at_utc": None,
        "available_at_utc": None,
        "availability_verified": False,
        "is_synthetic": False,
        "availability_note": "Retrieved/imported now; verified historical availability unknown",
        **extra,
    }


def stage_context(root: Path) -> list[dict]:
    catalog = json.loads(
        (PROJECT / "data/processed/user_context/context_catalog.json").read_text()
    )
    assets = []
    for item in catalog["weather"]:
        source = Path(item["path"])
        if sha256(source) != item["sha256"]:
            raise ValueError("Imported weather checksum mismatch")
        relative = Path("context/weather") / item["model"] / source.name
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assets.append(
            record(
                root,
                target,
                f"weather_{item['model']}_{item['location']}",
                "user-supplied provider response; original hashes in context catalog",
                "input_candidate",
                units=item["units"],
                time_basis=item["time_basis"],
                observation_first=item["first_time"],
                observation_last=item["last_time"],
                original_sources_sha256=[s["sha256"] for s in item["sources"]],
            )
        )
    # Preserve the imported lineage without machine-specific absolute paths.
    lineage = root / "context/imported_lineage.json"
    save(lineage, [{k: v for k, v in x.items() if k != "path"} for x in assets])
    # Only the complete Rasuwa terrain tile is needed for this transfer.
    for item in catalog["terrain"]:
        west, south, east, north = item["bounds"]
        if west <= 85.35 <= east and south <= 28.15 <= north:
            source = Path(item["path"])
            if sha256(source) != item["sha256"]:
                raise ValueError("Rasuwa terrain checksum mismatch")
            target = root / "context/terrain" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            assets.append(
                record(
                    root,
                    target,
                    "rasuwa_copernicus_dsm",
                    configured_url("PRAVAH_DEM_README_URL"),
                    "context_only",
                    crs=item["crs"],
                    bounds=item["bounds"],
                    nodata=item["nodata"],
                    units=item["units"],
                    class_semantics="DSM surface height; not water depth or a calibrated channel model",
                )
            )
    replay = PROJECT / "data/processed/live_app/replay_assets.json"
    if replay.exists():
        target = root / "context/satellite_catalog.json"
        save(target, json.loads(replay.read_text()))
        assets.append(
            record(
                root,
                target,
                "imported_satellite_catalog",
                "Copernicus STAC archived by supplied package",
                "catalog_only",
                description="L2A/GRD metadata; no downloaded optical L1C model input",
            )
        )
    return assets


def get_bytes(client: httpx.Client, url: str, **kwargs) -> bytes:
    with client.stream("GET", url, **kwargs) as response:
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > 20_000_000:
                raise ValueError("Public response exceeds the 20 MB acquisition limit")
            chunks.append(chunk)
    return b"".join(chunks)


def collect(root: Path) -> tuple[list[dict], list[dict]]:
    assets, attempts = [], []
    with httpx.Client(timeout=20, follow_redirects=True) as client:
        for identity, url, kind, role in SOURCES:
            try:
                content = get_bytes(client, url)
                if kind == "pdf" and not content.startswith(b"%PDF"):
                    raise ValueError("Provider did not return a PDF")
                extra = {}
                if kind == "json":
                    data = json.loads(content)
                    if data.get("id") != "us7000tbwb":
                        raise ValueError("USGS event identity mismatch")
                    properties = data["properties"]
                    extra = {
                        "acquired_at_utc": datetime.fromtimestamp(
                            properties["time"] / 1000, timezone.utc
                        ).isoformat(),
                        "event_type": properties["type"],
                        "magnitude": properties["mag"],
                        "catalog_updated_at_utc": datetime.fromtimestamp(
                            properties["updated"] / 1000, timezone.utc
                        ).isoformat(),
                        "interpretation": "Current reviewed event type; not proof of an independently predictive earthquake",
                    }
                target = root / "sources" / (identity + "." + kind)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                assets.append(record(root, target, identity, url, role, **extra))
                attempts.append(
                    {
                        "asset_id": identity,
                        "status": "downloaded_verified",
                        "sha256": sha256(target),
                        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                attempts.append(
                    {"asset_id": identity, "status": "failed", "reason": str(exc)[:200]}
                )
        for name, latitude, longitude in [
            ("rasuwa", 28.15, 85.35),
            ("bhote_koshi_upstream", 28.43, 85.55),
        ]:
            identity = "august_era5_" + name
            try:
                url = configured_url("PRAVAH_OPEN_METEO_ARCHIVE_URL")
                params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_date": "2026-08-01",
                    "end_date": "2026-08-28",
                    "models": "era5",
                    "timezone": "Asia/Kathmandu",
                    "daily": "precipitation_sum,temperature_2m_mean",
                }
                content = get_bytes(client, url, params=params)
                data = json.loads(content)
                if data["daily"]["time"] != [
                    f"2026-08-{day:02d}" for day in range(1, 29)
                ]:
                    raise ValueError("Unexpected August weather dates")
                if data["daily_units"]["precipitation_sum"] != "mm":
                    raise ValueError("Unexpected precipitation units")
                target = root / "sources" / (identity + ".json")
                target.write_bytes(content)
                assets.append(
                    record(
                        root,
                        target,
                        identity,
                        url,
                        "input_candidate",
                        request=params,
                        units=data["daily_units"],
                        observation_first="2026-08-01",
                        observation_last="2026-08-28",
                        description="ERA5 retrospective reanalysis; daily rainfall, not a river gauge",
                    )
                )
                attempts.append(
                    {
                        "asset_id": identity,
                        "status": "downloaded_verified",
                        "sha256": sha256(target),
                    }
                )
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                attempts.append(
                    {"asset_id": identity, "status": "failed", "reason": str(exc)[:200]}
                )
        # Use only the provider's own station form and public historical view.
        station_url = "https://dhm.gov.np/hydrology/hms-Single/4913"
        try:
            page = client.get(station_url)
            page.raise_for_status()
            csrf = re.search(r'name="csrf_test_name" value="([^"]+)"', page.text)
            series = re.search(r'name="seriesid"[^>]*value="([0-9]+)"', page.text)
            if not csrf or not series:
                raise ValueError("Public station form changed")
            endpoint = configured_url("PRAVAH_DHM_HISTORY_URL")
            response = client.post(
                endpoint,
                data={
                    "csrf_test_name": csrf.group(1),
                    "seriesid": series.group(1),
                    "date": "2026-08-26",
                    "period": "1",
                },
                headers={"Referer": station_url, "X-Requested-With": "XMLHttpRequest"},
            )
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "success":
                raise ValueError("Historical gauge request did not succeed")
            # Do not persist public-session CSRF tokens or cookies in the package.
            target = root / "sources/dhm_station_4913_view.json"
            save(
                target,
                {
                    "station_page": station_url,
                    "series_id": series.group(1),
                    "date": "2026-08-26",
                    "period": "1",
                    "data": {k: data["data"][k] for k in ["table", "chart"]},
                },
            )
            assets.append(
                record(
                    root,
                    target,
                    "dhm_station_4913_view",
                    endpoint,
                    "evaluation_reference",
                    description="Provider-rendered historical view; raw samples/units require explicit parsing and review",
                )
            )
            attempts.append(
                {
                    "asset_id": "dhm_station_4913_view",
                    "status": "downloaded_provider_view",
                    "sha256": sha256(target),
                }
            )
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            attempts.append(
                {
                    "asset_id": "dhm_station_4913_view",
                    "status": "failed",
                    "reason": str(exc)[:200],
                }
            )
        try:
            endpoint = configured_url("PRAVAH_CDSE_STAC_URL")
            collection = json.loads(
                get_bytes(client, endpoint + "/collections/sentinel-2-l1c")
            )
            if collection["id"] != "sentinel-2-l1c":
                raise ValueError("L1C collection identity mismatch")
            query = {
                "collections": [collection["id"]],
                "bbox": [85.25, 28.05, 85.65, 28.5],
                "datetime": "2026-08-10T00:00:00Z/2026-09-01T23:59:59Z",
                "limit": 50,
            }
            response = client.post(endpoint + "/search", json=query)
            response.raise_for_status()
            listing = response.json()
            features = listing["features"]
            # Keep full source hashes; omit asset URLs that could be signed.
            summary = {
                "query": query,
                "search_extent_semantics": "Analyst-selected search rectangle, not a flood footprint",
                "response_sha256": hashlib.sha256(response.content).hexdigest(),
                "pagination_complete": not any(
                    x.get("rel") == "next" for x in listing.get("links", [])
                ),
                "items": [
                    {
                        "id": x["id"],
                        "collection": x["collection"],
                        "geometry": x["geometry"],
                        "properties": {
                            k: v
                            for k, v in x["properties"].items()
                            if k
                            in [
                                "datetime",
                                "start_datetime",
                                "end_datetime",
                                "eo:cloud_cover",
                                "platform",
                            ]
                        },
                        "asset_keys": list(x.get("assets", {})),
                    }
                    for x in features
                ],
            }
            target = root / "sources/rasuwa_l1c_catalog.json"
            save(target, summary)
            assets.append(
                record(
                    root,
                    target,
                    "rasuwa_l1c_catalog",
                    endpoint + "/search",
                    "catalog_only",
                    description="Metadata discovery only; no imagery download or availability validation",
                )
            )
            attempts.append(
                {
                    "asset_id": "rasuwa_l1c_catalog",
                    "status": "catalog_only",
                    "item_count": len(features),
                    "pagination_complete": summary["pagination_complete"],
                }
            )
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            attempts.append(
                {
                    "asset_id": "rasuwa_l1c_catalog",
                    "status": "failed",
                    "reason": str(exc)[:200],
                }
            )
    return assets, attempts


def prepare(root: Path, download: bool) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise ValueError(
            "Prepared evidence already exists; use a new --root to preserve it"
        )
    assets = stage_context(root)
    attempts = []
    if download:
        new, attempts = collect(root)
        assets.extend(new)
    manifest = {
        "schema_version": 1,
        "event_id": EVENT_ID,
        "assets": assets,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_source": configured_url("PRAVAH_WHO_EVENT_URL"),
        "forecast_model": None,
        "blockers": [
            "No time-eligible Nepal input manifest or historical availability receipts",
            "No trained downstream physics/forecast model or event-held-out forecast targets",
            "No downloaded matched Nepal pre/post L1C images or independent binary-water reference",
            "Fuzzball v4.3.1 service validation, registry images and live execution pending",
        ],
    }
    save(root / "manifest.json", manifest)
    save(root / "acquisition_attempts.json", attempts)
    summary = audit(manifest, DEFAULT_CUTOFF, root)
    save(
        root / "readiness.json",
        {**summary, "source_attempts": attempts, "blockers": manifest["blockers"]},
    )
    return {
        "root": str(root),
        "asset_count": len(assets),
        "eligible_input_count": summary["eligible_input_count"],
        "acquisition_attempts": attempts,
        "status": "prepared_with_documented_blockers",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "audit", "freeze", "reveal"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--download", action="store_true", help="Fetch bounded public source responses"
    )
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--run-id", default="rasuwa_before_source_failure")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.root, args.download)
        else:
            manifest = json.loads((args.root / "manifest.json").read_text())
            if args.command == "audit":
                result = audit(manifest, args.cutoff, args.root)
            elif args.command == "freeze":
                result = freeze(args.root, manifest, args.cutoff, args.run_id)
            else:
                result = register_evaluation(args.root, manifest, args.run_id)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
