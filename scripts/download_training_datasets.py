#!/usr/bin/env python3
"""Download a pinned, reviewed training-data plan inside the Fuzzball volume.

Only original provider files are downloaded. Preparing training pairs is a separate
step: a successful transfer is not a claim that a raster or its labels are usable.
"""

import argparse
import base64
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from http.client import HTTPException
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def target_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("Unsafe destination path")
    result = Path(root).joinpath(*path.parts)
    if not result.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Destination escapes volume")
    return result


def hashes(path):
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    blob = hashlib.sha1(usedforsecurity=False)
    blob.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
            blob.update(block)
    return {
        "sha256": sha.hexdigest(),
        "md5_base64": base64.b64encode(md5.digest()).decode(),
        "git_blob_sha1": blob.hexdigest(),
    }


def check_file(path, asset):
    if path.stat().st_size != asset["bytes"]:
        raise ValueError("Provider byte-size mismatch")
    measured = hashes(path)
    expected = asset["checksum"]
    if measured[expected["algorithm"]] != expected["value"]:
        raise ValueError("Provider checksum mismatch")
    return measured["sha256"]


def download_asset(asset, root, retries=3, timeout=90):
    target = target_path(root, asset["destination"])
    record = {
        **asset,
        "path": str(target),
        "checked_at_utc": utcnow(),
        "raster_open_status": "not_tested",
        "status": "failed",
    }
    try:
        if target.exists():
            # Existing scientific inputs are never silently replaced.
            record["sha256"] = check_file(target, asset)
            record["status"] = "verified_existing_transfer"
            return record
    except (ValueError, OSError) as exc:
        record["error"] = "Existing input failed integrity check: " + str(exc)
        return record
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    for attempt in range(retries):
        try:
            # GCS inventories contain literal spaces in a few original object keys.
            url = urlsplit(asset["url"])
            request_url = urlunsplit(
                (url.scheme, url.netloc, quote(url.path, safe="/%"), url.query, "")
            )
            request = Request(
                request_url, headers={"User-Agent": "PRAVAH-research-intake/1.0"}
            )
            with urlopen(request, timeout=timeout) as response:
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) != asset["bytes"]:
                    raise ValueError("Response byte size differs from pinned inventory")
                size = 0
                with temporary.open("wb") as output:
                    while block := response.read(4 * 1024 * 1024):
                        size += len(block)
                        if size > asset["bytes"]:
                            raise ValueError("Response exceeds pinned size")
                        output.write(block)
            record["sha256"] = check_file(temporary, asset)
            os.replace(temporary, target)
            record["status"] = "downloaded_integrity_verified"
            record["checked_at_utc"] = utcnow()
            record["attempts"] = attempt + 1
            return record
        except (OSError, ValueError, HTTPException) as exc:
            # Redirect URLs can be signed. Never write an exception's URL to logs.
            record["error"] = (
                type(exc).__name__
                + ": "
                + (str(exc) if isinstance(exc, ValueError) else "transfer/read failure")
            )
            if hasattr(exc, "code"):
                record["http_status"] = exc.code
            temporary.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    return record


def validate_plan(plan, root, max_download_bytes):
    seen = set()
    total = 0
    for asset in plan["assets"]:
        target = target_path(root, asset["destination"])
        if str(target) in seen:
            raise ValueError("Duplicate destination")
        seen.add(str(target))
        url = urlsplit(asset["url"])
        if (
            url.scheme != "https"
            or url.hostname
            not in {
                "storage.googleapis.com",
                "huggingface.co",
                "raw.githubusercontent.com",
            }
            or url.username
            or url.password
        ):
            raise ValueError("Unexpected provider URL")
        if not isinstance(asset["bytes"], int) or asset["bytes"] <= 0:
            raise ValueError("Missing provider size")
        if asset["checksum"]["algorithm"] not in {
            "sha256",
            "md5_base64",
            "git_blob_sha1",
        }:
            raise ValueError("Unsupported provider checksum")
        if not target.exists():
            total += asset["bytes"]
    if total > max_download_bytes:
        raise ValueError("Plan exceeds download budget")
    return total


def logical_bytes(root):
    total = 0
    for directory, _, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                try:
                    total += path.stat().st_size
                except FileNotFoundError:
                    pass  # Concurrent jobs may remove their own scratch files.
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--root", default="/data")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-download-gib", type=float, default=160)
    parser.add_argument("--volume-budget-bytes", type=int, default=500_000_000_000)
    parser.add_argument("--reserve-gib", type=float, default=80)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Append repairs to a completed audit without downloading verified files again",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("Use 1 to 8 download workers")
    root = Path(args.root)
    plan = json.loads(Path(args.plan).read_text())
    needed = validate_plan(plan, root, int(args.max_download_gib * 2**30))
    current = logical_bytes(root)
    reserve = int(args.reserve_gib * 2**30)
    if current + needed + reserve > args.volume_budget_bytes:
        raise RuntimeError("Plan plus reserve exceeds the Team 12 volume budget")
    if needed + reserve > shutil.disk_usage(root).free:
        raise RuntimeError("Insufficient filesystem free space")
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    assets = plan["assets"]
    completed = []
    if args.retry_failed:
        previous_summary = report_dir / "download_summary.json"
        if not previous_summary.is_file():
            raise RuntimeError(
                "Wait for the original transfer to finish before retrying"
            )
        previous = {}
        for line in (report_dir / "transfers.jsonl").read_text().splitlines():
            record = json.loads(line)
            previous[record["destination"]] = record
        if set(previous) != {a["destination"] for a in assets}:
            raise RuntimeError("Previous audit and plan have different destinations")
        for asset in assets:
            if any(
                previous[asset["destination"]][key] != asset[key]
                for key in ("url", "bytes", "checksum")
            ):
                raise RuntimeError("Cannot retry an altered provider asset")
        completed = [r for r in previous.values() if r["status"] != "failed"]
        assets = [a for a in assets if previous[a["destination"]]["status"] == "failed"]
        backup = report_dir / (
            "download_summary_before_retry_" + str(time.time_ns()) + ".json"
        )
        shutil.copyfile(previous_summary, backup)
    preflight = {
        "at_utc": utcnow(),
        "existing_logical_bytes": current,
        "planned_new_bytes": needed,
        "reserved_bytes": reserve,
        "volume_budget_bytes": args.volume_budget_bytes,
        "asset_count": len(plan["assets"]),
        "workers": args.workers,
        "plan_sha256": hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),
    }
    preflight_name = "preflight_retry.json" if args.retry_failed else "preflight.json"
    (report_dir / preflight_name).write_text(json.dumps(preflight, indent=2) + "\n")
    print("PREFLIGHT", json.dumps(preflight), flush=True)
    with (
        (report_dir / "transfers.jsonl").open("a" if args.retry_failed else "w") as log,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = [pool.submit(download_asset, asset, root) for asset in assets]
        for future in as_completed(futures):
            record = future.result()
            completed.append(record)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if len(completed) % 100 == 0 or record["status"] == "failed":
                print(
                    "PROGRESS",
                    len(completed),
                    "/",
                    len(plan["assets"]),
                    "bytes",
                    sum(r["bytes"] for r in completed if r["status"] != "failed"),
                    "failed",
                    sum(r["status"] == "failed" for r in completed),
                    flush=True,
                )
    summary = {"completed_at_utc": utcnow(), "datasets": {}, "failed_assets": []}
    for record in completed:
        dataset = summary["datasets"].setdefault(
            record["dataset"],
            {
                "downloaded_files": 0,
                "reused_files": 0,
                "verified_bytes": 0,
                "failed_files": 0,
            },
        )
        if record["status"] == "failed":
            dataset["failed_files"] += 1
            summary["failed_assets"].append(record)
        else:
            dataset["verified_bytes"] += record["bytes"]
            field = (
                "reused_files"
                if record["status"] == "verified_existing_transfer"
                else "downloaded_files"
            )
            dataset[field] += 1
    (report_dir / "download_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("DOWNLOAD_SUMMARY", json.dumps(summary["datasets"]), flush=True)
    if summary["failed_assets"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
