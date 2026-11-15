#!/usr/bin/env python3
"""Download a manually reviewed list of small public assets with honest statuses.

Example private/local input manifest:
[{"asset_id":"real_id", "uri":"https://official-provider/real-asset.tif",
  "filename":"real-asset.tif", "license":"source licence", "sensor":"S2"}]

Do not commit manifests containing signed short-lived URLs or credentials.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import rasterio
from pravah.catalog.downloads import download_file
from pravah.configuration import configured_url


def safe_filename(filename: str) -> str:
    if not filename or filename in (".", "..") or filename != Path(filename).name or "\\" in filename:
        raise ValueError("Unsafe asset filename")
    return filename


def canonical_uri(url: str) -> str:
    """Do not copy ephemeral signed query-string tokens into published audit output."""
    u = urlsplit(url)
    if u.scheme not in ("https", configured_url("PRAVAH_LOCAL_SCHEME")) or not u.netloc:
        raise ValueError("Source must be http(s) URL")
    return urlunsplit((u.scheme, u.netloc, u.path, "", ""))


def audit_one(record: dict, folder: Path, args) -> dict:
    aid = record["asset_id"]
    filename = safe_filename(record["filename"])
    source_uri = canonical_uri(record["uri"])
    common = {k:v for k,v in record.items() if k not in ("uri", "authorization", "headers", "password")}
    common["canonical_uri_without_query"] = source_uri
    if args.dry_run:
        return {**common, "status": "dry_run_not_downloaded"}
    result = download_file(record["uri"], str(folder / filename),
                           max_bytes=args.max_bytes, retries=args.retry,
                           timeout_s=args.timeout_seconds)
    if result["status"] == "downloaded_unaudited" and filename.lower().endswith((".tif", ".tiff")):
        try:
            with rasterio.open(folder / filename) as src:
                if src.crs is None:
                    raise ValueError("Raster missing CRS")
                result.update(raster_open_status="pass", crs=str(src.crs), bands=src.count,
                              bounds=list(src.bounds), width=src.width, height=src.height)
        except Exception as exc:
            result.update(status="failed_qa", raster_open_status="fail", error=type(exc).__name__ + ": " + str(exc))
    return {**common, **result}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--out", required=True, help="Local private audit status JSON")
    p.add_argument("--max-concurrency", type=int, default=3)
    p.add_argument("--max-bytes", type=int, default=2_000_000_000)
    p.add_argument("--timeout-seconds", type=float, default=45)
    p.add_argument("--retry", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    a=p.parse_args()
    if a.max_concurrency not in range(1,9) or a.max_bytes<=0 or a.retry<0:
        raise ValueError("Invalid concurrency, download size, or retry value")
    records=json.loads(Path(a.manifest).read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must be a nonempty asset list")
    if any(not all(k in item for k in ("asset_id", "uri", "filename", "license")) for item in records):
        raise ValueError("Each entry requires asset_id, uri, filename and license")
    if len({r['asset_id'] for r in records})!=len(records) or len({r['filename'] for r in records})!=len(records):
        raise ValueError("Duplicate asset_id or destination filename")
    for r in records:
        safe_filename(r['filename'])
        canonical_uri(r['uri'])
    folder=Path(a.dest)
    folder.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=a.max_concurrency) as pool:
        futures=[pool.submit(audit_one,r,folder,a) for r in records]
        results=[f.result() for f in futures]
    output=Path(a.out)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps({"completed":sum(r.get('status')=='downloaded_unaudited' for r in results),
                      "statuses":[{"asset_id":r['asset_id'], "status":r['status']} for r in results]},indent=2))
    if any(r["status"].startswith("failed") for r in results):
        raise SystemExit(2)

if __name__ == "__main__":
    main()
