#!/usr/bin/env python3
"""Resume a pinned public checkpoint in eight verified HTTP byte ranges.

The final file is published only after its complete provider SHA256 matches.
Individual ranges are scratch files, never accepted as model checkpoints.
"""

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    assets = json.loads(args.plan.read_text())["assets"]
    matches = [a for a in assets if a["destination"].endswith(".pt")]
    if len(matches) != 1:
        raise ValueError("Require exactly one pinned checkpoint")
    asset = matches[0]
    if (
        not asset["url"].startswith("https://huggingface.co/")
        or asset["checksum"]["algorithm"] != "sha256"
    ):
        raise ValueError("Require public Hugging Face URL and SHA256")
    target = (args.root / asset["destination"]).resolve()
    if not target.is_relative_to(args.root.resolve()):
        raise ValueError("Unsafe destination")
    expected = asset["checksum"]["value"]
    if target.exists():
        if target.stat().st_size != asset["bytes"] or digest(target) != expected:
            raise ValueError("Existing checkpoint failed verification")
        print("VERIFIED_EXISTING", target, flush=True)
        return
    scratch = target.with_name(target.name + ".ranges")
    scratch.mkdir(parents=True, exist_ok=True)
    # The prior sequential downloader must be stopped before this is invoked.
    prefix = scratch / "prefix.part"
    sequential = target.with_name(target.name + ".part")
    if sequential.exists():
        if prefix.exists():
            raise ValueError("Two partial prefixes exist; inspect before resuming")
        sequential.replace(prefix)
    prefix_bytes = prefix.stat().st_size if prefix.exists() else 0
    if prefix_bytes >= asset["bytes"]:
        raise ValueError("Unexpected partial-prefix size")
    pinned = {
        "url": asset["url"],
        "bytes": asset["bytes"],
        "sha256": expected,
        "prefix_bytes": prefix_bytes,
    }
    identity = scratch / "identity.json"
    if identity.exists() and json.loads(identity.read_text()) != pinned:
        raise ValueError("Scratch download identity changed")
    identity.write_text(json.dumps(pinned, indent=2) + "\n")
    chunk_size = 16 * 1024 * 1024
    jobs = [
        (start, min(asset["bytes"], start + chunk_size) - 1)
        for start in range(prefix_bytes, asset["bytes"], chunk_size)
    ]

    def fetch(job):
        start, end = job
        part = scratch / f"{start:012d}-{end:012d}.part"
        size = end - start + 1
        for attempt in range(4):
            have = part.stat().st_size if part.exists() else 0
            if have == size:
                return part
            if have > size:
                raise ValueError("Oversized scratch range")
            request_start = start + have
            try:
                request = Request(
                    asset["url"],
                    headers={
                        "Range": f"bytes={request_start}-{end}",
                        "User-Agent": "PRAVAH-research-intake/1.0",
                    },
                )
                with urlopen(request, timeout=90) as response:
                    wanted = f"bytes {request_start}-{end}/{asset['bytes']}"
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != wanted
                    ):
                        raise ValueError("Provider did not honor exact byte range")
                    with part.open("ab") as stream:
                        while block := response.read(1024 * 1024):
                            have += len(block)
                            if have > size:
                                raise ValueError("Response exceeded pinned range")
                            stream.write(block)
                if have != size:
                    raise OSError("Truncated range")
                return part
            except OSError as exc:
                if attempt == 3:
                    # Signed redirect URLs must not appear in logs.
                    raise RuntimeError(
                        f"Range {start}-{end} failed: {type(exc).__name__}"
                    ) from None
                time.sleep(2**attempt)
        raise RuntimeError("Range did not finish")

    started = time.monotonic()
    print(
        "RANGE_DOWNLOAD_STARTED",
        json.dumps({"prefix_bytes": prefix_bytes, "ranges": len(jobs), "workers": 8}),
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        for done, future in enumerate(
            as_completed([executor.submit(fetch, j) for j in jobs]), 1
        ):
            future.result()
            print("RANGE_VERIFIED_SIZE", done, "/", len(jobs), flush=True)
    assembled = scratch / "assembled.part"
    with assembled.open("wb") as output:
        sources = ([prefix] if prefix.exists() else []) + [
            scratch / f"{a:012d}-{b:012d}.part" for a, b in jobs
        ]
        for path in sources:
            with path.open("rb") as source:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    output.write(block)
    actual = digest(assembled)
    if assembled.stat().st_size != asset["bytes"] or actual != expected:
        raise ValueError("Complete checkpoint checksum mismatch; scratch preserved")
    os.replace(assembled, target)
    receipt = {
        **pinned,
        "path": str(target),
        "status": "downloaded_integrity_verified",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
    }
    target.with_suffix(".download.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print("CHECKPOINT_VERIFIED", json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
