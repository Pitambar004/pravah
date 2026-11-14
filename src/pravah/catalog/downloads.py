"""Safe, bounded single-file downloader with streamed SHA256 and failed status."""
import hashlib
import os
import time
from pathlib import Path

import httpx


def download_file(url: str, destination: str, max_bytes: int = 2_000_000_000,
                  retries: int = 2, timeout_s: float = 30.0) -> dict:
    """Return status, bytes and checksum. Does not imply provider licence approval.

    Caller must record canonical source, acquisition/availability time and license.
    Avoid committing signed URLs in public manifests.
    """
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    last_error = ""
    for attempt in range(retries + 1):
        sha = hashlib.sha256()
        size = 0
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=timeout_s) as response:
                response.raise_for_status()
                expected = response.headers.get("content-length")
                if expected is not None and int(expected) > max_bytes:
                    raise ValueError("Declared download exceeds max_bytes")
                with open(temporary, "wb") as out:
                    for block in response.iter_bytes(chunk_size=1024 * 1024):
                        size += len(block)
                        if size > max_bytes:
                            raise ValueError("Download exceeds max_bytes")
                        sha.update(block)
                        out.write(block)
                if expected is not None and int(expected) != size:
                    raise OSError("Incomplete download (content-length mismatch)")
            os.replace(temporary, target)
            return {"status": "downloaded_unaudited", "path": str(target),
                    "bytes": size, "sha256": sha.hexdigest(), "raster_open_status": "not_tested"}
        except (httpx.HTTPError, ValueError, OSError) as exc:
            last_error = type(exc).__name__ + ": " + str(exc)
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 4))
    return {"status": "failed", "path": str(target), "error": last_error,
            "bytes": None, "sha256": None}
