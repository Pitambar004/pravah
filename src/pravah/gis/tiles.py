"""Deterministic tiled workload partitioning, independent of any scheduler."""
import hashlib
import json


def make_tiles(width: int, height: int, tile_size: int = 256, overlap: int = 32) -> list[dict]:
    if width <= 0 or height <= 0:
        raise ValueError("Raster dimensions must be positive")
    if tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError("Require tile_size > overlap >= 0")
    stride = tile_size - overlap
    tiles = []
    for row_off in range(0, height, stride):
        for col_off in range(0, width, stride):
            i = len(tiles)
            tiles.append({"tile_id": f"tile_{i:06d}", "index": i,
                          "row_off": row_off, "col_off": col_off,
                          "height": min(tile_size, height - row_off),
                          "width": min(tile_size, width - col_off)})
    return tiles


def partition(tiles: list[dict], rank: int, count: int) -> list[dict]:
    if count <= 0 or not 0 <= rank < count:
        raise ValueError("count must be positive and 0 <= rank < count")
    return [t for t in tiles if t["index"] % count == rank]


def fingerprint(obj: object) -> str:
    encoded = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
