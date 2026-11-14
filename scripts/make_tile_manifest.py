#!/usr/bin/env python3
"""Write a deterministic tile manifest for a verified local GeoTIFF."""
import argparse
import json
from pathlib import Path
import rasterio
from pravah.gis.tiles import make_tiles, fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    with rasterio.open(args.raster) as ds:
        if ds.crs is None:
            raise ValueError("Cannot tile raster without a CRS")
        tiles = make_tiles(ds.width, ds.height, args.tile_size, args.overlap)
        doc = {"run_id": args.run_id, "raster": str(Path(args.raster).resolve()),
               "crs": ds.crs.to_string(), "transform": list(ds.transform)[:6],
               "raster_width": ds.width, "raster_height": ds.height,
               "raster_bands": ds.count, "tile_size": args.tile_size,
               "overlap": args.overlap, "tiles": tiles}
    doc["tile_manifest_sha256"] = fingerprint(doc)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"tile_count": len(tiles), "manifest_sha256": doc["tile_manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
