#!/usr/bin/env python3
"""Polygonize inferred-water class and compute GIS exposure; not damage."""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from pravah.gis.exposure import summarize_exposure


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mask", required=True, help="GeoTIFF with 0 no water, 1 inferred water, 255 invalid")
    p.add_argument("--buildings", help="GeoPackage/GeoJSON with mapped building footprints")
    p.add_argument("--roads", help="GeoPackage/GeoJSON with mapped roads")
    p.add_argument("--bridges", help="Optional separately deduplicated mapped bridge features")
    p.add_argument("--osm-date", default="unknown", help="Source snapshot date or unknown")
    p.add_argument("--out", required=True)
    p.add_argument("--polygons-out", required=True)
    args=p.parse_args()
    with rasterio.open(args.mask) as src:
        if src.count!=1 or src.crs is None:
            raise ValueError("Need one-band georeferenced model water mask")
        z=src.read(1)
        if not np.isin(z,(0,1,255)).all():
            raise ValueError("Unsupported classes; expected 0/1/255 only")
        valid_pixels=int(np.isin(z,(0,1)).sum())
        if valid_pixels==0:
            raise ValueError("No valid observed pixels; cannot report zero exposure")
        iswater=(z==1)
        geoms=[shape(geometry) for geometry,value in shapes(z,mask=iswater,transform=src.transform) if value==1]
        crs=src.crs
    water=gpd.GeoDataFrame(geometry=geoms,crs=crs)
    vectors=lambda path: gpd.read_file(path) if path else None
    summary=summarize_exposure(water,vectors(args.buildings),vectors(args.roads),vectors(args.bridges))
    summary.update({"source_type":"model_inferred_water", "mask_artifact":args.mask,
                    "osm_snapshot_timestamp":args.osm_date,
                    "valid_pixel_count":valid_pixels,
                    "inference_is_not_damage":True,
                    "unavailable_fields":[name for name in ("buildings","roads","bridges") if getattr(args,name) is None]})
    outfile=Path(args.out); outfile.parent.mkdir(parents=True,exist_ok=True)
    outfile.write_text(json.dumps(summary,indent=2)+"\n")
    polygons=Path(args.polygons_out); polygons.parent.mkdir(parents=True,exist_ok=True)
    # Geopandas supports writing empty GeoJSON in current releases.
    water.to_file(polygons,driver="GeoJSON")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
