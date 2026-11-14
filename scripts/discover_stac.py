#!/usr/bin/env python3
"""Inventory and search STAC collections for a SOURCE-VERIFIED polygon.

Requires `pip install -e '.[catalog]'`, network access, actual AOI and real
collection ID; never prints or invents a matching satellite scene.
"""
import argparse
import json
from pathlib import Path
from shapely.geometry import mapping, shape
from shapely.ops import unary_union


def read_geom(path: str):
    doc=json.loads(Path(path).read_text())
    if doc["type"]=="FeatureCollection":
        geoms=[shape(item["geometry"]) for item in doc["features"]]
        if not geoms:
            raise ValueError("AOI has no polygon")
        geom=unary_union(geoms)
    elif doc["type"]=="Feature":
        geom=shape(doc["geometry"])
    else:
        geom=shape(doc)
    if geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("AOI must be a valid nonempty polygon")
    return mapping(geom)


def main():
    from pystac_client import Client
    p=argparse.ArgumentParser()
    p.add_argument("--endpoint",default="https://stac.dataspace.copernicus.eu/v1/")
    p.add_argument("--collections",action="store_true",help="List official collection IDs first")
    p.add_argument("--collection",help="Verified collection ID from --collections")
    p.add_argument("--aoi",help="Actual EPSG:4326 polygon GeoJSON")
    p.add_argument("--datetime",default="2026-08-10/2026-09-05")
    p.add_argument("--out",help="Store returned full STAC items as GeoJSON")
    p.add_argument("--limit",type=int,default=100)
    args=p.parse_args()
    client=Client.open(args.endpoint)
    if args.collections:
        print(json.dumps([{"id":c.id,"title":c.title} for c in client.get_collections()],indent=2))
        return
    if not args.collection or not args.aoi or not args.out:
        p.error("Search requires --collection, --aoi and --out")
    if args.limit<1 or args.limit>500:
        raise ValueError("Limit out of range")
    geo=read_geom(args.aoi)
    items=list(client.search(collections=[args.collection],intersects=geo,
                             datetime=args.datetime,max_items=args.limit).items())
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"type":"FeatureCollection","features":[i.to_dict() for i in items]},indent=2)+"\n")
    print(json.dumps({"count":len(items),"ids":[i.id for i in items],"status":"metadata_only_not_downloaded"},indent=2))

if __name__=="__main__":
    main()
