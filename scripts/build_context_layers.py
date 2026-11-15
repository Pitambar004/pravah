#!/usr/bin/env python3
"""Build map context layers (rivers, administrative boundaries) from the OSM extract.

Inputs come from scripts/extract_osm_bbox.py (or p3_extract_osm.sh). Outputs are
EPSG:4326 GeoJSON in ``data/processed/context/`` that the dashboard shows on every
run. Both are OSM snapshot features — context only, not hydrological truth or an
official administrative dataset.

    python scripts/build_context_layers.py --osm-dir data/processed/osm --snapshot 2026-01-01
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

BBOX = (85.10, 27.85, 85.70, 28.35)


def _write(frame: gpd.GeoDataFrame, path: Path, meta: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    gj = json.loads(frame.to_json(drop_id=True)) if len(frame) else {"type": "FeatureCollection", "features": []}
    gj["pravah_layer"] = meta
    path.write_text(json.dumps(gj) + "\n")
    return len(frame)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--osm-dir", type=Path, default=Path("data/processed/osm"))
    p.add_argument("--out", type=Path, default=Path("data/processed/context"))
    p.add_argument("--snapshot", required=True, help="OSM snapshot date, e.g. 2026-01-01")
    p.add_argument("--simplify-deg", type=float, default=0.0002, help="~20 m; display only")
    a = p.parse_args(argv)
    area = box(*BBOX)

    lines = gpd.read_file(a.osm_dir / "osm_lines.gpkg")
    rivers = lines[lines["waterway"].isin(["river"])][["osm_id", "name", "waterway", "geometry"]].copy()
    rivers["geometry"] = rivers.geometry.intersection(area).simplify(a.simplify_deg)
    rivers = rivers[~rivers.geometry.is_empty]
    rivers["label"] = rivers["name"].fillna("").where(rivers["name"].notna(), "River (unnamed)")
    n_r = _write(rivers, a.out / "rivers.geojson",
                 {"source": f"OpenStreetMap snapshot {a.snapshot} (Geofabrik)", "filter": "waterway=river",
                  "note": "Mapped river centrelines; not flood extent or hydrological model output."})

    polys = gpd.read_file(a.osm_dir / "osm_multipolygons.gpkg")
    # districts (6) and municipalities / rural municipalities (7); wards (9) would clutter the map
    admin = polys[(polys["boundary"] == "administrative") & polys["admin_level"].isin(["6", "7"])].copy()
    admin = admin[["osm_id", "osm_way_id", "name", "admin_level", "geometry"]]
    admin["geometry"] = admin.geometry.boundary.intersection(area).simplify(a.simplify_deg)
    admin = admin[~admin.geometry.is_empty]
    admin["label"] = admin["name"].fillna("Unnamed") + " (admin_level " + admin["admin_level"].astype(str) + ")"
    n_a = _write(admin, a.out / "admin_boundaries.geojson",
                 {"source": f"OpenStreetMap snapshot {a.snapshot} (Geofabrik)", "filter": "boundary=administrative, admin_level 6 (district) and 7 (municipality)",
                  "note": "OSM-mapped boundaries; not an official survey dataset. Clipped to the provisional bbox."})
    print(json.dumps({"rivers": n_r, "admin_boundaries": n_a, "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
