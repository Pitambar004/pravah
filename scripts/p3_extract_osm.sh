#!/usr/bin/env bash
# Reconstructed from P3's successful WSL extraction commands; not executed in this repository.
set -euo pipefail
DATA_ROOT="${PRAVAH_DATA_ROOT:-/mnt/c/PRAVAH_DATA}"
SOURCE="${DATA_ROOT}/raw/osm/nepal-260101.osm.pbf"
OUTPUT="${DATA_ROOT}/processed/osm"
BBOX=(85.10 27.85 85.70 28.35)
command -v ogr2ogr >/dev/null || { echo "Missing ogr2ogr (GDAL)" >&2; exit 1; }
test -f "$SOURCE" || { echo "Missing OSM source: $SOURCE" >&2; exit 1; }
mkdir -p "$OUTPUT"
for layer in points lines multipolygons multilinestrings other_relations; do
  echo "Extracting $layer"
  ogr2ogr -f GPKG "$OUTPUT/osm_${layer}.gpkg" "$SOURCE" "$layer" \
    -spat "${BBOX[@]}" -nln "$layer" -overwrite -progress
done
echo "P3 OSM bbox selection complete: $OUTPUT"
echo "WARNING: -spat selects intersecting features; it does not clip them."
