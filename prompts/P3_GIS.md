# P3 — GIS, validation, real exposure owner (Claude/Codex)

Read `AGENTS.md`, `docs/IMPLEMENTATION_RUNBOOK.md` sections 3, 5, 8, 10, 13, 18–21 and `docs/contracts.md`. Work on `feat/p3-gis`; own `src/pravah/gis` exposure (coordinate tiles with P4), `scripts/exposure.py`, `scripts/gis_validate.py`, GIS source metadata.

1. Derive verified core AOI and corridor AOI from UNOSAT/river sources, no guessed footprint.
2. Obtain OSM buildings/roads/bridge features and disclose date; historical OSM pre-Aug26 if replay-sensitive, current OSM retrospective only. Check EPSG:32645 appropriate for exact AOI before metric overlays.
3. Validate raster/vector alignment, invalid pixels, nonoverlapping coverage and projected area.
4. Convert model raster into connected valid water polygons; count unique building footprints and intersected line *length*, not damage.
5. Compare with UNOSAT overlap carefully: their debris/mudflow class differs from binary water. Report area/coverage/intersection, avoid unsupported IoU.
6. Unit-test edge cases CRS mismatches, empty inputs, dedup and invalid/no-observation.

Acceptance: timestamped source layers + geospatial output and defensible exposure JSON with no fabricated values.
