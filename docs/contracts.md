# Starter implementation contracts (not yet science validated)

- **Pairs JSON** for baseline training: array of `{image_path, label_path, split, event_id}` using authentic official Sen1Floods11 split files; absolute or repo-root-relative paths. Train and validation required, independent test recommended. Unreadable/missing data are errors.
- **S2 input**: 13-band Sen1Floods11 Sentinel-2 Level 1C TIFF (select 0-based band indices 1, 2, 3, 8, 11, 12), OR 6-band TIFF already in `[B2,B3,B4,B8A,B11,B12]` order. This is baseline small U-Net **only**, not Prithvi. Reflectances scaled `/10000` as a project baseline; quantify correctness on real data.
- **Label**: single-band binary water `0/1`, ignore `-1`. No prediction of debris-flow damage.
- **Tile manifest** from `scripts/make_tile_manifest.py`: contains source absolute raster path and tile geometry; identical for one and two worker runs.
- **Worker results**: per tile `{tile_id}.npz`, probability float32 and valid uint8 arrays; partitioned output prefixes must be distinct.
- **Stitch output**: `observed_water.tif` water inferred `1`, no water `0`, invalid `255`; `invalid_mask.tif`, `water_probability.tif`, `observed_metadata.json`.
- **Historical eligibility**: only acquisition **AND** verified availability precede cutoff. Missing availability excluded.
- **Benchmark**: one- vs two-node compute time requires same tile manifest hash, model hash, task count and CUDA verification; node IDs separate. This is **compute-only**, not wall-clock.
- **External GIS exposure**: metric projected CRS, OSM snapshot date and AOI coverage; intersecting building != destroyed building.

Every source must carry provider URI, product, acquisition/publishing timestamps, licensing and local checksum in production. Runbook defines final scientific metadata requirements beyond this starter's simplified output contracts.
