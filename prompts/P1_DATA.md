# P1 — Data acquisition and provenance owner (Claude/Codex)

Before coding read `AGENTS.md`, `docs/IMPLEMENTATION_RUNBOOK.md` sections 3, 5, 7, 8, 13, 18–21 and `docs/contracts.md`. Work only on `feat/p1-data-catalog` and own `src/pravah/catalog`, `scripts/discover_*`, `scripts/download_*`, `data/manifests/` (no raw binaries committed).

1. Verify Nepal Rasuwa AOI based on UNOSAT coverage and official geography. Record authoritative source and EPSG; do not invent coordinates.
2. Fetch only the smallest required verified Sentinel-2 L1C Sen1Floods11 sample chips and human-labels with official splits. Verify source bucket and filenames; do not mix training/test events. Save SHA256 and file-open QA.
3. Enumerate Sentinel-1 and Sentinel-2 acquisitions for Aug 2026 case with acquisition and actual availability timestamps (unknown means excluded from strict replay).
4. Verify downloadable UNOSAT affected/analysis/frame polygons; don't use them as historical pre-event input or equate mudflow with water.
5. Produce inspected pairs manifest `data/manifests/s2_pairs.json` with real paths and split IDs.
6. Test `scripts/check_s2_pair.py` on representative real imagery. Document download license, integrity, missing scenes and exact commands.

Acceptance: real auditable dataset asset + manifests; independently reproducible acquisition; failing sources marked failed, not substituted. No synthetic output portrayed as evidence.
