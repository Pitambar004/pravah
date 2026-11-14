# P4 — CIQ Fuzzball / two-node HPC owner (Claude/Codex)

Read `AGENTS.md`, `docs/IMPLEMENTATION_RUNBOOK.md` sections 4, 7, 11–12, 14–16, 18–21 and `docs/contracts.md`. Work on `feat/p4-hpc`; own `workflows/`, `infra/`, `scripts/make_tile_manifest.py`, `scripts/infer_tiles.py`, `scripts/stitch.py`, `scripts/benchmark.py`.

Tonight before 9 AM Sunday: develop deterministic partitioned inference locally on a **real** 6/13-band GeoTIFF with trained small-U-Net model; run 1-part vs 2-part on local CPU and verify identical mask. Keep CPU timing labeled CPU. Inspect docs for Fuzzball v4.3.1 but do not invent YAML or endpoint, or claim CUDA access. Assemble ARM64/CUDA checklist and mentor request.

After 9 AM: confirm Team 12 allocation, GPU names, storage, endpoints, actual official sample workflow; launch smoke job, 1-node inference, 2-node partitioned inference with actual log times, matched tiles/model, distinct node IDs. Then assist P2 with 2-node DDP if approved and reachable. Store run manifest + timing/queue/startup/I/O separately.

Acceptance: evidenced 2-worker GPU execution in Team 12, output checks and honest paired benchmark; blocked conditions documented and never faked.
