# P2 — Scientific ML training/inference owner (Claude/Codex)

Read `AGENTS.md`, `docs/IMPLEMENTATION_RUNBOOK.md` sections 6, 7, 9, 12, 15–16, 18–21 and `docs/contracts.md`. Work on `feat/p2-ml`; own `src/pravah/ml`, `scripts/train_ddp.py`, pretrained Prithvi adapter and evaluation scripts. Coordinate manifest contract with P1, tile output contract with P4.

1. Smoke-test small 6-band U-Net forward/backward and first non-synthetic training batch.
2. Run at least a CPU mini epoch over authentic official split pairs; log train/validation IoU, F1 and source IDs; keep test split untouched until final assessment.
3. Research/download exact IBM/NASA Prithvi 300M Sen1Floods11 fine-tune revision; run official TerraTorch inference using model-card preprocessing, not the small U-Net normalization as a substitute. If too complex, label as blocked, continue baseline.
4. Maintain DDP launcher and checksum-pinned checkpoints; prepare 1/2 node training commands, not claims of success.
5. Test first on single GB10 when accessible; validate cross-node training using authorized Torchrun rendezvous + NCCL only if CIQ networking supports it.
6. Verify label semantics, weights/source provenance and metrics. Never report upstream model-card accuracy as measured Nepal accuracy.

Acceptance: real model checkpoint, honest inference and metric provenance, DDP readiness and independently verified device logs.
