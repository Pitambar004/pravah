# PRAVAH Team 12 agent collaboration and scientific integrity rules

Read `docs/IMPLEMENTATION_RUNBOOK.md` and `README.md` before changes.
This is a scientific **retrospective flood/water evidence** prototype, not a validated disaster forecast.
Never use post-cutoff imagery, UNOSAT maps or reports as inputs allegedly known before an earlier cutoff.
A source is cutoff-eligible only when both acquisition and verified availability times are <= cutoff.
Keep optical Sentinel-2 and SAR Sentinel-1 models/bands/units distinct. Document CRS, nodata and class semantics.
Never invent data downloads, accuracy, GPU utilization, hardware IDs, vendor storage, timings, licenses, results or scene IDs.
Work on one role/issue and your feature branch; do not edit another person's owned modules without review.
Run tests, record exact commands/output, report implemented/tested/blocked, and request peer review before merge.
Keep secrets, signed URLs, large raw archives, checkpoints, snapshots and credentials out of Git.
Treat all unit-test fixtures as synthetic and never export them as disaster observations.
GPU/Fuzzball templates must be validated on Team 12 v4.3.1; do not claim the sample scripts prove deployment.
