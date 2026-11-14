# P5 — Team lead, integration, CI, dashboard and final demo owner (Claude/Codex)

Read `AGENTS.md`, `docs/IMPLEMENTATION_RUNBOOK.md` sections 2, 4, 10, 13, 17–22 and `docs/contracts.md`. Work on `feat/p5-ui`; own `app/`, `src/pravah/replay`, `src/pravah/reporting`, `.github/`, documentation, final presenter artifacts.

1. Create team-owned GitHub private repo, invite all five separate GitHub user accounts as collaborators, enable branch protection and checks if GitHub plan permits; do not share Claude/Codex credentials.
2. Give each person a GitHub issue and their feature branch; establish PR review rotation and source manifest/contract agreement before parallel changes.
3. Build Streamlit dashboard **against explicit empty/missing data state**, no sample fixture passed as live disaster output.
4. Add historical replay eligibility and later reveal separation, evidence dates, disclaimers, run/download statuses, and external-source links.
5. Integration-test emitted GeoTIFF, exposure JSON, logs and runtime against source contracts; merge approved PRs in small batches.
6. Prepare five-minute demo narrative, link evidence for every observed metric, show real cluster logs/benchmark if available, no fabricated GPU figures.

Acceptance: one-command demo consuming actual artifacts, green CI, peer-reviewed main and honest science/hardware claims.
