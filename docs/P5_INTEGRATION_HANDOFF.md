# Pravah — Team 12 Unified Design System & Integration Handoff

**Author:** Person 5 (Principal Product Designer & Lead Integration Engineer)  
**Date:** Saturday, 14 November 2026  
**Status:** Unified Design System Implemented & 100% Tested  
**Slogan:** "See upstream. Protect downstream."  
**Case Study:** 26 August 2026 Rasuwa / Bhote Koshi–Trishuli Flood & Debris Cascade, Nepal  

---

## 1. Quick Start

Ensure the virtual environment is active, then launch the Streamlit dashboard:

```bash
source .venv/bin/activate
python -m streamlit run app/Home.py
```

To run the full automated test suite:

```bash
pytest -v
```

Last recorded run (14 Nov 2026, Python 3.14, Streamlit 1.67, no torch): `pytest -q` → 47 passed, 1 skipped (U-Net test, torch not installed); `ruff check app/ tests/ src/` → clean.

---

## 2. Design System

All tokens live in `app/theme.py` (`TOKENS`) and are emitted as `--pv-*` CSS custom properties; `.streamlit/config.toml` mirrors the base colours and sets Inter / JetBrains Mono. Pages must not hardcode colours or spacing.

- **Surfaces:** app `#090E18`, sidebar `#0D1420`, surface `#111B2A`, raised `#172335`, hover `#1D2B3E`, borders `#263448` / `#1C293A`.
- **Text:** primary `#F3F6FA`, secondary `#A4B2C5`, muted `#718198`, disabled `#536176`.
- **Semantics:** cyan `#38BDF8` = primary/active/model-derived water; green `#34D399` = available/verified; amber `#FBBF24` = warnings, potential exposure, retrospective reference; red `#FB7185` = errors/excluded; purple `#A78BFA` = hypothetical scenarios. Status is never colour-only (labels, marker shapes, icons).
- **Type:** page title 28/36, section 16/24, card 14/22, body 14/22, caption 12/18; tabular numerals for metrics.
- **Spacing / radius:** 8px scale (4–48); controls 8px, cards 12px, large containers 16px.
- **Icons:** Material Symbols Rounded only (native to Streamlit navigation; loaded in the map iframe too).
- **Breakpoints:** ≤640px mobile (drawer nav, stacked cards), 641–1180px tablet (2-up metrics, map/side panels stack), desktop beyond.

## 3. Architecture

```text
app/
├── Home.py            # shell: page config, theme, sidebar (brand, nav, data source), st.navigation routing
├── context.py         # AppContext (artifacts + replay catalog + provenance), shared topbar/page header
├── theme.py           # tokens + global stylesheet
├── data_loader.py     # exception-safe artifact reader; load_replay_catalog_with_source()
├── views/             # one thin route file per page (URL paths: /, /map, /replay, /exposure, /compute, /methodology)
├── components/
│   ├── layout.py      # html(), icon(), badge, topbar, page/section headers, callout, card, kv_list, status_list, footer
│   ├── metric_card.py # MetricCard — None renders "Not available"/"Awaiting processing", never 0
│   ├── empty_state.py # standard empty/awaiting state
│   ├── map_view.py    # Leaflet map workspace (layers panel, opacity, basemaps, feature detail panel, legend) — client-side, no reruns
│   └── timeline.py    # evidence timeline SVG + evidence cards (strict vs retrospective)
├── sections/          # page renderers, each takes AppContext
└── fixtures/          # synthetic fixtures (only when the sidebar toggle is on; always labelled)
```

## 4. Integrity rules enforced in the UI

1. Synthetic fixtures load only when "Show synthetic fixtures" is on; every page then shows a fixture banner and the map shows a fixture chip. With fixtures off, the replay catalog no longer falls back to synthetic scenes.
2. Processing status is derived from files actually present; nothing is pre-marked "audited" or "enforced".
3. Strict replay never renders assets acquired or made available after the cutoff; unknown availability is excluded (`pravah.replay.eligibility`).
4. Map layers without an artifact (admin boundaries, rivers, imagery preview, infrastructure features, scenarios) are listed but disabled with the reason.
5. Compute page shows planned hardware as *planned*; speedup/efficiency only appear from `benchmark.json`.
6. Exposure is labelled "potential exposure"; population is "modelled estimate, not a headcount".
7. The sources table does not assert licence terms — each must be verified at source.

## 5. Teammate Handoff & Expected Artifacts

The frontend consumes outputs structured under `outputs/<run_id>/` (defaulting to `outputs/latest/`):

| Contributor | Owned Module | Expected Contract Output | Dashboard Consumer |
|---|---|---|---|
| **Person 1 (Data)** | `src/pravah/catalog/` | `data/manifests/assets.json`, `config/aois/analysis_coverage.geojson` | Historical Replay & Flood Map |
| **Person 2 (AI/ML)** | `src/pravah/ml/` | `outputs/<run_id>/observed_metadata.json`, model manifests | Overview & Sources |
| **Person 3 (GIS)** | `src/pravah/gis/` | `outputs/<run_id>/observed_water.tif`, `exposure.geojson`, `exposure_summary.json` | Flood Map & Exposure |
| **Person 4 (HPC/GB10)** | `infra/`, `workflows/` | `outputs/<run_id>/benchmark.json`, `benchmarks/benchmarks.csv` | GPU Performance & Overview |

---

## 6. Scientific Integrity & Local Development Constraints

> [!IMPORTANT]
> **Zero Git / Remote Operations**: All work was completed strictly inside the local repository. No git branches, commits, pulls, pushes, or remotes were modified, complying with the local development rule in effect until Sunday, November 15, 2026, at 9:00 AM CT.
