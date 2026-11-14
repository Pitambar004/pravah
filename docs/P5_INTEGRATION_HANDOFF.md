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

All 39 tests across data contracts, temporal replay gating, geospatial summaries, ML baseline, UI loaders, and Streamlit AppTest pass cleanly in ~2.5 seconds.

---

## 2. Visual Design System Specifications

The application uses an enterprise-grade SaaS design language inspired by Linear, Vercel, and Stripe Dashboard, engineered specifically for scientific geospatial intelligence.

### Color Tokens
- **Application Background:** `#0B1120`
- **Sidebar & Elevated Surfaces:** `#111827`
- **Card Backgrounds:** `#151F30`
- **Borders:** `#263244` (Subtle 1px solid)
- **Primary Text:** `#F1F5F9` (High-contrast, clean legibility)
- **Secondary Text:** `#94A3B8` (Descriptions and body)
- **Muted Text:** `#64748B` (Labels, metadata, and timestamps)
- **Primary Accent:** `#38BDF8` (Sky blue for water and primary active states)
- **Success:** `#34D399` (Pre-cutoff eligible, verified runs, CUDA active)
- **Warning:** `#FBBF24` (Retrospective reveals, demo fixture alerts, potential overlap)
- **Error:** `#FB7185` (Excluded unknown availability, bridge overlaps)

### Typography Scale
- **Page Heading:** 28px, font-weight 600, letter-spacing -0.02em (`.pv-page-title`)
- **Section Heading:** 18px, font-weight 600, letter-spacing -0.01em
- **Card Title / Metric Label:** 12px, font-weight 500, uppercase, letter-spacing 0.05em (`.pv-metric-label`)
- **Metric Value:** 26px, font-weight 700 (`.pv-metric-value`)
- **Body & Tables:** 13–14px, font-weight 400, line-height 1.5
- **Monospace:** `JetBrains Mono` / system monospace for hashes, timestamps, and commands

---

## 3. Modular Architecture & Design System Primitives

```text
app/
├── Home.py                      # Persistent sidebar, brand header, run selector & router
├── theme.py                     # Design tokens & global CSS stylesheet
├── data_loader.py               # Robust, exception-safe artifact reader for outputs/<run_id>/
├── components/
│   ├── layout.py                # Standardized page headers, section headers, badges & callouts
│   ├── metric_card.py           # Reusable KPI metric card primitive (.pv-metric-card)
│   ├── empty_state.py           # Standardized awaiting / empty state card (.pv-empty-state)
│   └── map_view.py              # Geospatial Folium map with dark-mode legend & controls
├── sections/
│   ├── overview.py              # Section 1: Mission Overview, 5 KPI cards, corridor map preview
│   ├── flood_map.py             # Section 2: Full-canvas Flood Evidence Map with layer controls
│   ├── replay.py                # Section 3: Historical Replay with ISO 8601 pre-cutoff temporal gating
│   ├── exposure.py              # Section 4: Infrastructure Exposure & OSM feature intersection
│   ├── gpu_benchmarks.py        # Section 5: GPU scaling, 1-node vs 2-node comparison & Fuzzball specs
│   └── methodology.py           # Section 6: AI models, satellite sources, licensing & provenance export
└── fixtures/
    └── synthetic_sample.py      # Clearly labeled [SYNTHETIC TEST FIXTURE] for offline evaluation
```

---

## 4. Section Design Highlights

1. **Overview:** Standardized 5-card metric row (`Inferred Water`, `Buildings Exposed`, `Roads Exposed`, `Bridge Overlaps`, `Multi-Node Scaling`), 3:2 split with corridor map canvas and verified pipeline checklist.
2. **Flood Evidence Map:** Full-width map canvas (`#263244` border), compact top toolbar, unified Folium palette, on-map legend (`#151F30` background), and attribute table.
3. **Historical Replay:** Cutoff toolbar with disaster milestone presets, 3 summary cards (`Eligible`, `Retrospective`, `Excluded`), and clean tabbed tables enforcing strict temporal eligibility (`acquired_at_utc <= cutoff AND available_at_utc <= cutoff`).
4. **Infrastructure Exposure:** Same standardized metric card primitive as Overview, prominent `exposure != damage` callout, and layer audit table.
5. **GPU Performance:** Speedup and parallel efficiency cards, clean execution time chart (`st.bar_chart`), cryptographic workload equivalence verification table, and an honest awaiting state for Sunday's GB10 cluster window.
6. **Sources & Methodology:** Tabbed documentation covering Prithvi EO 2.0 300M, Team 12 U-Net, Copernicus/UNOSAT/OSM/WorldPop licensing, sensor limitations, and single-click JSON provenance export.

---

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
