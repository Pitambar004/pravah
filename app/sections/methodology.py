"""Sources & Methodology — documentation, provenance and limitations."""
from __future__ import annotations

import json

import streamlit as st

from app.components.layout import card, html, kv_list, render_badge
from app.context import AppContext, render_page_frame
from pravah.configuration import configured_url

SOURCES = [
    {"Source": "Sentinel-2 MSI Level-1C", "Role": "Optical model input (6 bands)", "Evidence type": "Observed",
     "Provider": "Copernicus Data Space Ecosystem",
     "Link": configured_url("PRAVAH_CDSE_DOCS_URL")},
    {"Source": "Sen1Floods11 hand-labelled subset", "Role": "Baseline training / validation labels", "Evidence type": "Reference labels",
     "Provider": "Cloud to Street et al.", "Link": configured_url("PRAVAH_SEN1FLOODS_URL")},
    {"Source": "Prithvi-EO-2.0-300M-TL-Sen1Floods11", "Role": "Pretrained water segmentation", "Evidence type": "Model",
     "Provider": "IBM / NASA", "Link": configured_url("PRAVAH_PRITHVI_MODEL_URL")},
    {"Source": "OpenStreetMap extract", "Role": "Buildings, roads, bridges for exposure", "Evidence type": "Context",
     "Provider": "OpenStreetMap contributors", "Link": configured_url("PRAVAH_OSM_ATTRIBUTION_URL")},
    {"Source": "UNOSAT mudflow / rockflow extent", "Role": "Retrospective comparison only", "Evidence type": "Retrospective reference",
     "Provider": "UNITAR-UNOSAT / UNESCO IHP-WINS",
     "Link": configured_url("PRAVAH_UNOSAT_DATA_URL")},
    {"Source": "WorldPop population grid", "Role": "Optional modelled resident population", "Evidence type": "Context (modelled)",
     "Provider": "WorldPop", "Link": configured_url("PRAVAH_WORLDPOP_URL")},
]


def render_methodology_section(ctx: AppContext) -> None:
    render_page_frame(
        ctx,
        "Sources & Methodology",
        "Sources & Methodology",
        "How PRAVAH produces its evidence, which sources it relies on, and where its results stop being reliable.",
        location=False,
    )
    t_method, t_sources, t_models, t_replay, t_limits, t_prov = st.tabs(
        ["Methodology", "Data sources", "Models", "Replay rules", "Limitations", "Provenance"]
    )

    with t_method:
        html(
            '<div class="pv-doc">'
            "<h3>Pipeline</h3><ol>"
            "<li><strong>Source audit.</strong> Candidate scenes are discovered via STAC; each records provider, product, "
            "acquisition time, verified availability time, licence and checksum.</li>"
            "<li><strong>Pre-processing.</strong> Sentinel-2 L1C reflectances are reordered to "
            "<code>[B2, B3, B4, B8A, B11, B12]</code> and scaled by 1/10,000. Sentinel-1 SAR is never fed to optical models.</li>"
            "<li><strong>Tiled inference.</strong> A deterministic tile manifest is processed by the configured runtime; each tile yields "
            "water probability and a validity mask.</li>"
            "<li><strong>Stitching.</strong> Tiles are merged into <code>observed_water.tif</code>: 1 = model-inferred water, "
            "0 = no water, 255 = invalid source. Reference cloud labels are used for scoring only.</li>"
            "<li><strong>Exposure.</strong> Water polygons are intersected with OSM infrastructure in a recorded local metric CRS so that "
            "areas and lengths are metric.</li>"
            "<li><strong>Replay.</strong> Every source is gated by acquisition and verified availability time before it can "
            "appear in a historical view.</li></ol>"
            "<h3>Evidence vocabulary</h3>"
            f"<p>{render_badge('Observed', 'accent')} satellite acquisitions · "
            f"{render_badge('Model-derived', 'accent')} segmentation output · "
            f"{render_badge('Derived exposure', 'warning')} geometric intersections · "
            f"{render_badge('Retrospective reference', 'warning')} later reference products.</p>"
            "</div>"
        )

    with t_sources:
        st.dataframe(
            SOURCES,
            hide_index=True,
            width="stretch",
            column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open source")},
        )
        st.caption(
            "Licences and redistribution terms must be verified at each source before judge-facing use; this table "
            "does not assert licence terms. Download status: "
            + ("synthetic fixtures only." if ctx.is_synthetic else "see data/manifests/ for audited downloads.")
        )

    with t_models:
        c1, c2 = st.columns(2)
        with c1:
            html(card("Prithvi-EO-2.0 300M (Sen1Floods11 fine-tune)", kv_list([
                ("Role", "Primary pretrained water segmentation"),
                ("Input", "Sentinel-2 L1C, 6 bands"),
                ("Band order", "B2, B3, B4, B8A, B11, B12"),
                ("Status", "Separate optional model; not used by this U-Net service"),
                ("Model ID", "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"),
            ], mono_keys=("Model ID",)), trailing=render_badge("Pretrained", "accent")))
        with c2:
            html(card("Team 12 baseline U-Net", kv_list([
                ("Role", "Transparent independent baseline"),
                ("Input", "Same 6 optical bands"),
                ("Loss", "Weighted cross entropy with ignore label −1"),
                ("Status", "Completed training record loaded" if (ctx.artifacts.observed_metadata or {}).get("training_evaluation") else "See the selected run's checkpoint provenance"),
                ("Code", "src/pravah/ml/unet.py"),
            ], mono_keys=("Code",)), trailing=render_badge("Baseline", "neutral")))
        st.caption("Held-out metrics from the completed run appear on Compute & Performance. A single example score is not aggregate accuracy.")

    with t_replay:
        html(
            '<div class="pv-doc">'
            "<h3>Eligibility rule</h3>"
            "<p>A source is eligible at cutoff <em>C</em> only when <strong>acquired ≤ C</strong> and "
            "<strong>verified available ≤ C</strong>. Both timestamps must be timezone-aware.</p>"
            "<h3>Unknown availability</h3>"
            "<p>If the publication/availability time is missing or unverified the source is <strong>excluded</strong>. "
            "It never silently passes the strict filter.</p>"
            "<h3>Retrospective reveal</h3>"
            "<p>Later evidence (e.g. a post-event Sentinel-2 pass or the UNOSAT product published 28 Aug 2026) is shown only "
            "in retrospective mode, labelled as unavailable at the cutoff, and is never used as model input for that cutoff.</p>"
            "<p>Implementation: <code>src/pravah/replay/eligibility.py</code>.</p></div>"
        )

    with t_limits:
        html(
            '<div class="pv-doc"><ol>'
            "<li><strong>Retrospective, not a forecast.</strong> PRAVAH maps evidence after acquisition; it does not predict "
            "flood arrival or provide early warning.</li>"
            "<li><strong>Cloud occlusion.</strong> Source nodata is invalid (255). Reference cloud masks restrict scoring; "
            "predictions under unmasked cloud are unvalidated.</li>"
            "<li><strong>Terrain effects.</strong> Steep valleys cause shadow in optical data and layover in SAR.</li>"
            "<li><strong>Debris vs water.</strong> Sediment-laden flows differ spectrally from clear water; UNOSAT mudflow "
            "extent is a different class from model water.</li>"
            "<li><strong>Exposure ≠ damage.</strong> Intersections indicate potential exposure only.</li>"
            "<li><strong>Population is modelled.</strong> Gridded estimates are not headcounts of affected people.</li>"
            "</ol></div>"
        )

    with t_prov:
        a = ctx.artifacts
        bundle = {
            "application": "PRAVAH",
            "case_study": f"{ctx.study['region']} ({ctx.study['event_date']})",
            "run_id": a.run_id,
            "is_synthetic": a.is_synthetic,
            "replay_catalog_source": ctx.catalog_source,
            "observed_metadata": a.observed_metadata,
            "exposure_summary": a.exposure_summary,
            "benchmark": a.benchmark,
            "missing_files": a.missing_files,
        }
        html(card("Run provenance", kv_list([
            ("Run", a.run_id),
            ("Data mode", ctx.data_status[0]),
            ("Replay catalog", ctx.catalog_source or "None"),
            ("Missing files", ", ".join(a.missing_files) if a.missing_files else "None"),
        ], mono_keys=("Run",))))
        st.download_button(
            "Download provenance JSON",
            data=json.dumps(bundle, indent=2, default=str),
            file_name=f"pravah_provenance_{a.run_id}.json",
            mime="application/json",
            icon=":material/download:",
        )
