"""Sources & Methodology — Provenance, Model Cards, Attribution, and Scientific Limitations."""
import json

import streamlit as st

from app.components.layout import (
    render_page_header,
    render_section_header,
)
from app.data_loader import RunArtifacts


def render_methodology_section(artifacts: RunArtifacts):
    """Render the standardized Sources & Methodology documentation interface."""
    render_page_header(
        title="Sources & Scientific Methodology",
        subtitle="Audited catalog manifests, model card architectures, licensing attribution, and inherent sensor uncertainties.",
        badge="Scientific Audit",
        badge_variant="neutral",
    )

    tab_models, tab_sources, tab_limits, tab_export = st.tabs([
        "AI Model Cards",
        "Data Sources & Licensing",
        "Sensor Uncertainties & Limits",
        "Provenance Export",
    ])

    with tab_models:
        render_section_header("Model Architecture Specifications", "Scientific models evaluated for optical water segmentation.")

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown(
                """
                <div class="pv-card">
                    <div class="pv-card-header">
                        <span class="pv-card-title">Prithvi EO 2.0 300M (Fine-Tuned)</span>
                        <span class="pv-badge pv-badge-accent">Foundation Model</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 0.75rem; line-height: 1.45;">
                        Geospatial foundation model developed by IBM Research and NASA, fine-tuned on Sen1Floods11.
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 0.35rem; font-size: 0.775rem;">
                        <div><strong>Architecture:</strong> Temporal Vision Transformer (ViT-B) with U-Net Decoder</div>
                        <div><strong>Parameter Count:</strong> ~300M parameters</div>
                        <div><strong>Input Optical Channels (6):</strong> <code>[B2, B3, B4, B8A, B11, B12]</code></div>
                        <div><strong>Normalization:</strong> Level 1C TOA reflectances / 10,000</div>
                        <div><strong>License:</strong> Apache 2.0</div>
                        <div><strong>HuggingFace ID:</strong> <code>ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11</code></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_m2:
            st.markdown(
                """
                <div class="pv-card">
                    <div class="pv-card-header">
                        <span class="pv-card-title">Team 12 Sentinel-2 Baseline U-Net</span>
                        <span class="pv-badge pv-badge-success">Trained Baseline</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 0.75rem; line-height: 1.45;">
                        Transparent 6-channel convolutional network trained independently on the official hand-labeled Sen1Floods11 subset.
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 0.35rem; font-size: 0.775rem;">
                        <div><strong>Architecture:</strong> Classical U-Net with skip connections</div>
                        <div><strong>Parameter Count:</strong> ~1.2M parameters</div>
                        <div><strong>Input Channels:</strong> 6 optical channels (same order as Prithvi)</div>
                        <div><strong>Loss Function:</strong> Binary Cross-Entropy with <code>ignore_index=-1</code></div>
                        <div><strong>Training Pipeline:</strong> CPU baseline smoke test & PyTorch 2-GPU DDP script</div>
                        <div><strong>Verification:</strong> Unit-tested forward/backward passes passing</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_sources:
        render_section_header("Audited Source Inventory & Licensing", "Authoritative provenance for datasets.")

        sources_data = [
            {
                "Data Source": "Sentinel-2 MSI Level 1C",
                "Provider": "Copernicus Data Space Ecosystem (CDSE)",
                "Spatial Resolution": "10m / 20m optical multispectral",
                "License": "CC-BY-4.0 Copernicus Sentinel Data",
                "Application": "Primary optical inference input for water segmentation",
            },
            {
                "Data Source": "Sen1Floods11 Hand-Labeled v1.1",
                "Provider": "Cloud to Street / NASA / Radiant Earth",
                "Spatial Resolution": "446 512×512 hand-verified chips",
                "License": "CC-BY-4.0",
                "Application": "Training & validation ground truth for surface water",
            },
            {
                "Data Source": "OpenStreetMap Historical Extract",
                "Provider": "Geofabrik (2026-01-01 Snapshot)",
                "Spatial Resolution": "Vector buildings, roads, bridges",
                "License": "Open Database License (ODbL 1.0)",
                "Application": "Pre-event infrastructure exposure intersections",
            },
            {
                "Data Source": "UNOSAT Satellite-Detected Mud/Rockflow",
                "Provider": "UNESCO IHP-WINS / UNITAR-UNOSAT",
                "Spatial Resolution": "Published 2026-08-28 (DOI 10.63253/1y26ihdq)",
                "License": "CC-BY-NC-SA 3.0 IGO",
                "Application": "Retrospective post-event qualitative reference comparator",
            },
            {
                "Data Source": "WorldPop Gridded Population Density",
                "Provider": "WorldPop / University of Southampton",
                "Spatial Resolution": "100m gridded resident counts",
                "License": "CC-BY-4.0",
                "Application": "Contextual modelled resident population in study AOI",
            },
        ]
        st.dataframe(sources_data, use_container_width=True, hide_index=True)

    with tab_limits:
        render_section_header("Scientific Caveats & Methodological Uncertainties", "Transparent reporting of limitations.")

        st.markdown(
            """
            <div class="pv-card">
                <ol style="margin: 0; padding-left: 1.25rem; font-size: 0.8rem; color: #94A3B8; line-height: 1.6;">
                    <li style="margin-bottom: 0.5rem;">
                        <strong style="color: #F1F5F9;">Retrospective Evidence vs Operational Forecasting:</strong>
                        PRAVAH produces satellite-derived water evidence after sensor acquisitions. It is not an advance forecasting or early warning system.
                    </li>
                    <li style="margin-bottom: 0.5rem;">
                        <strong style="color: #F1F5F9;">Monsoon Cloud Occlusion:</strong>
                        Optical Sentinel-2 imagery cannot image through cloud decks. Flash flood peaks obscured by cloud are masked as invalid (<code>255</code>), never zero.
                    </li>
                    <li style="margin-bottom: 0.5rem;">
                        <strong style="color: #F1F5F9;">Himalayan Terrain Distortion:</strong>
                        Steep gorge walls create severe radiometric shadows in optical data and layover/foreshortening in SAR backscatter.
                    </li>
                    <li style="margin-bottom: 0.5rem;">
                        <strong style="color: #F1F5F9;">Optical Water Mask vs Sediment-Laden Debris:</strong>
                        Himalayan flood cascades carry heavy mud and debris, which have distinct spectral signatures from clear surface water.
                    </li>
                    <li>
                        <strong style="color: #F1F5F9;">Exposure Definition (Exposure &ne; Damage):</strong>
                        Spatial overlap with building or road footprints indicates inundation exposure, not structural destruction or collapse.
                    </li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_export:
        render_section_header("Download Provenance Bundle", "Export machine-readable run metadata for scientific reproducibility.")

        bundle = {
            "application": "PRAVAH",
            "version": "0.1.0",
            "case_study": "Rasuwa-Nuwakot, Nepal (2026-08-26)",
            "run_id": artifacts.run_id,
            "is_synthetic": artifacts.is_synthetic,
            "observed_metadata": artifacts.observed_metadata,
            "exposure_summary": artifacts.exposure_summary,
            "benchmark": artifacts.benchmark,
        }

        st.download_button(
            label="⬇ Download Run Provenance JSON",
            data=json.dumps(bundle, indent=2),
            file_name=f"pravah_provenance_{artifacts.run_id}.json",
            mime="application/json",
        )
