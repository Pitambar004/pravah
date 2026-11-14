"""Shared per-run application context.

``Home.py`` builds the context once per rerun (from the sidebar data controls)
and stores it in session state; every page reads it via ``get_context()``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st

from app.components.layout import render_badge, render_page_header, render_topbar
from app.data_loader import RunArtifacts, load_replay_catalog_with_source

CASE_STUDY = {
    "id": "rasuwa-2026-08-26",
    "region": "Rasuwa, Nepal",
    "corridor": "Bhote Koshi–Trishuli corridor",
    "event_date": "26 Aug 2026",
}

_KEY = "pv_context"


@dataclass
class AppContext:
    artifacts: RunArtifacts
    catalog: list[dict[str, Any]]
    catalog_source: str | None

    @property
    def is_synthetic(self) -> bool:
        return self.artifacts.is_synthetic

    @property
    def catalog_is_synthetic(self) -> bool:
        return self.catalog_source == "synthetic_fixture"

    @property
    def data_status(self) -> tuple[str, str]:
        """(label, badge variant) describing what the UI is showing."""
        if self.is_synthetic:
            return "Synthetic fixtures", "warning"
        if self.artifacts.has_any_artifacts:
            return "Run artifacts loaded", "success"
        return "Awaiting artifacts", "neutral"

    def source_note(self, artifact_file: str) -> str:
        """Short provenance string for metric card footers."""
        if self.is_synthetic:
            return "Synthetic fixture · not an observation"
        return f"{self.artifacts.run_id}/{artifact_file}"


def build_context(artifacts: RunArtifacts) -> AppContext:
    catalog, source = load_replay_catalog_with_source(allow_synthetic=artifacts.is_synthetic)
    ctx = AppContext(artifacts=artifacts, catalog=catalog, catalog_source=source)
    st.session_state[_KEY] = ctx
    return ctx


def get_context() -> AppContext:
    ctx = st.session_state.get(_KEY)
    if ctx is None:  # page executed without the shell (should not happen via navigation)
        from app.data_loader import load_run_artifacts

        ctx = build_context(load_run_artifacts("outputs/latest", allow_synthetic_fallback=False))
    return ctx


def render_page_frame(ctx: AppContext, page: str, title: str, subtitle: str | None = None) -> None:
    """Topbar (breadcrumb + region + data status) and page header shared by all pages."""
    label, variant = ctx.data_status
    render_topbar(
        ["PRAVAH", page],
        [
            render_badge(CASE_STUDY["region"], "outline"),
            render_badge(f"Historical case study · {CASE_STUDY['event_date']}", "outline"),
            render_badge(label, variant, dot=True),
        ],
    )
    render_page_header(title, subtitle)
