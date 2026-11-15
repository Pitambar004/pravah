"""Maps — the searched place, before/after comparison and every run on one map.

A place picked in the location bar gets its own map first, opened on that run's image. Every run
folder in the outputs directory (before, after, predictions) is a layer group on the combined map,
so no map is hidden; toggle any combination.
"""
from __future__ import annotations

import streamlit as st

from app.components.compare_map import MapSide, render_compare_map
from app.components.context_layers import with_downloaded_context
from app.components.empty_state import render_empty_state
from app.components.layout import render_section_header
from app.components.location_bar import PLACE_KEY, same_point, short_name
from app.components.map_view import (
    build_layers,
    build_multi_run_layers,
    render_map_workspace,
    run_layers,
    short_run_label,
)
from app.context import AppContext, render_page_frame
from app.data_loader import RunArtifacts
from app.runs import all_runs, run_role, run_title


def place_run(place: dict | None, runs: list[RunArtifacts], active: RunArtifacts) -> RunArtifacts | None:
    """Newest finished run for the searched place (the active run wins when it matches)."""
    if not place:
        return None

    def at_place(a: RunArtifacts) -> bool:
        return a.is_other_place and same_point((a.observed_metadata or {}).get("case_study"), place)

    if at_place(active):
        return active
    matches = [a for a in runs if at_place(a)]
    return max(matches, key=lambda a: (a.acquired_at or "", a.run_id)) if matches else None


def render_place_map(place: dict, run: RunArtifacts | None) -> None:
    name = short_name(place)
    if run is None:
        render_section_header(name, "The map appears here when the imagery download and model run above finish.")
        return
    render_section_header(name, f"Sentinel-2 {(run.acquired_at or '')[:10]} · {run.run_id} · "
                                "model-derived water, not ground truth")
    render_map_workspace(build_layers(run, show_retrospective=False), height=620, mode="full")


def render_gallery_section(ctx: AppContext) -> None:
    render_page_frame(ctx, "Maps", "Maps", "Before and after the event, and every map produced, in one place.")
    if ctx.is_synthetic:
        render_map_workspace(build_layers(ctx.artifacts), height=520, mode="full", synthetic=True)
        return
    runs = all_runs()
    place = st.session_state.get(PLACE_KEY)
    if place:
        render_place_map(place, place_run(place, runs, ctx.artifacts))
    if not runs:
        if place:
            return
        render_empty_state("No maps yet", "Run folders appear here automatically.",
                           expected_files=["outputs/<run_id>/"], icon_name="map", compact=True)
        return

    pres = [r for r in runs if run_role(r) == "pre_event"]
    posts = [r for r in runs if run_role(r) == "post_event"]
    if pres and posts:
        by_id = {r.run_id: r for r in runs}
        c1, c2, _ = st.columns([1, 1, 2])
        with c2:
            post_id = st.selectbox("After", [r.run_id for r in posts], format_func=short_run_label, key="pv_ba_post")
        paired = (by_id[post_id].change_summary or {}).get("pre_run_id")
        pre_ids = [r.run_id for r in pres]
        with c1:
            pre_id = st.selectbox("Before", pre_ids, format_func=short_run_label, key=f"pv_ba_pre_{post_id}",
                                  index=pre_ids.index(paired) if paired in pre_ids else 0)
        pre, post = by_id[pre_id], by_id[post_id]
        render_compare_map(MapSide("Before", short_run_label(pre_id), with_downloaded_context(run_layers(pre, "pre"), light=True),
                                   "Satellite"),
                           MapSide("After", short_run_label(post_id), with_downloaded_context(run_layers(post, "post"), light=True),
                                   "Satellite"),
                           height=520)

    render_section_header("Every run", f"{len(runs)} runs · one layer group each · toggle in the layer panel")
    render_map_workspace(with_downloaded_context(build_multi_run_layers(runs, ctx.artifacts, run_title)), height=620, mode="full")
