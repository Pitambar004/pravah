"""Maps — before/after comparison and every run on one map.

Every run folder in the outputs directory (before, after, predictions) is a layer
group on the combined map, so no map is hidden; toggle any combination.
"""
from __future__ import annotations

import streamlit as st

from app.components.compare_map import MapSide, render_compare_map
from app.components.empty_state import render_empty_state
from app.components.layout import render_section_header
from app.components.map_view import (
    build_layers,
    build_multi_run_layers,
    render_map_workspace,
    run_layers,
    short_run_label,
)
from app.context import AppContext, render_page_frame
from app.runs import all_runs, run_role, run_title


def render_gallery_section(ctx: AppContext) -> None:
    render_page_frame(ctx, "Maps", "Maps", "Before and after the event, and every map produced, in one place.")
    if ctx.is_synthetic:
        render_map_workspace(build_layers(ctx.artifacts), height=520, mode="full", synthetic=True)
        return
    runs = all_runs()
    if not runs:
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
        render_compare_map(MapSide("Before", short_run_label(pre_id), run_layers(pre, "pre"), "Satellite"),
                           MapSide("After", short_run_label(post_id), run_layers(post, "post"), "Satellite"),
                           height=520)

    render_section_header("Every run", f"{len(runs)} runs · one layer group each · toggle in the layer panel")
    render_map_workspace(build_multi_run_layers(runs, ctx.artifacts, run_title), height=620, mode="full")
