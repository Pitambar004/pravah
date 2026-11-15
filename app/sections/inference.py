"""Run Inference — submit scenes to the MiniUNet API and follow jobs.

Everything shown comes from the live API (``/api/v1/status``, ``/scenes``,
``/predictions``); when the API is unreachable or has no checkpoint the page
says so instead of showing placeholders. Device and GPU fields are exactly what
the API process reports from ``torch``.
"""
from __future__ import annotations

from html import escape

import streamlit as st

from app import api_client
from app.components.empty_state import render_empty_state
from app.components.layout import (
    card,
    html,
    kv_list,
    render_badge,
    render_callout,
    render_section_header,
)
from app.components.metric_card import render_metric_card
from app.context import AppContext, render_page_frame

RUN_KEY = "pv_run"  # sidebar run selectbox key (Home.py)


def _open_run(run_id: str) -> None:
    st.session_state[RUN_KEY] = run_id
    st.session_state["pv_demo"] = False


def _fmt_secs(v) -> str | None:
    return f"{v:.2f}" if isinstance(v, (int, float)) else None


def _runtime_card(st_json: dict) -> None:
    rt = st_json.get("runtime") or {}
    model = st_json.get("model") or {}
    storage = st_json.get("storage") or {}
    if st_json.get("model_loaded"):
        badge = render_badge("CUDA verified", "success", dot=True) if rt.get("cuda_verified") else \
            render_badge("CPU (no CUDA)", "warning", dot=True)
    else:
        badge = render_badge("Model not loaded", "error", dot=True)
    c1, c2 = st.columns(2)
    with c1:
        html(card("Inference runtime", kv_list([
            ("API", api_client.api_url()),
            ("Device", rt.get("device") or "—"),
            ("GPU", rt.get("gpu_name") or "None reported"),
            ("Torch / CUDA", f"{rt.get('torch_version') or '—'} / {rt.get('torch_cuda_version') or 'none'}"),
            ("Host · arch", f"{rt.get('hostname') or '—'} · {rt.get('machine') or '—'}"),
            ("CUDA required", "Yes" if st_json.get("require_cuda") else "No"),
        ], mono_keys=("API", "Host · arch")), trailing=badge))
    with c2:
        ckpt = storage.get("checkpoint") or {}
        sha = model.get("checkpoint_sha256")
        html(card("Model checkpoint", kv_list([
            ("Architecture", model.get("architecture") or "MiniUNet (not loaded)"),
            ("Checkpoint", ckpt.get("path") or "—"),
            ("SHA-256", (sha[:16] + "…") if sha else "—"),
            ("Epoch · steps", f"{model.get('epoch', '—')} · {model.get('steps', '—')}" if model else "—"),
            ("Bands", ", ".join(model.get("bands") or []) or "B2, B3, B4, B8A, B11, B12"),
            ("Outputs", (storage.get("outputs_dir") or {}).get("path") or "—"),
        ], mono_keys=("Checkpoint", "SHA-256", "Outputs")),
            trailing=render_badge("Baseline U-Net · not Prithvi", "neutral")))
    if not st_json.get("model_loaded"):
        render_callout(
            escape(st_json.get("model_error") or "Unknown error") +
            ". Stage the trained checkpoint on the Pravah-Dataset volume (see docs/DEPLOYMENT.md) and reload. "
            "The API never serves randomly initialised weights.",
            title="No trained checkpoint loaded", variant="error")


def _submit_form(scene_rows: list[dict], model_loaded: bool, on_gpu: bool) -> None:
    compatible = [s for s in scene_rows if s.get("model_compatible")]
    if not scene_rows:
        render_empty_state(
            "No scenes on the storage volume",
            "Place georeferenced Sentinel-2 L1C GeoTIFFs (13-band, or 6-band B2,B3,B4,B8A,B11,B12) in the scenes "
            "directory reported by the API.",
            expected_files=["raw/sentinel2/<scene>.tif"], icon_name="satellite_alt", compact=True)
        return
    with st.form("pv_predict", border=False):
        c1, c2, c3 = st.columns([3, 1.4, 1.2], vertical_alignment="bottom")
        with c1:
            scene = st.selectbox(
                "Scene", [s["scene_id"] for s in compatible] or ["No model-compatible scene"],
                help="Only georeferenced 6- or 13-band Sentinel-2 L1C rasters are accepted. SAR is rejected.")
        with c2:
            threshold = st.slider("Water threshold", 0.05, 0.95, 0.5, 0.05,
                                  help="Probability ≥ threshold → class 1 (model-inferred water)")
        with c3:
            label = "Run on GPU" if on_gpu else ("Run on CPU" if model_loaded else "Run inference")
            go = st.form_submit_button(label, type="primary", icon=":material/play_arrow:",
                                       disabled=not (model_loaded and compatible), width="stretch")
    if go:
        data, err = api_client.submit_prediction(scene, threshold)
        if err:
            st.error(err, icon=":material/error:")
        else:
            st.toast(f"Queued {data['run_id']}", icon=":material/schedule:")
    skipped = [s for s in scene_rows if not s.get("model_compatible")]
    if skipped:
        with st.expander(f"{len(skipped)} file(s) not usable by this model"):
            st.dataframe([{"Scene": s["scene_id"], "Bands": s.get("band_count"), "CRS": s.get("crs"),
                           "Reason": s.get("error") or "Needs CRS and 6 or 13 Sentinel-2 bands"} for s in skipped],
                         hide_index=True, width="stretch")


@st.fragment(run_every=3)
def _jobs_panel() -> None:
    data, err = api_client.jobs()
    if err:
        st.caption(f"Jobs unavailable: {err}")
        return
    rows = data.get("jobs", [])
    if not rows:
        st.caption("No predictions submitted since the API started. Finished runs remain on the volume and in "
                   "the sidebar run list.")
        return
    for j in rows[:10]:
        res = j.get("result") or {}
        timings = res.get("timings_seconds") or {}
        runtime = res.get("runtime") or {}
        variant = {"succeeded": "success", "failed": "error", "running": "accent"}.get(j["status"], "neutral")
        c1, c2, c3, c4 = st.columns([3, 1.3, 2.2, 1.4], vertical_alignment="center")
        with c1:
            html(f'<div class="pv-mono" style="font-size:13px">{escape(j["run_id"])}</div>'
                 f'<div style="color:var(--pv-text-muted);font-size:12px">{escape(j["scene_id"])} · '
                 f'threshold {j["threshold"]}</div>')
        with c2:
            html(render_badge(j["status"].capitalize(), variant, dot=True))
        with c3:
            if j["status"] == "running" and j.get("tiles_total"):
                st.progress(j["tiles_done"] / j["tiles_total"], text=f"{j['tiles_done']}/{j['tiles_total']} tiles")
            elif j["status"] == "succeeded":
                st.caption(f"{runtime.get('device', '?')} · model {_fmt_secs(timings.get('model_compute'))} s · "
                           f"total {_fmt_secs(timings.get('end_to_end'))} s (measured)")
            elif j["status"] == "failed":
                st.caption(j.get("error") or "Failed")
            else:
                st.caption("Waiting for the GPU worker")
        with c4:
            if j["status"] == "succeeded":
                st.button("Open run", key=f"open_{j['job_id']}", on_click=_open_run, args=(j["run_id"],),
                          icon=":material/map:", width="stretch")


def render_inference_section(ctx: AppContext) -> None:
    render_page_frame(
        ctx, "Run Inference", "Run Inference",
        "Send a Sentinel-2 scene from the Pravah-Dataset volume to the MiniUNet service. Output is a "
        "georeferenced water mask with measured timings — model evidence, not a forecast.",
    )
    st_json, err = api_client.status()
    if err:
        render_callout(
            f"{escape(err)}. Start the API (<code>make api</code>) or set <code>PRAVAH_API_URL</code>. On Fuzzball "
            "the dashboard reaches it at <code>http://api.service:8000</code>.",
            title="Inference API unavailable", variant="error")
        return

    _runtime_card(st_json)
    q = st_json.get("queue") or {}
    rt = st_json.get("runtime") or {}
    m1, m2, m3 = st.columns(3)
    with m1:
        render_metric_card("Execution device", rt.get("device"), subtext="Reported by torch in the API process",
                           icon_name="memory")
    with m2:
        render_metric_card("Jobs pending", q.get("pending"), subtext=f"Single GPU worker · max {q.get('max')}",
                           icon_name="pending_actions")
    with m3:
        out_ok = ((st_json.get("storage") or {}).get("outputs_dir") or {}).get("writable")
        render_metric_card("Output storage", "Writable" if out_ok else "Not writable",
                           badge="OK" if out_ok else "Check", badge_variant="success" if out_ok else "error",
                           subtext="Pravah-Dataset outputs directory", icon_name="database")

    render_section_header("New prediction", "Scenes discovered on the storage volume")
    sc, sc_err = api_client.scenes()
    if sc_err:
        st.error(sc_err, icon=":material/error:")
    else:
        _submit_form(sc.get("scenes", []), bool(st_json.get("model_loaded")), bool(rt.get("cuda_verified")))

    render_section_header("Jobs", "Live from the API · refreshes every 3 s")
    _jobs_panel()
