"""Tests for design-system components: metric card, map workspace, timeline."""
import json

import pytest

pytest.importorskip("streamlit", reason="Streamlit required for UI tests")

from app.components.map_view import build_layers, build_map_html
from app.components.metric_card import render_metric_card
from app.components.timeline import build_timeline_svg, evidence_card
from app.data_loader import RunArtifacts
from app.fixtures.synthetic_sample import SYNTHETIC_REPLAY_ASSETS
from pravah.replay.eligibility import classify_asset

CUTOFF = "2026-08-26T11:00:00+05:45"


def test_metric_card_html_rendering():
    html = render_metric_card(label="Water Area", value=34.8, unit="ha", subtext="Optical",
                              badge="Live", badge_variant="success", as_html=True)
    assert "pv-metric-card" in html
    assert "Water Area" in html and "34.8" in html and "ha" in html
    assert "pv-badge-success" in html


def test_metric_card_unknown_is_not_zero():
    html = render_metric_card(label="Buildings", value=None, unit="footprints", as_html=True)
    assert "Not available" in html
    assert ">0<" not in html
    assert "footprints" not in html  # unit hidden when value is unknown


def test_metric_card_escapes_text():
    html = render_metric_card(label="<script>x</script>", value="<b>", as_html=True)
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_map_layers_from_fixture(tmp_path):
    from app.data_loader import load_run_artifacts

    a = load_run_artifacts(tmp_path, allow_synthetic_fallback=True)
    layers = {layer.id: layer for layer in build_layers(a)}
    assert layers["water"].available and layers["aoi"].available and layers["unosat"].available
    assert "susceptibility" not in layers and "inundation" not in layers
    assert layers["unosat"].visible is False  # retrospective layer is off by default
    html = build_map_html(list(layers.values()), synthetic=True)
    assert "leaflet" in html.lower()
    assert "Synthetic fixture" in html


def test_map_strict_context_omits_retrospective(tmp_path):
    from app.data_loader import load_run_artifacts

    a = load_run_artifacts(tmp_path, allow_synthetic_fallback=True)
    assert "unosat" not in {layer.id for layer in build_layers(a, show_retrospective=False)}


def test_map_empty_has_no_data_layers(tmp_path):
    a = RunArtifacts(run_id="empty", folder=tmp_path)
    layers = build_layers(a)
    assert not any(layer.available for layer in layers)
    html = build_map_html(layers)
    cfg = json.loads(html.split("const CFG = ", 1)[1].split(";\n", 1)[0])
    assert all(layer["data"] is None for layer in cfg["layers"])


def test_map_html_escapes_script_close():
    a = RunArtifacts(run_id="x", folder=None)
    a.water_geojson = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"label": "</script><script>alert(1)</script>"},
        "geometry": {"type": "Point", "coordinates": [85.3, 28.1]}}]}
    html = build_map_html(build_layers(a))
    assert "</script><script>alert(1)" not in html


def _classified():
    return [{**x, "_status": classify_asset(x, CUTOFF)} for x in SYNTHETIC_REPLAY_ASSETS]


def test_strict_timeline_omits_post_cutoff_assets():
    svg = build_timeline_svg(_classified(), CUTOFF, strict=True)
    assert "S2B_MSIL1C_20260827T051219" not in svg
    assert "UNOSAT_NEPAL" not in svg
    assert "Hidden in strict replay" in svg


def test_retrospective_timeline_includes_later_assets():
    svg = build_timeline_svg(_classified(), CUTOFF, strict=False)
    assert "S2B_MSIL1C_20260827T051219" in svg
    assert "UNOSAT_NEPAL" in svg


def test_evidence_card_labels_status():
    later = next(x for x in _classified() if x["_status"] == "retrospective_reveal")
    assert "Later evidence" in evidence_card(later)
    unknown = next(x for x in _classified() if x["_status"] == "excluded_unknown_availability")
    assert "Unverified" in evidence_card(unknown)
