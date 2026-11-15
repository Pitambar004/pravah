"""Downloaded-context layers for Model evidence maps, on tiny SYNTHETIC tmp-dir bundles only.

The loaders under test refuse data marked synthetic, so these fixtures omit the flag;
they live only in pytest's tmp_path and are never exported as observations.
"""
import hashlib
import json

import pytest

pytest.importorskip("streamlit")
import streamlit as st

from app.components import context_layers as cl
from app.components.map_view import MapLayer

INSIDE = [[85.30, 28.10], [85.31, 28.11]]  # within RASUWA_BBOX
OUTSIDE = [[84.00, 27.00], [84.01, 27.01]]  # far outside


def _fc(*lines):
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": f"f{i}"}, "geometry": {"type": "LineString", "coordinates": c}}
        for i, c in enumerate(lines)]}


def _write(root, name, doc):
    path = root / f"{name}.geojson"
    path.write_text(json.dumps(doc))
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def bundles(tmp_path, monkeypatch):
    geo, nat, lakes = tmp_path / "geo", tmp_path / "nat", tmp_path / "lakes"
    for d in (geo, nat, lakes):
        d.mkdir()
    manifest = {"layers": {k: _write(geo, k, _fc(INSIDE, OUTSIDE)) for k in ("rivers", "roads", "bridges", "buildings")},
                "retrieved_at_utc": "T"}
    (geo / "manifest.json").write_text(json.dumps(manifest))
    rec = _write(nat, "major_rivers", _fc(INSIDE))
    (nat / "manifest.json").write_text(json.dumps({**rec, "is_synthetic": False, "retrieved_at_utc": "T"}))
    (lakes / "manifest.json").write_text(json.dumps({"is_synthetic": False, "layers": {"s2": _write(lakes, "s2", _fc(INSIDE))}}))
    monkeypatch.setenv("PRAVAH_NEPAL_GEOGRAPHY", str(geo))
    monkeypatch.setenv("PRAVAH_NATIONAL_ROOT", str(nat))
    monkeypatch.setenv("PRAVAH_LAKE_ROOT", str(lakes))
    st.cache_data.clear()
    yield geo
    st.cache_data.clear()


def test_layers_clipped_to_study_region(bundles):
    layers = {lyr.id: lyr for lyr in cl.downloaded_context_layers()}
    assert {"ctx_national_rivers", "ctx_rivers", "ctx_roads", "ctx_bridges", "ctx_buildings", "ctx_glacial_lakes"} <= set(layers)
    assert len(layers["ctx_rivers"].data["features"]) == 1  # the outside feature is clipped away
    assert all(lyr.kind == "context" for lyr in layers.values())
    assert layers["ctx_rivers"].visible and not layers["ctx_buildings"].visible


def test_light_set_skips_heavy_layers(bundles):
    ids = {lyr.id for lyr in cl.downloaded_context_layers(light=True)}
    assert "ctx_rivers" in ids and "ctx_bridges" in ids
    assert not ids & {"ctx_roads", "ctx_buildings", "ctx_national_rivers"}


def test_tampered_bundle_yields_no_osm_layers(bundles):
    (bundles / "rivers.geojson").write_text(json.dumps(_fc(INSIDE)))  # checksum no longer matches
    st.cache_data.clear()
    ids = {lyr.id for lyr in cl.downloaded_context_layers()}
    assert not any(i in ids for i in ("ctx_rivers", "ctx_roads", "ctx_buildings"))


def test_missing_bundles_yield_nothing(tmp_path, monkeypatch):
    for var in ("PRAVAH_NEPAL_GEOGRAPHY", "PRAVAH_NATIONAL_ROOT", "PRAVAH_LAKE_ROOT"):
        monkeypatch.setenv(var, str(tmp_path / "absent"))
    st.cache_data.clear()
    assert cl.downloaded_context_layers() == []


def test_replaces_old_rivers_layer(bundles):
    old = MapLayer(id="rivers", name="River network", group="Geographic context", kind="context", color="#000")
    water = MapLayer(id="water", name="Water", group="Satellite evidence", kind="model", color="#000")
    ids = [lyr.id for lyr in cl.with_downloaded_context([old, water])]
    assert "rivers" not in ids and "water" in ids and "ctx_rivers" in ids
