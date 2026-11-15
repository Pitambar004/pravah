"""Two Leaflet maps side by side with locked pan/zoom.

Used for "prediction vs observed reality" and "before vs after". Each side gets its
own list of ``MapLayer`` (real artifacts only; empty layers are listed as unavailable
in the legend). Pan or zoom either map and the other follows, so the same valley
is always compared at the same place and scale.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import streamlit as st

from app.components.map_view import (
    BASEMAPS,
    LEAFLET_VERSION,
    RASUWA_CENTER,
    MapLayer,
    _payload,
    configured_basemaps,
    essential_layers,
)
from app.theme import FONT_MONO, FONT_SANS, TOKENS
from pravah.configuration import configured_url


@dataclass
class MapSide:
    title: str
    subtitle: str
    layers: list[MapLayer] = field(default_factory=list)
    badge: str = ""  # e.g. "Prediction", "Observed"


def build_compare_html(left: MapSide, right: MapSide, basemap: str = "imagery", synthetic: bool = False) -> str:
    cfg = {
        "sides": [
            {"title": s.title, "subtitle": s.subtitle, "badge": s.badge, "layers": _payload(essential_layers(s.layers))}
            for s in (left, right)
        ],
        "basemaps": configured_basemaps(),
        "basemap": basemap if basemap in BASEMAPS else "imagery",
        "center": RASUWA_CENTER,
        "synthetic": synthetic,
    }
    data = json.dumps(cfg, separators=(",", ":")).replace("</", "<\\/")
    css_vars = ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in TOKENS.items())
    return (_TEMPLATE.replace("%LEAFLET_CSS%", configured_url("PRAVAH_LEAFLET_CSS_URL", version=LEAFLET_VERSION))
            .replace("%LEAFLET_JS%", configured_url("PRAVAH_LEAFLET_JS_URL", version=LEAFLET_VERSION))
            .replace("%CSSVARS%", css_vars)
            .replace("%FONT%", FONT_SANS).replace("%MONO%", FONT_MONO).replace("%CONFIG%", data))


def render_compare_map(left: MapSide, right: MapSide, height: int = 520, basemap: str = "imagery",
                       synthetic: bool = False) -> None:
    st.components.v1.html(build_compare_html(left, right, basemap, synthetic), height=height)


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<link rel="stylesheet" href="%LEAFLET_CSS%">
<script src="%LEAFLET_JS%"></script>
<style>
:root{%CSSVARS%;--font:%FONT%;--mono:%MONO%}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg-app);color:var(--text);font:13px/20px var(--font)}
#wrap{position:absolute;inset:0;display:grid;grid-template-columns:1fr 1fr;gap:6px}
.side{position:relative;border:1px solid var(--border);border-radius:12px;overflow:hidden}
.map{position:absolute;inset:0;background:#0b111c}
.head{position:absolute;z-index:500;top:10px;left:10px;right:10px;display:flex;gap:8px;align-items:flex-start;
  pointer-events:none}
.title{background:rgba(17,27,42,.94);border:1px solid var(--border);border-radius:10px;padding:8px 12px;max-width:100%}
.title b{font-size:13px;font-weight:600}
.title small{display:block;color:var(--text-muted);font-size:11.5px;line-height:16px}
.badge{font-size:10.5px;padding:1px 7px;border-radius:6px;border:1px solid var(--border);margin-left:6px;
  color:var(--accent)}
.legend{position:absolute;z-index:500;left:10px;bottom:10px;background:rgba(17,27,42,.94);
  border:1px solid var(--border);border-radius:10px;padding:8px 10px;font-size:12px;max-width:calc(100% - 20px)}
.legend label{display:flex;align-items:center;gap:8px;cursor:pointer;line-height:22px}
.legend .off{color:var(--text-disabled);cursor:default}
.sw{width:10px;height:10px;border-radius:3px;flex:none}
.bm{position:absolute;z-index:600;top:10px;right:10px;display:flex;gap:4px}
.bm button{height:26px;border-radius:7px;border:1px solid var(--border);background:rgba(23,35,53,.96);
  color:var(--text-secondary);font:500 11px var(--font);cursor:pointer;padding:0 8px}
.bm button[aria-pressed=true]{color:var(--accent);border-color:rgba(56,189,248,.45)}
.leaflet-container{font:12px var(--font)}
.leaflet-popup-content-wrapper,.leaflet-popup-tip{background:var(--surface-raised);color:var(--text)}
.fx{position:absolute;z-index:700;left:50%;transform:translateX(-50%);bottom:10px;background:rgba(251,191,36,.15);
  color:var(--warning);border:1px solid rgba(251,191,36,.4);border-radius:8px;padding:2px 10px;font-size:11.5px}
</style></head><body>
<div id="wrap"></div>
<script>
const CFG = %CONFIG%;
const wrap = document.getElementById('wrap');
const maps = [], bases = [];
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function el(t, a, ...kids){const n=document.createElement(t);for(const k in a||{})n.setAttribute(k,a[k]);
  kids.forEach(c=>n.append(c));return n;}
CFG.sides.forEach((side, i) => {
  const box = el('div', {class:'side'});
  const m = el('div', {class:'map'});
  box.append(m);
  const t = el('div', {class:'title'}, el('b', {}, side.title));
  if (side.badge) t.querySelector('b').append(el('span', {class:'badge'}, side.badge));
  t.append(el('small', {}, side.subtitle));
  box.append(el('div', {class:'head'}, t));
  wrap.append(box);
  const map = L.map(m, {zoomControl: i === 1, attributionControl: i === 1}).setView(CFG.center, 11);
  const bm = CFG.basemaps[CFG.basemap];
  bases.push(L.tileLayer(bm.url, {attribution: bm.attr, maxZoom: 18}).addTo(map));
  const legend = el('div', {class:'legend'});
  let bounds = null;
  side.layers.forEach(spec => {
    const sw = el('span', {class:'sw', style:`background:${spec.color}`});
    if (!spec.available) {
      legend.append(el('label', {class:'off', title: spec.unavailable_reason}, sw, spec.name + ' — no data'));
      return;
    }
    const lyr = L.geoJSON(spec.data, {
      style: () => ({color: spec.color, weight: spec.fill ? 1 : 1.6, fillColor: spec.color,
                     fillOpacity: spec.fill ? spec.opacity : 0, opacity: Math.min(1, spec.opacity + .3),
                     dashArray: spec.dash || null}),
      pointToLayer: (f, ll) => L.circleMarker(ll, {radius: 4}),
      onEachFeature: (f, l) => l.bindPopup(`<b>${esc(spec.name)}</b><br><small>${esc((f.properties||{}).label)}` +
                                           `<br>${esc(spec.timestamp)}</small>`),
    });
    const cb = el('input', {type:'checkbox'});
    cb.checked = spec.visible;
    if (spec.visible) { lyr.addTo(map); }
    cb.onchange = () => cb.checked ? lyr.addTo(map) : map.removeLayer(lyr);
    legend.append(el('label', {title: spec.limitation}, cb, sw, spec.name));
    if (spec.kind !== 'context') { const b = lyr.getBounds(); if (b.isValid()) bounds = bounds ? bounds.extend(b) : b; }
  });
  box.append(legend);
  maps.push({map, bounds});
});
// Basemap switch (applies to both sides)
const bmBox = el('div', {class:'bm'});
Object.entries(CFG.basemaps).forEach(([k, v]) => {
  const b = el('button', {'aria-pressed': String(k === CFG.basemap)}, v.label);
  b.onclick = () => { bases.forEach((t, i) => { t.setUrl(v.url); });
    bmBox.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', String(x === b))); };
  bmBox.append(b);
});
wrap.children[0].append(bmBox);
// Lock views together
let syncing = false;
maps.forEach((a, i) => a.map.on('move', () => {
  if (syncing) return; syncing = true;
  const o = maps[1 - i].map; o.setView(a.map.getCenter(), a.map.getZoom(), {animate: false});
  syncing = false;
}));
const b = maps[0].bounds || maps[1].bounds;
if (b) { maps[0].map.fitBounds(b, {padding: [20, 20]}); }
if (CFG.synthetic) document.body.append(el('div', {class:'fx'}, 'Synthetic fixture — not an observation'));
</script></body></html>"""
