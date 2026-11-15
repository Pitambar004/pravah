"""PRAVAH map workspace.

A self-contained Leaflet view rendered in one iframe. Layer toggles, opacity,
basemap switching, feature selection and the detail panel all run client-side,
so they never trigger a Streamlit rerun and the viewport is preserved.

Layers are described in Python (``MapLayer``) from real artifacts only. Layers
without an artifact are still listed but disabled, with the reason shown.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import streamlit as st

from app.theme import FONT_MONO, FONT_SANS, TOKENS

RASUWA_CENTER = [28.150, 85.350]
LEAFLET_VERSION = "1.9.4"

GROUP_ORDER = ["Geographic context", "Satellite evidence", "Change vs pre-event", "Infrastructure", "Scenarios",
               "Validation"]

KIND_LABELS = {
    "context": "Context",
    "observed": "Observed",
    "model": "Model-derived",
    "derived": "Derived exposure",
    "scenario": "Hypothetical scenario",
    "reference": "Retrospective reference",
}

BASEMAPS = {
    "dark": {
        "label": "Dark",
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri, HERE, Garmin, &copy; OpenStreetMap contributors",
    },
    "light": {
        "label": "Light",
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri, HERE, Garmin, &copy; OpenStreetMap contributors",
    },
    "imagery": {
        "label": "Imagery",
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri World Imagery (basemap only — not PRAVAH evidence)",
    },
    "terrain": {
        "label": "Terrain",
        "url": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": "&copy; OpenStreetMap contributors, SRTM | OpenTopoMap (CC-BY-SA)",
    },
}


@dataclass
class MapLayer:
    id: str
    name: str
    group: str
    kind: str
    color: str
    data: dict[str, Any] | None = None
    unavailable_reason: str = "No artifact in the selected run"
    visible: bool = True
    opacity: float = 0.55
    fill: bool = True
    dash: str | None = None
    source: str = "—"
    timestamp: str = "—"
    description: str = ""
    limitation: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return bool(self.data and self.data.get("features"))


def _round_coords(obj: Any, ndigits: int = 6) -> Any:
    """Reduce coordinate precision (~0.1 m) to keep the embedded payload small."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, list):
        return [_round_coords(v, ndigits) for v in obj]
    if isinstance(obj, dict):
        return {k: (_round_coords(v, ndigits) if k in ("coordinates", "geometry", "geometries") else v)
                for k, v in obj.items()}
    return obj


def _slim_geojson(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if not data:
        return None
    feats = data.get("features") if data.get("type") == "FeatureCollection" else [
        data if data.get("type") == "Feature" else {"type": "Feature", "properties": {}, "geometry": data}
    ]
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": f.get("properties") or {}, "geometry": _round_coords(f.get("geometry"))}
            for f in (feats or [])
            if f.get("geometry")
        ],
    }


def _context_layer(artifacts: Any, key: str, layer_id: str, name: str, color: str) -> MapLayer:
    data = (getattr(artifacts, "context_layers", None) or {}).get(key)
    meta = (data or {}).get("pravah_layer") or {}
    return MapLayer(id=layer_id, name=name, group="Geographic context", kind="context", color=color,
                    data=_slim_geojson(data), fill=False, opacity=0.8, visible=layer_id == "rivers",
                    source=meta.get("source", "—"), timestamp=meta.get("source", "—").split("snapshot ")[-1],
                    description=meta.get("filter", ""), limitation=meta.get("note", ""),
                    unavailable_reason="Not yet ingested (scripts/build_context_layers.py)")


def _change_layers(artifacts: Any) -> list[MapLayer]:
    """Pre/post change layers; only present when a change analysis exists for the run."""
    summary = getattr(artifacts, "change_summary", None)
    if not summary:
        return []
    pre = (summary.get("pre_acquired_at_utc") or "")[:10]
    post = (summary.get("post_acquired_at_utc") or "")[:10]
    span = f"{pre} → {post}"
    src = f"change_vs_pre.tif ({summary.get('pre_run_id')} vs {summary.get('post_run_id')})"
    limit = ("Change between two model outputs on pixels cloud-free on both dates. Cloud/terrain shadow and "
             "different viewing geometry can create apparent change. Not a forecast.")
    return [
        MapLayer(id="change_new", name="Water newly detected", group="Change vs pre-event", kind="model",
                 color=TOKENS["danger"], data=_slim_geojson(getattr(artifacts, "change_new_water_geojson", None)),
                 opacity=0.6, source=src, timestamp=span,
                 description="Pixels classed no-water on the pre-event date and water on the post-event date.",
                 limitation=limit, unavailable_reason="No newly detected water in comparable pixels"),
        MapLayer(id="change_lost", name="Water no longer detected", group="Change vs pre-event", kind="model",
                 color=TOKENS["text_muted"], data=_slim_geojson(getattr(artifacts, "change_lost_water_geojson", None)),
                 opacity=0.45, visible=False, source=src, timestamp=span,
                 description="Pixels classed water on the pre-event date and no-water on the post-event date.",
                 limitation=limit, unavailable_reason="No water-loss pixels in comparable area"),
    ]


def _exposure_layers(artifacts: Any, fixture: str) -> list[MapLayer]:
    feats = getattr(artifacts, "exposure_features", None) or {}
    specs = [("buildings", "Buildings in mapped extent", "OSM building footprints intersecting the mapped water extent."),
             ("roads", "Roads in mapped extent", "OSM highway segments clipped to the mapped water extent."),
             ("bridges", "Bridges in mapped extent", "OSM bridge-tagged ways intersecting the mapped water extent.")]
    exp = getattr(artifacts, "exposure_summary", None) or {}
    osm_date = exp.get("osm_snapshot_timestamp") or exp.get("source_snapshot") or "—"
    return [
        MapLayer(id=cat, name=name, group="Infrastructure", kind="derived", color=TOKENS["map_exposure"],
                 data=_slim_geojson(feats.get(cat)), fill=cat == "buildings", opacity=0.7,
                 source=f"exposure_{cat}.geojson · OSM {osm_date}{fixture}", timestamp=f"OSM snapshot {osm_date}",
                 description=desc, limitation="Intersection indicates potential exposure, not damage.",
                 unavailable_reason="No exposure feature layer exported")
        for cat, name, desc in specs
    ]


def build_layers(artifacts: Any, show_retrospective: bool = True) -> list[MapLayer]:
    """Build the standard layer stack from loaded run artifacts.

    Only artifacts that actually exist are given data. ``show_retrospective=False``
    omits post-event reference layers entirely (strict replay contexts).
    """
    fixture = " (synthetic fixture)" if getattr(artifacts, "is_synthetic", False) else ""
    meta = artifacts.observed_metadata or {}
    layers = [
        MapLayer(
            id="aoi", name="Study area (AOI)", group="Geographic context", kind="context",
            color=TOKENS["text_secondary"], data=_slim_geojson(artifacts.aoi_geojson), fill=False, dash="6 4",
            opacity=0.9, source="config/aois/" + fixture, description="Bounded analysis corridor for the Bhote Koshi–Trishuli valleys.",
            limitation="Pixels outside the AOI or occluded by cloud are invalid (class 255), never 'no water'.",
            unavailable_reason="No AOI GeoJSON in config/aois/",
        ),
        _context_layer(artifacts, "admin_boundaries", "admin", "Administrative boundaries", TOKENS["text_muted"]),
        _context_layer(artifacts, "rivers", "rivers", "River network", TOKENS["text_secondary"]),
        MapLayer(
            id="water", name="Model-derived water extent", group="Satellite evidence", kind="model",
            color=TOKENS["map_water"], data=_slim_geojson(artifacts.water_geojson), opacity=0.55,
            source=(meta.get("source_raster") or "observed_water.tif → polygons") + fixture,
            timestamp=meta.get("acquired_at_utc") or meta.get("created_at_utc") or "See run metadata",
            description="Binary water segmentation from Sentinel-2 L1C optical bands (threshold "
            f"{meta.get('threshold', '—')}).",
            limitation="Model output, not ground truth. Sediment-laden flow and cloud shadow can be misclassified.",
            unavailable_reason="No stitched water polygons (exposure.geojson / polygons.geojson)",
        ),
        MapLayer(id="imagery", name="Sentinel-2 scene preview", group="Satellite evidence", kind="observed",
                 color=TOKENS["accent"], unavailable_reason="No preview tiles generated (GeoTIFFs are not loaded in-browser)"),
        *_change_layers(artifacts),
        *_exposure_layers(artifacts, fixture),
        MapLayer(id="susceptibility", name="Terrain susceptibility", group="Scenarios", kind="scenario",
                 color=TOKENS["map_scenario"], unavailable_reason="No scenario model has been run"),
        MapLayer(id="inundation", name="Hypothetical inundation", group="Scenarios", kind="scenario",
                 color=TOKENS["map_scenario"], unavailable_reason="No scenario model has been run"),
    ]
    if show_retrospective:
        unosat = artifacts.unosat_geojson
        props = (unosat or {}).get("features", [{}])[0].get("properties", {}) if unosat else {}
        layers.append(
            MapLayer(
                id="unosat", name="UNOSAT affected extent", group="Validation", kind="reference",
                color=TOKENS["map_reference"], data=_slim_geojson(unosat), visible=False, opacity=0.3, dash="4 3",
                source=(props.get("citation") or "UNOSAT / UNESCO IHP-WINS") + fixture,
                timestamp=props.get("published_date") or "—",
                description="Satellite-detected mudflow / rockflow extent, published after the event.",
                limitation="Retrospective only. A different class definition from water — compare qualitatively.",
                unavailable_reason="UNOSAT geometry not downloaded",
            )
        )
    return layers


def _payload(layers: list[MapLayer]) -> list[dict[str, Any]]:
    out = []
    for layer in layers:
        d = asdict(layer)
        d["available"] = layer.available
        d["kind_label"] = KIND_LABELS.get(layer.kind, layer.kind)
        out.append(d)
    return out


def build_map_html(
    layers: list[MapLayer],
    mode: str = "full",
    synthetic: bool = False,
    initial_basemap: str = "dark",
) -> str:
    """Return the full HTML document for the map workspace.

    ``mode='full'`` shows the layer panel, toolbar and detail panel.
    ``mode='preview'`` shows a compact, non-scroll-zoom map with a legend only.
    """
    config = {
        "layers": _payload(layers),
        "groups": GROUP_ORDER,
        "basemaps": BASEMAPS,
        "basemap": initial_basemap if initial_basemap in BASEMAPS else "dark",
        "mode": mode,
        "synthetic": synthetic,
        "center": RASUWA_CENTER,
        "zoom": 11,
    }
    data_json = json.dumps(config, separators=(",", ":")).replace("</", "<\\/")
    tokens = {f"--{k.replace('_', '-')}": v for k, v in TOKENS.items()}
    css_vars = ";".join(f"{k}:{v}" for k, v in tokens.items())
    return (
        _TEMPLATE.replace("%LEAFLET%", LEAFLET_VERSION)
        .replace("%CSSVARS%", css_vars)
        .replace("%FONT%", FONT_SANS)
        .replace("%MONO%", FONT_MONO)
        .replace("%CONFIG%", data_json)
    )


def render_map_workspace(
    layers: list[MapLayer],
    height: int = 680,
    mode: str = "full",
    synthetic: bool = False,
) -> None:
    """Render the map workspace iframe."""
    st.components.v1.html(build_map_html(layers, mode=mode, synthetic=synthetic), height=height)


# Back-compat helper used by earlier sections/tests.
def render_flood_map(artifacts: Any, height: int = 440, mode: str = "preview", show_retrospective: bool = True) -> None:
    render_map_workspace(
        build_layers(artifacts, show_retrospective=show_retrospective),
        height=height, mode=mode, synthetic=getattr(artifacts, "is_synthetic", False),
    )


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@%LEAFLET%/dist/leaflet.css">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,300..500,0,0&display=block">
<script src="https://unpkg.com/leaflet@%LEAFLET%/dist/leaflet.js"></script>
<style>
:root{%CSSVARS%;--font:%FONT%;--mono:%MONO%}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg-app);color:var(--text);font:13px/20px var(--font);-webkit-font-smoothing:antialiased}
#map{position:absolute;inset:0;background:#0b111c}
.ico{font-family:'Material Symbols Rounded';font-size:18px;line-height:1;font-feature-settings:'liga';font-variation-settings:'FILL' 0,'wght' 400;display:inline-block;width:18px;height:18px;overflow:hidden;user-select:none}
.panel{position:absolute;z-index:500;background:rgba(17,27,42,.96);border:1px solid var(--border);border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,.35);backdrop-filter:blur(6px)}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;height:32px;min-width:32px;padding:0 8px;border-radius:8px;border:1px solid var(--border);background:rgba(23,35,53,.96);color:var(--text-secondary);cursor:pointer;font:500 12px/1 var(--font);transition:background .15s,color .15s}
.btn:hover{background:var(--surface-hover);color:var(--text)}
.btn:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.btn[aria-pressed=true]{color:var(--accent);border-color:rgba(56,189,248,.45)}
/* Layer panel */
#layers{top:12px;left:12px;width:280px;max-height:calc(100% - 24px);display:flex;flex-direction:column;transition:transform .18s cubic-bezier(.2,0,0,1),opacity .18s}
#layers.closed{transform:translateX(-8px);opacity:0;pointer-events:none}
.ph{display:flex;align-items:center;justify-content:space-between;padding:12px 12px 10px 14px;border-bottom:1px solid var(--border-subtle)}
.ph h2{margin:0;font-size:13px;font-weight:600}
.pbody{overflow-y:auto;padding:4px 0 8px}
.grp{padding:10px 14px 2px;font-size:11px;font-weight:500;color:var(--text-disabled)}
.row{padding:6px 14px}
.row label{display:flex;align-items:flex-start;gap:10px;cursor:pointer}
.row input[type=checkbox]{appearance:none;width:16px;height:16px;margin:2px 0 0;border:1px solid var(--border);border-radius:4px;background:var(--surface);flex:none;display:grid;place-items:center;cursor:pointer}
.row input[type=checkbox]:checked{background:var(--accent);border-color:var(--accent)}
.row input[type=checkbox]:checked::after{content:'';width:8px;height:4px;border:2px solid #04121D;border-top:0;border-right:0;transform:rotate(-45deg) translate(1px,-1px)}
.row input:disabled{opacity:.4;cursor:not-allowed}
.swatch{width:10px;height:10px;border-radius:3px;flex:none;margin-top:5px}
.lname{flex:1;min-width:0;color:var(--text);font-size:13px}
.lname small{display:block;font-size:11.5px;line-height:16px;color:var(--text-muted)}
.row.off .lname{color:var(--text-disabled)}
.row.off .lname small{color:var(--text-disabled)}
.tag{font-size:10.5px;line-height:16px;padding:0 6px;border-radius:5px;border:1px solid var(--border-subtle);color:var(--text-muted);white-space:nowrap;margin-top:1px}
.tag.model{color:var(--accent);border-color:rgba(56,189,248,.3)}
.tag.reference{color:var(--warning);border-color:rgba(251,191,36,.3)}
.tag.scenario{color:var(--scenario);border-color:rgba(167,139,250,.3)}
.tag.derived{color:var(--warning);border-color:rgba(251,191,36,.3)}
.opacity{display:flex;align-items:center;gap:8px;margin:6px 0 2px 26px;font-size:11px;color:var(--text-muted)}
.opacity input{flex:1;accent-color:var(--accent);height:16px}
.bm{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;padding:10px 14px 4px}
.bm button{height:28px;border-radius:7px;border:1px solid var(--border-subtle);background:var(--surface);color:var(--text-secondary);font:500 11.5px var(--font);cursor:pointer}
.bm button[aria-pressed=true]{background:rgba(56,189,248,.1);color:var(--accent);border-color:rgba(56,189,248,.4)}
/* Toolbar */
#tools{top:12px;right:12px;display:flex;flex-direction:column;gap:6px;z-index:600;transition:right .18s cubic-bezier(.2,0,0,1)}
body.detail-open #tools{right:336px}
#openLayers{position:absolute;top:12px;left:12px;z-index:600;display:none}
#layers.closed ~ #openLayers{display:inline-flex}
/* Detail panel */
#detail{top:12px;right:12px;width:312px;max-height:calc(100% - 24px);display:flex;flex-direction:column;transform:translateX(12px);opacity:0;pointer-events:none;transition:transform .18s cubic-bezier(.2,0,0,1),opacity .18s}
body.detail-open #detail{transform:none;opacity:1;pointer-events:auto}
#detail .pbody{padding:12px 14px 14px}
.dt-kind{display:flex;gap:6px;align-items:center;margin-bottom:6px}
.dt-title{font-size:14px;font-weight:600;line-height:20px;margin:0 0 4px}
.dt-desc{color:var(--text-secondary);font-size:12.5px;line-height:18px;margin:0 0 12px}
.kv{display:grid;grid-template-columns:42% 58%;font-size:12px;line-height:18px;border-top:1px solid var(--border-subtle)}
.kv div{padding:6px 0;border-bottom:1px solid var(--border-subtle);word-break:break-word}
.kv .k{color:var(--text-muted);padding-right:8px}
.kv .v{color:var(--text);font-variant-numeric:tabular-nums}
.note{margin-top:12px;padding:10px 12px;border-radius:8px;background:rgba(251,191,36,.06);border:1px solid rgba(251,191,36,.2);color:var(--text-secondary);font-size:12px;line-height:18px}
.note b{color:var(--warning);font-weight:600}
/* Legend & status */
#legend{left:12px;bottom:28px;padding:10px 12px;min-width:180px;max-width:260px;transition:left .18s}
body.layers-open #legend{left:304px}
#legend h3{margin:0 0 6px;font-size:11px;font-weight:500;color:var(--text-muted)}
.lg{display:flex;align-items:center;gap:8px;font-size:12px;line-height:20px;color:var(--text-secondary)}
.lg i{width:14px;height:10px;border-radius:2px;flex:none}
#empty{left:50%;top:50%;transform:translate(-50%,-50%);padding:16px 18px;max-width:320px;text-align:center;color:var(--text-secondary);display:none}
#fixture{left:50%;top:12px;transform:translateX(-50%);padding:5px 10px;font-size:11.5px;color:var(--warning);border-color:rgba(251,191,36,.35);display:none;white-space:nowrap}
#coords{right:56px;bottom:10px;padding:2px 8px;border-radius:6px;font:11px/18px var(--mono);color:var(--text-muted)}
.leaflet-control-zoom{border:1px solid var(--border)!important;border-radius:8px!important;overflow:hidden}
.leaflet-control-zoom a{background:rgba(23,35,53,.96)!important;color:var(--text-secondary)!important;border-bottom:1px solid var(--border)!important;width:32px!important;height:32px!important;line-height:30px!important}
.leaflet-control-zoom a:hover{background:var(--surface-hover)!important;color:var(--text)!important}
.leaflet-control-scale-line{background:rgba(17,27,42,.8);color:var(--text-muted);border-color:var(--text-muted);font:10.5px var(--font)}
.leaflet-control-attribution{background:rgba(9,14,24,.75)!important;color:var(--text-disabled)!important;font-size:10px}
.leaflet-control-attribution a{color:var(--text-muted)!important}
.leaflet-tooltip.pv{background:var(--surface-raised);border:1px solid var(--border);color:var(--text);font:12px var(--font);border-radius:6px;box-shadow:none;padding:4px 8px}
.leaflet-tooltip.pv::before{display:none}
body.preview #layers,body.preview #detail,body.preview #openLayers,body.preview #coords{display:none!important}
body.preview #legend{left:12px!important;bottom:12px}
@media (max-width:720px){
  #layers{width:calc(100% - 24px);max-height:60%}
  #detail{left:12px;right:12px;width:auto;top:auto;bottom:12px;max-height:55%;transform:translateY(12px)}
  body.detail-open #tools{right:12px}
  body.layers-open #legend{display:none}
  #coords{display:none}
  #fixture{top:auto;bottom:12px}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head>
<body>
<div id="map" role="application" aria-label="Flood evidence map"></div>
<section id="layers" class="panel" aria-label="Map layers">
  <div class="ph"><h2>Layers</h2><button class="btn" id="closeLayers" title="Hide layer panel" aria-label="Hide layer panel"><span class="ico">left_panel_close</span></button></div>
  <div class="pbody" id="layerBody"></div>
</section>
<button class="btn panel" id="openLayers" title="Show layers" aria-label="Show layer panel"><span class="ico">layers</span> Layers</button>
<div id="tools" class="panel" style="padding:4px;border-radius:10px">
  <button class="btn" id="fitAoi" title="Fit study area" aria-label="Fit study area"><span class="ico">crop_free</span></button>
  <button class="btn" id="resetView" title="Reset view" aria-label="Reset view"><span class="ico">restart_alt</span></button>
  <button class="btn" id="toggleLegend" title="Toggle legend" aria-label="Toggle legend" aria-pressed="true"><span class="ico">list</span></button>
</div>
<aside id="detail" class="panel" aria-label="Feature details" aria-live="polite">
  <div class="ph"><h2>Feature details</h2><button class="btn" id="closeDetail" aria-label="Close feature details" title="Close (Esc)"><span class="ico">close</span></button></div>
  <div class="pbody" id="detailBody"></div>
</aside>
<div id="legend" class="panel" aria-label="Legend"><h3>Legend</h3><div id="legendBody"></div></div>
<div id="empty" class="panel" role="status"></div>
<div id="fixture" class="panel" role="status">Synthetic fixture — not real observations</div>
<div id="coords" class="panel" aria-hidden="true">—</div>
<script>
const CFG = %CONFIG%;
const $ = (id) => document.getElementById(id);
const el = (tag, attrs = {}, text) => { const e = document.createElement(tag); for (const [k, v] of Object.entries(attrs)) { if (k === 'class') e.className = v; else e.setAttribute(k, v); } if (text !== undefined) e.textContent = text; return e; };
const preview = CFG.mode === 'preview';
if (preview) document.body.classList.add('preview'); else document.body.classList.add('layers-open');
if (CFG.synthetic) $('fixture').style.display = 'block';
if (preview) { ['resetView'].forEach(id => $(id).style.display = 'none'); }

const map = L.map('map', { zoomControl: false, scrollWheelZoom: !preview, attributionControl: true, preferCanvas: false })
  .setView(CFG.center, CFG.zoom);
L.control.zoom({ position: 'bottomright' }).addTo(map);
L.control.scale({ position: 'bottomright', imperial: false }).addTo(map);

let baseLayer = null;
function setBasemap(key) {
  const b = CFG.basemaps[key]; if (!b) return;
  if (baseLayer) map.removeLayer(baseLayer);
  baseLayer = L.tileLayer(b.url, { attribution: b.attr, subdomains: 'abc', maxZoom: 16 }).addTo(map);
  baseLayer.bringToBack();
  document.querySelectorAll('.bm button').forEach(btn => btn.setAttribute('aria-pressed', btn.dataset.key === key));
}

const state = {};
let selected = null;
function styleFor(spec, s) {
  return { color: spec.color, weight: spec.fill ? 1.25 : 2, opacity: Math.min(1, s.opacity + 0.35), dashArray: spec.dash || null,
           fillColor: spec.color, fillOpacity: spec.fill ? s.opacity : 0 };
}
function fmt(v) {
  if (v === null || v === undefined || v === '') return '—';
  if (typeof v === 'number') return Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}
function openDetail(spec, feature) {
  const body = $('detailBody'); body.replaceChildren();
  const kind = el('div', { class: 'dt-kind' });
  const sw = el('span', { class: 'swatch' }); sw.style.background = spec.color; sw.style.marginTop = '0';
  kind.append(sw, el('span', { class: 'tag ' + spec.kind }, spec.kind_label));
  body.append(kind, el('p', { class: 'dt-title' }, spec.name));
  if (spec.description) body.append(el('p', { class: 'dt-desc' }, spec.description));
  const kv = el('div', { class: 'kv' });
  const add = (k, v) => { kv.append(el('div', { class: 'k' }, k), el('div', { class: 'v' }, fmt(v))); };
  const props = (feature && feature.properties) || {};
  add('Feature type', (feature && feature.geometry && feature.geometry.type) || '—');
  add('Identifier', props.id ?? props.osm_id ?? props.label ?? props.name ?? '—');
  for (const [k, v] of Object.entries(props)) {
    if (['id', 'osm_id', 'label', 'name'].includes(k)) continue;
    add(k.replace(/_/g, ' '), v);
  }
  add('Source', spec.source);
  add('Timestamp', spec.timestamp);
  body.append(kv);
  if (spec.kind === 'derived') {
    body.append(el('p', { class: 'dt-desc', style: 'margin-top:12px' }, 'Exposure relationship: geometry intersects the mapped water extent. This is potential exposure, not observed damage.'));
  }
  if (spec.limitation) { const n = el('div', { class: 'note' }); n.append(el('b', {}, 'Limitation  '), document.createTextNode(spec.limitation)); body.append(n); }
  document.body.classList.add('detail-open');
}
function closeDetail() {
  document.body.classList.remove('detail-open');
  if (selected) { selected.layer.setStyle(styleFor(selected.spec, state[selected.spec.id])); selected = null; }
}
$('closeDetail').onclick = closeDetail;
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDetail(); });

// Build Leaflet layers
CFG.layers.forEach(spec => {
  const s = { on: spec.available && spec.visible, opacity: spec.opacity, layer: null };
  state[spec.id] = s;
  if (!spec.available) return;
  s.layer = L.geoJSON(spec.data, {
    style: () => styleFor(spec, s),
    onEachFeature: (f, lyr) => {
      const p = f.properties || {};
      lyr.bindTooltip(String(p.label || p.name || spec.name), { className: 'pv', sticky: true, direction: 'top' });
      if (!preview) lyr.on('click', (ev) => {
        L.DomEvent.stopPropagation(ev);
        if (selected) selected.layer.setStyle(styleFor(selected.spec, state[selected.spec.id]));
        selected = { layer: lyr, spec };
        lyr.setStyle({ weight: 3, color: '#F3F6FA', opacity: 1 });
        openDetail(spec, f);
      });
    }
  });
  if (s.on) s.layer.addTo(map);
});
map.on('click', closeDetail);

// Layer panel
function renderPanel() {
  const body = $('layerBody'); body.replaceChildren();
  body.append(el('div', { class: 'grp' }, 'Basemap'));
  const bm = el('div', { class: 'bm', role: 'group', 'aria-label': 'Basemap' });
  for (const [key, b] of Object.entries(CFG.basemaps)) {
    const btn = el('button', { type: 'button', 'data-key': key, 'aria-pressed': key === CFG.basemap }, b.label);
    btn.onclick = () => setBasemap(key); bm.append(btn);
  }
  body.append(bm);
  CFG.groups.forEach(g => {
    const specs = CFG.layers.filter(l => l.group === g); if (!specs.length) return;
    body.append(el('div', { class: 'grp' }, g));
    specs.forEach(spec => {
      const s = state[spec.id];
      const row = el('div', { class: 'row' + (spec.available ? '' : ' off') });
      const lab = el('label');
      const cb = el('input', { type: 'checkbox', 'aria-describedby': 'd-' + spec.id });
      cb.checked = s.on; cb.disabled = !spec.available;
      const sw = el('span', { class: 'swatch' });
      sw.style.background = spec.fill ? spec.color : 'transparent'; sw.style.border = '1.5px ' + (spec.dash ? 'dashed ' : 'solid ') + spec.color;
      if (!spec.available) { sw.style.background = 'transparent'; sw.style.borderColor = 'var(--text-disabled)'; }
      const name = el('span', { class: 'lname' }, spec.name);
      name.append(el('small', { id: 'd-' + spec.id }, spec.available ? spec.kind_label : spec.unavailable_reason));
      lab.append(cb, sw, name);
      if (spec.available && (spec.kind === 'reference' || spec.kind === 'scenario')) lab.append(el('span', { class: 'tag ' + spec.kind }, spec.kind === 'reference' ? 'Post-event' : 'Scenario'));
      row.append(lab);
      if (spec.available && spec.fill) {
        const op = el('div', { class: 'opacity' });
        op.append(el('span', {}, 'Opacity'));
        const rng = el('input', { type: 'range', min: '0', max: '100', value: String(Math.round(s.opacity * 100)), 'aria-label': spec.name + ' opacity' });
        rng.oninput = () => { s.opacity = rng.value / 100; s.layer.setStyle(styleFor(spec, s)); };
        op.append(rng); op.hidden = !s.on; row.append(op);
        cb.addEventListener('change', () => { op.hidden = !cb.checked; });
      }
      cb.addEventListener('change', () => {
        s.on = cb.checked;
        if (s.on) s.layer.addTo(map); else { map.removeLayer(s.layer); if (selected && selected.spec.id === spec.id) closeDetail(); }
        renderLegend(); updateEmpty();
      });
      body.append(row);
    });
  });
}
function renderLegend() {
  const body = $('legendBody'); body.replaceChildren();
  const on = CFG.layers.filter(l => state[l.id].on);
  if (!on.length) { body.append(el('div', { class: 'lg' }, 'No layers visible')); return; }
  on.forEach(l => {
    const row = el('div', { class: 'lg' });
    const sw = el('i');
    sw.style.background = l.fill ? l.color + '88' : 'transparent';
    sw.style.border = '1.5px ' + (l.dash ? 'dashed ' : 'solid ') + l.color;
    row.append(sw, document.createTextNode(l.name + (l.kind === 'reference' ? ' (post-event)' : '')));
    body.append(row);
  });
}
function updateEmpty() {
  const anyData = CFG.layers.some(l => l.available && l.id !== 'aoi');
  const box = $('empty');
  if (!anyData) {
    box.replaceChildren(el('div', { style: 'font-weight:600;color:var(--text);margin-bottom:4px' }, 'No mapped evidence yet'),
      document.createTextNode('Water polygons will appear after tile inference and stitching produce an artifact for this run.'));
    box.style.display = 'block';
  } else box.style.display = 'none';
}
function boundsOf(ids) {
  let b = null;
  ids.forEach(id => { const s = state[id]; if (s && s.layer) { const lb = s.layer.getBounds(); if (lb.isValid()) b = b ? b.extend(lb) : L.latLngBounds(lb.getSouthWest(), lb.getNorthEast()); } });
  return b;
}
function fitAoi(animate) {
  const b = boundsOf(['aoi']) || boundsOf(CFG.layers.map(l => l.id));
  if (b) map.fitBounds(b, { padding: preview ? [16, 16] : [40, 40], paddingTopLeft: (!preview && window.innerWidth > 720) ? [300, 40] : undefined, animate });
  else map.setView(CFG.center, CFG.zoom, { animate });
}
$('fitAoi').onclick = () => fitAoi(true);
$('resetView').onclick = () => { closeDetail(); fitAoi(true); };
$('toggleLegend').onclick = (e) => { const lg = $('legend'); const show = lg.style.display === 'none'; lg.style.display = show ? '' : 'none'; e.currentTarget.setAttribute('aria-pressed', show); };
$('closeLayers').onclick = () => { $('layers').classList.add('closed'); document.body.classList.remove('layers-open'); $('openLayers').focus(); };
$('openLayers').onclick = () => { $('layers').classList.remove('closed'); document.body.classList.add('layers-open'); };
if (!preview && window.innerWidth <= 720) { $('layers').classList.add('closed'); document.body.classList.remove('layers-open'); }
map.on('mousemove', e => { $('coords').textContent = e.latlng.lat.toFixed(4) + '°N  ' + e.latlng.lng.toFixed(4) + '°E'; });

setBasemap(CFG.basemap);
renderPanel(); renderLegend(); updateEmpty(); fitAoi(false);
</script></body></html>
"""
