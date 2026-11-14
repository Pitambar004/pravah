"""Interactive Geospatial Folium Map for PRAVAH Design System."""
from typing import Any

import folium
import streamlit as st
from folium import plugins

RASUWA_CENTER = [28.150, 85.350]


def create_folium_map(
    water_geojson: dict[str, Any] | None = None,
    aoi_geojson: dict[str, Any] | None = None,
    unosat_geojson: dict[str, Any] | None = None,
    center: list[float] = RASUWA_CENTER,
    zoom_start: int = 12,
    show_unosat: bool = False,
) -> folium.Map:
    """Create a standardized Folium map matching the PRAVAH design tokens."""
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True,
    )

    # 1. Base Tile Layers
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://carto.com/">CARTO</a> dark_matter',
        name="CartoDB Dark",
        subdomains="abcd",
        show=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite (Esri)",
        show=False,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap contributors",
        name="OpenStreetMap",
        show=False,
    ).add_to(m)

    # 2. Study Area Corridor (Core AOI)
    if aoi_geojson and aoi_geojson.get("features"):
        folium.GeoJson(
            aoi_geojson,
            name="📍 Study Boundary (Core AOI)",
            style_function=lambda _: {
                "fillColor": "transparent",
                "color": "#38BDF8",
                "weight": 2,
                "dashArray": "5, 5",
            },
        ).add_to(m)

    # 3. Retrospective UNOSAT Layer (Published Aug 28, 2026 - Post-Event Context)
    if unosat_geojson and unosat_geojson.get("features"):
        folium.GeoJson(
            unosat_geojson,
            name="🟠 UNOSAT Affected Extent [Retrospective]",
            show=show_unosat,
            style_function=lambda _: {
                "fillColor": "#FBBF24",
                "color": "#D97706",
                "weight": 1.5,
                "fillOpacity": 0.28,
            },
            popup=folium.GeoJsonPopup(
                fields=["classification", "published_date", "note"]
                if "classification" in unosat_geojson.get("features", [{}])[0].get("properties", {})
                else [],
                aliases=["Classification:", "Published:", "Note:"],
                style="background-color: #111827; color: #F1F5F9; font-family: sans-serif; font-size: 11px; padding: 6px;",
            ),
        ).add_to(m)

    # 4. Model-Inferred Optical Water Layer
    if water_geojson and water_geojson.get("features"):
        folium.GeoJson(
            water_geojson,
            name="💧 Model Inferred Water Extent",
            show=True,
            style_function=lambda _: {
                "fillColor": "#38BDF8",
                "color": "#0284C7",
                "weight": 1.5,
                "fillOpacity": 0.65,
            },
            popup=folium.GeoJsonPopup(
                fields=["label", "area_m2"]
                if "area_m2" in water_geojson.get("features", [{}])[0].get("properties", {})
                else [],
                aliases=["Feature:", "Area (m²):"],
                style="background-color: #111827; color: #38BDF8; font-family: sans-serif; font-size: 11px; padding: 6px;",
            ),
        ).add_to(m)

    # 5. Clean Embedded Legend matching #151F30 / #263244
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 24px; 
        right: 24px; 
        width: 220px; 
        background-color: rgba(21, 31, 48, 0.94);
        border: 1px solid #263244;
        border-radius: 6px;
        padding: 10px 12px;
        color: #F1F5F9;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 11px;
        z-index: 9999;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    ">
        <div style="font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8; font-size: 10px; margin-bottom: 6px;">
            Evidence Layers
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <span style="display: inline-block; width: 12px; height: 12px; background: rgba(56, 189, 248, 0.65); border: 1px solid #0284C7; margin-right: 8px; border-radius: 2px;"></span>
            <span>Inferred Water</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <span style="display: inline-block; width: 12px; height: 12px; background: rgba(251, 191, 36, 0.28); border: 1px solid #D97706; margin-right: 8px; border-radius: 2px;"></span>
            <span>UNOSAT Extent (Retrospective)</span>
        </div>
        <div style="display: flex; align-items: center;">
            <span style="display: inline-block; width: 12px; height: 12px; border: 1.5px dashed #38BDF8; margin-right: 8px; border-radius: 2px;"></span>
            <span>Study Boundary</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Controls
    folium.LayerControl(position="topright", collapsed=True).add_to(m)
    plugins.Fullscreen(position="topleft").add_to(m)
    return m


def render_flood_map(
    water_geojson: dict[str, Any] | None = None,
    aoi_geojson: dict[str, Any] | None = None,
    unosat_geojson: dict[str, Any] | None = None,
    height: int = 540,
    show_unosat: bool = False,
):
    """Render Folium map inside a styled border container."""
    m = create_folium_map(
        water_geojson=water_geojson,
        aoi_geojson=aoi_geojson,
        unosat_geojson=unosat_geojson,
        show_unosat=show_unosat,
    )
    st.components.v1.html(m._repr_html_(), height=height)
