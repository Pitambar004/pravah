"""Tests for UI components — metric card and Folium map generation."""
from app.components.map_view import create_folium_map
from app.components.metric_card import render_metric_card
from app.fixtures.synthetic_sample import (
    SYNTHETIC_AOI_GEOJSON,
    SYNTHETIC_UNOSAT_GEOJSON,
    SYNTHETIC_WATER_GEOJSON,
)


def test_metric_card_html_rendering():
    html = render_metric_card(
        label="Water Area",
        value=34.8,
        unit="ha",
        subtext="Optical Inferred",
        badge="Live",
        badge_variant="success",
        as_html=True,
    )
    assert html is not None
    assert "pv-metric-card" in html
    assert "Water Area" in html
    assert "34.8" in html
    assert "ha" in html
    assert "pv-badge-success" in html


def test_folium_map_generation_with_layers():
    m = create_folium_map(
        water_geojson=SYNTHETIC_WATER_GEOJSON,
        aoi_geojson=SYNTHETIC_AOI_GEOJSON,
        unosat_geojson=SYNTHETIC_UNOSAT_GEOJSON,
        show_unosat=True,
    )
    html = m._repr_html_()
    assert html is not None
    assert len(html) > 5000
    assert "leaflet" in html.lower()


def test_folium_map_generation_empty():
    m = create_folium_map(water_geojson=None, aoi_geojson=None, unosat_geojson=None)
    html = m._repr_html_()
    assert html is not None
    assert len(html) > 1000
