"""Alert grouping. All features and settlements here are SYNTHETIC fixtures."""
from pravah.gis.alerts import (
    UNASSIGNED,
    Settlement,
    draft_message,
    group_by_settlement,
    tier_for,
)


def _pt(lon, lat, **props):
    return {"type": "Feature", "properties": props, "geometry": {"type": "Point", "coordinates": [lon, lat]}}


def test_groups_by_nearest_settlement_with_range_limit():
    sets = [Settlement("A", 85.30, 28.10), Settlement("B", 85.40, 28.20), Settlement("nan", 85.31, 28.10)]
    exposure = {
        "buildings": {"features": [_pt(85.301, 28.101, osm_id="1"), _pt(85.302, 28.100, osm_id="2"),
                                   _pt(85.399, 28.199, osm_id="3"), _pt(86.5, 29.0, osm_id="4")]},
        "roads": {"features": [_pt(85.30, 28.10, clipped_length_m=120.5)]},
        "bridges": {"features": [_pt(85.40, 28.20)]},
    }
    areas = {a.settlement: a for a in group_by_settlement(exposure, sets, max_km=3)}
    assert areas["A"].buildings == 2 and areas["A"].road_m == 120.5
    assert areas["B"].buildings == 1 and areas["B"].bridges == 1
    assert areas[UNASSIGNED].buildings == 1  # far away, never forced onto a settlement
    assert "nan" not in areas


def test_tiers_and_message_are_labelled_simulated():
    assert tier_for(0) == "None" and tier_for(1) == "Advisory" and tier_for(10) == "Watch" and tier_for(50) == "Warning"
    areas = group_by_settlement({"buildings": {"features": [_pt(85.3, 28.1)]}}, [Settlement("A", 85.3, 28.1)])
    msg = draft_message(areas[0], "observed", "2026-08-27")
    assert "SIMULATED" in msg and "NOT AN OFFICIAL WARNING" in msg and "AFTER the event" in msg
