"""Synthetic clocks and configuration fixtures; no disaster measurements."""

import pytest

from pravah.replay.presentation import presentation_stage


def test_retained_replay_session_loads_images_without_resetting_its_cutoff(monkeypatch):
    from app.sections import judge_demo

    snapshot = {"prediction": None, "cutoff": "synthetic frozen cutoff"}
    demo = {"snapshot": snapshot, "started_monotonic": 17, "finished": False}
    images = {"is_synthetic": True, "pairs": {}}
    calls = []
    monkeypatch.setattr(judge_demo, "get_image_pairs", lambda: (calls.append("load") or images, None))
    assert judge_demo.replay_images(demo) == (images, None)
    assert judge_demo.replay_images(demo) == (images, None)
    assert calls == ["load"]
    assert demo["snapshot"] is snapshot and demo["started_monotonic"] == 17


def test_retained_replay_missing_images_can_recover_on_next_rerun(monkeypatch):
    from app.sections import judge_demo

    demo = {"snapshot": {"prediction": None}}
    monkeypatch.setattr(judge_demo, "get_image_pairs", lambda: (None, "Synthetic unavailable source"))
    assert judge_demo.replay_images(demo) == (None, "Synthetic unavailable source")
    assert "images" not in demo
    images = {"is_synthetic": True, "pairs": {}}
    monkeypatch.setattr(judge_demo, "get_image_pairs", lambda: (images, None))
    assert judge_demo.replay_images(demo) == (images, None)


@pytest.mark.parametrize("elapsed,phase,assets,evaluation", [
    (0, "history", False, False), (31.9, "history", False, False),
    (32, "glacier", False, False), (44, "lake", False, False),
    (56, "seismic", False, False), (66, "gauge", False, False),
    (78, "water", False, False),
    (88, "office", True, False), (100, "assets", True, False),
    (110, "evaluation", True, True), (120, "complete", True, True),
])
def test_two_minute_clock_keeps_exploratory_assets_and_later_evidence_gated(elapsed, phase, assets, evaluation):
    stage = presentation_stage(elapsed)
    assert stage["phase"] == phase
    assert stage["show_exploratory_assets"] is assets
    assert stage["evaluation_revealed"] is evaluation
    assert stage["forecast"] is None and stage["accuracy"] is None


def test_lead_up_visits_fifteen_dates_and_clock_finishes_without_wraparound():
    dates = {presentation_stage(second)["date"] for second in range(32)}
    assert len(dates) == 15 and min(dates) == "2026-08-11" and max(dates) == "2026-08-25"
    assert presentation_stage(120)["date"] == "2026-08-26"
    assert presentation_stage(180)["remaining_seconds"] == 0
    for value in [-1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            presentation_stage(value)


def test_carto_credential_is_bound_at_runtime_without_mutating_default_sources(monkeypatch):
    from app.components.map_view import BASEMAPS, configured_basemaps

    monkeypatch.setenv("PRAVAH_CARTO_BASEMAP_API_KEY", "synthetic test credential")
    maps = configured_basemaps()
    assert "rastertiles/dark_all/" in maps["dark"]["url"]
    assert "synthetic%20test%20credential" in maps["dark"]["url"]
    assert "?key=" not in BASEMAPS["dark"]["url"]
