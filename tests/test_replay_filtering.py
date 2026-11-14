"""Tests for strict historical replay filtering and temporal gating in PRAVAH."""
import pytest

from pravah.replay.eligibility import classify_asset, eligible, parse_time

CUTOFF_NPT = "2026-08-26T11:00:00+05:45"
CUTOFF_UTC = "2026-08-26T05:15:00Z"


def test_time_parsing_timezone_awareness():
    # Valid timezone-aware ISO strings
    dt1 = parse_time(CUTOFF_NPT)
    assert dt1.tzinfo is not None
    assert dt1.year == 2026
    assert dt1.month == 8
    assert dt1.day == 26

    dt2 = parse_time(CUTOFF_UTC)
    assert dt2.tzinfo is not None

    # Blank and None handling
    assert parse_time(None) is None
    assert parse_time("") is None

    # Naive timestamp must be rejected
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_time("2026-08-26T11:00:00")


def test_strict_eligibility_both_timestamps_required():
    # Both before cutoff -> ELIGIBLE
    assert eligible(
        "2026-08-25T12:00:00Z",
        "2026-08-25T18:00:00Z",
        CUTOFF_UTC
    )

    # Acquired before, but made available AFTER cutoff -> INELIGIBLE
    assert not eligible(
        "2026-08-25T12:00:00Z",
        "2026-08-27T01:00:00Z",
        CUTOFF_UTC
    )

    # Acquired after cutoff -> INELIGIBLE
    assert not eligible(
        "2026-08-27T05:00:00Z",
        "2026-08-27T09:00:00Z",
        CUTOFF_UTC
    )

    # Missing availability timestamp -> INELIGIBLE
    assert not eligible("2026-08-25T12:00:00Z", None, CUTOFF_UTC)
    assert not eligible(None, "2026-08-25T12:00:00Z", CUTOFF_UTC)


def test_classify_asset_three_way_partition():
    # 1. Known before cutoff
    asset_pre = {
        "asset_id": "PRE_SCENE_01",
        "acquired_at_utc": "2026-08-20T05:00:00Z",
        "available_at_utc": "2026-08-20T08:00:00Z",
    }
    assert classify_asset(asset_pre, CUTOFF_UTC) == "known_before_cutoff"

    # 2. Retrospective reveal (acquired after or available after)
    asset_post = {
        "asset_id": "POST_SCENE_01",
        "acquired_at_utc": "2026-08-27T05:00:00Z",
        "available_at_utc": "2026-08-27T09:00:00Z",
    }
    assert classify_asset(asset_post, CUTOFF_UTC) == "retrospective_reveal"

    # 3. Excluded due to unknown availability
    asset_unverified = {
        "asset_id": "UNVERIFIED_SCENE_01",
        "acquired_at_utc": "2026-08-24T05:00:00Z",
        "available_at_utc": None,
    }
    assert classify_asset(asset_unverified, CUTOFF_UTC) == "excluded_unknown_availability"
