import pytest

from pravah.replay.eligibility import classify_asset, eligible, parse_time

CUTOFF = "2026-08-26T11:00:00+05:45"


def test_replay_gates_on_both_times():
    assert eligible("2026-08-25T12:00:00Z", "2026-08-26T01:00:00Z", CUTOFF)
    assert not eligible("2026-08-26T06:00:00Z", "2026-08-26T06:01:00Z", CUTOFF)
    assert not eligible("2026-08-24T01:00:00Z", "2026-08-27T01:00:00Z", CUTOFF)
    assert not eligible("2026-08-24T01:00:00Z", None, CUTOFF)


def test_replay_asset_must_have_availability():
    assert classify_asset({"acquired_at_utc": "2026-08-25T00:00:00Z"}, CUTOFF) == "excluded_unknown_availability"
    assert classify_asset({"acquired_at_utc": "2026-08-25T00:00:00Z", "available_at_utc": "2026-08-28T00:00:00Z"}, CUTOFF) == "retrospective_reveal"


def test_naive_time_rejected():
    with pytest.raises(ValueError):
        parse_time("2026-08-26T11:00:00")
