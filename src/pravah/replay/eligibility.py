"""Strict historical availability gate: no data leakage across cutoff."""
from collections.abc import Mapping
from datetime import datetime
from typing import Any


def parse_time(value: str | datetime | None) -> datetime | None:
    if value is None or value == "":
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Replay timestamps must be timezone-aware")
    return dt


def eligible(acquired: str | datetime | None, available: str | datetime | None,
             cutoff: str | datetime | None) -> bool:
    """Both acquisition AND verified public availability must precede cutoff."""
    a, v, c = parse_time(acquired), parse_time(available), parse_time(cutoff)
    return a is not None and v is not None and c is not None and a <= c and v <= c


def classify_asset(asset: Mapping[str, Any], cutoff: str | datetime) -> str:
    acquired, available = asset.get("acquired_at_utc"), asset.get("available_at_utc")
    if acquired is None or available is None:
        return "excluded_unknown_availability"
    return "known_before_cutoff" if eligible(acquired, available, cutoff) else "retrospective_reveal"
