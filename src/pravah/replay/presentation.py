"""Two-minute presentation clock, separate from measured historical signals."""

import math
from datetime import date, timedelta

DURATION_SECONDS = 120
PHASES = [
    (0, "history", "15-day satellite evidence review", "catalog_context"),
    (32, "glacier", "Glacier image review", "retrospective_image_review"),
    (44, "lake", "Upstream lake-change watch", "exploratory_signal"),
    (56, "seismic", "Regional disturbance watch", "exploratory_signal"),
    (66, "gauge", "Gauge data-gap alert", "exploratory_signal"),
    (78, "water", "Unusually high river-level alert", "exploratory_signal"),
    (88, "office", "Rasuwa office warning", "exploratory_response"),
    (100, "assets", "Downstream infrastructure review", "computed_scenario"),
    (110, "evaluation", "Reveal later event evidence", "later_evaluation"),
    (120, "complete", "Replay complete", "presentation_complete"),
]


def presentation_stage(elapsed_seconds):
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("Presentation time must be finite and nonnegative")
    elapsed = min(elapsed_seconds, DURATION_SECONDS)
    start, phase, title, kind = next(row for row in reversed(PHASES) if row[0] <= elapsed)
    day_offset = min(14, int(elapsed / 32 * 15)) if phase == "history" else 15
    selected = date(2026, 8, 11) + timedelta(days=day_offset)
    return {"phase": phase, "title": title, "kind": kind, "phase_start_seconds": start,
            "date": selected.isoformat(), "elapsed_seconds": elapsed,
            "remaining_seconds": max(0, DURATION_SECONDS - math.floor(elapsed)),
            "progress": elapsed / DURATION_SECONDS,
            "show_exploratory_assets": elapsed >= 88,
            "evaluation_revealed": elapsed >= 110,
            "forecast": None, "accuracy": None}
