#!/usr/bin/env python3
"""Score a flood prediction against post-event satellite observations (Prediction vs Reality).

    python scripts/validate_prediction.py \
        --prediction outputs/pred_rasuwa_v1 \
        --reference outputs/p3_prithvi_nepal_20260827_T45RUL \
        --baseline  outputs/p3_prithvi_nepal_20260811_T45RUL \
        --target new_water --event-time 2026-08-26T05:15:00Z

The prediction folder must hold ``prediction_metadata.json`` and ``predicted_flood.tif``
(0/1/255) or ``predicted_flood_probability.tif`` (+ ``--threshold``). See
docs/PREDICTION_CONTRACT.md. Exit codes: 0 scored, 2 input error, 3 integrity rejection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pravah.analysis.prediction_validation import TARGETS, validate_prediction_run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prediction", required=True, type=Path, help="Prediction run folder")
    p.add_argument("--reference", required=True, type=Path, action="append",
                   help="Post-event observation run folder (repeat for several tiles)")
    p.add_argument("--baseline", type=Path, action="append", default=None,
                   help="Pre-event observation run on the same grid as each --reference (same order)")
    p.add_argument("--target", choices=TARGETS, default="water_extent")
    p.add_argument("--event-time", default=None, help="Documented event start, ISO-8601 UTC")
    p.add_argument("--threshold", type=float, default=None, help="For probability predictions only")
    a = p.parse_args(argv)
    try:
        s = validate_prediction_run(a.prediction, a.reference, a.baseline, a.target, a.event_time, a.threshold)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if s["status"] == "rejected_integrity":
        print("REJECTED — prediction is not cutoff-eligible:", file=sys.stderr)
        for v in s["integrity"]["violations"]:
            print(f"  - {v}", file=sys.stderr)
        return 3
    print(json.dumps({"status": s["status"], "integrity": s["integrity"], "combined": s.get("combined")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
