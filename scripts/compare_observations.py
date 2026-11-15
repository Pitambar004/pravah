#!/usr/bin/env python3
"""Pre-event vs post-event water change for two run folders on the same grid.

    python scripts/compare_observations.py --pre outputs/<pre_run> --post outputs/<post_run>

Writes change outputs into the post-event run folder (see pravah.analysis.change)
and prints the summary. Retrospective change analysis only; not a forecast.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pravah.analysis.change import compare_runs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pre", required=True, type=Path)
    p.add_argument("--post", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None, help="Default: the post run folder")
    a = p.parse_args(argv)
    try:
        s = compare_runs(a.pre, a.post, a.out)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in s.items() if k != "interpretation_limits"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
