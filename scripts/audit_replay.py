#!/usr/bin/env python3
"""Audit JSON manifest items for strict historical replay eligibility."""
import argparse
import json
from pathlib import Path
from pravah.replay.eligibility import classify_asset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", required=True)
    p.add_argument("--cutoff", required=True, help="ISO 8601 timezone-aware timestamp")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    items = json.loads(Path(args.assets).read_text())
    if not isinstance(items, list):
        raise ValueError("Expected list of asset objects")
    result = [{"asset_id": a.get("asset_id"), "eligibility": classify_asset(a, args.cutoff)} for a in items]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
