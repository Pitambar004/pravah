#!/usr/bin/env python3
"""Download verified real provider previews for current and Rasuwa image pairs."""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from pravah.configuration import configured_url
from pravah.replay.image_pairs import make_pair
from pravah.replay.imagery import fetch_preview

PROJECT = Path(__file__).resolve().parents[2]


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    ranges = {"current": (now-timedelta(days=25)).isoformat()+"/"+now.isoformat(),
              "event": "2026-08-20T00:00:00Z/2026-08-30T23:59:59Z"}
    documents = {}
    with httpx.Client(timeout=15, follow_redirects=True) as client:
        for key, period in ranges.items():
            response = client.post(configured_url("PRAVAH_CDSE_SEARCH_URL"), json={
                "collections": ["sentinel-2-l1c"], "bbox": [85.25,28.05,85.65,28.5],
                "datetime": period, "limit": 80, "sortby": [{"field":"datetime","direction":"desc"}]})
            response.raise_for_status()
            items = [f for f in response.json()["features"] if "R119_T45RUM" in f["id"]]
            items.sort(key=lambda f: f["properties"].get("datetime") or f["properties"]["start_datetime"])
            if key == "current":
                latest = items[-1]
                target = datetime.fromisoformat(latest["properties"]["datetime"].replace("Z", "+00:00"))-timedelta(days=2)
                earlier = [f for f in items if datetime.fromisoformat(f["properties"]["datetime"].replace("Z", "+00:00")) <= target]
                selected = {"current_before": earlier[-1], "current_latest": latest}
            else:
                before = [f for f in items if f["properties"]["datetime"] < "2026-08-26T00:00:00Z"]
                after = [f for f in items if f["properties"]["datetime"] > "2026-08-26T23:59:59Z"]
                selected = {"event_earlier": before[-2], "event_before": before[-1], "event_after": after[0]}
            for name, item in selected.items():
                documents[name] = fetch_preview(item, root, now, client)
                if not documents[name].get("preview_data_url"):
                    raise ValueError("Actual provider preview unavailable: " + item["id"])
                print("Downloaded real preview", name, item["id"], flush=True)
    pairs = {"current": make_pair(root, documents["current_before"], documents["current_latest"], "Latest vs at least two days earlier"),
             "pre_event": make_pair(root, documents["event_earlier"], documents["event_before"], "Two actual pre-flood acquisitions"),
             "post_event": make_pair(root, documents["event_before"], documents["event_after"], "Real pre/post Rasuwa acquisitions")}
    doc = {"is_synthetic": False, "retrieved_at_utc": now.isoformat(), "pairs": pairs,
           "selection": "Same MGRS tile 45RUM and relative orbit 119. Actual acquisition dates retained.",
           "scope": "Real provider visual previews; georeferenced six-band crops require separate preparation"}
    (root/"pairs.json").write_text(json.dumps(doc,indent=2)+'\n')
    return {name:{"before":pair["before"]["acquired_at_utc"],"current":pair["current"]["acquired_at_utc"],"review":pair["summary"]} for name,pair in pairs.items()}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=PROJECT/"data/processed/nepal_imagery")
    args=parser.parse_args()
    print(json.dumps(prepare(args.output),indent=2))
