#!/usr/bin/env python3
"""Export actual major Nepal river centrelines with documented map simplification."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pyogrio

from pravah.replay.experiment import sha256

PROJECT = Path(__file__).resolve().parents[2]
MAJOR = re.compile(r"koshi|kosi|karnali|gandaki|narayani|mahakali|bagmati|rapti|babai|bheri|seti|trish|trisuli|marsyangdi|arun|tamor|tamakoshi|sunkoshi|mechi|kankai|kamala|कोशी|कोसी|कर्णाली|गण्डकी|गंडकी|नारायणी|महाकाली|बागमती|राप्ती|बबई|भेरी|सेती|त्रिश|त्रिस|मर्स्याङ्दी|अरुण|तमोर|तामाकोशी|सुनकोशी|मेची|कन्काई|कमला", re.IGNORECASE)


def main():
    raw = PROJECT / "data/raw/nepal_monitor"
    pbf = raw / "nepal.osm.pbf"
    receipt = json.loads((raw / "receipt.json").read_text())
    if sha256(pbf) != receipt["sha256"]:
        raise ValueError("National source checksum mismatch")
    frame = pyogrio.read_dataframe(pbf, layer="lines", where="waterway='river'")
    labels = frame["name"].fillna("") + " " + frame["other_tags"].fillna("")
    frame = frame[labels.map(lambda s: bool(MAJOR.search(s)))].copy()
    frame = frame.to_crs("EPSG:32645")
    frame.geometry = frame.geometry.simplify(30, preserve_topology=True)
    frame = frame.to_crs("EPSG:4326")
    frame = frame[["osm_id", "name", "waterway", "geometry"]]
    root = PROJECT / "data/processed/nepal_national"; root.mkdir(parents=True, exist_ok=True)
    path = root / "major_rivers.geojson"; path.write_text(frame.to_json(drop_id=True) + "\n")
    metadata = {"is_synthetic": False, "file": path.name, "sha256": sha256(path), "features": len(frame),
                "source": receipt, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "export_crs": "EPSG:4326",
                "selection": "Named major rivers from actual OSM waterway=river records", "simplification_m": 30,
                "limitation": "Mapped centrelines simplified for country view; not measured width, connected flow routing or complete hydrological coverage."}
    (root / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
