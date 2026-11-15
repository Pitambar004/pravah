#!/usr/bin/env python3
"""Fuzzball/local API factory: real water segmentation plus audited Nepal replay.

  PRAVAH_REPLAY_ROOT=/data/pravah_replay/evidence \
  PRAVAH_REAL_APP_CONFIG=/data/pravah_replay/live/config.json \
  python -m uvicorn scripts.serve_nepal_replay:api --factory --host 0.0.0.0

Inference remains water segmentation. Missing forecast models remain missing.
"""

import json
import os
from pathlib import Path

from pravah.api.main import create_app
from pravah.api.settings import Settings
from pravah.replay.service import attach_replay_routes

PROJECT = Path(__file__).resolve().parents[1]


def api():
    settings = Settings.from_env()
    config_path = os.environ.get("PRAVAH_REAL_APP_CONFIG")
    if config_path:
        from pravah.integration.real_engine import RealDataEngine

        config = json.loads(Path(config_path).read_text())
        project = Path(config.get("project_root", PROJECT))
        app = create_app(settings, lambda s: RealDataEngine(s, project, config))
    else:
        app = create_app(settings)
    root = Path(
        os.environ.get("PRAVAH_REPLAY_ROOT", PROJECT / "outputs/nepal_replay/prepared")
    )
    return attach_replay_routes(app, root, settings.api_token)
