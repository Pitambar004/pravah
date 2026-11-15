"""Environment configuration shared by the API, UI and preparation commands."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_environment():
    """Session variables take precedence over private and example settings."""
    private = Path(os.environ.get("PRAVAH_ENV_FILE", PROJECT_ROOT / ".env"))
    load_dotenv(private, override=False)
    load_dotenv(PROJECT_ROOT / ".env.example", override=False)


def configured_url(name, **parts):
    load_environment()
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError("Missing endpoint configuration: " + name)
    return value.format(**parts) if parts else value
