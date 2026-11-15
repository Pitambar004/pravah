"""pravah must import on GPU images without python-dotenv (env vars still configure URLs)."""
import importlib
import sys

import pytest


def test_imports_and_reads_env_without_dotenv(monkeypatch):
    monkeypatch.setitem(sys.modules, "dotenv", None)  # simulate: package not installed
    import pravah.configuration as cfg

    cfg = importlib.reload(cfg)
    assert cfg.load_dotenv is None
    monkeypatch.setenv("PRAVAH_TEST_URL", "https://example.org/{x}")
    assert cfg.configured_url("PRAVAH_TEST_URL", x="a") == "https://example.org/a"
    monkeypatch.delenv("PRAVAH_TEST_URL")
    with pytest.raises(ValueError, match="Missing endpoint"):
        cfg.configured_url("PRAVAH_TEST_URL_ABSENT")
    monkeypatch.delitem(sys.modules, "dotenv")
    importlib.reload(cfg)  # restore the real loader for other tests
