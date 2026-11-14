"""End-to-end Streamlit AppTest — every route in fixture mode and empty mode."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="Streamlit required for AppTest suite")
from streamlit.testing.v1 import AppTest

HOME_PATH = (Path(__file__).parent.parent / "app" / "Home.py").resolve()

PAGES = [
    "views/overview.py",
    "views/flood_map.py",
    "views/replay.py",
    "views/exposure.py",
    "views/compute.py",
    "views/methodology.py",
]


def _app(page: str, demo: bool) -> AppTest:
    at = AppTest.from_file(str(HOME_PATH), default_timeout=30)
    at.run()
    if not demo:
        at.sidebar.toggle[0].set_value(False)
        at.run()
    at.switch_page(page)
    at.run()
    return at


def _text(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def test_home_page_renders():
    at = AppTest.from_file(str(HOME_PATH), default_timeout=30)
    at.run()
    assert not at.exception


@pytest.mark.parametrize("page", PAGES)
def test_route_renders_with_fixtures(page):
    at = _app(page, demo=True)
    assert not at.exception, f"{page} failed in fixture mode: {at.exception}"
    assert "Synthetic fixture mode" in _text(at)


@pytest.mark.parametrize("page", PAGES)
def test_route_renders_empty(page):
    at = _app(page, demo=False)
    assert not at.exception, f"{page} failed in empty mode: {at.exception}"
    assert "Synthetic fixture" not in _text(at), f"{page} shows fixture content with fixtures disabled"


def test_empty_mode_never_shows_synthetic_scene_ids():
    """With fixtures off, replay must not fall back to synthetic scenes."""
    at = _app("views/replay.py", demo=False)
    text = _text(at)
    assert "S2A_MSIL1C_20260816" not in text
    assert "No source manifest available" in text


def test_strict_replay_hides_later_evidence():
    at = _app("views/replay.py", demo=True)
    text = _text(at)
    # Default cutoff 26 Aug 11:00 NPT: post-event scene and UNOSAT must be hidden.
    assert "S2B_MSIL1C_20260827T051219" not in text
    assert "UNOSAT_NEPAL_MUDFLOW_ROCKFLOW_EXTENT_AUG28" not in text
    # Pre-event baseline is eligible and shown.
    assert "S2A_MSIL1C_20260816T045701" in text
    # Unknown-availability event pass acquired before cutoff is shown as excluded.
    assert "S2A_MSIL1C_20260826T050122_EVENT_UNCONFIRMED_AVAILABILITY" in text
