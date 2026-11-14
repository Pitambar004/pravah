"""End-to-end Streamlit AppTest — verifying all sections in both demo and empty modes."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="Streamlit required for AppTest suite")
from streamlit.testing.v1 import AppTest

HOME_PATH = (Path(__file__).parent.parent / "app" / "Home.py").resolve()

PAGES = [
    "Overview",
    "Flood Evidence Map",
    "Historical Replay",
    "Infrastructure Exposure",
    "GPU Performance",
    "Sources & Methodology",
]


def test_home_page_renders():
    at = AppTest.from_file(HOME_PATH)
    at.run()
    assert not at.exception


@pytest.mark.parametrize("page", PAGES)
def test_section_renders_in_demo_mode(page):
    at = AppTest.from_file(HOME_PATH)
    at.run()
    at.sidebar.radio[0].set_value(page)
    at.run()
    assert not at.exception, f"{page} failed in demo mode: {at.exception}"


@pytest.mark.parametrize("page", PAGES)
def test_section_renders_in_empty_mode(page):
    at = AppTest.from_file(HOME_PATH)
    at.run()
    at.sidebar.radio[0].set_value(page)
    at.sidebar.toggle[0].set_value(False)
    at.run()
    assert not at.exception, f"{page} failed in empty mode: {at.exception}"
