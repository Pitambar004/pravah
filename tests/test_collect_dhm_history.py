"""collect_dhm_history parsing/merging on SYNTHETIC provider snippets; no network."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("collect", Path(__file__).parent.parent / "scripts" / "collect_dhm_history.py")
collect = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collect)

CHART = ("<script>var river = '\\x5B\\x7B\"datetime\":\"2026-11-14T00:05:00+00:00\",\"value\":3.1\\x7D,"
         "\\x7B\"datetime\":\"2026-11-14T00:15:00+00:00\",\"value\":null\\x7D\\x5D';</script>")


def test_parse_river_decodes_escaped_json_and_drops_nulls():
    assert collect.parse_river(CHART) == [{"datetime_utc": "2026-11-14T00:05:00+00:00", "level_m": 3.1}]


def test_parse_river_refuses_changed_format():
    with pytest.raises(ValueError, match="format changed"):
        collect.parse_river("<script>nothing here</script>")


def test_merge_deduplicates_and_sorts(tmp_path):
    path = tmp_path / "1.csv"
    assert collect.merge(path, [{"datetime_utc": "b", "level_m": 2.0}, {"datetime_utc": "a", "level_m": 1.0}]) == (2, 2)
    assert collect.merge(path, [{"datetime_utc": "b", "level_m": 2.5}, {"datetime_utc": "c", "level_m": 3.0}]) == (1, 3)
    assert path.read_text().splitlines() == ["datetime_utc,level_m", "a,1.0", "b,2.5", "c,3.0"]
