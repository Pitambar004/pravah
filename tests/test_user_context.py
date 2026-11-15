"""Synthetic archive/number fixtures; never exported as observations."""

import hashlib
import json
import runpy
import stat
import zipfile
from pathlib import Path

import pytest

MODULE = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/import_user_context.py")
)


def test_archive_rejects_traversal_and_symlinks():
    for name in ["../outside.tif", "/absolute.tif", "a\\outside.tif"]:
        with pytest.raises(ValueError, match="unsafe"):
            MODULE["safe_member"](zipfile.ZipInfo(name))
    info = zipfile.ZipInfo("link.tif")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError, match="symlinks"):
        MODULE["safe_member"](info)


def test_incomplete_and_macos_metadata_are_never_imported():
    assert (
        MODULE["safe_member"](zipfile.ZipInfo("terrain/file.tif.part")) == "incomplete"
    )
    assert MODULE["safe_member"](zipfile.ZipInfo("__MACOSX/._file.tif")) == "metadata"
    assert MODULE["safe_member"](zipfile.ZipInfo("terrain/.DS_Store")) == "metadata"


def test_missing_weather_never_becomes_zero_rainfall():
    clean = MODULE["clean_number"]
    assert clean(None) == clean("") == clean(float("nan")) == clean(-999, -999) == ""
    assert clean(0, -999) == 0.0
    assert clean(-5, -999) == -5.0  # Valid subzero temperature, not a missing sentinel.


def test_nepal_time_conversion_preserves_date_boundary():
    assert (
        MODULE["utc_timestamp"]("2026-02-01T00:00", "Asia/Kathmandu")
        == "2026-01-31T18:15:00+00:00"
    )


def test_internal_manifest_checksum_is_enforced_before_publication(tmp_path):
    archive = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("sample/value.txt", b"wrong")
        z.writestr(
            "sample/manifest.json",
            json.dumps(
                {
                    "success": [
                        {
                            "path": "value.txt",
                            "sha256": hashlib.sha256(b"right").hexdigest(),
                            "bytes": 5,
                            "status": "downloaded",
                        }
                    ]
                }
            ),
        )
    with pytest.raises(ValueError, match="supplied manifest"):
        MODULE["extract_archive"](archive, tmp_path / "out")
    assert not (tmp_path / "out/sample/value.txt").exists()
