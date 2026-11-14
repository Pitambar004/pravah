from pathlib import Path

from pravah.catalog.downloads import download_file


def test_failed_download_does_not_create_real_asset(tmp_path: Path):
    dest = tmp_path / "imaginary.tif"
    outcome = download_file("http://127.0.0.1:1/nonexistent", str(dest), retries=0, timeout_s=0.5)
    assert outcome["status"] == "failed"
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()
