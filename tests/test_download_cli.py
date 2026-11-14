import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from download_assets import canonical_uri, safe_filename


def test_prevent_path_traversal_and_strip_query():
    for name in ("../evil", "dir/evil", "\\foo", "", ".."): 
        with pytest.raises(ValueError): safe_filename(name)
    assert canonical_uri("https://example.org/data.tif?token=private#fragment")=="https://example.org/data.tif"
    with pytest.raises(ValueError):canonical_uri("file:///tmp/private")
