import numpy as np
import pytest

from pravah.gis.tiles import fingerprint, make_tiles, partition


def test_full_grid_coverage_and_no_partition_duplicates():
    tiles = make_tiles(57, 41, tile_size=32, overlap=8)
    grid = np.zeros((41, 57), dtype=np.int32)
    for tile in tiles:
        row, col = tile["row_off"], tile["col_off"]
        grid[row: row + tile["height"], col: col + tile["width"]] += 1
    assert np.all(grid >= 1)
    one = partition(tiles, 0, 2)
    two = partition(tiles, 1, 2)
    assert {t["tile_id"] for t in one}.isdisjoint({t["tile_id"] for t in two})
    assert len(one) + len(two) == len(tiles)
    assert fingerprint(tiles) == fingerprint(tiles)


def test_invalid_tiling_rejected():
    with pytest.raises(ValueError):
        make_tiles(12, 12, 16, 16)
    with pytest.raises(ValueError):
        partition([], 2, 2)
