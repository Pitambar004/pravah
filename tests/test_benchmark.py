import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark import compare


def test_benchmark_paired_speedup():
    base = {"tile_manifest_sha256": "manifest", "model_sha256": "checkpoint", "tile_count": 100,
            "inference_seconds": 120, "cuda_verified": True, "nodes_used": 1, "node_ids": ["gb10-a"]}
    two = dict(base, inference_seconds=75, nodes_used=2, node_ids=["gb10-a", "gb10-b"])
    result = compare(base, two)
    assert result["speedup_compute"] == pytest.approx(1.6)
    assert result["parallel_efficiency"] == pytest.approx(0.8)
    two["tile_count"] = 200
    with pytest.raises(ValueError, match="mismatch"):
        compare(base, two)
