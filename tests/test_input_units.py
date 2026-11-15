"""Input-unit guard for the inference engine. Rasters and weights are SYNTHETIC (random)."""
import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from pravah.inference.engine import InferenceEngine, check_input_units
from pravah.ml.unet import MiniUNet


def test_guard_rejects_mismatched_units():
    refl = np.random.default_rng(0).uniform(0.01, 0.4, (6, 64, 64)).astype(np.float32)
    with pytest.raises(ValueError, match="reflectance"):
        check_input_units(refl, "dn")
    check_input_units(refl, "reflectance")
    with pytest.raises(ValueError, match="DN"):
        check_input_units(refl * 10000, "reflectance")


def test_reflectance_and_dn_inputs_give_identical_maps(tmp_path):
    ck = tmp_path / "ck.pt"
    torch.manual_seed(0)
    torch.save({"model": MiniUNet().state_dict(), "architecture": "MiniUNet",
                "bands": ["B2", "B3", "B4", "B8A", "B11", "B12"]}, ck)
    refl = np.random.default_rng(1).uniform(0.01, 0.4, (6, 64, 64)).astype(np.float32)
    kw = {"driver": "GTiff", "width": 64, "height": 64, "count": 6, "crs": "EPSG:32645",
          "transform": from_origin(0, 0, 10, 10)}
    with rasterio.open(tmp_path / "r.tif", "w", dtype="float32", **kw) as d:
        d.write(refl)
    with rasterio.open(tmp_path / "d.tif", "w", dtype="float32", **kw) as d:
        d.write(refl * 10000)
    eng = InferenceEngine(ck, device="cpu")
    a = eng.predict(tmp_path / "r.tif", tmp_path / "ra", run_id="ra", input_units="reflectance")
    b = eng.predict(tmp_path / "d.tif", tmp_path / "db", run_id="db", input_units="dn")
    with rasterio.open(tmp_path / "ra/observed_water.tif") as x, rasterio.open(tmp_path / "db/observed_water.tif") as y:
        assert np.array_equal(x.read(1), y.read(1))
    assert a.metadata["input_units"] == "reflectance" and b.metadata["input_units"] == "dn"
    with pytest.raises(ValueError):
        eng.predict(tmp_path / "r.tif", tmp_path / "bad", run_id="bad", input_units="dn")
