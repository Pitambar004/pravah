"""Fuzzball template rendering: structure checks only (not proof of v4.3.1 validity)."""
import importlib.util
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("render_workflow", ROOT / "scripts/fuzzball/render_workflow.py")
rw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rw)

TMPL = ROOT / "workflows/fuzzball"
IMG = ["--set", "API_IMAGE=docker://registry.example/pravah-api:test",
       "--set", "DASHBOARD_IMAGE=docker://registry.example/pravah-dashboard:test"]


def test_serve_renders_gpu_api_and_dashboard(tmp_path):
    out = tmp_path / "serve.yaml"
    assert rw.main([str(TMPL / "07_serve.yaml.tmpl"), *IMG, "--out", str(out),
                    "--env", "dashboard:PRAVAH_DASHBOARD_PASSCODE=x"]) == 0
    doc = yaml.safe_load(out.read_text())
    api, dash = doc["services"]["api"], doc["services"]["dashboard"]
    assert doc["version"] == "v4"
    assert api["resource"]["devices"]["nvidia.com/gpu"] == 1
    assert "devices" not in dash["resource"]  # dashboard needs no GPU
    assert "PRAVAH_REQUIRE_CUDA=1" in api["env"]
    assert "PRAVAH_API_URL=http://api.service:8000" in dash["env"]
    assert "PRAVAH_DASHBOARD_PASSCODE=x" in dash["env"]
    assert api["mounts"] == {"/data": {"volume": "dataset"}} and doc["volumes"]["dataset"]["name"] == "Pravah-Dataset"
    assert all(ep["scope"] == "group" for s in (api, dash) for ep in s["network"]["endpoints"])
    assert dash["depends-on"][0]["name"] == "api" and api["persist"] and dash["persist"]


def test_missing_values_and_bad_inputs_are_refused(tmp_path, capsys):
    assert rw.main([str(TMPL / "07_serve.yaml.tmpl"), "--out", str(tmp_path / "x.yaml")]) == 2
    assert "Missing values" in capsys.readouterr().err
    assert rw.main([str(TMPL / "07_serve.yaml.tmpl"), *IMG, "--set", "ENDPOINT_SCOPE=everyone",
                    "--out", str(tmp_path / "x.yaml")]) == 2
    assert rw.main([str(TMPL / "07_serve.yaml.tmpl"), "--set", "API_IMAGE=pravah-api",
                    "--set", "DASHBOARD_IMAGE=docker://r/d:t", "--out", str(tmp_path / "x.yaml")]) == 2
    assert rw.main([str(TMPL / "06_inference_batch.yaml.tmpl"), *IMG[:2], "--set", "RUN_ID=r1",
                    "--set", "SCENE_ID=../../etc/passwd.tif", "--out", str(tmp_path / "x.yaml")]) == 2
    assert not (tmp_path / "x.yaml").exists()


def test_batch_and_smoke_render(tmp_path):
    out = tmp_path / "b.yaml"
    assert rw.main([str(TMPL / "06_inference_batch.yaml.tmpl"), *IMG[:2], "--set", "RUN_ID=rasuwa_post_01",
                    "--set", "SCENE_ID=s2/scene.tif", "--out", str(out)]) == 0
    cmd = yaml.safe_load(out.read_text())["jobs"]["infer"]["command"]
    assert "--require-cuda" in cmd and "/data/raw/sentinel2/s2/scene.tif" in cmd
    assert rw.main([str(TMPL / "00_gpu_smoke.yaml.tmpl"), *IMG[:2], "--out", str(tmp_path / "s.yaml")]) == 0


def test_public_scope_is_flagged(tmp_path, capsys):
    assert rw.main([str(TMPL / "07_serve.yaml.tmpl"), *IMG, "--set", "ENDPOINT_SCOPE=public",
                    "--out", str(tmp_path / "p.yaml")]) == 0
    assert "WARNING" in capsys.readouterr().out
