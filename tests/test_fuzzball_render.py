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


def test_training_templates_render_with_matching_workload(tmp_path):
    one, two = tmp_path / "1.yaml", tmp_path / "2.yaml"
    common = [*IMG[:2], "--set", "GLOBAL_BATCH=16", "--set", "EPOCHS=2"]
    assert rw.main([str(TMPL / "05_train_1node.yaml.tmpl"), *common, "--set", "RUN_ID=gb10_1node_a",
                    "--out", str(one)]) == 0
    assert rw.main([str(TMPL / "05_train_2node_ddp.yaml.tmpl"), *common, "--set", "RUN_ID=gb10_2node_a",
                    "--out", str(two)]) == 0
    j1 = yaml.safe_load(one.read_text())["jobs"]["train-1node"]
    j2 = yaml.safe_load(two.read_text())["jobs"]["train-2node-ddp"]
    assert j2["multinode"] == {"nodes": 2, "implementation": "generic"}
    assert "--nnodes=2" in j2["script"] and "--global-batch-size 16" in j2["script"]
    assert "16" in j1["command"] and "--require-cuda" not in j1["command"] and "cuda" in j1["command"]
    assert j1["resource"]["devices"]["nvidia.com/gpu"] == 1 == j2["resource"]["devices"]["nvidia.com/gpu"]


def test_training_template_rejects_odd_batch_and_bad_pairs(tmp_path):
    base = [str(TMPL / "05_train_2node_ddp.yaml.tmpl"), *IMG[:2], "--set", "RUN_ID=x", "--out", str(tmp_path / "x")]
    assert rw.main([*base, "--set", "GLOBAL_BATCH=3"]) == 2
    assert rw.main([*base, "--set", "PAIRS=/etc/passwd"]) == 2


WORLD = "/data/training/expanded_20261114"


def _render(tmp_path, name, *sets):
    out = tmp_path / (name + ".yaml")
    args = [str(TMPL / name), *IMG, "--out", str(out)]
    for s in sets:
        args += ["--set", s]
    assert rw.main(args) == 0
    return yaml.safe_load(out.read_text())


def test_world_download_and_prepare_are_cpu_jobs_on_the_volume(tmp_path):
    dl = _render(tmp_path, "03_download_world.yaml.tmpl")["jobs"]["download-world"]
    prep = _render(tmp_path, "04_prepare_world.yaml.tmpl", f"PAIRS={WORLD}/manifests/optical_supervised.json")["jobs"]["prepare-world"]
    for job in (dl, prep):
        assert "devices" not in job["resource"] and job["mounts"] == {"/data": {"volume": "dataset"}}
    assert f'ROOT="{WORLD}"' in dl["script"] and "--retry-failed" in dl["script"] and "sha256sum -c" in dl["script"]
    assert "refusing" in dl["script"]  # never overwrite a different plan on the volume
    assert "--dry-run" in prep["script"] and "S2WaterDataset" in prep["script"]
    assert f"{WORLD}/manifests/optical_supervised.json" in prep["script"]


@pytest.mark.parametrize("bad", ["WORLD_ROOT=/tmp/x", "WORLD_ROOT=/data/../etc", "DOWNLOAD_WORKERS=0",
                                 "DOWNLOAD_WORKERS=9", "RETRY_FAILED=yes"])
def test_world_inputs_are_validated(tmp_path, bad):
    assert rw.main([str(TMPL / "03_download_world.yaml.tmpl"), *IMG, "--set", bad,
                    "--out", str(tmp_path / "x.yaml")]) == 2


def test_world_training_runs_share_the_same_workload(tmp_path):
    pairs = f"PAIRS={WORLD}/manifests/optical_supervised.json"
    one = _render(tmp_path, "05_train_1node.yaml.tmpl", pairs, "RUN_ID=world_1node_t", "EPOCHS=5", "MAX_STEPS=100")
    two = _render(tmp_path, "05_train_2node_ddp.yaml.tmpl", pairs, "RUN_ID=world_2node_t", "EPOCHS=5", "MAX_STEPS=100")
    cmd1 = " ".join(one["jobs"]["train-1node"]["command"])
    cmd2 = two["jobs"]["train-2node-ddp"]["script"]
    for flag in (f"--pairs {WORLD}/manifests/optical_supervised.json", "--global-batch-size 8", "--epochs 5",
                 "--max-steps 100"):
        assert flag in cmd1 and flag in " ".join(cmd2.split()), flag


def test_pinned_world_plan_matches_its_checksum():
    import gzip
    import hashlib

    plan = ROOT / "config/training/world_download_plan.json.gz"
    expected = (ROOT / "config/training/world_download_plan.json.sha256").read_text().split()[0]
    assert hashlib.sha256(gzip.decompress(plan.read_bytes())).hexdigest() == expected


SHA = "a" * 64


def test_bundle_mode_runs_repo_code_on_stock_image(tmp_path):
    uri, prefix = "fb://group/pravah/world/t/bundle.tar.gz", "fb://group/pravah/world/t"
    common = ["--bundle", uri, "--bundle-sha256", SHA, "--results-prefix", prefix]

    def render(name, *sets):
        out = tmp_path / (name + ".yaml")
        args = [str(TMPL / name), *IMG, *common, "--out", str(out)]
        for s in sets:
            args += ["--set", s]
        assert rw.main(args) == 0
        return yaml.safe_load(out.read_text())

    pairs = f"PAIRS={WORLD}/manifests/optical_supervised.json"
    dl = render("03_download_world.yaml.tmpl")
    one = render("05_train_1node.yaml.tmpl", pairs, "RUN_ID=world_1node_t")
    two = render("05_train_2node_ddp.yaml.tmpl", pairs, "RUN_ID=world_2node_t")
    for doc in (dl, one, two):
        job = next(iter(doc["jobs"].values()))
        assert job["image"]["uri"] == rw.BUNDLE_IMAGE and "/opt/pravah" not in job["script"]
        assert f'echo "{SHA}  /transfer/in/bundle.tar.gz" | sha256sum -c -' in job["script"]
        assert doc["volumes"]["transfer"]["ingress"][0]["source"]["uri"] == uri
        assert job["mounts"]["/transfer"] == {"volume": "transfer"}
    assert "rasterio==1.5.2" not in dl["jobs"]["download-world"]["script"]  # stdlib-only job
    s1 = one["jobs"]["train-1node"]["script"]
    assert "command" not in one["jobs"]["train-1node"] and "rasterio==1.5.2" in s1 and "$CODE/scripts/train_ddp.py" in s1
    assert one["volumes"]["transfer"]["egress"][0]["destination"]["uri"] == f"{prefix}/world_1node_t.tar.gz"
    s2 = two["jobs"]["train-2node-ddp"]["script"]
    assert 'exec_all "(cd' in s2 and "PYTHONPATH=$CODE/src torchrun" in s2  # every node installs + sees code


@pytest.mark.parametrize("bad", [["--bundle", "https://x/b.tar.gz", "--bundle-sha256", SHA],
                                 ["--bundle", "fb://group/pravah/b.zip", "--bundle-sha256", SHA],
                                 ["--bundle", "fb://group/pravah/b.tar.gz", "--bundle-sha256", "abc"],
                                 ["--bundle", "fb://group/pravah/b.tar.gz"]])
def test_bundle_mode_rejects_bad_inputs(tmp_path, bad):
    assert rw.main([str(TMPL / "03_download_world.yaml.tmpl"), *IMG, *bad, "--out", str(tmp_path / "x.yaml")]) == 2


def test_private_api_service_stages_checked_model(tmp_path):
    out = tmp_path / "api.yaml"
    ck = "fb://group/pravah/models/m/checkpoint_best.pt"
    assert rw.main([str(TMPL / "08_api_gpu.yaml.tmpl"), *IMG, "--bundle", "fb://group/pravah/b.tar.gz",
                    "--bundle-sha256", SHA, "--ingress", f"{ck}=checkpoint_best.pt",
                    "--set", f"CHECKPOINT_SHA256={'b' * 64}", "--out", str(out)]) == 0
    doc = yaml.safe_load(out.read_text())
    api = doc["services"]["api"]
    assert "endpoints" not in api["network"]  # private: reach it only via port-forward
    assert api["resource"]["devices"]["nvidia.com/gpu"] == 1 and api["image"]["uri"] == rw.BUNDLE_IMAGE
    assert [i["source"]["uri"] for i in doc["volumes"]["transfer"]["ingress"]][1] == ck
    assert f'echo "{"b" * 64}  /transfer/in/checkpoint_best.pt" | sha256sum -c -' in api["script"]
    assert "PRAVAH_INPUT_UNITS=reflectance" in api["env"]


@pytest.mark.parametrize("bad", [["--ingress", "https://x/c.pt=c.pt"], ["--ingress", "fb://g/c.pt=../c.pt"],
                                 ["--set", "CHECKPOINT_SHA256=xyz"]])
def test_private_api_rejects_bad_inputs(tmp_path, bad):
    args = [str(TMPL / "08_api_gpu.yaml.tmpl"), *IMG, "--bundle", "fb://group/pravah/b.tar.gz", "--bundle-sha256", SHA,
            "--set", f"CHECKPOINT_SHA256={'b' * 64}", *bad, "--out", str(tmp_path / "x.yaml")]
    assert rw.main(args) == 2
