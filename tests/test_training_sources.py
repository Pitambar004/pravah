"""Training page discovery of local-trainer runs; SYNTHETIC tmp-dir files only."""
import json

from app import runs
from app.data_loader import load_training_run


def _run(d, status, test_iou=None):
    d.mkdir(parents=True)
    ev = {"status": status, "optimizer_steps": 10}
    if test_iou is not None:
        ev["official_test"] = {"metrics": {"iou": test_iou, "f1": 0.9, "precision": 0.9, "recall": 0.9}}
    (d / "final_evaluation.json").write_text(json.dumps(ev))
    (d / "train_metrics.csv").write_text("epoch,step,mean_training_loss,elapsed_seconds\n1,5,0.4,2.0\n")


def test_any_finished_status_is_completed(tmp_path):
    _run(tmp_path / "pilot", "finished_validation_only")
    run = load_training_run(tmp_path / "pilot")
    assert run["status"] == "completed"
    assert run["metrics"][0]["loss_rank0"] == 0.4  # local-trainer columns are aliased


def test_local_trainer_folder_is_discovered(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    _run(out / "local_demo" / "unet_x", "finished", test_iou=0.8)
    monkeypatch.setenv("PRAVAH_OUTPUTS_DIR", str(out))
    for var in ("PRAVAH_TRAINING_DIR", "PRAVAH_MODELS_DIR", "PRAVAH_LOCAL_MODELS_DIR"):
        monkeypatch.delenv(var, raising=False)
    found = runs.training_runs()
    assert [r["label"] for r in found] == ["unet_x"] and found[0]["status"] == "completed"
