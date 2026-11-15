"""Synthetic numerical/selection fixtures, never exported as satellite evidence."""

import runpy
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

MODULE = runpy.run_path(str(Path(__file__).parents[1] / "scripts/train_local.py"))


def test_auto_selects_mac_gpu_when_available(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert MODULE["choose_device"]("auto").type == "mps"


def test_explicit_mac_gpu_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS is unavailable"):
        MODULE["choose_device"]("mps")


def test_class_balance_never_reads_validation_or_test_labels():
    rows = [
        {"split": "train", "water_pixels": 10, "valid_pixels": 100},
        {"split": "val", "water_pixels": 100000, "valid_pixels": 100000},
        {"split": "test", "water_pixels": 100000, "valid_pixels": 100000},
    ]
    weights, counts = MODULE["training_class_weights"](rows)
    assert weights == [1.0, 3.0]
    assert counts == {"water": 10, "non_water": 90, "valid": 100}


def test_evaluation_excludes_invalid_reference_pixels():
    class FixedModel(torch.nn.Module):
        def forward(self, image):
            return torch.tensor([[[[0.0, 1.0, 0.0]], [[1.0, 0.0, 1.0]]]])

    data = TensorDataset(torch.zeros(1, 6, 1, 3), torch.tensor([[[-1, 0, 1]]]))
    result = MODULE["evaluate"](FixedModel(), DataLoader(data), torch.device("cpu"))
    assert result["counts"] == {"tp": 1, "tn": 1, "fp": 0, "fn": 0, "valid": 2}


def test_nonfinite_predictions_fail_instead_of_becoming_dry_pixels():
    class BrokenModel(torch.nn.Module):
        def forward(self, image):
            return torch.full((1, 2, 1, 1), float("nan"))

    data = TensorDataset(
        torch.zeros(1, 6, 1, 1), torch.zeros(1, 1, 1, dtype=torch.int64)
    )
    with pytest.raises(RuntimeError, match="Nonfinite"):
        MODULE["evaluate"](BrokenModel(), DataLoader(data), torch.device("cpu"))
