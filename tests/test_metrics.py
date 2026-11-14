import numpy as np

from pravah.ml.metrics import confusion, score


def test_ignore_void_and_score():
    y = np.array([[1, 0, -1], [1, 0, 0]])
    p = np.array([[1, 1, 1], [0, 0, 1]])
    counts = confusion(y, p)
    assert counts == {"tp": 1, "tn": 1, "fp": 2, "fn": 1, "valid": 5}
    metrics = score(counts)
    assert metrics["iou"] == 0.25
    assert metrics["precision"] == 1 / 3
    assert metrics["recall"] == 0.5


def test_zero_valid_handled():
    assert score(confusion(np.array([-1]), np.array([1])))["iou"] is None
