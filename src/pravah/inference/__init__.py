"""Checkpoint-backed MiniUNet inference producing georeferenced water evidence.

This package serves an *existing trained* ``MiniUNet`` checkpoint (written by
``scripts/train_ddp.py``). It never initialises random weights for serving: if no
valid checkpoint is present, prediction is refused.
"""
from pravah.inference.engine import (
    CheckpointError,
    InferenceEngine,
    ModelInfo,
    resolve_device,
)

__all__ = ["CheckpointError", "InferenceEngine", "ModelInfo", "resolve_device"]
