"""Sen1Floods11 L1C raster pairs from a reviewed JSON pairs manifest.

Manifest format: [{"image_path":".../x_S2Hand.tif", "label_path":".../x_LabelHand.tif",
                   "split":"train|val|test", "event_id":"source_event_id"}]
The manifest is created from the actual official split files, never guessed.
"""
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

S2_INDEXES = (1, 2, 3, 8, 11, 12)


class S2WaterDataset(Dataset):
    def __init__(self, manifest_path: str, split: str):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        data = json.loads(Path(manifest_path).read_text())
        self.records = [r for r in data if r.get("split") == split]
        if not self.records:
            raise ValueError(f"No actual {split} records in manifest")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        with rasterio.open(r["image_path"]) as src:
            if src.count == 13:
                data = src.read([i + 1 for i in S2_INDEXES])
            elif src.count == 6:
                data = src.read()
            else:
                raise ValueError(f"Expected 13 or 6 S2 channels, got {src.count}")
            if src.crs is None:
                raise ValueError("Source image must be georeferenced")
        with rasterio.open(r["label_path"]) as mask_src:
            if mask_src.count != 1:
                raise ValueError("Expected single-band label")
            label = mask_src.read(1).astype(np.int64)
            if label.shape != data.shape[1:]:
                raise ValueError("Label and image shapes differ")
            if mask_src.crs is None:
                raise ValueError("Label missing CRS")
        if not np.isin(label, (-1, 0, 1)).all():
            raise ValueError("Unsupported label value")
        # Project-owned SMALL-U-NET baseline scaling, NOT Prithvi pretrained preprocessing.
        arr = np.clip(data.astype(np.float32) / 10000.0, 0.0, 1.0)
        return torch.from_numpy(arr), torch.from_numpy(label)
