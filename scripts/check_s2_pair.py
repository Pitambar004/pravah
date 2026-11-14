#!/usr/bin/env python3
"""Print real Sentinel-2/label shape and class information (no invented data)."""
import argparse
import json
import numpy as np
import rasterio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--label", required=True)
    args = p.parse_args()
    with rasterio.open(args.image) as im, rasterio.open(args.label) as mask:
        if im.count != 13 or mask.count != 1:
            raise ValueError("Expected original 13-band S2 L1C chip and 1-band label")
        if im.crs != mask.crs or im.transform != mask.transform or im.shape != mask.shape:
            raise ValueError("Image/label grid misalignment")
        y = mask.read(1)
        if not np.isin(y, (-1, 0, 1)).all():
            raise ValueError("Unsupported label classes")
        print(json.dumps({"image_bands": im.count, "shape": list(im.shape),
                          "crs": str(im.crs), "label_classes": np.unique(y).tolist(),
                          "valid_pixels": int((y != -1).sum())}, indent=2))


if __name__ == "__main__":
    main()
