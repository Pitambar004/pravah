import os
from pathlib import Path


DATA_ROOT = Path(
    os.getenv("PRAVAH_DATA_ROOT", "/data")
)


DIRECTORIES = [
    "raw/sentinel1",
    "raw/sentinel2",
    "raw/labels",

    "processed/train",
    "processed/validation",
    "processed/test",

    "manifests",

    "models/checkpoints",

    "outputs/predictions",
    "outputs/evaluation"
]


def initialize_storage():

    print("Initializing Pravah storage...")

    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(
            f"Storage mount not found: {DATA_ROOT}"
        )

    for directory in DIRECTORIES:

        path = DATA_ROOT / directory

        path.mkdir(
            parents=True,
            exist_ok=True
        )

        print(f"Created: {path}")

    print("\nStorage initialization completed.")


if __name__ == "__main__":
    initialize_storage()