import os
from pathlib import Path


DATA_ROOT = Path(
    os.getenv("PRAVAH_DATA_ROOT", "/data")
)


def inspect_storage():

    print("=" * 60)
    print("PRAVAH - FUZZBALL STORAGE INSPECTION")
    print("=" * 60)

    print(f"Storage location: {DATA_ROOT}")

    if not DATA_ROOT.exists():
        print("ERROR: Storage path does not exist.")
        return False

    if not DATA_ROOT.is_dir():
        print("ERROR: Storage path is not a directory.")
        return False

    print("\nStorage contents:")

    try:
        items = list(DATA_ROOT.iterdir())

        if len(items) == 0:
            print("Storage volume is empty.")

        for item in sorted(items):

            if item.is_dir():
                print(f"[DIRECTORY] {item.name}")

            elif item.is_file():

                size = item.stat().st_size

                size_mb = size / (1024 * 1024)

                print(
                    f"[FILE] {item.name} "
                    f"({size_mb:.2f} MB)"
                )

        print("\nStorage inspection successful.")

        return True

    except PermissionError:

        print("ERROR: Permission denied.")

        return False


if __name__ == "__main__":

    success = inspect_storage()

    if not success:
        raise SystemExit(1)