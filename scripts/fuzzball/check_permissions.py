import os
from pathlib import Path


DATA_ROOT = Path(
    os.getenv("PRAVAH_DATA_ROOT", "/data")
)


def check_permissions():

    print("PRAVAH STORAGE PERMISSION CHECK")

    print("Current UID:", os.getuid())
    print("Current GID:", os.getgid())
    print("Supplementary GIDs:", os.getgroups())

    print("Storage:", DATA_ROOT)

    if not DATA_ROOT.exists():
        print("Storage does not exist.")
        return

    print(
        "Readable:",
        os.access(DATA_ROOT, os.R_OK)
    )

    print(
        "Writable:",
        os.access(DATA_ROOT, os.W_OK)
    )

    print(
        "Executable:",
        os.access(DATA_ROOT, os.X_OK)
    )


if __name__ == "__main__":
    check_permissions()