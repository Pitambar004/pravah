from pathlib import Path
import os
import socket
import json
import time

# Use local project storage on Mac.
# Use /data when running inside Fuzzball.
DATA = Path(os.environ.get("PRAVAH_DATA_DIR", "./data"))

TEST_DIR = DATA / "verification"
TEST_DIR.mkdir(parents=True, exist_ok=True)

test_file = TEST_DIR / "storage_test.json"

result = {
    "hostname": socket.gethostname(),
    "timestamp": time.time(),
    "message": "PRAVAH storage verification",
    "storage_path": str(DATA.resolve()),
}

test_file.write_text(json.dumps(result, indent=2))

print("Storage path:", DATA.resolve())
print("Writable:", os.access(DATA, os.W_OK))
print("Test file:", test_file)
print("Readback:", test_file.read_text())

print("\nSUCCESS: Storage write/read test passed!")
