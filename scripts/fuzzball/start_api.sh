#!/bin/sh
# Run from the verified API bundle. Never install into the training environment.
set -eu
cd "${PRAVAH_CODE_ROOT:?Missing verified bundle directory}"
sha256sum -c --quiet SHA256SUMS
python -c 'import platform, sys; assert platform.machine() == "aarch64"; assert sys.version_info[:2] == (3, 12)'
python -m venv --system-site-packages .venv-api
.venv-api/bin/python -m pip install --disable-pip-version-check --no-index --no-deps --ignore-installed wheels/*.whl
export PYTHONPATH="$PRAVAH_CODE_ROOT/src"
export PRAVAH_CHECKPOINT="$PRAVAH_CODE_ROOT/assets/checkpoint.pt"
export PRAVAH_ENV_FILE="$PRAVAH_CODE_ROOT/.env.example"
mkdir -p "$PRAVAH_SCENES_DIR" "$PRAVAH_OUTPUTS_DIR"
if [ -d assets/scenes ]; then
  # A unique content-addressed subdirectory; no overwrite of a user's scene.
  SCENE_SET=$(sha256sum assets/scenes/*.tif | sha256sum | cut -c 1-16)
  DEST="$PRAVAH_SCENES_DIR/bundled-$SCENE_SET"
  mkdir -p "$DEST"
  for FILE in assets/scenes/*; do
    if [ -e "$DEST/$(basename "$FILE")" ]; then
      cmp "$FILE" "$DEST/$(basename "$FILE")"
    else
      cp "$FILE" "$DEST/"
    fi
  done
fi
.venv-api/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA required"; print("GPU:", torch.cuda.get_device_name(0), "torch:", torch.__version__, flush=True)'
# One process owns one GPU and serializes the bounded inference queue.
exec .venv-api/bin/python -m uvicorn --factory pravah.api.main:create_app \
  --host 0.0.0.0 --port 8000 --workers 1 --no-access-log
