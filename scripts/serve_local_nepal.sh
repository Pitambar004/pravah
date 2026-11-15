#!/usr/bin/env bash
# Local Nepal dashboard + API without a prepared live_app config.
#
# Same two services as scripts/serve_nepal_replay.py (API :8006, dashboard :8506),
# but uses the standard inference engine on local paths instead of the
# Sen1Floods11 benchmark catalog in outputs/live_app/config.json.
# Scenes: data/raw/sentinel2/*.tif (prepared TOA reflectance, see
# scripts/prepare_s2_reflectance.py). Synthetic fixtures are disabled.
#
#   CHECKPOINT=outputs/local_demo/<run>/checkpoint_best.pt scripts/serve_local_nepal.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT="$(pwd)"
PY="${PYTHON:-$PROJECT/.venv/bin/python}"
API_PORT="${API_PORT:-8006}"
UI_PORT="${UI_PORT:-8506}"
if [[ -z "${CHECKPOINT:-}" ]]; then
  shopt -s nullglob
  candidates=(outputs/local_demo/*/checkpoint_best.pt outputs/models/*/checkpoint_best.pt)
  shopt -u nullglob
  CHECKPOINT="$( ((${#candidates[@]})) && ls -t "${candidates[@]}" | head -1 || true)"
fi
if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
  echo "No checkpoint found; set CHECKPOINT=..." >&2
  exit 1
fi
LOGS="$PROJECT/outputs/nepal_monitor"
mkdir -p "$LOGS"

export PYTHONPATH="$PROJECT/src:$PROJECT"
export PRAVAH_DATA_ROOT="$PROJECT/data"
export PRAVAH_SCENES_DIR="$PROJECT/data/raw/sentinel2"
export PRAVAH_OUTPUTS_DIR="$PROJECT/outputs"
export PRAVAH_CHECKPOINT="$PROJECT/$CHECKPOINT"
export PRAVAH_INPUT_UNITS=reflectance
export PRAVAH_DEVICE="${DEVICE:-auto}"
export PRAVAH_REQUIRE_CUDA=0
export PRAVAH_OSM_DIR="$PROJECT/data/processed/osm/thematic"
export PRAVAH_ENABLE_TEST_FIXTURES=0
export PRAVAH_RUN_SOURCE=local
export PRAVAH_API_URL="http://127.0.0.1:$API_PORT"
export PRAVAH_NEPAL_API_URL="http://127.0.0.1:$API_PORT"
unset PRAVAH_REAL_APP_CONFIG

"$PY" -m uvicorn scripts.serve_nepal_replay:api --factory --host 127.0.0.1 --port "$API_PORT" \
  >>"$LOGS/api.log" 2>&1 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT INT TERM
echo "API: $PRAVAH_API_URL (log $LOGS/api.log) · checkpoint $CHECKPOINT"
echo "Dashboard: http://127.0.0.1:$UI_PORT"
"$PY" -m streamlit run app/Home.py --server.address 127.0.0.1 --server.port "$UI_PORT" \
  --server.headless true --browser.gatherUsageStats false
