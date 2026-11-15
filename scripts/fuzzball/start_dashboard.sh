#!/bin/sh
# Start the PRAVAH dashboard behind a Fuzzball endpoint.
#
# Fuzzball "path" endpoints serve the app at https://endpoints.<domain>/endpoints/<id>/
# and (per the CIQ Jupyter example, which sets its base URL to that path) forward
# the prefix unchanged, so Streamlit must know its base path. The path is taken
# from, in order: PRAVAH_BASE_URL_PATH, FB_ENDPOINT_URL_DASHBOARD (injected by
# Fuzzball for an endpoint named "dashboard"), else "" (subdomain endpoint/local).
# VALIDATE ON TEAM 12 FUZZBALL v4.3.1: confirm the variable name and prefix handling.
set -eu

BASE="${PRAVAH_BASE_URL_PATH:-}"
if [ -z "$BASE" ] && [ -n "${FB_ENDPOINT_URL_DASHBOARD:-}" ]; then
  BASE=$(python -c 'import os, urllib.parse as u; print(u.urlparse(os.environ["FB_ENDPOINT_URL_DASHBOARD"]).path.strip("/"))')
fi
echo "PRAVAH dashboard: base path='/${BASE}' api=${PRAVAH_API_URL:-unset} outputs=${PRAVAH_OUTPUTS_DIR:-unset}"

exec python -m streamlit run app/Home.py \
  --server.address 0.0.0.0 \
  --server.port "${PRAVAH_DASHBOARD_PORT:-8501}" \
  --server.headless true \
  --server.baseUrlPath "$BASE"
