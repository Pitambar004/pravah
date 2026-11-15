#!/bin/sh
# Watch several Fuzzball workflows at once: status + latest log line, refreshed every INTERVAL seconds.
# Read-only (describe + log). Ctrl+C stops watching; the workflows keep running.
#
#   scripts/fuzzball/watch_runs.sh <workflow-id>:<job-name> [<workflow-id>:<job-name> ...]
#   e.g. scripts/fuzzball/watch_runs.sh 014ead45-...:download-world <id>:train-2node-ddp
FB="${FUZZBALL:-$HOME/bin/fuzzball}"
INTERVAL="${INTERVAL:-30}"
[ $# -ge 1 ] || { echo "usage: $0 <workflow-id>:<job-name> ..." >&2; exit 2; }
while true; do
  clear 2>/dev/null || true
  echo "PRAVAH Fuzzball runs — $(date -u +%H:%M:%SZ) (refresh ${INTERVAL}s, Ctrl+C to stop watching)"
  for item in "$@"; do
    wf="${item%%:*}"; job="${item#*:}"
    status=$("$FB" workflow describe "$wf" 2>&1 | awk '/^Status:/{print $2}')
    last=$("$FB" workflow log "$wf" "$job" -n 1 --timestamp 2>&1 | tail -1 | cut -c1-200)
    printf '\n%s  [%s]\n  %s\n' "$job" "${status:-unknown}" "$last"
  done
  sleep "$INTERVAL"
done
