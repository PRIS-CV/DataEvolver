#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${1:-$(pwd)}"
RUNTIME_DIR="${RUNTIME_DIR:-$CODE_ROOT/runtime/full20_scene_20260404/render_tuner}"
LOG_DIR="${LOG_DIR:-$CODE_ROOT/runtime/full20_scene_20260404/logs}"
SESSION_NAME="${SESSION_NAME:-full20_tuner}"
STALL_SECONDS="${STALL_SECONDS:-1200}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-120}"
TUNER_DAEMON="${TUNER_DAEMON:-$CODE_ROOT/scripts/run_render_feedback_tuner_daemon.sh}"
QUEUE_PATH="${QUEUE_PATH:-$CODE_ROOT/pipeline/data/regeneration_queue.json}"
EXPORT_DONE_MARKER="${EXPORT_DONE_MARKER:-$CODE_ROOT/pipeline/data/export_scene_v7_full20_rotation8_best_multiview_20260405_horizontal/_export_done.json}"

mkdir -p "$LOG_DIR"

latest_activity_epoch() {
  local latest=0
  local path
  for path in \
    "$RUNTIME_DIR/render_tuner_active.json" \
    "$(find "$RUNTIME_DIR" -path '*/reviews/*_trace.json' -type f 2>/dev/null | sort | tail -n 1)"; do
    [[ -n "${path:-}" ]] || continue
    [[ -f "$path" ]] || continue
    local ts
    ts="$(stat -c %Y "$path" 2>/dev/null || echo 0)"
    if [[ "$ts" -gt "$latest" ]]; then
      latest="$ts"
    fi
  done
  echo "$latest"
}

queue_has_active_jobs() {
  QUEUE_PATH="$QUEUE_PATH" python3 - <<'PY'
import json, os, pathlib, sys
p = pathlib.Path(os.environ["QUEUE_PATH"])
if not p.exists():
    sys.exit(1)
try:
    data = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    sys.exit(0)
active = [item for item in data if str(item.get("status") or "") in {"queued", "running", "smoke_failed", "promoted"}]
sys.exit(0 if active else 1)
PY
}

restart_tuner() {
  local reason="$1"
  echo "[tuner-watchdog] $(date '+%F %T') restart reason=$reason"
  tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
  tmux new-session -d -s "$SESSION_NAME" \
    "cd $CODE_ROOT && bash $TUNER_DAEMON . > $LOG_DIR/render_tuner.log 2>&1"
}

archive_and_restart_cycle() {
  local reason="$1"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  echo "[tuner-watchdog] $(date '+%F %T') archive_and_restart reason=$reason archive_ts=$ts"
  tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
  if [[ -d "$RUNTIME_DIR" ]]; then
    ARCHIVE_DST="${RUNTIME_DIR}_archive_${ts}" RUNTIME_SRC="$RUNTIME_DIR" python3 - <<'PY'
import os, shutil
src = os.environ["RUNTIME_SRC"]
dst = os.environ["ARCHIVE_DST"]
if os.path.isdir(src):
    shutil.move(src, dst)
PY
  fi
  mkdir -p "$RUNTIME_DIR"
  tmux new-session -d -s "$SESSION_NAME" \
    "cd $CODE_ROOT && bash $TUNER_DAEMON . > $LOG_DIR/render_tuner.log 2>&1"
}

while true; do
  if [[ -f "$RUNTIME_DIR/render_tuner_done.json" ]]; then
    status="$(DONE_PATH="$RUNTIME_DIR/render_tuner_done.json" python3 - <<'PY'
import json, os
p=os.environ["DONE_PATH"]
try:
    data=json.load(open(p,"r",encoding="utf-8"))
    print(data.get("status","unknown"))
except Exception:
    print("unknown")
PY
)"
    if [[ "$status" == "good_enough" ]]; then
      if [[ -f "$EXPORT_DONE_MARKER" ]] && ! queue_has_active_jobs; then
        echo "[tuner-watchdog] $(date '+%F %T') done-marker-good-enough-and-export-done"
        exit 0
      fi
      echo "[tuner-watchdog] $(date '+%F %T') stale-good-enough-marker-continue"
      archive_and_restart_cycle "stale_good_enough"
      sleep "$CHECK_INTERVAL_SECONDS"
      continue
    fi
    archive_and_restart_cycle "done_status_${status}"
    sleep "$CHECK_INTERVAL_SECONDS"
    continue
  fi

  if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    restart_tuner "session_missing"
    sleep "$CHECK_INTERVAL_SECONDS"
    continue
  fi

  now_epoch="$(date +%s)"
  latest_epoch="$(latest_activity_epoch)"
  if [[ "$latest_epoch" -gt 0 ]]; then
    age=$(( now_epoch - latest_epoch ))
    echo "[tuner-watchdog] $(date '+%F %T') latest_age_seconds=$age"
    if [[ "$age" -gt "$STALL_SECONDS" ]]; then
      restart_tuner "stalled_${age}s"
    fi
  else
    echo "[tuner-watchdog] $(date '+%F %T') no_activity_yet"
  fi

  sleep "$CHECK_INTERVAL_SECONDS"
done
