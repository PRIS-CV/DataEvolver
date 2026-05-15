#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${1:-$(pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$CODE_ROOT/pipeline/data/evolution_scene_v7_full20_rotation4_agent_round0_nostage35_20260404}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BLENDER_BIN="${BLENDER_BIN:-blender}"
PROFILE_PATH="${PROFILE_PATH:-$CODE_ROOT/configs/dataset_profiles/scene_v7_full20_loop.json}"
MESHES_DIR="${MESHES_DIR:-$CODE_ROOT/pipeline/data/meshes}"
TEMPLATE_PATH="${TEMPLATE_PATH:-$CODE_ROOT/configs/scene_template.json}"
RUNTIME_DIR="${RUNTIME_DIR:-$CODE_ROOT/runtime/full20_scene_20260404/render_tuner}"
DEVICE="${DEVICE:-cuda:0}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10}"
BENCHMARK_LIMIT="${BENCHMARK_LIMIT:-4}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"

cd "$CODE_ROOT"
mkdir -p "$RUNTIME_DIR"

DONE_MARKER="$RUNTIME_DIR/render_tuner_done.json"

read_done_status() {
  DONE_PATH="$DONE_MARKER" python3 - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ["DONE_PATH"])
if not p.exists():
    print("missing")
else:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        print(data.get("status", "unknown"))
    except Exception:
        print("unknown")
PY
}

archive_runtime_cycle() {
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  echo "[render-tuner] $(date '+%F %T') archive-runtime-cycle ts=$ts"
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
}

while true; do
  echo "[render-tuner] $(date '+%F %T') tick"
  if [[ -f "$DONE_MARKER" ]]; then
    status="$(read_done_status)"
    echo "[render-tuner] done-marker-exists status=$status"
    if [[ "$status" == "good_enough" ]]; then
      exit 0
    fi
    archive_runtime_cycle
    sleep "$SLEEP_SECONDS"
    continue
  fi

  if "$PYTHON_BIN" scripts/run_render_feedback_tuner.py \
    --root-dir "$SOURCE_ROOT" \
    --template-path "$TEMPLATE_PATH" \
    --profile "$PROFILE_PATH" \
    --meshes-dir "$MESHES_DIR" \
    --python-bin "$PYTHON_BIN" \
    --blender-bin "$BLENDER_BIN" \
    --device "$DEVICE" \
    --runtime-dir "$RUNTIME_DIR" \
    --max-iterations "$MAX_ITERATIONS" \
    --benchmark-limit "$BENCHMARK_LIMIT"; then
    :
  else
    rc=$?
    echo "[render-tuner] worker-failed:rc=$rc"
  fi

  sleep "$SLEEP_SECONDS"
done
