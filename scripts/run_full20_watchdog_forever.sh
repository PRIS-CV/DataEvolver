#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${1:-$(pwd)}"
SOURCE_ROOT="${2:-$CODE_ROOT/pipeline/data/evolution_scene_v7_full20_rotation4_agent_round0_nostage35_20260404}"
EXPORT_DIR="${3:-$CODE_ROOT/pipeline/data/export_scene_v7_full20_rotation8_best_multiview_20260405_horizontal}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BLENDER_BIN="${BLENDER_BIN:-blender}"
PROFILE_PATH="${PROFILE_PATH:-$CODE_ROOT/configs/dataset_profiles/scene_v7_full20_loop.json}"
SCENE_TEMPLATE_PATH="${SCENE_TEMPLATE_PATH:-$CODE_ROOT/configs/scene_template.json}"
SCENE_GPUS="${SCENE_GPUS:-0,2,1}"
WATCHDOG_SLEEP_SECONDS="${WATCHDOG_SLEEP_SECONDS:-15}"
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-5}"
LOG_DIR="${LOG_DIR:-$CODE_ROOT/runtime/full20_scene_20260404/logs}"

mkdir -p "$LOG_DIR"
cd "$CODE_ROOT"

while true; do
  echo "[watchdog-supervisor] $(date '+%F %T') starting watchdog"
  "$PYTHON_BIN" scripts/run_full20_watchdog.py \
    --root-dir "$SOURCE_ROOT" \
    --export-dir "$EXPORT_DIR" \
    --python "$PYTHON_BIN" \
    --blender "$BLENDER_BIN" \
    --profile "$PROFILE_PATH" \
    --scene-template "$SCENE_TEMPLATE_PATH" \
    --scene-gpus "$SCENE_GPUS" \
    --sleep-seconds "$WATCHDOG_SLEEP_SECONDS"
  rc=$?
  echo "[watchdog-supervisor] $(date '+%F %T') watchdog exited rc=$rc"
  sleep "$RESTART_DELAY_SECONDS"
done
