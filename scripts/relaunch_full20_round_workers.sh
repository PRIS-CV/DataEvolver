#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${1:-<run-root>"
ROOT_DIR="${CODE_ROOT}/pipeline/data/evolution_scene_v7_full20_rotation4_agent_round0_nostage35_20260404"
LOG_DIR="${CODE_ROOT}/runtime/full20_scene_20260404/logs"
PYTHON_BIN="${PYTHON_BIN:-python}"
BLENDER_BIN="${BLENDER_BIN:-blender}"
PROFILE="${CODE_ROOT}/configs/dataset_profiles/scene_v7_full20_loop.json"
SCENE_TEMPLATE="${CODE_ROOT}/configs/scene_template.json"
MESHES_DIR="${CODE_ROOT}/pipeline/data/meshes"
MONITOR_SCRIPT="${CODE_ROOT}/scripts/run_scene_agent_monitor.py"

mkdir -p "${LOG_DIR}"
cd "${CODE_ROOT}"

pkill -f "run_render_feedback_tuner.py" || true
pkill -f "run_render_feedback_tuner_daemon.sh" || true
pkill -f "run_render_feedback_tuner_watchdog.sh" || true
pkill -f "run_scene_agent_monitor.py --worker-mode --root-dir ./pipeline/data/evolution_scene_v7_full20_rotation4_agent_round0_nostage35_20260404" || true

tmux kill-session -t full20_rounds_g0 2>/dev/null || true
tmux kill-session -t full20_rounds_g1 2>/dev/null || true
tmux kill-session -t full20_rounds_g2 2>/dev/null || true

CODE_ROOT="${CODE_ROOT}" ROOT_DIR="${ROOT_DIR}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

from scripts.run_scene_agent_monitor import prepare_worker_shard_dir

root_dir = Path(os.environ["ROOT_DIR"])
log_root = root_dir / "_monitor_logs"
shard_root = root_dir / "_shards"

for gpu in ("0", "1", "2"):
    out = prepare_worker_shard_dir(shard_root, log_root, gpu)
    print(f"gpu_{gpu}: {out}")
PY

for gpu in 0 1 2; do
  tmux new -ds "full20_rounds_g${gpu}" \
    "cd '${CODE_ROOT}' && '${PYTHON_BIN}' '${MONITOR_SCRIPT}' \
      --worker-mode \
      --root-dir '${ROOT_DIR}' \
      --gpus '${gpu}' \
      --python '${PYTHON_BIN}' \
      --blender '${BLENDER_BIN}' \
      --meshes-dir '${MESHES_DIR}' \
      --profile '${PROFILE}' \
      --scene-template '${SCENE_TEMPLATE}' \
      --max-rounds 10 \
      --pass-sleep-seconds 30 \
      --worker-gpu '${gpu}' \
      --worker-shard-dir '${ROOT_DIR}/_monitor_logs/merged_worker_shards/gpu_${gpu}' \
      --worker-log-path '${ROOT_DIR}/_monitor_logs/monitor_gpu_${gpu}.json' \
      --allow-unsafe-single-gpu-review \
      --disable-asset-deprecation \
      --ignore-registry-blocks \
      > '${LOG_DIR}/rounds_g${gpu}.log' 2>&1"
done

tmux ls | grep "full20_rounds_" || true
