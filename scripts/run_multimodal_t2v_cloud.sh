#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/jiazhuangzhuang/miniconda3/envs/aris/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/_tmp/multimodal_t2v_loop_cloud}"
DEVICE="${DEVICE:-cuda:0}"
MODEL_PATH="${WAN_T2V_MODEL_PATH:-/data/jiazhuangzhuang/Wan2.1-T2V-1.3B-Diffusers}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-/huggingface/model_hub/Qwen3.6-27B}"
USER_REQUEST="${USER_REQUEST:-A ceramic mug slowly rotates on a wooden workbench under warm lamp light}"
MAX_ROUNDS="${MAX_ROUNDS:-5}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-33}"
FPS="${FPS:-16}"
STEPS="${STEPS:-20}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
SEED="${SEED:-2026}"
DRY_RUN="${DRY_RUN:-0}"
VLM_EVAL="${VLM_EVAL:-1}"

cmd=(
  "$PYTHON_BIN" "$ROOT/scripts/run_multimodal_t2v_vlm_loop.py"
  --output-root "$OUTPUT_ROOT"
  --initial-request "$USER_REQUEST"
  --max-rounds "$MAX_ROUNDS"
  --python "$PYTHON_BIN"
  --device "$DEVICE"
  --generator wan-t2v
  --model-path "$MODEL_PATH"
  --vlm-model-path "$VLM_MODEL_PATH"
  --height "$HEIGHT"
  --width "$WIDTH"
  --num-frames "$NUM_FRAMES"
  --fps "$FPS"
  --steps "$STEPS"
  --guidance-scale "$GUIDANCE_SCALE"
  --seed "$SEED"
)

if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi
if [[ "$VLM_EVAL" == "0" ]]; then
  cmd+=(--no-vlm-eval)
fi

echo "[run_multimodal_t2v_cloud] ${cmd[*]}"
"${cmd[@]}"
