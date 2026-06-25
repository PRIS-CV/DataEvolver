#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/jiazhuangzhuang/miniconda3/envs/aris/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/_tmp/multimodal_t2i_dataset_cloud}"
DEVICE="${DEVICE:-cuda:0}"
MODEL_PATH="${QWEN_IMAGE_MODEL_PATH:-/data/wuwenzhuo/Qwen-Image-2512}"
USER_REQUEST="${USER_REQUEST:-Generate clean product-style daily object images on simple backgrounds}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
STEPS="${STEPS:-8}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
PROMPT_FILE="${PROMPT_FILE:-}"
VLM_EVAL="${VLM_EVAL:-0}"
VLM_MAX_SAMPLES="${VLM_MAX_SAMPLES:-1}"

cmd=(
  "$PYTHON_BIN" "$ROOT/pipeline/multimodal/run_multimodal_dataset.py"
  --route t2i
  --user-request "$USER_REQUEST"
  --output-root "$OUTPUT_ROOT"
  --generator qwen-image
  --allow-model-inference
  --model-path "$MODEL_PATH"
  --device "$DEVICE"
  --height "$HEIGHT"
  --width "$WIDTH"
  --steps "$STEPS"
  --num-samples "$NUM_SAMPLES"
)

if [[ -n "$PROMPT_FILE" ]]; then
  cmd+=(--prompt-file "$PROMPT_FILE")
fi
if [[ "$VLM_EVAL" == "1" ]]; then
  cmd+=(--vlm-eval --vlm-device "$DEVICE" --vlm-max-samples "$VLM_MAX_SAMPLES")
fi

echo "[run_multimodal_t2i_cloud] ${cmd[*]}"
"${cmd[@]}"
