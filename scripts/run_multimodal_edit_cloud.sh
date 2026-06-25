#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/jiazhuangzhuang/miniconda3/envs/aris/bin/python}"
INPUT_IMAGE_DIR="${INPUT_IMAGE_DIR:?INPUT_IMAGE_DIR is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/_tmp/multimodal_edit_dataset_cloud}"
DEVICE="${DEVICE:-cuda:0}"
MODEL_PATH="${QWEN_IMAGE_EDIT_MODEL_PATH:-/data/wuwenzhuo/Qwen-Image-Edit-2511}"
USER_REQUEST="${USER_REQUEST:-Edit the main object while preserving background and layout}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
STEPS="${STEPS:-40}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
PROMPT_FILE="${PROMPT_FILE:-}"
VLM_EVAL="${VLM_EVAL:-0}"
VLM_MAX_SAMPLES="${VLM_MAX_SAMPLES:-1}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-/huggingface/model_hub/Qwen3.6-27B}"

cmd=(
  "$PYTHON_BIN" "$ROOT/pipeline/multimodal/run_multimodal_dataset.py"
  --route edit
  --user-request "$USER_REQUEST"
  --input-image-dir "$INPUT_IMAGE_DIR"
  --output-root "$OUTPUT_ROOT"
  --generator qwen-image-edit
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
  cmd+=(
    --vlm-eval
    --vlm-device "$DEVICE"
    --vlm-model-path "$VLM_MODEL_PATH"
    --vlm-max-samples "$VLM_MAX_SAMPLES"
  )
fi

echo "[run_multimodal_edit_cloud] ${cmd[*]}"
"${cmd[@]}"
