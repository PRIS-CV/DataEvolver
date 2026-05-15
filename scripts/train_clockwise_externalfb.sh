#!/usr/bin/env bash
# Train Qwen-Image-Edit clockwise LoRA on an ARIS rotation split dataset.
#
# Required environment:
#   DATASET_ROOT       ARIS train-ready/split dataset root.
#   TRAIN_OUTPUT_DIR  Output directory for checkpoints/logs.
#
# This wrapper avoids relying on train_clockwise.sh defaults, which may point to
# older dataset roots. It launches the DiffSynth training script with explicit
# paths derived from DATASET_ROOT.

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-}"

if [[ -z "${DATASET_ROOT}" ]]; then
  echo "DATASET_ROOT is required." >&2
  exit 1
fi
if [[ -z "${TRAIN_OUTPUT_DIR}" ]]; then
  echo "TRAIN_OUTPUT_DIR is required." >&2
  exit 1
fi

DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-<diffsynth-root>}"
TRAIN_PY="${TRAIN_PY:-${DIFFSYNTH_ROOT}/train_clockwise.py}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${DIFFSYNTH_ROOT}/accelerate_config_3gpu.yaml}"

DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-${DATASET_ROOT}/pairs/train_pairs.jsonl}"
MODEL_ROOT="${MODEL_ROOT:-/data/wuwenzhuo/Qwen-Image-Edit-2511}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_ROOT}/tokenizer}"
PROCESSOR_PATH="${PROCESSOR_PATH:-${MODEL_ROOT}/processor}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
NUM_PROCESSES="${NUM_PROCESSES:-3}"
LEARNING_RATE="${LEARNING_RATE:-${LR:-1e-4}}"
NUM_EPOCHS="${NUM_EPOCHS:-${EPOCHS:-30}}"
DATASET_REPEAT="${DATASET_REPEAT:-10}"
LORA_RANK="${LORA_RANK:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1}"
REMOVE_PREFIX_IN_CKPT="${REMOVE_PREFIX_IN_CKPT:-pipe.dit.}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-2}"
FIND_UNUSED_PARAMETERS="${FIND_UNUSED_PARAMETERS:-true}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-1}"
EVAL_NUM_INFERENCE_STEPS="${EVAL_NUM_INFERENCE_STEPS:-30}"
EVAL_SEED="${EVAL_SEED:-42}"
TRAIN_SEED="${TRAIN_SEED:-42}"
TENSORBOARD_FLUSH_SECS="${TENSORBOARD_FLUSH_SECS:-10}"

export CUDA_VISIBLE_DEVICES
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTHONPATH="${DIFFSYNTH_ROOT}:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-<repo-root>/runtime/model_cache/diffsynth_models}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-<repo-root>/runtime/model_cache/modelscope}"
export HF_HOME="${HF_HOME:-<repo-root>/runtime/model_cache/huggingface}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "Training script not found: ${TRAIN_PY}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_METADATA_PATH}" ]]; then
  echo "Training metadata not found: ${DATASET_METADATA_PATH}" >&2
  exit 1
fi
mkdir -p "${DIFFSYNTH_MODEL_BASE_PATH}" "${MODELSCOPE_CACHE}" "${HF_HOME}"

MODEL_PATHS="${MODEL_PATHS:-}"
if [[ -z "${MODEL_PATHS}" ]]; then
  MODEL_PATHS="$(python - "${MODEL_ROOT}" <<'PY'
import glob
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
transformer = sorted(glob.glob(str(root / "transformer" / "diffusion_pytorch_model-*.safetensors")))
text_encoder = sorted(glob.glob(str(root / "text_encoder" / "model-*.safetensors")))
vae = str(root / "vae" / "diffusion_pytorch_model.safetensors")
missing = []
if not transformer:
    missing.append(str(root / "transformer" / "diffusion_pytorch_model-*.safetensors"))
if not text_encoder:
    missing.append(str(root / "text_encoder" / "model-*.safetensors"))
if not Path(vae).exists():
    missing.append(vae)
if missing:
    raise SystemExit("Missing local model files: " + ", ".join(missing))
print(json.dumps([transformer, text_encoder, vae]))
PY
)"
fi

EVAL_IMAGE_PATH="${EVAL_IMAGE_PATH:-}"
if [[ -z "${EVAL_IMAGE_PATH}" ]]; then
  if [[ -f "${DATASET_ROOT}/bbox_views/obj_001/yaw000.png" ]]; then
    EVAL_IMAGE_PATH="${DATASET_ROOT}/bbox_views/obj_001/yaw000.png"
  elif [[ -f "${DATASET_ROOT}/views/obj_001/yaw000.png" ]]; then
    EVAL_IMAGE_PATH="${DATASET_ROOT}/views/obj_001/yaw000.png"
  fi
fi

TB_DIR="${TRAIN_OUTPUT_DIR}/tensorboard"
mkdir -p "${TRAIN_OUTPUT_DIR}" "${TB_DIR}"

echo ""
echo "=================================================================="
echo "External-feedback clockwise training"
echo "DATASET_ROOT: ${DATASET_ROOT}"
echo "METADATA:     ${DATASET_METADATA_PATH}"
echo "OUT:          ${TRAIN_OUTPUT_DIR}"
echo "TB:           ${TB_DIR}"
echo "EVAL IMAGE:   ${EVAL_IMAGE_PATH:-<none>}"
echo "DIFFSYNTH:    ${DIFFSYNTH_ROOT}"
echo "MODEL_ROOT:   ${MODEL_ROOT}"
echo "=================================================================="

accelerate_cmd=(accelerate launch --num_processes "${NUM_PROCESSES}")
if [[ -n "${ACCELERATE_CONFIG}" && -f "${ACCELERATE_CONFIG}" ]]; then
  accelerate_cmd+=(--config_file "${ACCELERATE_CONFIG}")
fi

cd "${DIFFSYNTH_ROOT}"

train_args=(
  "${TRAIN_PY}"
  --dataset_base_path "${DATASET_ROOT}"
  --dataset_metadata_path "${DATASET_METADATA_PATH}"
  --data_file_keys "image,edit_image"
  --extra_inputs "edit_image"
  --max_pixels 1048576
  --dataset_repeat "${DATASET_REPEAT}"
  --model_paths "${MODEL_PATHS}"
  --tokenizer_path "${TOKENIZER_PATH}"
  --processor_path "${PROCESSOR_PATH}"
  --learning_rate "${LEARNING_RATE}"
  --num_epochs "${NUM_EPOCHS}"
  --save_every_epochs "${SAVE_EVERY_EPOCHS}"
  --remove_prefix_in_ckpt "${REMOVE_PREFIX_IN_CKPT}"
  --output_path "${TRAIN_OUTPUT_DIR}"
  --lora_base_model "dit"
  --lora_target_modules "${LORA_TARGET_MODULES}"
  --lora_rank "${LORA_RANK}"
  --use_gradient_checkpointing
  --dataset_num_workers "${DATASET_NUM_WORKERS}"
  --tensorboard_log_dir "${TB_DIR}"
  --tensorboard_flush_secs "${TENSORBOARD_FLUSH_SECS}"
  --eval_every_epochs "${EVAL_EVERY_EPOCHS}"
  --eval_num_inference_steps "${EVAL_NUM_INFERENCE_STEPS}"
  --eval_seed "${EVAL_SEED}"
  --seed "${TRAIN_SEED}"
  --zero_cond_t
)

if [[ -n "${EVAL_IMAGE_PATH}" ]]; then
  train_args+=(--eval_image_path "${EVAL_IMAGE_PATH}")
fi
if [[ "${FIND_UNUSED_PARAMETERS}" == "true" ]]; then
  train_args+=(--find_unused_parameters)
fi

"${accelerate_cmd[@]}" "${train_args[@]}"
