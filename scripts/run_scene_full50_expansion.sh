#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
BLENDER_BIN="${BLENDER_BIN:-}"
STAGE1_MODE="${STAGE1_MODE:-template-only}"
T2I_DEVICE="${T2I_DEVICE:-cuda:0}"
SAM_DEVICE="${SAM_DEVICE:-cuda:0}"
I23D_DEVICES="${I23D_DEVICES:-cuda:0,cuda:1,cuda:2}"
BOOTSTRAP_GPUS="${BOOTSTRAP_GPUS:-0,1,2}"
EXPORT_GPUS="${EXPORT_GPUS:-0,1,2}"
ASSET_MODE="${ASSET_MODE:-symlink}"
TRAIN_OBJECTS="${TRAIN_OBJECTS:-35}"
VAL_OBJECTS="${VAL_OBJECTS:-7}"
TEST_OBJECTS="${TEST_OBJECTS:-8}"
SEED="${SEED:-42}"

if [[ -z "${BLENDER_BIN}" ]]; then
  echo "BLENDER_BIN is required."
  echo "Example:"
  echo "  BLENDER_BIN=/path/to/blender bash scripts/run_scene_full50_expansion.sh"
  exit 1
fi

cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/build_scene_full50_expansion_pipeline.py \
  --python "${PYTHON_BIN}" \
  --blender "${BLENDER_BIN}" \
  --stage1-mode "${STAGE1_MODE}" \
  --t2i-device "${T2I_DEVICE}" \
  --sam-device "${SAM_DEVICE}" \
  --i23d-devices "${I23D_DEVICES}" \
  --bootstrap-gpus "${BOOTSTRAP_GPUS}" \
  --export-gpus "${EXPORT_GPUS}" \
  --asset-mode "${ASSET_MODE}" \
  --train-objects "${TRAIN_OBJECTS}" \
  --val-objects "${VAL_OBJECTS}" \
  --test-objects "${TEST_OBJECTS}" \
  --seed "${SEED}" \
  "$@"
