#!/usr/bin/env bash
# Orchestrate the outer loop:
#   optional Stage1->dataset generation -> train -> eval/bench feedback -> augmented dataset -> optional next train.
#
# This script intentionally keeps the training/eval commands pluggable. The
# ARIS repo owns dataset generation and feedback construction, while the actual
# LoRA trainer may live in DiffSynth-Studio or another training project.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_stage1_to_externalfb_train_loop.sh [options]

Core options:
  --round NAME                 Run name, default timestamped.
  --dataset-root PATH          Starting split/train-ready dataset root.
  --run-stage1                 Run Stage1->dataset generation before training.
  --skip-train                 Do not run TRAIN_CMD.
  --skip-feedback              Do not build/apply external feedback.
  --train-feedback             Run TRAIN_CMD once more on the feedback dataset.
  --dry-run                    Print commands without executing heavy steps.

Feedback input:
  --eval-zip PATH              Eval metrics zip consumed by build_external_feedback_loop.py.
  --eval-dir PATH              Eval metrics directory consumed by build_external_feedback_loop.py.
  --current CSV                Current metrics CSV name, default ours_objinfo_metrics.csv.
  --baseline CSV               Baseline metrics CSV name, default base_metrics.csv.

Dataset output:
  --feedback-dataset PATH      Output root for feedback-augmented dataset.
  --oversample-factor N        Total copies for weak-angle train pairs, default 2.
  --asset-mode symlink|copy|none

Environment hooks:
  TRAIN_CMD                    Training command. Receives DATASET_ROOT and TRAIN_OUTPUT_DIR env vars.
  TRAIN_FEEDBACK_CMD           Optional second training command. Defaults to TRAIN_CMD.
  EVAL_CMD                     Optional eval/bench command after first training.
  STAGE1_DATASET_CMD           Optional command for Stage1->dataset generation.
  PYTHON_BIN                   Python executable, default python.
  BLENDER_BIN                  Blender executable for default Stage1 command.

Example:
  TRAIN_CMD='bash <diffsynth-root>/train_clockwise.sh' \
  EVAL_CMD='bash run_eval_inference.sh all' \
  bash scripts/run_stage1_to_externalfb_train_loop.sh \
    --dataset-root pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410 \
    --eval-zip /path/to/eval_spatialedit_metrics.zip \
    --train-feedback
EOF
}

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

log() {
  echo "[$(date '+%F %T')] $*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

shell_quote() {
  printf "%q" "$1"
}

run_cmd() {
  local name="$1"
  local cmd="$2"
  local log_file="$3"

  log "=== ${name} ==="
  log "Command: ${cmd}"
  log "Log: ${log_file}"

  mkdir -p "$(dirname "${log_file}")"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] ${cmd}" | tee "${log_file}"
    return 0
  fi

  set +e
  bash -lc "${cmd}" 2>&1 | tee "${log_file}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    die "${name} failed with exit code ${rc}; see ${log_file}"
  fi
}

write_state() {
  local status="$1"
  "${PYTHON_BIN}" - "$STATE_FILE" "$status" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
status = sys.argv[2]
payload = {
    "status": status,
    "round": os.environ.get("ROUND_NAME", ""),
    "run_root": os.environ.get("RUN_ROOT", ""),
    "dataset_root": os.environ.get("DATASET_ROOT", ""),
    "feedback_dataset": os.environ.get("FEEDBACK_DATASET", ""),
    "feedback_dir": os.environ.get("FEEDBACK_DIR", ""),
    "train_output_dir": os.environ.get("TRAIN_OUTPUT_DIR", ""),
    "train_feedback_output_dir": os.environ.get("TRAIN_FEEDBACK_OUTPUT_DIR", ""),
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

ROUND_NAME="externalfb_$(timestamp)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BLENDER_BIN="${BLENDER_BIN:-blender}"

RUN_STAGE1=0
SKIP_TRAIN=0
SKIP_FEEDBACK=0
TRAIN_FEEDBACK=0
DRY_RUN=0

DATASET_ROOT=""
EVAL_ZIP=""
EVAL_DIR=""
CURRENT_METRICS="ours_objinfo_metrics.csv"
BASELINE_METRICS="base_metrics.csv"
OVERSAMPLE_FACTOR=2
ASSET_MODE="symlink"
FEEDBACK_DATASET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --round) ROUND_NAME="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --run-stage1) RUN_STAGE1=1; shift ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --skip-feedback) SKIP_FEEDBACK=1; shift ;;
    --train-feedback) TRAIN_FEEDBACK=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --eval-zip) EVAL_ZIP="$2"; shift 2 ;;
    --eval-dir) EVAL_DIR="$2"; shift 2 ;;
    --current) CURRENT_METRICS="$2"; shift 2 ;;
    --baseline) BASELINE_METRICS="$2"; shift 2 ;;
    --feedback-dataset) FEEDBACK_DATASET="$2"; shift 2 ;;
    --oversample-factor) OVERSAMPLE_FACTOR="$2"; shift 2 ;;
    --asset-mode) ASSET_MODE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

RUN_ROOT="${REPO_ROOT}/runtime/external_feedback_loop/${ROUND_NAME}"
LOG_DIR="${RUN_ROOT}/logs"
FEEDBACK_DIR="${RUN_ROOT}/feedback"
STATE_FILE="${RUN_ROOT}/LOOP_STATE.json"

if [[ -z "${DATASET_ROOT}" ]]; then
  DATASET_ROOT="pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410"
fi
if [[ -z "${FEEDBACK_DATASET}" ]]; then
  FEEDBACK_DATASET="pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_externalfb_${ROUND_NAME}"
fi

export ROUND_NAME RUN_ROOT LOG_DIR FEEDBACK_DIR STATE_FILE
export DATASET_ROOT FEEDBACK_DATASET
export TRAIN_OUTPUT_DIR="${RUN_ROOT}/train_first"
export TRAIN_FEEDBACK_OUTPUT_DIR="${RUN_ROOT}/train_feedback"

mkdir -p "${LOG_DIR}" "${FEEDBACK_DIR}" "${TRAIN_OUTPUT_DIR}" "${TRAIN_FEEDBACK_OUTPUT_DIR}"
write_state "started"

if [[ "${RUN_STAGE1}" == "1" ]]; then
  DEFAULT_STAGE1_CMD="BLENDER_BIN=$(shell_quote "${BLENDER_BIN}") PYTHON_BIN=$(shell_quote "${PYTHON_BIN}") PYTHONPATH=<diffsynth-root>:\${PYTHONPATH:-} bash scripts/run_scene_full50_expansion.sh"
  STAGE1_DATASET_CMD="${STAGE1_DATASET_CMD:-${DEFAULT_STAGE1_CMD}}"
  run_cmd "stage1_to_dataset_generation" "${STAGE1_DATASET_CMD}" "${LOG_DIR}/stage1_to_dataset_generation.log"
else
  log "Skip Stage1->dataset generation. Using DATASET_ROOT=${DATASET_ROOT}"
fi

[[ -e "${DATASET_ROOT}" || "${DRY_RUN}" == "1" ]] || die "Dataset root does not exist: ${DATASET_ROOT}"

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  if [[ -z "${TRAIN_CMD:-}" ]]; then
    die "TRAIN_CMD is required unless --skip-train is set. Example: TRAIN_CMD='bash <diffsynth-root>/train_clockwise.sh'"
  fi
  run_cmd "train_first" "${TRAIN_CMD}" "${LOG_DIR}/train_first.log"
else
  log "Skip training by --skip-train"
fi

if [[ -n "${EVAL_CMD:-}" ]]; then
  run_cmd "eval_after_first_train" "${EVAL_CMD}" "${LOG_DIR}/eval_after_first_train.log"
else
  log "No EVAL_CMD provided. If --eval-zip/--eval-dir is also absent, feedback will be skipped."
fi

if [[ "${SKIP_FEEDBACK}" != "1" ]]; then
  if [[ -z "${EVAL_ZIP}" && -z "${EVAL_DIR}" ]]; then
    die "Feedback requires --eval-zip or --eval-dir, or pass --skip-feedback."
  fi

  FEEDBACK_BUILD_CMD=("${PYTHON_BIN}" "scripts/build_external_feedback_loop.py")
  if [[ -n "${EVAL_ZIP}" ]]; then
    FEEDBACK_BUILD_CMD+=("--eval-zip" "${EVAL_ZIP}")
  fi
  if [[ -n "${EVAL_DIR}" ]]; then
    FEEDBACK_BUILD_CMD+=("--eval-dir" "${EVAL_DIR}")
  fi
  FEEDBACK_BUILD_CMD+=(
    "--output-dir" "${FEEDBACK_DIR}"
    "--current" "${CURRENT_METRICS}"
    "--baseline" "${BASELINE_METRICS}"
  )
  run_cmd "build_external_feedback" "$(printf "%q " "${FEEDBACK_BUILD_CMD[@]}")" "${LOG_DIR}/build_external_feedback.log"

  APPLY_CMD=(
    "${PYTHON_BIN}" "scripts/apply_external_feedback_to_dataset.py"
    "--source-dataset" "${DATASET_ROOT}"
    "--requirements" "${FEEDBACK_DIR}/augmentation_requirements.json"
    "--weak-samples" "${FEEDBACK_DIR}/weak_samples.csv"
    "--output-dir" "${FEEDBACK_DATASET}"
    "--oversample-factor" "${OVERSAMPLE_FACTOR}"
    "--asset-mode" "${ASSET_MODE}"
  )
  run_cmd "apply_external_feedback_to_dataset" "$(printf "%q " "${APPLY_CMD[@]}")" "${LOG_DIR}/apply_external_feedback_to_dataset.log"
else
  log "Skip feedback by --skip-feedback"
fi

if [[ "${TRAIN_FEEDBACK}" == "1" ]]; then
  if [[ -z "${TRAIN_FEEDBACK_CMD:-${TRAIN_CMD:-}}" ]]; then
    die "TRAIN_FEEDBACK_CMD or TRAIN_CMD is required for --train-feedback."
  fi
  export DATASET_ROOT="${FEEDBACK_DATASET}"
  export TRAIN_OUTPUT_DIR="${TRAIN_FEEDBACK_OUTPUT_DIR}"
  run_cmd "train_feedback_dataset" "${TRAIN_FEEDBACK_CMD:-${TRAIN_CMD}}" "${LOG_DIR}/train_feedback_dataset.log"
fi

write_state "completed"

log "Loop completed."
log "State: ${STATE_FILE}"
log "Feedback dataset: ${FEEDBACK_DATASET}"
