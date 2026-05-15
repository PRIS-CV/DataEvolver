#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${1:-$(pwd)}"
QUEUE_PATH="${QUEUE_PATH:-$CODE_ROOT/pipeline/data/regeneration_queue.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BLENDER_BIN="${BLENDER_BIN:-blender}"
PROFILE_PATH="${PROFILE_PATH:-configs/dataset_profiles/scene_v7_full20_loop.json}"
SCENE_TEMPLATE_PATH="${SCENE_TEMPLATE_PATH:-configs/scene_template.json}"
SLEEP_SECONDS="${SLEEP_SECONDS:-30}"
IDLE_MB="${IDLE_MB:-10000}"
TARGET_GPU_IDS="${TARGET_GPU_IDS:-1}"
GENERATION_DEVICE="${GENERATION_DEVICE:-cuda:1}"
SMOKE_DEVICE="${SMOKE_DEVICE:-cuda:1}"
SMOKE_VISIBLE_GPUS="${SMOKE_VISIBLE_GPUS:-1}"
MAX_STAGE1_RESTARTS_PER_OBJECT="${MAX_STAGE1_RESTARTS_PER_OBJECT:-8}"
STAGE1_MODE="${STAGE1_MODE:-auto}"
MANUAL_REQUEUE_JOBS="${MANUAL_REQUEUE_JOBS:-2}"
MANUAL_REQUEUE_MAX_STAGE1_RESTARTS="${MANUAL_REQUEUE_MAX_STAGE1_RESTARTS:-999}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$CODE_ROOT"
export PYTORCH_CUDA_ALLOC_CONF

queue_has_active_jobs() {
  python3 -c "import json, pathlib, sys; p=pathlib.Path(r'$QUEUE_PATH'); data=json.load(open(p,'r',encoding='utf-8')) if p.exists() else []; active=[j for j in data if j.get('status') in {'queued','running'}]; print(len(active)); sys.exit(0 if active else 1)"
}

maybe_requeue_manual_reprompt_jobs() {
  "$PYTHON_BIN" scripts/requeue_manual_reprompt_jobs.py \
    --max-jobs "$MANUAL_REQUEUE_JOBS" \
    --max-stage1-restarts "$MANUAL_REQUEUE_MAX_STAGE1_RESTARTS"
}

target_gpus_idle() {
  python3 -c "import subprocess, sys; targets={int(x) for x in r'$TARGET_GPU_IDS'.split(',') if x.strip()}; out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'], text=True); usage={int(line.split(',')[0].strip()): int(line.split(',')[1].strip()) for line in out.splitlines() if line.strip()}; sys.exit(0 if targets and all(usage.get(g, 10**9) < int(r'$IDLE_MB') for g in targets) else 1)"
}

while true; do
  echo "[regen-daemon] $(date '+%F %T') tick"
  if ! target_gpus_idle; then
    echo "[regen-daemon] target-gpus-busy"
    sleep "$SLEEP_SECONDS"
    continue
  fi

  if ! queue_has_active_jobs; then
    echo "[regen-daemon] queue-empty-before-requeue"
    if maybe_requeue_manual_reprompt_jobs; then
      :
    else
      echo "[regen-daemon] manual-requeue-script-failed"
    fi
  fi

  if ! queue_has_active_jobs; then
    echo "[regen-daemon] queue-empty"
    sleep "$SLEEP_SECONDS"
    continue
  fi

  if "$PYTHON_BIN" scripts/run_asset_regeneration_queue.py \
    --max-jobs 1 \
    --max-stage1-restarts-per-object "$MAX_STAGE1_RESTARTS_PER_OBJECT" \
    --python-bin "$PYTHON_BIN" \
    --generation-device "$GENERATION_DEVICE" \
    --smoke-device "$SMOKE_DEVICE" \
    --smoke-visible-gpus "$SMOKE_VISIBLE_GPUS" \
    --blender-bin "$BLENDER_BIN" \
    --profile "$PROFILE_PATH" \
    --scene-template "$SCENE_TEMPLATE_PATH" \
    --stage1-mode "$STAGE1_MODE" \
    --allow-unsafe-single-gpu-review; then
    :
  else
    rc=$?
    echo "[regen-daemon] worker-failed:rc=$rc"
  fi

  sleep "$SLEEP_SECONDS"
done
