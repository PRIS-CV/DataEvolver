#!/usr/bin/env bash
# ============================================================================
# resume_and_rerun_scenes.sh
#
# 1. 场景4(outdoor13) & 场景5(outdoor16): 从 angle_gate 断点续跑
# 2. 场景1(indoor_4blend) & 场景2(outdoor15): 全流程重跑
#
# 所有场景串行执行（VLM 需要独占一张 GPU）
# ============================================================================
set -euo pipefail

REPO_ROOT="$HOME/ARIS"
PYTHON="/home/jiazhuangzhuang/miniconda3/envs/aris/bin/python"
BLENDER="/aaaidata/zhangqisong/blender-4.24/blender"
RUNTIME_ROOT="/aaaidata/jiazhuangzhuang/ARIS_runtime"
BASELINE_SPLIT="${REPO_ROOT}/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410"
DEVICE="cuda:0"
EXPORT_GPUS="0,1,2"
TARGET_ROTATIONS="45,90,135,180,225,270,315"
ANGLE_GATE_THRESHOLD="0.50"
ANGLE_GATE_MAX_ROUNDS=2

MASTER_LOG="${RUNTIME_ROOT}/resume_rerun_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"; }

run_step() {
    local step_name="$1"
    local log_file="$2"
    shift 2
    local cmd=("$@")

    log "  [step] ${step_name}"
    log "  CMD: ${cmd[*]}"

    local start_time=$(date +%s)
    if "${cmd[@]}" 2>&1 | tee "$log_file"; then
        local elapsed=$(( $(date +%s) - start_time ))
        log "  ✓ ${step_name} 完成 (${elapsed}s)"
        return 0
    else
        local elapsed=$(( $(date +%s) - start_time ))
        log "  ✗ ${step_name} 失败 (${elapsed}s)，日志: ${log_file}"
        return 1
    fi
}

# ── 断点续跑函数：从 angle_gate 开始 ──────────────────────────────────────────
resume_from_angle_gate() {
    local SCENE_NAME="$1"
    local RUN_ROOT="$2"

    local LOGS_DIR="${RUN_ROOT}/logs"
    local ASSETS_ROOT="${RUN_ROOT}/stage1_assets"
    local ROT8_ROOT="${RUN_ROOT}/rotation8_consistent"
    local ROT8_ENHANCED_ROOT="${RUN_ROOT}/rotation8_enhanced"
    local TRAINREADY_ROOT="${RUN_ROOT}/trainready"
    local AUGMENTED_ROOT="${RUN_ROOT}/augmented_trainready_split"
    local PINNED_SPLIT="${RUN_ROOT}/pinned_split.json"
    local GATE_PROFILE_LOCAL="${RUN_ROOT}/scene_v7_gate_profile.json"
    local PROMPTS_FILE="${ASSETS_ROOT}/prompts.json"
    local SCENE_TEMPLATE="${ROT8_ROOT}/scene_template_fixed_camera.json"

    log ""
    log "╔══════════════════════════════════════════════════════════════╗"
    log "║  断点续跑: ${SCENE_NAME}"
    log "║  运行目录: ${RUN_ROOT}"
    log "║  从 angle_gate (step 5.5) 开始"
    log "╚══════════════════════════════════════════════════════════════╝"

    # 清理失败的 rotation8_enhanced
    if [[ -d "${ROT8_ENHANCED_ROOT}" ]]; then
        log "  [cleanup] 清理旧的 rotation8_enhanced"
        rm -rf "${ROT8_ENHANCED_ROOT}"
    fi
    mkdir -p "$LOGS_DIR"

    # Step 5.5: angle_gate
    run_step "rotation8_angle_gate" "${LOGS_DIR}/05.5_rotation8_angle_gate.log" \
        "$PYTHON" "${REPO_ROOT}/scripts/run_rotation8_angle_gate.py" \
        --source-root "$ROT8_ROOT" \
        --output-dir "$ROT8_ENHANCED_ROOT" \
        --meshes-dir "${ASSETS_ROOT}/meshes" \
        --blender "$BLENDER" \
        --profile "$GATE_PROFILE_LOCAL" \
        --device "$DEVICE" \
        --angle-threshold "$ANGLE_GATE_THRESHOLD" \
        --max-enhance-rounds "$ANGLE_GATE_MAX_ROUNDS" \
        --copy-mode symlink || return 1

    # Step 6: trainready
    run_step "build_trainready" "${LOGS_DIR}/06_build_trainready.log" \
        "$PYTHON" "${REPO_ROOT}/scripts/build_rotation8_trainready_dataset.py" \
        --source-root "$ROT8_ENHANCED_ROOT" \
        --output-dir "$TRAINREADY_ROOT" \
        --copy-mode symlink \
        --target-rotations "$TARGET_ROTATIONS" \
        --prompts-json "$PROMPTS_FILE" || return 1

    # Steps 7-9: augment to baseline
    if [[ -d "$BASELINE_SPLIT" ]]; then
        run_step "build_pinned_split" "${LOGS_DIR}/07_build_pinned_split.log" \
            "$PYTHON" "${REPO_ROOT}/scripts/feedback_loop/build_pinned_split.py" \
            --dataset-root "$BASELINE_SPLIT" \
            --output "$PINNED_SPLIT" \
            --source-note "resume_angle_gate ${SCENE_NAME}" || return 1

        run_step "build_augmented_dataset" "${LOGS_DIR}/08_build_augmented_dataset.log" \
            "$PYTHON" "${REPO_ROOT}/scripts/feedback_loop/build_augmented_rotation_dataset.py" \
            --baseline-split-root "$BASELINE_SPLIT" \
            --new-trainready-root "$TRAINREADY_ROOT" \
            --output-dir "$AUGMENTED_ROOT" \
            --pinned-split-path "$PINNED_SPLIT" \
            --asset-mode symlink \
            --target-rotations "$TARGET_ROTATIONS" || return 1

        run_step "validate_final_dataset" "${LOGS_DIR}/09_validate_final_dataset.log" \
            "$PYTHON" "${REPO_ROOT}/scripts/feedback_loop/validate_dataset.py" \
            "$AUGMENTED_ROOT" \
            --output "${RUN_ROOT}/validation_report.json" \
            --pinned-split-path "$PINNED_SPLIT" \
            --expected-manifest "${AUGMENTED_ROOT}/augmented_manifest.json" || return 1
    fi

    log "  ✓ ${SCENE_NAME} 断点续跑完成"
}

# ══════════════════════════════════════════════════════════════════════════════
# 开始执行
# ══════════════════════════════════════════════════════════════════════════════
log "============================================================"
log "  4 场景 恢复 + 重跑 ($(date))"
log "  场景 4 & 5: 从 angle_gate 断点续跑"
log "  场景 1 & 2: 全流程重跑"
log "============================================================"

RESULTS=()

# ── 场景 4: outdoor13 (断点续跑) ─────────────────────────────────────────────
log ""
log ">>> [1/4] 场景 4: outdoor13 — 断点续跑 angle_gate"
T0=$(date +%s)
if resume_from_angle_gate "outdoor13" \
    "/aaaidata/jiazhuangzhuang/ARIS_runtime/new_asset_force_rot8_20260501_011748"; then
    ELAPSED=$(( $(date +%s) - T0 ))
    RESULTS+=("场景4 outdoor13: OK (${ELAPSED}s)")
else
    ELAPSED=$(( $(date +%s) - T0 ))
    RESULTS+=("场景4 outdoor13: FAIL (${ELAPSED}s)")
fi

# GPU 清理
log "[cleanup] 清理 GPU 内存..."
$PYTHON -c "import torch,gc; gc.collect(); torch.cuda.empty_cache(); [torch.cuda.empty_cache() for _ in range(3)]" 2>/dev/null || true
sleep 5

# ── 场景 5: outdoor16 (断点续跑) ─────────────────────────────────────────────
log ""
log ">>> [2/4] 场景 5: outdoor16 — 断点续跑 angle_gate"
T0=$(date +%s)
if resume_from_angle_gate "outdoor16" \
    "/aaaidata/jiazhuangzhuang/ARIS_runtime/new_asset_force_rot8_20260501_022950"; then
    ELAPSED=$(( $(date +%s) - T0 ))
    RESULTS+=("场景5 outdoor16: OK (${ELAPSED}s)")
else
    ELAPSED=$(( $(date +%s) - T0 ))
    RESULTS+=("场景5 outdoor16: FAIL (${ELAPSED}s)")
fi

# GPU 清理
log "[cleanup] 清理 GPU 内存..."
$PYTHON -c "import torch,gc; gc.collect(); torch.cuda.empty_cache(); [torch.cuda.empty_cache() for _ in range(3)]" 2>/dev/null || true
sleep 5

# ── 场景 1: indoor_4blend (全流程重跑) ───────────────────────────────────────
log ""
log ">>> [3/4] 场景 1: indoor_4blend — 全流程重跑"
T0=$(date +%s)
if bash "${REPO_ROOT}/scripts/launch_new_asset_force_rot8.sh" \
    --seed-count 6 \
    --seed-start-index 1494 \
    --scene-template "${REPO_ROOT}/configs/scene_template.json" \
    --gate-max-rounds 5 \
    --augment-baseline 2>&1 | tee -a "$MASTER_LOG"; then
    ELAPSED=$(( $(date +%s) - T0 ))
    RESULTS+=("场景1 indoor_4blend: OK (${ELAPSED}s)")
else
    ELAPSED=$(( $(date +%s) - T0 ))
    RESULTS+=("场景1 indoor_4blend: FAIL (${ELAPSED}s)")
fi

# GPU 清理
log "[cleanup] 清理 GPU 内存..."
$PYTHON -c "import torch,gc; gc.collect(); torch.cuda.empty_cache(); [torch.cuda.empty_cache() for _ in range(3)]" 2>/dev/null || true
sleep 5

# ── 场景 2: outdoor15 (全流程重跑) ───────────────────────────────────────────
log ""
log ">>> [4/4] 场景 2: outdoor15 — 全流程重跑"
T0=$(date +%s)
if bash "${REPO_ROOT}/scripts/launch_new_asset_force_rot8.sh" \
    --seed-count 6 \
    --seed-start-index 1500 \
    --scene-template "${REPO_ROOT}/configs/scene_template_outdoor15.json" \
    --gate-max-rounds 5 \
    --augment-baseline 2>&1 | tee -a "$MASTER_LOG"; then
    ELAPSED=$(( $(date +%s) - T0 ))
    RESULTS+=("场景2 outdoor15: OK (${ELAPSED}s)")
else
    ELAPSED=$(( $(date +%s) - T0 ))
    RESULTS+=("场景2 outdoor15: FAIL (${ELAPSED}s)")
fi

# ══════════════════════════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════════════════════════
log ""
log "============================================================"
log "  全部完成 — 汇总"
log ""
for r in "${RESULTS[@]}"; do
    log "  $r"
done
log ""
log "  总日志: ${MASTER_LOG}"
log "============================================================"
