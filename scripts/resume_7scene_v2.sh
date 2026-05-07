#!/usr/bin/env bash
# ============================================================================
# resume_7scene_v2.sh
#
# 从断点恢复 7场景×5物体 pipeline (20260503_115024)
# - 跳过 Step 01-02 (资产已存在)
# - Step 03: 仅单卡 GPU 2 跑 26 个未完成的 VLM gate 对象
# - Step 03.5+: force_accept → rebuild → rotation8 → angle gate → trainready → augment → validate
#
# 使用: tmux new -s pipeline-7scene-resume
#       bash ~/ARIS/scripts/resume_7scene_v2.sh 2>&1 | tee ~/resume_7scene_v2.log
# ============================================================================

set -euo pipefail

# ── 环境 ──────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=2
export ANTHROPIC_API_KEY='sk-7834aaf7864cbd4fbef111f71fe835de14e124fe3807a0e5d8a2e154ef865c81'

REPO_ROOT="$HOME/ARIS"
PYTHON="/home/jiazhuangzhuang/miniconda3/envs/aris/bin/python"
BLENDER="/aaaidata/zhangqisong/blender-4.24/blender"

# ── 现有运行目录（不创建新的）────────────────────────────────────────────────
RUN_ROOT="/aaaidata/jiazhuangzhuang/ARIS_runtime/new_asset_force_rot8_20260503_115024"
LOGS_DIR="${RUN_ROOT}/logs"
SEED_FILE="${RUN_ROOT}/seed_concepts.json"
ASSETS_ROOT="${RUN_ROOT}/stage1_assets"
PROMPTS_FILE="${ASSETS_ROOT}/prompts.json"
GATE_ROOT="${RUN_ROOT}/gate_loop"
ACCEPTED_ROOT="${RUN_ROOT}/accepted_pairs"
ROT8_ROOT="${RUN_ROOT}/rotation8_consistent"
ROT8_ENHANCED_ROOT="${RUN_ROOT}/rotation8_enhanced"
TRAINREADY_ROOT="${RUN_ROOT}/trainready"
AUGMENTED_ROOT="${RUN_ROOT}/augmented_trainready_split"
GATE_PROFILE_LOCAL="${RUN_ROOT}/scene_v7_gate_profile.json"
BASELINE_SPLIT="${REPO_ROOT}/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410"
PINNED_SPLIT="${RUN_ROOT}/pinned_split.json"
LOOP_STATE="${RUN_ROOT}/LOOP_STATE.json"

SCENE_TEMPLATE="${REPO_ROOT}/configs/scene_template.json"
SCENE_POOL="${REPO_ROOT}/configs/scene_pool.json"
GATE_MAX_ROUNDS=3
GATE_THRESHOLD="0.78"
GATE_PATIENCE=0
ANGLE_GATE_THRESHOLD="0.50"
ANGLE_GATE_MAX_ROUNDS=3
EXPORT_ROTATIONS="0,45,90,135,180,225,270,315"
TARGET_ROTATIONS="45,90,135,180,225,270,315"

mkdir -p "$LOGS_DIR"

log() { echo "[$(date '+%F %T')] $*"; }

run_step() {
    local step_num="$1"
    local step_name="$2"
    shift 2
    local cmd=("$@")

    log "=== [${step_num}] ${step_name} ==="
    log "CMD: ${cmd[*]}"

    local log_file="${LOGS_DIR}/${step_num}_${step_name}_resume.log"
    local start_time=$(date +%s)
    if "${cmd[@]}" 2>&1 | tee "$log_file"; then
        local elapsed=$(( $(date +%s) - start_time ))
        log "  ✓ ${step_name} 完成 (${elapsed}s)"
    else
        local elapsed=$(( $(date +%s) - start_time ))
        log "  ✗ ${step_name} 失败 (${elapsed}s)，查看日志: ${log_file}"
        return 1
    fi
}

echo "============================================================"
echo "[resume] 恢复 7场景×5物体 pipeline"
echo "[resume] 运行目录: ${RUN_ROOT}"
echo "[resume] GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[resume] Step 01-02: 跳过 (资产已存在)"
echo "[resume] Step 03: VLM gate 补跑 26 个缺失对象 (单卡)"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 03 (resume): VLM 质量门控 — 单卡补跑
# 使用 --resume 标志，只处理未完成的 object
# 不用 --gpus（单卡模式），device=cuda:0 因为 CUDA_VISIBLE_DEVICES=2
# ══════════════════════════════════════════════════════════════════════════════
run_step "03-resume" "run_vlm_quality_gate" \
    "$PYTHON" "${REPO_ROOT}/scripts/run_vlm_quality_gate_loop.py" \
    --root-dir "$GATE_ROOT" \
    --objects-file "$SEED_FILE" \
    --rotations 0 \
    --threshold-start "$GATE_THRESHOLD" \
    --threshold-step 0.0 \
    --threshold-max "$GATE_THRESHOLD" \
    --score-field hybrid_score \
    --max-rounds "$GATE_MAX_ROUNDS" \
    --min-vlm-only-score 0.0 \
    --step-scale 0.25 \
    --patience "$GATE_PATIENCE" \
    --rollback-score-drop-threshold 0.05 \
    --device "cuda:0" \
    --blender "$BLENDER" \
    --meshes-dir "${ASSETS_ROOT}/meshes" \
    --profile "$GATE_PROFILE_LOCAL" \
    --scene-template "$SCENE_TEMPLATE" \
    --scene-pool "$SCENE_POOL" \
    --resume

# ══════════════════════════════════════════════════════════════════════════════
# STEP 03.5: 强制接受 — 将被拒绝的 object 也标记为 accepted
# ══════════════════════════════════════════════════════════════════════════════
log "=== [03.5] force_accept_all ==="
log "无论 VLM 是否接受，将所有 object 强制标记为 accepted"

$PYTHON "${REPO_ROOT}/scripts/resume_force_accept.py" "$GATE_ROOT"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 04: 构建 accepted pair root
# ══════════════════════════════════════════════════════════════════════════════
run_step "04-resume" "build_accepted_pair_root" \
    "$PYTHON" "${REPO_ROOT}/scripts/build_accepted_pair_root_from_gate.py" \
    --gate-root "$GATE_ROOT" \
    --output-dir "$ACCEPTED_ROOT" \
    --asset-mode symlink \
    --overwrite

# ══════════════════════════════════════════════════════════════════════════════
# STEP 05: 导出 rotation8 一致性数据集
# 单卡模式 (CUDA_VISIBLE_DEVICES=2, --gpus 0 表示可见卡索引0)
# ══════════════════════════════════════════════════════════════════════════════
run_step "05-resume" "export_rotation8" \
    "$PYTHON" "${REPO_ROOT}/scripts/export_rotation8_from_best_object_state.py" \
    --source-root "$ACCEPTED_ROOT" \
    --output-dir "$ROT8_ROOT" \
    --python "$PYTHON" \
    --blender "$BLENDER" \
    --meshes-dir "${ASSETS_ROOT}/meshes" \
    --gpus "0" \
    --fallback-to-best-any-angle \
    --rotations "$EXPORT_ROTATIONS" \
    --scene-template "$SCENE_TEMPLATE" \
    --scene-pool "$SCENE_POOL" \
    --scene-assignments "${GATE_ROOT}/scene_assignments.json"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 05.5: Per-angle 质量门控 + 弱角度增强
# ══════════════════════════════════════════════════════════════════════════════
run_step "05.5-resume" "rotation8_angle_gate" \
    "$PYTHON" "${REPO_ROOT}/scripts/run_rotation8_angle_gate.py" \
    --source-root "$ROT8_ROOT" \
    --output-dir "$ROT8_ENHANCED_ROOT" \
    --meshes-dir "${ASSETS_ROOT}/meshes" \
    --blender "$BLENDER" \
    --profile "$GATE_PROFILE_LOCAL" \
    --device "cuda:0" \
    --angle-threshold "$ANGLE_GATE_THRESHOLD" \
    --max-enhance-rounds "$ANGLE_GATE_MAX_ROUNDS" \
    --copy-mode symlink

# ══════════════════════════════════════════════════════════════════════════════
# STEP 06: 构建 train-ready 数据集
# ══════════════════════════════════════════════════════════════════════════════
run_step "06-resume" "build_trainready" \
    "$PYTHON" "${REPO_ROOT}/scripts/build_rotation8_trainready_dataset.py" \
    --source-root "$ROT8_ENHANCED_ROOT" \
    --output-dir "$TRAINREADY_ROOT" \
    --copy-mode symlink \
    --target-rotations "$TARGET_ROTATIONS" \
    --prompts-json "$PROMPTS_FILE"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 07-09: 合并到 baseline 数据集
# ══════════════════════════════════════════════════════════════════════════════
if [[ -d "$BASELINE_SPLIT" ]]; then
    log "=== 合并到 baseline 数据集 ==="

    run_step "07-resume" "build_pinned_split" \
        "$PYTHON" "${REPO_ROOT}/scripts/feedback_loop/build_pinned_split.py" \
        --dataset-root "$BASELINE_SPLIT" \
        --output "$PINNED_SPLIT" \
        --source-note "new_asset_force_rot8 baseline split"

    run_step "08-resume" "build_augmented_dataset" \
        "$PYTHON" "${REPO_ROOT}/scripts/feedback_loop/build_augmented_rotation_dataset.py" \
        --baseline-split-root "$BASELINE_SPLIT" \
        --new-trainready-root "$TRAINREADY_ROOT" \
        --output-dir "$AUGMENTED_ROOT" \
        --pinned-split-path "$PINNED_SPLIT" \
        --asset-mode symlink \
        --target-rotations "$TARGET_ROTATIONS"

    run_step "09-resume" "validate_final_dataset" \
        "$PYTHON" "${REPO_ROOT}/scripts/feedback_loop/validate_dataset.py" \
        "$AUGMENTED_ROOT" \
        --output "${RUN_ROOT}/validation_report.json" \
        --pinned-split-path "$PINNED_SPLIT" \
        --expected-manifest "${AUGMENTED_ROOT}/augmented_manifest.json"
fi

echo ""
echo "============================================================"
echo "[完成] 7场景×5物体 pipeline 恢复完毕"
echo ""
echo "  运行根目录:       ${RUN_ROOT}"
echo "  Accepted pairs:   ${ACCEPTED_ROOT}"
echo "  Rotation8 渲染:   ${ROT8_ROOT}"
echo "  Rotation8 增强:   ${ROT8_ENHANCED_ROOT}"
echo "  Train-ready:      ${TRAINREADY_ROOT}"
echo "  增强数据集:       ${AUGMENTED_ROOT}"
echo "  日志:             ${LOGS_DIR}"
echo "============================================================"
