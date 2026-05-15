#!/usr/bin/env bash
# ============================================================================
# launch_new_asset_force_rot8.sh
#
# 在 <ssh-alias> 服务器上一键启动：
#   1. 生成 1 个新数字资产 (seed concept → T2I → SAM2 → 3D mesh)
#   2. VLM 渲染优化 loop（最多 5 轮）
#   3. 无论 VLM 是否接受，强制进入 rotation8 导出
#   4. 构建 train-ready 数据集
#
# 用法：
#   # 在 <ssh-alias> 上手动运行：
#   tmux new -s new-asset-rot8
#   export ANTHROPIC_API_KEY="sk-ant-..."   # 或 export STAGE1_API_KEY="..."
#   bash ~/ARIS/scripts/launch_new_asset_force_rot8.sh [选项]
#
#   # 或直接从本地 SSH 启动（需要先设置好 API key）：
#   ssh <ssh-alias> 'tmux new -d -s new-asset-rot8 "
#     export ANTHROPIC_API_KEY=sk-ant-...;
#     bash ~/ARIS/scripts/launch_new_asset_force_rot8.sh 2>&1 | tee ~/run.log
#   "'
#
# 选项：
#   --device DEVICE          GPU 设备 (默认: cuda:0)
#   --seed-count N           生成新资产数量 (默认: 1)
#   --seed-start-index N     obj_id 起始编号 (默认: 自动递增)
#   --seed-model MODEL       Stage1 概念生成模型 (默认: claude-opus-4-6)
#   --api-provider PROVIDER  API provider (默认: anthropic)
#   --api-base-url URL       API 代理 URL (默认: https://dragoncode.codes/v1)
#   --gate-max-rounds N      VLM 渲染优化最大轮数 (默认: 5)
#   --gate-threshold FLOAT   VLM 接受阈值 (默认: 0.78)
#   --template-only          使用预设概念列表，不调 LLM API
#   --scene-pool PATH        场景池 JSON 文件路径，每个物体随机分配不同场景
#   --augment-baseline       合并到已有 baseline 数据集
#   --dry-run                仅打印命令，不执行
# ============================================================================

set -euo pipefail

# ── 环境 ──────────────────────────────────────────────────────────────────────
REPO_ROOT="$HOME/ARIS"
PYTHON="python"
BLENDER="blender"
RUNTIME_ROOT="<runtime-root>"
BASELINE_SPLIT="${REPO_ROOT}/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410"
GATE_PROFILE="${REPO_ROOT}/configs/dataset_profiles/scene_v7.json"
SCENE_TEMPLATE="${REPO_ROOT}/configs/scene_template.json"
SCENE_POOL=""

# ── 默认参数 ──────────────────────────────────────────────────────────────────
DEVICE="cuda:0"
I23D_DEVICES="cuda:0,cuda:1,cuda:2"
EXPORT_GPUS="0,1,2"
SEED_COUNT=1
SEED_START_INDEX=""   # 空 = 自动计算
SEED_MODEL="claude-opus-4-6"
API_PROVIDER="anthropic"
API_BASE_URL="https://dragoncode.codes/v1"
API_TIMEOUT=300
GATE_MAX_ROUNDS=5
GATE_THRESHOLD="0.78"
GATE_PATIENCE=0       # 0 = 用完 max_rounds 全部预算，不提前因 patience 退出
TEMPLATE_ONLY=0
AUGMENT_BASELINE=1
DRY_RUN=0
EXPORT_ROTATIONS="${EXPORT_ROTATIONS:-0,45,90,135,180,225,270,315}"
TARGET_ROTATIONS="${TARGET_ROTATIONS:-45,90,135,180,225,270,315}"
ANGLE_GATE_THRESHOLD="0.50"
ANGLE_GATE_MAX_ROUNDS=2
STOP_AFTER_GATE=0
FORCE_ACCEPT=1
RESEARCH_PRIOR_PATH=""

# ── 解析参数 ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)           DEVICE="$2"; shift 2 ;;
        --i23d-devices)     I23D_DEVICES="$2"; shift 2 ;;
        --export-gpus)      EXPORT_GPUS="$2"; shift 2 ;;
        --seed-count)       SEED_COUNT="$2"; shift 2 ;;
        --seed-start-index) SEED_START_INDEX="$2"; shift 2 ;;
        --seed-model)       SEED_MODEL="$2"; shift 2 ;;
        --api-provider)     API_PROVIDER="$2"; shift 2 ;;
        --api-base-url)     API_BASE_URL="$2"; shift 2 ;;
        --research-prior-path) RESEARCH_PRIOR_PATH="$2"; shift 2 ;;
        --gate-max-rounds)  GATE_MAX_ROUNDS="$2"; shift 2 ;;
        --gate-threshold)   GATE_THRESHOLD="$2"; shift 2 ;;
        --gate-patience)    GATE_PATIENCE="$2"; shift 2 ;;
        --angle-gate-threshold) ANGLE_GATE_THRESHOLD="$2"; shift 2 ;;
        --angle-gate-max-rounds) ANGLE_GATE_MAX_ROUNDS="$2"; shift 2 ;;
        --template-only)    TEMPLATE_ONLY=1; shift ;;
        --augment-baseline) AUGMENT_BASELINE=1; shift ;;
        --no-augment)       AUGMENT_BASELINE=0; shift ;;
        --stop-after-gate)  STOP_AFTER_GATE=1; shift ;;
        --no-force-accept)  FORCE_ACCEPT=0; shift ;;
        --scene-template)   SCENE_TEMPLATE="$2"; shift 2 ;;
        --scene-pool)       SCENE_POOL="$2"; shift 2 ;;
        --target-rotations) TARGET_ROTATIONS="$2"; shift 2 ;;
        --export-rotations) EXPORT_ROTATIONS="$2"; shift 2 ;;
        --dry-run)          DRY_RUN=1; shift ;;
        -h|--help)
            head -35 "$0" | tail -30
            exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

# ── 自动计算起始 index ────────────────────────────────────────────────────────
if [[ -z "$SEED_START_INDEX" ]]; then
    SEED_START_INDEX=$(find "$RUNTIME_ROOT" -name 'seed_concepts.json' -exec grep -h '"id"' {} + 2>/dev/null \
        | sed 's/.*obj_\([0-9]*\).*/\1/' | sort -n | tail -1)
    SEED_START_INDEX=$(( ${SEED_START_INDEX:-1100} + 1 ))
    echo "[init] Auto seed start index: $SEED_START_INDEX"
fi

# ── 创建运行目录 ──────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="new_asset_force_rot8_${TIMESTAMP}"
RUN_ROOT="${RUNTIME_ROOT}/${RUN_NAME}"
LOGS_DIR="${RUN_ROOT}/logs"
mkdir -p "$LOGS_DIR"

echo "============================================================"
echo "[launch] 新资产 + 强制 rotation8 流水线"
echo "[launch] 运行目录: ${RUN_ROOT}"
echo "[launch] 设备: ${DEVICE}"
echo "[launch] 生成数量: ${SEED_COUNT}, 起始 ID: obj_${SEED_START_INDEX}"
echo "[launch] VLM 最大轮数: ${GATE_MAX_ROUNDS}, 接受阈值: ${GATE_THRESHOLD}"
echo "[launch] 强制进入 rotation8: 是（无论 VLM 是否接受）"
echo "[launch] 角度质量门控: 阈值=${ANGLE_GATE_THRESHOLD}, 最大增强轮数=${ANGLE_GATE_MAX_ROUNDS}"
if [[ -n "$SCENE_POOL" ]]; then
echo "[launch] 场景池: ${SCENE_POOL}"
else
echo "[launch] 场景模板: ${SCENE_TEMPLATE}"
fi
echo "============================================================"

SEED_FILE="${RUN_ROOT}/seed_concepts.json"
ASSETS_ROOT="${RUN_ROOT}/stage1_assets"
PROMPTS_FILE="${ASSETS_ROOT}/prompts.json"
GATE_ROOT="${RUN_ROOT}/gate_loop"
ACCEPTED_ROOT="${RUN_ROOT}/accepted_pairs"
ROT8_ROOT="${RUN_ROOT}/rotation8_consistent"
ROT8_ENHANCED_ROOT="${RUN_ROOT}/rotation8_enhanced"
TRAINREADY_ROOT="${RUN_ROOT}/trainready"
AUGMENTED_ROOT="${RUN_ROOT}/augmented_trainready_split"
PINNED_SPLIT="${RUN_ROOT}/pinned_split.json"
LOOP_STATE="${RUN_ROOT}/LOOP_STATE.json"
GATE_PROFILE_LOCAL="${RUN_ROOT}/scene_v7_gate_profile.json"

# ── 辅助函数 ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

run_step() {
    local step_num="$1"
    local step_name="$2"
    shift 2
    local cmd=("$@")

    log "=== [${step_num}] ${step_name} ==="
    log "CMD: ${cmd[*]}"

    local log_file="${LOGS_DIR}/${step_num}_${step_name}.log"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] ${cmd[*]}" | tee "$log_file"
        return 0
    fi

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

# ── 写入 gate profile（设置 reference_images_dir）────────────────────────────
$PYTHON -c "
import json
profile = json.load(open('${GATE_PROFILE}'))
profile['reference_images_dir'] = '${ASSETS_ROOT}/images'
profile['max_rounds'] = ${GATE_MAX_ROUNDS}
profile['accept_threshold'] = ${GATE_THRESHOLD}
profile['patience'] = ${GATE_PATIENCE}
with open('${GATE_PROFILE_LOCAL}', 'w') as f:
    json.dump(profile, f, indent=2)
print('[init] Gate profile written:', '${GATE_PROFILE_LOCAL}')
"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: 生成 seed concepts
# ══════════════════════════════════════════════════════════════════════════════
STAGE1_ARGS=(
    "$PYTHON" "${REPO_ROOT}/scripts/stage1_generate_ai_seed_concepts.py"
    --output-file "$SEED_FILE"
    --count "$SEED_COUNT"
    --start-index "$SEED_START_INDEX"
)
if [[ -n "$RESEARCH_PRIOR_PATH" ]]; then
    STAGE1_ARGS+=( --research-prior-path "$RESEARCH_PRIOR_PATH" )
fi

if [[ "$TEMPLATE_ONLY" == "1" ]]; then
    STAGE1_ARGS+=( --template-only )
else
    STAGE1_ARGS+=(
        --model "$SEED_MODEL"
        --api-provider "$API_PROVIDER"
        --api-base-url "$API_BASE_URL"
        --api-timeout "$API_TIMEOUT"
    )
fi

run_step "01" "generate_seed_concepts" "${STAGE1_ARGS[@]}"

log "Seed concepts:"
cat "$SEED_FILE"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: 构建 Stage1→Stage3 资产 (T2I → SAM2 → 3D mesh)
# ══════════════════════════════════════════════════════════════════════════════
STAGE1_BUILD_ARGS=(
    "$PYTHON" "${REPO_ROOT}/scripts/build_scene_assets_from_stage1.py"
    --objects-file "$SEED_FILE"
    --output-root "$ASSETS_ROOT"
    --python "$PYTHON"
    --t2i-device "$DEVICE"
    --sam-device "$DEVICE"
    --i23d-devices "$I23D_DEVICES"
    --no-skip
)
if [[ -n "$RESEARCH_PRIOR_PATH" ]]; then
    STAGE1_BUILD_ARGS+=( --research-prior-path "$RESEARCH_PRIOR_PATH" )
fi

if [[ "$TEMPLATE_ONLY" == "0" ]]; then
    STAGE1_BUILD_ARGS+=(
        --stage1-mode api
        --stage1-model "$SEED_MODEL"
        --stage1-api-provider "$API_PROVIDER"
        --stage1-api-base-url "$API_BASE_URL"
        --stage1-api-timeout "$API_TIMEOUT"
    )
else
    STAGE1_BUILD_ARGS+=( --stage1-mode template-only )
fi

run_step "02" "build_stage1_assets" "${STAGE1_BUILD_ARGS[@]}"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: VLM 质量门控 loop（最多 GATE_MAX_ROUNDS 轮渲染优化）
# ══════════════════════════════════════════════════════════════════════════════
GATE_CMD=(
    "$PYTHON" "${REPO_ROOT}/scripts/run_vlm_quality_gate_loop.py"
    --root-dir "$GATE_ROOT"
    --objects-file "$SEED_FILE"
    --rotations 0
    --threshold-start "$GATE_THRESHOLD"
    --threshold-step 0.0
    --threshold-max "$GATE_THRESHOLD"
    --score-field hybrid_score
    --max-rounds "$GATE_MAX_ROUNDS"
    --min-vlm-only-score 0.0
    --step-scale 1.0
    --patience "$GATE_PATIENCE"
    --rollback-score-drop-threshold 0.05
    --device "$DEVICE"
    --gpus "$EXPORT_GPUS"
    --blender "$BLENDER"
    --meshes-dir "${ASSETS_ROOT}/meshes"
    --profile "$GATE_PROFILE_LOCAL"
    --scene-template "$SCENE_TEMPLATE"
)
if [[ -n "$SCENE_POOL" ]]; then
    GATE_CMD+=(--scene-pool "$SCENE_POOL")
fi
run_step "03" "run_vlm_quality_gate" "${GATE_CMD[@]}"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3.5: 强制接受 — 将被拒绝的 object 也标记为 accepted
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$FORCE_ACCEPT" == "1" ]]; then
log "=== [03.5] force_accept_all ==="
log "无论 VLM 是否接受，将所有 object 强制标记为 accepted"

$PYTHON - "$GATE_ROOT" <<'FORCE_ACCEPT_PY'
import json
import sys
from pathlib import Path

gate_root = Path(sys.argv[1])
manifest_path = gate_root / "gate_manifest.json"
if not manifest_path.exists():
    print(f"[force_accept] ERROR: {manifest_path} not found")
    sys.exit(1)

manifest = json.loads(manifest_path.read_text())
rejected = manifest.get("rejected", [])
accepted = manifest.get("accepted", [])
force_promoted = []

for item in rejected:
    obj_id = item.get("obj_id", "unknown")
    pair_dir = item.get("pair_dir", "")
    best_score = item.get("best_score", 0)
    best_round_idx = item.get("best_round_idx", 0)

    promoted = {
        "status": "accepted",
        "obj_id": obj_id,
        "rotation_deg": item.get("rotation_deg", 0),
        "pair_dir": pair_dir,
        "accepted_round_idx": best_round_idx,
        "accepted_score": best_score,
        "accepted_threshold": 0.0,
        "score_field": item.get("score_field", "hybrid_score"),
        "force_accepted": True,
        "original_reject_reason": item.get("reason", "unknown"),
        "best_score": best_score,
        "best_round_idx": best_round_idx,
        "best_state_out": item.get("best_state_out"),
        "best_render_dir": item.get("best_render_dir"),
        "best_review_agg_path": item.get("best_review_agg_path"),
        "state_out": item.get("best_state_out"),
        "render_dir": item.get("best_render_dir"),
        "review_agg_path": item.get("best_review_agg_path"),
        "rounds": item.get("rounds", []),
        "elapsed_seconds": item.get("elapsed_seconds", 0),
    }
    accepted.append(promoted)
    force_promoted.append(obj_id)
    print(f"  [force_accept] {obj_id}: rejected -> accepted (best_score={best_score:.3f}, round={best_round_idx})")

    gate_status_path = Path(pair_dir) / "gate_status.json"
    if gate_status_path.exists():
        gs = json.loads(gate_status_path.read_text())
        gs["status"] = "accepted"
        gs["force_accepted"] = True
        gs["accepted_round_idx"] = best_round_idx
        gs["accepted_score"] = best_score
        gs["accepted_threshold"] = 0.0
        gate_status_path.write_text(json.dumps(gs, indent=2, ensure_ascii=False))

manifest["accepted"] = accepted
manifest["rejected"] = []
manifest["accepted_count"] = len(accepted)
manifest["rejected_count"] = 0
manifest["force_promoted"] = force_promoted

manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(f"[force_accept] 完成: {len(force_promoted)} objects 强制接受, 总计 {len(accepted)} accepted")
FORCE_ACCEPT_PY

else
    log "=== [03.5] skip force_accept (--no-force-accept) ==="
    log "被拒绝的 object 不会被强制接受，将被淘汰"
fi

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: 构建 accepted pair root
# ══════════════════════════════════════════════════════════════════════════════
run_step "04" "build_accepted_pair_root" \
    "$PYTHON" "${REPO_ROOT}/scripts/build_accepted_pair_root_from_gate.py" \
    --gate-root "$GATE_ROOT" \
    --output-dir "$ACCEPTED_ROOT" \
    --asset-mode symlink \
    --overwrite

# ── 如果只需要跑到 gate，这里退出 ──
if [[ "$STOP_AFTER_GATE" == "1" ]]; then
    log "=== [stop-after-gate] 步骤 01-04 完成，跳过 rotation8 及后续步骤 ==="
    $PYTHON - <<WRITE_GATE_STATE
import json
from datetime import datetime
state = {
    "schema_version": "new_asset_force_rot8_v1",
    "loop_root": "${RUN_ROOT}",
    "seed_concepts_file": "${SEED_FILE}",
    "research_prior_path": "${RESEARCH_PRIOR_PATH}" or None,
    "stage1_assets_root": "${ASSETS_ROOT}",
    "gate_root": "${GATE_ROOT}",
    "accepted_pair_root": "${ACCEPTED_ROOT}",
    "stopped_after": "gate",
    "status": "gate_completed",
    "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
with open("${LOOP_STATE}", "w") as f:
    json.dump(state, f, indent=2, ensure_ascii=False)
WRITE_GATE_STATE
    echo ""
    echo "============================================================"
    echo "[完成] Gate 阶段完成（--stop-after-gate）"
    echo "  运行根目录:       ${RUN_ROOT}"
    echo "  Accepted pairs:   ${ACCEPTED_ROOT}"
    echo "  LOOP_STATE:       ${LOOP_STATE}"
    echo "============================================================"
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: 导出 rotation8 一致性数据集（8 个 yaw 角度渲染）
# ══════════════════════════════════════════════════════════════════════════════
ROT8_CMD=(
    "$PYTHON" "${REPO_ROOT}/scripts/export_rotation8_from_best_object_state.py"
    --source-root "$ACCEPTED_ROOT"
    --output-dir "$ROT8_ROOT"
    --python "$PYTHON"
    --blender "$BLENDER"
    --meshes-dir "${ASSETS_ROOT}/meshes"
    --gpus "$EXPORT_GPUS"
    --fallback-to-best-any-angle
    --rotations "$EXPORT_ROTATIONS"
    --scene-template "$SCENE_TEMPLATE"
)
if [[ -n "$SCENE_POOL" ]]; then
    ROT8_CMD+=(--scene-pool "$SCENE_POOL")
    # Reuse gate loop's scene assignments if available
    if [[ -f "${GATE_ROOT}/scene_assignments.json" ]]; then
        ROT8_CMD+=(--scene-assignments "${GATE_ROOT}/scene_assignments.json")
    fi
fi
run_step "05" "export_rotation8" "${ROT8_CMD[@]}"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5.5: Per-angle 质量门控 + 弱角度增强
# ══════════════════════════════════════════════════════════════════════════════
run_step "05.5" "rotation8_angle_gate" \
    "$PYTHON" "${REPO_ROOT}/scripts/run_rotation8_angle_gate.py" \
    --source-root "$ROT8_ROOT" \
    --output-dir "$ROT8_ENHANCED_ROOT" \
    --meshes-dir "${ASSETS_ROOT}/meshes" \
    --blender "$BLENDER" \
    --profile "$GATE_PROFILE_LOCAL" \
    --device "$DEVICE" \
    --angle-threshold "$ANGLE_GATE_THRESHOLD" \
    --max-enhance-rounds "$ANGLE_GATE_MAX_ROUNDS" \
    --copy-mode symlink

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: 构建 train-ready 数据集（从增强后的 rotation8）
# ══════════════════════════════════════════════════════════════════════════════
run_step "06" "build_trainready" \
    "$PYTHON" "${REPO_ROOT}/scripts/build_rotation8_trainready_dataset.py" \
    --source-root "$ROT8_ENHANCED_ROOT" \
    --output-dir "$TRAINREADY_ROOT" \
    --copy-mode symlink \
    --target-rotations "$TARGET_ROTATIONS" \
    --prompts-json "$PROMPTS_FILE"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7-9: （可选）合并到 baseline 数据集
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$AUGMENT_BASELINE" == "1" && -d "$BASELINE_SPLIT" ]]; then
    log "=== 合并到 baseline 数据集 ==="

    run_step "07" "build_pinned_split" \
        "$PYTHON" "${REPO_ROOT}/scripts/feedback_loop/build_pinned_split.py" \
        --dataset-root "$BASELINE_SPLIT" \
        --output "$PINNED_SPLIT" \
        --source-note "new_asset_force_rot8 baseline split"

    run_step "08" "build_augmented_dataset" \
        "$PYTHON" "${REPO_ROOT}/scripts/feedback_loop/build_augmented_rotation_dataset.py" \
        --baseline-split-root "$BASELINE_SPLIT" \
        --new-trainready-root "$TRAINREADY_ROOT" \
        --output-dir "$AUGMENTED_ROOT" \
        --pinned-split-path "$PINNED_SPLIT" \
        --asset-mode symlink \
        --target-rotations "$TARGET_ROTATIONS"

    run_step "09" "validate_final_dataset" \
        "$PYTHON" "${REPO_ROOT}/scripts/feedback_loop/validate_dataset.py" \
        "$AUGMENTED_ROOT" \
        --output "${RUN_ROOT}/validation_report.json" \
        --pinned-split-path "$PINNED_SPLIT" \
        --expected-manifest "${AUGMENTED_ROOT}/augmented_manifest.json"
fi

# ══════════════════════════════════════════════════════════════════════════════
# 写入 LOOP_STATE.json
# ══════════════════════════════════════════════════════════════════════════════
$PYTHON - <<WRITE_STATE
import json
from datetime import datetime
state = {
    "schema_version": "new_asset_force_rot8_v1",
    "loop_root": "${RUN_ROOT}",
    "seed_concepts_file": "${SEED_FILE}",
    "research_prior_path": "${RESEARCH_PRIOR_PATH}" or None,
    "stage1_assets_root": "${ASSETS_ROOT}",
    "gate_root": "${GATE_ROOT}",
    "accepted_pair_root": "${ACCEPTED_ROOT}",
    "rotation8_export_root": "${ROT8_ROOT}",
    "rotation8_enhanced_root": "${ROT8_ENHANCED_ROOT}",
    "trainready_root": "${TRAINREADY_ROOT}",
    "augmented_root": "${AUGMENTED_ROOT}" if ${AUGMENT_BASELINE} else None,
    "gate_max_rounds": ${GATE_MAX_ROUNDS},
    "gate_threshold": ${GATE_THRESHOLD},
    "force_accept_all": True,
    "angle_gate_threshold": ${ANGLE_GATE_THRESHOLD},
    "angle_gate_max_rounds": ${ANGLE_GATE_MAX_ROUNDS},
    "export_rotations": "${EXPORT_ROTATIONS}",
    "target_rotations": "${TARGET_ROTATIONS}",
    "device": "${DEVICE}",
    "status": "completed",
    "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
with open("${LOOP_STATE}", "w") as f:
    json.dump(state, f, indent=2, ensure_ascii=False)
WRITE_STATE

# ══════════════════════════════════════════════════════════════════════════════
# 完成
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "[完成] 新资产 + 强制 rotation8 流水线"
echo ""
echo "  运行根目录:       ${RUN_ROOT}"
echo "  Seed concepts:    ${SEED_FILE}"
echo "  Stage1 资产:      ${ASSETS_ROOT}"
echo "  VLM Gate:         ${GATE_ROOT}"
echo "  Accepted pairs:   ${ACCEPTED_ROOT}"
echo "  Rotation8 渲染:   ${ROT8_ROOT}"
echo "  Rotation8 增强:   ${ROT8_ENHANCED_ROOT}"
echo "  Train-ready:      ${TRAINREADY_ROOT}"
if [[ "$AUGMENT_BASELINE" == "1" ]]; then
echo "  增强数据集:       ${AUGMENTED_ROOT}"
fi
echo ""
echo "  LOOP_STATE:       ${LOOP_STATE}"
echo "  日志:             ${LOGS_DIR}"
echo "============================================================"
