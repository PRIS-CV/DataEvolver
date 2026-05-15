#!/usr/bin/env bash
# ============================================================================
# launch_30obj_sequential_scenes.sh
#
# 30 个物体按场景串行执行全流程：5 个场景 × 6 个物体
# 每个场景的全流程（T2I → SAM3 → 3D → VLM gate → rotation8 → trainready）
# 跑完后清理 GPU 内存，再启动下一个场景，规避 OOM。
#
# 用法：
#   tmux new -s scene-sequential
#   bash ~/ARIS/scripts/launch_30obj_sequential_scenes.sh [选项]
#
# 选项：
#   --seed-start-index N   obj_id 起始编号 (默认: 自动计算)
#   --dry-run              仅打印命令，不执行
#   --gate-max-rounds N    VLM gate 最大轮数 (默认: 5)
#   --no-augment           不合并到 baseline
#   --resume-from N        从第 N 个场景恢复 (1-5, 默认: 1)
# ============================================================================

set -eo pipefail

REPO_ROOT="$HOME/ARIS"
PYTHON="python"
RUNTIME_ROOT="<runtime-root>"
LAUNCH_SCRIPT="${REPO_ROOT}/scripts/launch_new_asset_force_rot8.sh"

export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-<REDACTED_API_KEY>}"

# ── 场景定义（与 scene_pool.json 一致）──────────────────────────────────────
SCENE_NAMES=(
    "indoor_4blend"
    "outdoor15"
    "outdoor9"
    "outdoor13"
    "outdoor16"
)
SCENE_TEMPLATES=(
    "${REPO_ROOT}/configs/scene_template.json"
    "${REPO_ROOT}/configs/scene_template_outdoor15.json"
    "${REPO_ROOT}/configs/scene_template_outdoor9.json"
    "${REPO_ROOT}/configs/scene_template_outdoor13.json"
    "${REPO_ROOT}/configs/scene_template_outdoor16.json"
)
OBJECTS_PER_SCENE=6

# ── 默认参数 ──────────────────────────────────────────────────────────────────
SEED_START_INDEX=""
DRY_RUN=0
GATE_MAX_ROUNDS=5
AUGMENT_FLAG="--augment-baseline"
RESUME_FROM=1
EXTRA_ARGS=()

# ── 解析参数 ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed-start-index) SEED_START_INDEX="$2"; shift 2 ;;
        --dry-run)          DRY_RUN=1; shift ;;
        --gate-max-rounds)  GATE_MAX_ROUNDS="$2"; shift 2 ;;
        --no-augment)       AUGMENT_FLAG="--no-augment"; shift ;;
        --resume-from)      RESUME_FROM="$2"; shift 2 ;;
        *)                  EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── 自动计算起始 index ────────────────────────────────────────────────────────
if [[ -z "$SEED_START_INDEX" ]]; then
    SEED_START_INDEX=$(find "$RUNTIME_ROOT" -name 'seed_concepts.json' -exec grep -h '"id"' {} + 2>/dev/null \
        | sed 's/.*obj_\([0-9]*\).*/\1/' | sort -n | tail -1)
    SEED_START_INDEX=$(( ${SEED_START_INDEX:-1100} + 1 ))
fi

# ── 总运行目录 ────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_ROOT="${RUNTIME_ROOT}/sequential_30obj_5scenes_${TIMESTAMP}"
MASTER_LOG="${MASTER_ROOT}/master.log"
mkdir -p "$MASTER_ROOT"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"; }

log "============================================================"
log "30 物体 × 5 场景 串行全流程"
log "总运行目录:    ${MASTER_ROOT}"
log "起始 obj_id:   obj_${SEED_START_INDEX}"
log "每场景物体数:  ${OBJECTS_PER_SCENE}"
log "VLM 最大轮数:  ${GATE_MAX_ROUNDS}"
log "从场景 #${RESUME_FROM} 开始"
log "============================================================"

cleanup_gpu() {
    log "[cleanup] 清理 GPU 内存..."
    # 杀死所有 python 占用 GPU 的残留进程（排除当前 shell）
    local pids
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u || true)
    if [[ -n "$pids" ]]; then
        for pid in $pids; do
            local cmdline
            cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ' || true)
            if [[ "$cmdline" == *python* ]] || [[ "$cmdline" == *blender* ]]; then
                log "[cleanup]   killing PID $pid: $cmdline"
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    fi
    sleep 5
    log "[cleanup] GPU 状态:"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$MASTER_LOG"
}

check_gpu_clean() {
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s}')
    if [[ "${used:-0}" -gt 500 ]]; then
        log "[WARNING] GPU 内存仍有 ${used}MiB 占用，尝试再次清理..."
        cleanup_gpu
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s}')
        if [[ "${used:-0}" -gt 500 ]]; then
            log "[WARNING] GPU 清理后仍有 ${used}MiB 占用，继续执行..."
        fi
    fi
    log "[check] GPU 内存占用: ${used:-0}MiB — OK"
}

# ── 保存运行清单 ──────────────────────────────────────────────────────────────
MANIFEST="${MASTER_ROOT}/scene_manifest.json"
$PYTHON - "$MANIFEST" "$SEED_START_INDEX" "$OBJECTS_PER_SCENE" \
    "${SCENE_NAMES[@]}" "---" "${SCENE_TEMPLATES[@]}" <<'MANIFEST_PY'
import json, sys

out_path = sys.argv[1]
start = int(sys.argv[2])
per = int(sys.argv[3])

rest = sys.argv[4:]
sep = rest.index("---")
names = rest[:sep]
templates = rest[sep+1:]

scenes = []
for i, (name, tmpl) in enumerate(zip(names, templates)):
    obj_start = start + i * per
    scenes.append({
        "scene_index": i + 1,
        "scene_name": name,
        "template": tmpl,
        "obj_start_index": obj_start,
        "obj_end_index": obj_start + per - 1,
        "obj_count": per,
    })
manifest = {
    "total_objects": per * len(names),
    "total_scenes": len(names),
    "objects_per_scene": per,
    "scenes": scenes,
}
with open(out_path, "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(json.dumps(manifest, indent=2, ensure_ascii=False))
MANIFEST_PY
log "场景清单已写入: ${MANIFEST}"

# ── 主循环：逐场景串行 ────────────────────────────────────────────────────────
TOTAL_SCENES=${#SCENE_NAMES[@]}
SCENE_RESULTS=()
OVERALL_START=$(date +%s)

for i in $(seq 0 $((TOTAL_SCENES - 1))); do
    SCENE_NUM=$((i + 1))
    SCENE_NAME="${SCENE_NAMES[$i]}"
    SCENE_TMPL="${SCENE_TEMPLATES[$i]}"
    OBJ_START=$((SEED_START_INDEX + i * OBJECTS_PER_SCENE))

    if [[ $SCENE_NUM -lt $RESUME_FROM ]]; then
        log "[skip] 场景 #${SCENE_NUM} (${SCENE_NAME}) — 在 resume-from 之前，跳过"
        SCENE_RESULTS+=("SKIPPED")
        continue
    fi

    log ""
    log "╔══════════════════════════════════════════════════════════════╗"
    log "║  场景 #${SCENE_NUM}/${TOTAL_SCENES}: ${SCENE_NAME}"
    log "║  模板: ${SCENE_TMPL}"
    log "║  物体: obj_${OBJ_START} ~ obj_$((OBJ_START + OBJECTS_PER_SCENE - 1))"
    log "╚══════════════════════════════════════════════════════════════╝"

    # 检查 GPU 内存是否干净
    check_gpu_clean

    SCENE_START=$(date +%s)

    LAUNCH_CMD=(
        bash "$LAUNCH_SCRIPT"
        --seed-count "$OBJECTS_PER_SCENE"
        --seed-start-index "$OBJ_START"
        --scene-template "$SCENE_TMPL"
        --gate-max-rounds "$GATE_MAX_ROUNDS"
        $AUGMENT_FLAG
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        LAUNCH_CMD+=(--dry-run)
    fi

    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        LAUNCH_CMD+=("${EXTRA_ARGS[@]}")
    fi

    SCENE_LOG="${MASTER_ROOT}/scene_${SCENE_NUM}_${SCENE_NAME}.log"

    log "[scene-${SCENE_NUM}] 启动命令: ${LAUNCH_CMD[*]}"

    if "${LAUNCH_CMD[@]}" 2>&1 | tee "$SCENE_LOG"; then
        SCENE_ELAPSED=$(( $(date +%s) - SCENE_START ))
        log "[scene-${SCENE_NUM}] ✓ ${SCENE_NAME} 完成 (${SCENE_ELAPSED}s / $(( SCENE_ELAPSED / 60 ))min)"
        SCENE_RESULTS+=("OK:${SCENE_ELAPSED}s")
    else
        SCENE_ELAPSED=$(( $(date +%s) - SCENE_START ))
        log "[scene-${SCENE_NUM}] ✗ ${SCENE_NAME} 失败 (${SCENE_ELAPSED}s)，日志: ${SCENE_LOG}"
        SCENE_RESULTS+=("FAIL:${SCENE_ELAPSED}s")
    fi

    # 场景完成后清理 GPU 内存
    if [[ $SCENE_NUM -lt $TOTAL_SCENES ]]; then
        log "[scene-${SCENE_NUM}] 场景完成，清理 GPU 准备下一个场景..."
        cleanup_gpu
    fi
done

# ── 汇总 ──────────────────────────────────────────────────────────────────────
OVERALL_ELAPSED=$(( $(date +%s) - OVERALL_START ))

log ""
log "============================================================"
log "  30 物体 × 5 场景 串行全流程 — 完成"
log ""
for i in $(seq 0 $((TOTAL_SCENES - 1))); do
    log "  场景 #$((i+1)) ${SCENE_NAMES[$i]}: ${SCENE_RESULTS[$i]}"
done
log ""
log "  总耗时: ${OVERALL_ELAPSED}s ($(( OVERALL_ELAPSED / 3600 ))h$(( (OVERALL_ELAPSED % 3600) / 60 ))m)"
log "  总运行目录: ${MASTER_ROOT}"
log "  日志: ${MASTER_LOG}"
log "============================================================"
