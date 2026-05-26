#!/usr/bin/env bash
# ============================================================================
# rerun_scene1_scene2.sh
#
# 重跑场景1(indoor_4blend) 和 场景2(outdoor15)
# ============================================================================
set -euo pipefail

export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY before launching the VLM gate}"

REPO_ROOT="$HOME/ARIS"
PYTHON="/home/jiazhuangzhuang/miniconda3/envs/aris/bin/python"
RUNTIME_ROOT="/aaaidata/jiazhuangzhuang/ARIS_runtime"
MASTER_LOG="${RUNTIME_ROOT}/rerun_scene12_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"; }

RESULTS=()

# ── 场景 1: indoor_4blend (全流程重跑) ───────────────────────────────────────
log ""
log ">>> [1/2] 场景 1: indoor_4blend — 全流程重跑"
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

# ── 场景 2: outdoor15 (全流程重跑) ───────────────────────────────────────
log ""
log ">>> [2/2] 场景 2: outdoor15 — 全流程重跑"
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
log "  完成 — 汇总"
log ""
for r in "${RESULTS[@]}"; do
    log "  $r"
done
log ""
log "  总日志: ${MASTER_LOG}"
log "============================================================"
