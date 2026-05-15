#!/usr/bin/env bash
# run_all.sh — End-to-end pipeline: Text → Image → 3D → Render → Metadata
#
# Usage:
#   bash pipeline/run_all.sh                  # Run all stages
#   bash pipeline/run_all.sh --from-stage 2   # Resume from Stage 2
#   bash pipeline/run_all.sh --stages 1,2     # Run only Stage 1 and 2
#
# Prerequisites (run once):
#   conda activate <local-path>
#   pip install diffsynth trimesh rembg[gpu] Pillow
#   cd <large-file-root> && git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
#   cd Hunyuan3D-2.1 && pip install -r requirements.txt

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
LOG_DIR="${DATA_DIR}/logs"
CONDA_ENV="<local-path>"
PYTHON="${CONDA_ENV}/bin/python"
BLENDER_BIN="${BLENDER_BIN:-blender}"

FROM_STAGE=1
RUN_STAGES="1,2,2.5,3,4,5"
DEVICE="${DEVICE:-cuda:0}"
SKIP_STAGE1=0
FORCE_STAGE1=0

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from-stage) FROM_STAGE="$2"; shift 2 ;;
        --stages)     RUN_STAGES="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        --skip-stage1) SKIP_STAGE1=1; shift 1 ;;
        --force-stage1) FORCE_STAGE1=1; shift 1 ;;
        *) echo "[run_all] Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="${LOG_DIR}/run_all_${TIMESTAMP}.log"

echo "[run_all] Pipeline started at $(date)" | tee "$MASTER_LOG"
echo "[run_all] Stages: $RUN_STAGES (from stage $FROM_STAGE)" | tee -a "$MASTER_LOG"
echo "[run_all] Device: $DEVICE" | tee -a "$MASTER_LOG"
echo "[run_all] Log: $MASTER_LOG" | tee -a "$MASTER_LOG"

# ── Helper: run a stage ───────────────────────────────────────────────────────
should_run() {
    local stage=$1
    # Support decimal stage numbers (e.g. 3.5) via awk for numeric comparison
    local ok_from
    ok_from=$(awk -v s="$stage" -v f="$FROM_STAGE" 'BEGIN { print (s >= f) ? "yes" : "no" }')
    [[ "$ok_from" == "yes" ]] && echo "$RUN_STAGES" | tr ',' '\n' | grep -qx "${stage}"
}

run_stage() {
    local stage=$1
    local name=$2
    local cmd=$3

    if ! should_run "$stage"; then
        echo "[run_all] Skipping Stage $stage ($name)" | tee -a "$MASTER_LOG"
        return 0
    fi

    echo "" | tee -a "$MASTER_LOG"
    echo "════════════════════════════════════════════" | tee -a "$MASTER_LOG"
    echo "[run_all] Stage $stage: $name — $(date)" | tee -a "$MASTER_LOG"
    echo "════════════════════════════════════════════" | tee -a "$MASTER_LOG"

    local stage_log="${LOG_DIR}/stage${stage}_${TIMESTAMP}.log"
    if eval "$cmd" 2>&1 | tee "$stage_log" | tee -a "$MASTER_LOG"; then
        echo "[run_all] ✓ Stage $stage completed" | tee -a "$MASTER_LOG"
    else
        echo "[run_all] ✗ Stage $stage FAILED (see $stage_log)" | tee -a "$MASTER_LOG"
        exit 1
    fi
}

build_stage1_cmd() {
    if [[ "$SKIP_STAGE1" == "1" ]]; then
        echo "echo '[run_all] Stage 1 skipped by --skip-stage1'"
        return 0
    fi

    if [[ "$FORCE_STAGE1" == "1" ]]; then
        echo "$PYTHON ${SCRIPT_DIR}/stage1_text_expansion.py"
        return 0
    fi

    if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
        echo "$PYTHON ${SCRIPT_DIR}/stage1_text_expansion.py"
        return 0
    fi

    if [[ -f "${DATA_DIR}/prompts.json" ]]; then
        echo "echo '[run_all] ANTHROPIC_API_KEY not set; reusing existing prompts.json and skipping Stage 1'"
        return 0
    fi

    echo "echo '[run_all] ERROR: Stage 1 requires ANTHROPIC_API_KEY, or an existing ${DATA_DIR}/prompts.json. Generate prompts locally first, or rerun with --from-stage 2 / --skip-stage1 once prompts.json is uploaded.' >&2; exit 1"
}

# ── Stage 1: Text Expansion ───────────────────────────────────────────────────
STAGE1_CMD="$(build_stage1_cmd)"
run_stage 1 "Text Expansion" \
    "$STAGE1_CMD"

# ── Stage 2: Text-to-Image ────────────────────────────────────────────────────
run_stage 2 "Text-to-Image (Qwen-Image-2512)" \
    "$PYTHON ${SCRIPT_DIR}/stage2_t2i_generate.py --device $DEVICE"

# ── Stage 2.5: SAM2 Foreground Segmentation ───────────────────────────────────
run_stage 2.5 "SAM2 Foreground Segmentation" \
    "$PYTHON ${SCRIPT_DIR}/stage2_5_sam2_segment.py --device $DEVICE"

# ── Stage 3: Image-to-3D ─────────────────────────────────────────────────────
run_stage 3 "Image-to-3D (Hunyuan3D-2.1)" \
    "$PYTHON ${SCRIPT_DIR}/stage3_image_to_3d.py --device $DEVICE && mkdir -p \"${DATA_DIR}/meshes\" && find \"${DATA_DIR}/meshes_raw\" -maxdepth 1 -type f -name '*.glb' -exec cp -f {} \"${DATA_DIR}/meshes/\" \\; && echo '[run_all] Stage 3 canonical meshes refreshed from meshes_raw/'"

# ── Stage 4: Blender Rendering ───────────────────────────────────────────────
run_stage 4 "Multi-view Rendering (Blender EEVEE)" \
    "bash ${SCRIPT_DIR}/stage4_batch_render.sh"

# ── Stage 5: Merge Metadata ───────────────────────────────────────────────────
run_stage 5 "Merge Metadata & Generate CSVs" \
    "$PYTHON ${SCRIPT_DIR}/stage5_merge_metadata.py"

# ── Final summary ─────────────────────────────────────────────────────────────
echo "" | tee -a "$MASTER_LOG"
echo "════════════════════════════════════════════" | tee -a "$MASTER_LOG"
echo "[run_all] Pipeline complete at $(date)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Count outputs
if [[ -f "${DATA_DIR}/prompts.json" ]]; then
    echo "  Stage 1 ✓ prompts.json" | tee -a "$MASTER_LOG"
fi
IMG_COUNT=$(find "${DATA_DIR}/images" -name "*.png" 2>/dev/null | wc -l)
echo "  Stage 2 ✓ $IMG_COUNT images" | tee -a "$MASTER_LOG"
RGBA_COUNT=$(find "${DATA_DIR}/images_rgba" -name "*.png" 2>/dev/null | wc -l)
echo "  Stage 2.5 ✓ $RGBA_COUNT RGBA images" | tee -a "$MASTER_LOG"
RAW_MESH_COUNT=$(find "${DATA_DIR}/meshes_raw" -name "*.glb" 2>/dev/null | wc -l)
echo "  Stage 3 ✓ $RAW_MESH_COUNT raw meshes (meshes_raw/)" | tee -a "$MASTER_LOG"
MESH_COUNT=$(find "${DATA_DIR}/meshes" -name "*.glb" 2>/dev/null | wc -l)
echo "  Stage 3 ✓ $MESH_COUNT active meshes (meshes/)" | tee -a "$MASTER_LOG"
RENDER_COUNT=$(find "${DATA_DIR}/renders" -name "*.png" 2>/dev/null | wc -l)
echo "  Stage 4 ✓ $RENDER_COUNT renders" | tee -a "$MASTER_LOG"
if [[ -f "${DATA_DIR}/image_metadata.json" ]]; then
    echo "  Stage 5 ✓ image_metadata.json + pairs CSVs" | tee -a "$MASTER_LOG"
fi

echo "════════════════════════════════════════════" | tee -a "$MASTER_LOG"
echo "[run_all] All stages complete!" | tee -a "$MASTER_LOG"
