#!/usr/bin/env bash
# Stage 4: Batch Blender rendering — calls Blender for all .glb files
# Usage: bash pipeline/stage4_batch_render.sh [--single obj_001] [--resolution 512]
#                                              [--camera-distance 3.5]
#
# Blender 4.2+: /aaaidata/zhangqisong/blender-4.24/blender

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER_BIN="${BLENDER_BIN:-/aaaidata/zhangqisong/blender-4.24/blender}"
BLENDER_SCRIPT="${SCRIPT_DIR}/stage4_blender_render.py"
INPUT_DIR="${SCRIPT_DIR}/data/meshes"
OUTPUT_DIR="${SCRIPT_DIR}/data/renders"
RESOLUTION="${RESOLUTION:-512}"
ENGINE="${ENGINE:-CYCLES}"
CAMERA_DISTANCE="${CAMERA_DISTANCE:-3.5}"
FLOOR_ZCUT="${FLOOR_ZCUT:-0}"
SINGLE_OBJ=""
LOG_DIR="${SCRIPT_DIR}/data/logs"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --single)          SINGLE_OBJ="$2";       shift 2 ;;
        --resolution)      RESOLUTION="$2";        shift 2 ;;
        --engine)          ENGINE="$2";            shift 2 ;;
        --camera-distance) CAMERA_DISTANCE="$2";   shift 2 ;;
        --floor-zcut)      FLOOR_ZCUT="$2";        shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ ! -f "$BLENDER_BIN" ]]; then
    echo "[ERROR] Blender not found at $BLENDER_BIN"
    echo "  Set BLENDER_BIN=/path/to/blender"
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "[ERROR] Mesh directory not found: $INPUT_DIR"
    echo "  Run Stage 3 first to generate .glb files"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

GLB_FILES=( "$INPUT_DIR"/*.glb )
if [[ ${#GLB_FILES[@]} -eq 0 || ! -f "${GLB_FILES[0]}" ]]; then
    echo "[ERROR] No .glb files found in $INPUT_DIR"
    exit 1
fi

# ── Render ────────────────────────────────────────────────────────────────────
echo "[Stage 4] Blender: $BLENDER_BIN"
echo "[Stage 4] Input:   $INPUT_DIR (${#GLB_FILES[@]} meshes)"
echo "[Stage 4] Output:  $OUTPUT_DIR"
echo "[Stage 4] Resolution: ${RESOLUTION}x${RESOLUTION}, Engine: $ENGINE"
echo "[Stage 4] Camera distance: $CAMERA_DISTANCE"
echo "[Stage 4] Floor Z-cut: $FLOOR_ZCUT"

BLENDER_EXTRA_ARGS=()
if [[ "$FLOOR_ZCUT" != "0" ]]; then
    BLENDER_EXTRA_ARGS+=( --floor-zcut "$FLOOR_ZCUT" )
fi

if [[ -n "$SINGLE_OBJ" ]]; then
    echo "[Stage 4] Single object mode: $SINGLE_OBJ"
    LOG_FILE="${LOG_DIR}/render_${SINGLE_OBJ}.log"
    "$BLENDER_BIN" -b -P "$BLENDER_SCRIPT" -- \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --resolution "$RESOLUTION" \
        --engine "$ENGINE" \
        --camera-distance "$CAMERA_DISTANCE" \
        "${BLENDER_EXTRA_ARGS[@]}" \
        --obj-id "$SINGLE_OBJ" \
        2>&1 | tee "$LOG_FILE"
    echo "[Stage 4] Log → $LOG_FILE"
else
    echo "[Stage 4] Batch mode: rendering all ${#GLB_FILES[@]} objects in one Blender call"
    LOG_FILE="${LOG_DIR}/render_all_$(date +%Y%m%d_%H%M%S).log"
    "$BLENDER_BIN" -b -P "$BLENDER_SCRIPT" -- \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --resolution "$RESOLUTION" \
        --engine "$ENGINE" \
        --camera-distance "$CAMERA_DISTANCE" \
        "${BLENDER_EXTRA_ARGS[@]}" \
        2>&1 | tee "$LOG_FILE"
    echo "[Stage 4] Log → $LOG_FILE"
fi

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "[Stage 4] Verifying outputs ..."
TOTAL_RENDERS=0
MISSING_OBJS=0

for glb in "${GLB_FILES[@]}"; do
    OBJ_ID=$(basename "$glb" .glb)
    OBJ_DIR="${OUTPUT_DIR}/${OBJ_ID}"

    if [[ -n "$SINGLE_OBJ" && "$OBJ_ID" != "$SINGLE_OBJ" ]]; then
        continue
    fi

    if [[ ! -d "$OBJ_DIR" ]]; then
        echo "  [MISSING] $OBJ_ID — no output directory"
        ((MISSING_OBJS++))
        continue
    fi

    COUNT=$(find "$OBJ_DIR" -name "*.png" | wc -l)
    TOTAL_RENDERS=$((TOTAL_RENDERS + COUNT))
    echo "  $OBJ_ID: $COUNT renders"

    if [[ "$COUNT" -lt 48 ]]; then
        echo "  [WARN] Expected 48 renders, got $COUNT"
    fi
done

echo ""
echo "[Stage 4] Total renders: $TOTAL_RENDERS"
if [[ $MISSING_OBJS -gt 0 ]]; then
    echo "[Stage 4] WARNING: $MISSING_OBJS objects missing output"
    exit 1
fi
echo "[Stage 4] ✓ Rendering complete"
