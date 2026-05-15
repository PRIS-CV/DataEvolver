#!/bin/bash
# Batch render depth + normal maps for all 18 objects.
# Run on <user> server.
#
# Usage:
#   bash batch_render_depth_normal.sh

set -euo pipefail

BLENDER="blender"
RUNTIME_ROOT="<repo-root>/runtime/new_asset_force_rot8_20260429_203842"
OUTPUT_ROOT="${RUNTIME_ROOT}/trainready"
SCRIPT="<repo-root>/scripts/render_depth_normal.py"
LOG_DIR="${RUNTIME_ROOT}/logs/depth_normal_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"

OBJECTS=(
  obj_1471 obj_1472 obj_1473 obj_1474 obj_1475 obj_1476
  obj_1477 obj_1478 obj_1479 obj_1480 obj_1481 obj_1482
  obj_1483 obj_1484 obj_1485 obj_1486 obj_1487 obj_1488
)

echo "============================================"
echo "Batch depth+normal rendering"
echo "Objects: ${#OBJECTS[@]}"
echo "Logs: $LOG_DIR"
echo "Started at: $(date)"
echo "============================================"

FAILED=()
for obj_id in "${OBJECTS[@]}"; do
  LOG_FILE="${LOG_DIR}/${obj_id}.log"
  echo "[$(date +%H:%M:%S)] Starting $obj_id ..."

  if "$BLENDER" --background \
       --python "$SCRIPT" -- \
       --obj-id "$obj_id" \
       --runtime-root "$RUNTIME_ROOT" \
       --output-root "$OUTPUT_ROOT" \
       > "$LOG_FILE" 2>&1; then
    echo "[$(date +%H:%M:%S)] $obj_id DONE"
  else
    echo "[$(date +%H:%M:%S)] $obj_id FAILED (see $LOG_FILE)"
    FAILED+=("$obj_id")
  fi
done

echo "============================================"
echo "Finished at: $(date)"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED: ${FAILED[*]}"
  exit 1
else
  echo "ALL ${#OBJECTS[@]} objects succeeded"
fi
