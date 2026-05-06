#!/bin/bash
# Parallel depth+normal rendering across 3 GPUs.
# Run on jiazhuangzhuang server.
#
# Usage: bash batch_render_parallel.sh

set -euo pipefail

BLENDER="/aaaidata/zhangqisong/blender-4.24/blender"
RUNTIME_ROOT="/home/jiazhuangzhuang/ARIS/runtime/new_asset_force_rot8_20260429_203842"
OUTPUT_ROOT="${RUNTIME_ROOT}/trainready"
SCRIPT="/home/jiazhuangzhuang/ARIS/scripts/render_depth_normal.py"
LOG_DIR="${RUNTIME_ROOT}/logs/depth_normal_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"

OBJECTS=(
  obj_1471 obj_1472 obj_1473 obj_1474 obj_1475 obj_1476
  obj_1477 obj_1478 obj_1479 obj_1480 obj_1481 obj_1482
  obj_1483 obj_1484 obj_1485 obj_1486 obj_1487 obj_1488
)

run_on_gpu() {
  local gpu_id=$1
  local obj_id=$2
  local log="$LOG_DIR/${obj_id}_gpu${gpu_id}.log"
  echo "[$(date +%H:%M:%S)] GPU${gpu_id}: Starting $obj_id ..."
  if CUDA_VISIBLE_DEVICES=$gpu_id "$BLENDER" --background \
       --python "$SCRIPT" -- \
       --obj-id "$obj_id" \
       --runtime-root "$RUNTIME_ROOT" \
       --output-root "$OUTPUT_ROOT" \
       > "$log" 2>&1; then
    echo "[$(date +%H:%M:%S)] GPU${gpu_id}: $obj_id DONE"
    return 0
  else
    echo "[$(date +%H:%M:%S)] GPU${gpu_id}: $obj_id FAILED"
    return 1
  fi
}

echo "============================================"
echo "Parallel depth+normal rendering"
echo "Objects: ${#OBJECTS[@]}"
echo "GPUs: 3 (CUDA_VISIBLE_DEVICES=0,1,2)"
echo "Logs: $LOG_DIR"
echo "Started at: $(date)"
echo "============================================"

# Process objects in groups of 3
total=${#OBJECTS[@]}
for ((i=0; i<total; i+=3)); do
  pids=()
  for ((j=0; j<3 && i+j<total; j++)); do
    gpu_id=$j
    obj_id="${OBJECTS[i+j]}"
    run_on_gpu "$gpu_id" "$obj_id" &
    pids+=($!)
  done

  # Wait for this batch of 3 to complete
  for pid in "${pids[@]}"; do
    wait $pid || true
  done
  echo "  Batch $((i/3+1)) complete ($((i+3 < total ? i+3 : total))/$total objects)"
done

echo "============================================"
echo "Finished at: $(date)"

# Verify outputs
echo ""
echo "Output verification:"
missing=0
for obj_id in "${OBJECTS[@]}"; do
  for yaw in 000 045 090 135 180 225 270 315; do
    d="/home/jiazhuangzhuang/ARIS/runtime/new_asset_force_rot8_20260429_203842/trainready/views/${obj_id}/yaw${yaw}_depth.png"
    n="/home/jiazhuangzhuang/ARIS/runtime/new_asset_force_rot8_20260429_203842/trainready/views/${obj_id}/yaw${yaw}_normal.png"
    if [ ! -f "$d" ] || [ ! -f "$n" ]; then
      echo "  MISSING: $obj_id yaw${yaw}"
      missing=$((missing+1))
    fi
  done
done
if [ $missing -eq 0 ]; then
  echo "  ALL 288 files present (18 objects x 8 yaws x 2 passes)"
else
  echo "  $missing files missing!"
fi
