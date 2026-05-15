#!/bin/bash
OBJ1=<repo-root>/runtime/new_asset_force_rot8_20260503_184759/stage1_assets/meshes_raw/obj_1614.glb
OBJ2=<repo-root>/runtime/new_asset_force_rot8_20260504_115839/stage1_assets/meshes_raw/obj_1626.glb
BLENDER=blender
SCRIPT=<repo-root>/pipeline/demo_dual_object_placement.py
OUTDIR=<repo-root>/pipeline/data/dual_renders/batch_all_scenes

mkdir -p "$OUTDIR"

SCENES="
<large-file-root>
<large-file-root>
<large-file-root>
<large-file-root>
<large-file-root>
<large-file-root>
<large-file-root>
<large-file-root>
<large-file-root>
<large-file-root>
<large-file-root>
<large-file-root>
<large-file-root>
<large-file-root>
<large-file-root>
<large-file-root>
<large-file-root>
<large-file-root>
"

TOTAL=0
COUNT=0
FAIL=0
for s in $SCENES; do TOTAL=$((TOTAL+1)); done

for SCENE in $SCENES; do
  COUNT=$((COUNT+1))
  NAME=$(basename "$SCENE" .blend)
  OUT="$OUTDIR/${NAME}_obj1614_obj1626.png"
  echo "[$COUNT/$TOTAL] Rendering $NAME..."
  START=$(date +%s)

  $BLENDER "$SCENE" --background --python "$SCRIPT" -- \
    --obj1 "$OBJ1" --obj2 "$OBJ2" \
    --output "$OUT" \
    --seed 42 \
    > "$OUTDIR/${NAME}.log" 2>&1

  RC=$?
  END=$(date +%s)
  ELAPSED=$((END-START))

  if [ $RC -eq 0 ] && [ -f "$OUT" ]; then
    SIZE=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
    echo "  OK: ${ELAPSED}s, ${SIZE} bytes"
  else
    FAIL=$((FAIL+1))
    echo "  FAIL: exit code $RC after ${ELAPSED}s"
    tail -3 "$OUTDIR/${NAME}.log" 2>/dev/null
  fi
done

echo ""
echo "=== Batch complete: $((COUNT-FAIL))/$TOTAL succeeded, $FAIL failed ==="
ls "$OUTDIR"/*.png 2>/dev/null | wc -l
