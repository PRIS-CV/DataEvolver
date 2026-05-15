#!/bin/bash
# Full pipeline: consistent bright → trainready → split → bbox
# Run after render_bright_all_rotations.py completes
set -e

CODE_ROOT="<run-root>"
DATA_ROOT="$CODE_ROOT/pipeline/data"

CONSISTENT="$DATA_ROOT/dataset_scene_v7_full50_rotation8_consistent_bright_20260416"
TRAINREADY="$DATA_ROOT/dataset_scene_v7_full50_rotation8_trainready_front2others_bright_20260416"
SPLIT="$DATA_ROOT/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bright_20260416"
BBOX="$DATA_ROOT/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_bboxmask_bright_final_20260416"

cd "$CODE_ROOT"

echo "=== Step 1: Build trainready dataset ==="
python3 scripts/build_rotation8_trainready_dataset.py \
    --source-root "$CONSISTENT" \
    --output-dir "$TRAINREADY" \
    --copy-mode symlink

echo ""
echo "=== Step 2: Build object-disjoint split ==="
python3 scripts/build_object_split_for_rotation_dataset.py \
    --source-root "$TRAINREADY" \
    --output-dir "$SPLIT" \
    --train-objects 35 \
    --val-objects 7 \
    --test-objects 8 \
    --seed 42 \
    --asset-mode symlink

echo ""
echo "=== Step 3: Build bbox condition dataset ==="
python3 scripts/build_rotation_bbox_condition_dataset.py \
    --source-root "$SPLIT" \
    --output-dir "$BBOX" \
    --asset-mode symlink \
    --overwrite-pair-image-fields

echo ""
echo "=== Pipeline complete ==="
echo "Final dataset: $BBOX"
ls "$BBOX"
