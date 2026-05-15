#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
BLENDER=blender
REPO_ROOT=<repo-root>
RUN_ROOT=<runtime-root>/new_asset_force_rot8_20260425_204826
SEED_FILE=${RUN_ROOT}/seed_concepts.json
ASSETS_ROOT=${RUN_ROOT}/stage1_assets
GATE_ROOT=${RUN_ROOT}/gate_loop
ACCEPTED_ROOT=${RUN_ROOT}/accepted_pairs
ROT8_ROOT=${RUN_ROOT}/rotation8_consistent
TRAINREADY_ROOT=${RUN_ROOT}/trainready
AUGMENTED_ROOT=${RUN_ROOT}/augmented_trainready_split
PINNED_SPLIT=${RUN_ROOT}/pinned_split.json
LOOP_STATE=${RUN_ROOT}/LOOP_STATE.json
GATE_PROFILE_LOCAL=${RUN_ROOT}/scene_v7_gate_profile.json
BASELINE_SPLIT=${REPO_ROOT}/pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410
EXPORT_ROTATIONS="0,45,90,135,180,225,270,315"
TARGET_ROTATIONS="45,90,135,180,225,270,315"

log() { echo "[$(date '+%F %T')] $*"; }

# === Step 3: VLM gate ===
log "=== [03] run_vlm_quality_gate ==="
$PYTHON ${REPO_ROOT}/scripts/run_vlm_quality_gate_loop.py \
    --root-dir "$GATE_ROOT" \
    --objects-file "$SEED_FILE" \
    --rotations 0 \
    --threshold-start 0.78 \
    --threshold-step 0.0 \
    --threshold-max 0.78 \
    --score-field hybrid_score \
    --max-rounds 5 \
    --min-vlm-only-score 0.0 \
    --step-scale 0.25 \
    --patience 0 \
    --rollback-score-drop-threshold 0.05 \
    --device cuda:0 \
    --blender "$BLENDER" \
    --meshes-dir "${ASSETS_ROOT}/meshes" \
    --profile "$GATE_PROFILE_LOCAL" \
    2>&1 | tee ${RUN_ROOT}/logs/03_vlm_gate.log
log "Step 3 done"

# === Step 3.5: force accept ===
log "=== [03.5] force_accept_all ==="
$PYTHON ${REPO_ROOT}/scripts/force_accept_all.py "$GATE_ROOT"
log "Step 3.5 done"

# === Step 4: accepted pair root ===
log "=== [04] build_accepted_pair_root ==="
$PYTHON ${REPO_ROOT}/scripts/build_accepted_pair_root_from_gate.py \
    --gate-root "$GATE_ROOT" \
    --output-dir "$ACCEPTED_ROOT" \
    --asset-mode symlink \
    --overwrite \
    2>&1 | tee ${RUN_ROOT}/logs/04_accepted_pairs.log
log "Step 4 done"

# === Step 5: rotation8 export ===
log "=== [05] export_rotation8 ==="
$PYTHON ${REPO_ROOT}/scripts/export_rotation8_from_best_object_state.py \
    --source-root "$ACCEPTED_ROOT" \
    --output-dir "$ROT8_ROOT" \
    --python "$PYTHON" \
    --blender "$BLENDER" \
    --meshes-dir "${ASSETS_ROOT}/meshes" \
    --gpus 0,1,2 \
    --fallback-to-best-any-angle \
    --rotations "$EXPORT_ROTATIONS" \
    2>&1 | tee ${RUN_ROOT}/logs/05_rotation8.log
log "Step 5 done"

# === Step 6: trainready ===
log "=== [06] build_trainready ==="
$PYTHON ${REPO_ROOT}/scripts/build_rotation8_trainready_dataset.py \
    --source-root "$ROT8_ROOT" \
    --output-dir "$TRAINREADY_ROOT" \
    --copy-mode symlink \
    --target-rotations "$TARGET_ROTATIONS" \
    2>&1 | tee ${RUN_ROOT}/logs/06_trainready.log
log "Step 6 done"

# === Step 7: pinned split ===
log "=== [07] build_pinned_split ==="
$PYTHON ${REPO_ROOT}/scripts/feedback_loop/build_pinned_split.py \
    --dataset-root "$BASELINE_SPLIT" \
    --output "$PINNED_SPLIT" \
    --source-note "new_asset_force_rot8 baseline split" \
    2>&1 | tee ${RUN_ROOT}/logs/07_pinned_split.log
log "Step 7 done"

# === Step 8: augment ===
log "=== [08] build_augmented_dataset ==="
$PYTHON ${REPO_ROOT}/scripts/feedback_loop/build_augmented_rotation_dataset.py \
    --baseline-split-root "$BASELINE_SPLIT" \
    --new-trainready-root "$TRAINREADY_ROOT" \
    --output-dir "$AUGMENTED_ROOT" \
    --pinned-split-path "$PINNED_SPLIT" \
    --asset-mode symlink \
    --target-rotations "$TARGET_ROTATIONS" \
    2>&1 | tee ${RUN_ROOT}/logs/08_augment.log
log "Step 8 done"

# === Step 9: validate ===
log "=== [09] validate_final_dataset ==="
$PYTHON ${REPO_ROOT}/scripts/feedback_loop/validate_dataset.py \
    "$AUGMENTED_ROOT" \
    --output "${RUN_ROOT}/validation_report.json" \
    --pinned-split-path "$PINNED_SPLIT" \
    --expected-manifest "${AUGMENTED_ROOT}/augmented_manifest.json" \
    2>&1 | tee ${RUN_ROOT}/logs/09_validate.log
log "Step 9 done"

log "========================================"
log "Pipeline Steps 3-9 ALL COMPLETE"
log "========================================"
