#!/bin/bash
# Wait for Stage 3 to finish (10 .glb files), then run Stage 4
WORKDIR=/aaaidata/zhangqisong/data_build
MESHES_DIR=$WORKDIR/pipeline/data/meshes
RENDERS_DIR=$WORKDIR/pipeline/data/renders
LOGDIR=$WORKDIR/pipeline/data/logs

echo "[wait_stage4] Waiting for Stage 3 to finish (need 10 .glb files)..."

while true; do
    COUNT=$(ls -1 $MESHES_DIR/*.glb 2>/dev/null | wc -l)
    echo "[wait_stage4] $(date '+%H:%M:%S') meshes ready: $COUNT/10"
    if [[ $COUNT -ge 10 ]]; then
        echo "[wait_stage4] Stage 3 complete! Starting Stage 4..."
        break
    fi
    sleep 30
done

# Clear old renders (from black-image meshes)
echo "[wait_stage4] Clearing old renders..."
rm -rf $RENDERS_DIR/obj_*/

# Run Stage 4
LOG_FILE=$LOGDIR/stage4_rerun_$(date +%Y%m%d_%H%M%S).log
echo "[wait_stage4] Running Stage 4, log: $LOG_FILE"
cd $WORKDIR
bash pipeline/stage4_batch_render.sh 2>&1 | tee $LOG_FILE

echo "[wait_stage4] Stage 4 done."
