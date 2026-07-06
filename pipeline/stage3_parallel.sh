#!/bin/bash
# 3-GPU parallel Stage 3: split 10 objects across 3 GPUs
# GPU0: obj_001 obj_002 obj_003 obj_004
# GPU1: obj_005 obj_006 obj_007
# GPU2: obj_008 obj_009 obj_010

WORKDIR=/aaaidata/zhangqisong/data_build
PYTHON=/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python
SCRIPT=$WORKDIR/pipeline/stage3_image_to_3d.py
LOGDIR=$WORKDIR/pipeline/data/logs

mkdir -p $LOGDIR

echo "[stage3_parallel] Starting 3-GPU parallel Stage 3..."

# GPU 0: obj_001 ~ obj_004
CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --no-skip --ids obj_001,obj_002,obj_003,obj_004 --device cuda:0 \
    2>&1 | tee $LOGDIR/stage3_gpu0.log &
PID0=$!
echo "  GPU0 pid=$PID0  objs: 001~004"

sleep 3  # stagger startup to avoid model-hub race

# GPU 1: obj_005 ~ obj_007
CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT --no-skip --ids obj_005,obj_006,obj_007 --device cuda:0 \
    2>&1 | tee $LOGDIR/stage3_gpu1.log &
PID1=$!
echo "  GPU1 pid=$PID1  objs: 005~007"

sleep 3

# GPU 2: obj_008 ~ obj_010
CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT --no-skip --ids obj_008,obj_009,obj_010 --device cuda:0 \
    2>&1 | tee $LOGDIR/stage3_gpu2.log &
PID2=$!
echo "  GPU2 pid=$PID2  objs: 008~010"

echo "[stage3_parallel] All 3 workers launched. Waiting..."
wait $PID0 $PID1 $PID2
echo "[stage3_parallel] All done. Checking results..."

ls -lh $WORKDIR/pipeline/data/meshes/
