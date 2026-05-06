#!/bin/bash
cd /home/jiazhuangzhuang/ARIS
export STAGE1_API_KEY=sk-7834aaf7864cbd4fbef111f71fe835de14e124fe3807a0e5d8a2e154ef865c81

/home/jiazhuangzhuang/miniconda3/envs/aris/bin/python scripts/run_dual_pipeline_loop.py   --resume /aaaidata/jiazhuangzhuang/ARIS_runtime/dual_loop_20260427_202818   --target-objects 60   --full-objects 11 --weak-objects 7 --group-ratio 3:2 --weak-angles 225   --gate-threshold 0.51 --gate-max-rounds 3   --angle-gate-threshold 0.51 --angle-gate-max-rounds 3   --weak-gate-reject-threshold 0.45   --no-force-accept --skip-training   --gpus 0,1,2 --device cuda:0
