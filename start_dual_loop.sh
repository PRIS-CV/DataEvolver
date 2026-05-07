#!/bin/bash
cd /home/jiazhuangzhuang/ARIS
export STAGE1_API_KEY=sk-7834aaf7864cbd4fbef111f71fe835de14e124fe3807a0e5d8a2e154ef865c81
/home/jiazhuangzhuang/miniconda3/envs/aris/bin/python scripts/run_dual_pipeline_loop.py   --target-objects 60   --initial-baseline /aaaidata/jiazhuangzhuang/ARIS_runtime/dual_pipeline_20260427_170551/merged_trainready   --full-objects 3   --weak-objects 3   --weak-angles 225   --gate-threshold 0.51   --gate-max-rounds 3   --angle-gate-threshold 0.51   --angle-gate-max-rounds 3   --no-force-accept   --skip-training   2>&1 | tee /tmp/dual_loop.log
