#!/usr/bin/env bash
set -euo pipefail

export ANTHROPIC_API_KEY="sk-7834aaf7864cbd4fbef111f71fe835de14e124fe3807a0e5d8a2e154ef865c81"

exec bash ~/ARIS/scripts/launch_new_asset_force_rot8.sh \
    --seed-count 1 \
    --gate-max-rounds 3 \
    --scene-template ~/ARIS/configs/scene_template.json \
    --template-only \
    2>&1 | tee ~/ARIS/test_run_scene1.log
