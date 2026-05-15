#!/usr/bin/env bash
set -euo pipefail

export ANTHROPIC_API_KEY="<REDACTED_API_KEY>"

exec bash ~/ARIS/scripts/launch_new_asset_force_rot8.sh \
    --seed-count 1 \
    --gate-max-rounds 3 \
    --scene-template ~/ARIS/configs/scene_template.json \
    --template-only \
    2>&1 | tee ~/ARIS/test_run_scene1.log
