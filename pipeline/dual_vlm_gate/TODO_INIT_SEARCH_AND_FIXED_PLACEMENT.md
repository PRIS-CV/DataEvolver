# Dual VLM Gate TODO: Init Search And Fixed Placement

Status: baseline implementation added.

## Implemented

- Added scene/init prefilter script: `dual_init_search.py`.
- `dual_gate_loop.py` now runs init search before VLM rounds unless `--skip-init-search` is passed.
- Init search renders multiple candidate placements and writes the selected `placement_state.json`.
- Init search can optionally run VLM candidate review with `--use-vlm-init-review`.
- VLM candidate review writes `candidate_xx/init_vlm_review.json` and combines `rule_score` with `vlm_init_score`.
- `dual_scene_render.py` can write `placement_state.json` for generated placements.
- `dual_scene_render.py` can read `placement_state.json` and restore the same placement for later rounds.
- `dual_agent_step.py` passes the fixed placement state into every render round.
- `dual_agent_step.py` no longer changes render seed by `round_idx`; placement stability now comes from the placement state.
- `demo_dual_object_placement.py` now filters `Fog` and `Sky` mesh objects from scene relevance checks.
- `find_ground_position()` now raycasts inside the detected ground bounds instead of returning the median ground z directly, which is required for terrain scenes such as `outdoor6`.

## Required Follow-Ups

1. Persist and reuse round00 placement.
   - Each sample should have `<sample>/placement_state.json`.
   - It stores `template`, `obj1_position`, `obj2_position`, `side`, `height_offset`, `separation`, and `camera_azimuth`.
   - Round01+ must load this file and only apply scale, pair, camera, and lighting controls.

2. Improve scene init search.
   - Current implementation supports rule scoring and optional VLM scoring.
   - Tune prompt, `vlm_init_weight`, and hard-fail tags after more scene-pool runs.
   - Keep VLM refinement separate from init search; refinement should not resample placement.

3. Harden terrain placement.
   - `outdoor6` style terrain needs raycast height at sampled XY.
   - Keep filtering `Fog` and `Sky` from BVH/bounds.
   - Add slope checks if terrain still causes floating/intersection.

4. Add scene profiles.
   - `outdoor7`: restrict camera angles or use scene camera because dome/pad edges distort easily.
   - `outdoor11`: prefer scene camera or fixed camera azimuth because there is no HDRI azimuth anchor.
   - `outdoor13`: keep placement near center of large flat plane to avoid empty/vanishing-ground views.

5. Verify remotely.
   - Run a small one-scene smoke test on `<ssh-alias>`.
   - Check that round00-round04 metadata keep the same `pair.obj1_position`, `pair.obj2_position`, `template`, and `camera_azimuth` unless an explicit action changes camera controls.
