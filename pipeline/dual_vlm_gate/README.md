# Dual VLM Gate

This folder is a standalone dual-object VLM quality gate. It does not modify the existing single-object gate loop.

## What It Does

The loop renders two meshes into one Blender scene, asks the VLM to judge the combined result, applies dual-object actions, and repeats until the score reaches a threshold or `max_rounds` is exhausted.

It checks:

- object A and object B identity against optional reference images
- relative scale
- spacing and occlusion
- grounding/contact
- lighting and composition
- final usability of the two-object scene

## Files

- `dual_scene_render.py`: Blender entrypoint. Imports two meshes, places them, renders one image, writes `metadata.json`.
- `dual_agent_step.py`: one render + VLM review round.
- `dual_gate_loop.py`: iterative gate loop.
- `dual_actions.py`: dual control state, actions, and action selection.
- `dual_action_space.json`: actions exposed to the VLM prompt.
- `default_config.json`: default thresholds and render settings.
- `run_example.ps1`: local PowerShell wrapper.

## Basic Run

```powershell
python pipeline/dual_vlm_gate/dual_gate_loop.py `
  --root-dir pipeline/data/dual_vlm_gate_runs/test01 `
  --sample-id obj_a_obj_b_scene01 `
  --mesh-a D:/path/to/a.glb `
  --mesh-b D:/path/to/b.glb `
  --ref-a D:/path/to/a.png `
  --ref-b D:/path/to/b.png `
  --scene <large-file-root> `
  --device cuda:0 `
  --threshold 0.72 `
  --max-rounds 5
```

By default the loop runs an initialization search before `round00`. It renders
multiple candidate placements, selects one, writes `placement_state.json`, and
then keeps that placement fixed across VLM refinement rounds.

Useful init-search options:

```bash
--init-candidates 12
--use-vlm-init-review
--vlm-init-weight 0.55
--vlm-init-min-score 0.35
--skip-init-search
```

`--use-vlm-init-review` is off by default because it adds one VLM call per
candidate. Use it for difficult scenes where rule-only candidate selection is
not enough.

Remote/Linux:

```bash
python pipeline/dual_vlm_gate/dual_gate_loop.py \
  --root-dir <runtime-root>/dual_vlm_gate_test01 \
  --sample-id obj_a_obj_b_scene01 \
  --mesh-a /path/to/a.glb \
  --mesh-b /path/to/b.glb \
  --ref-a /path/to/a.png \
  --ref-b /path/to/b.png \
  --scene <large-file-root> \
  --device cuda:0 \
  --threshold 0.72 \
  --max-rounds 5
```

## Output Layout

```text
<root>/<sample-id>/
  gate_status.json
  gate_decision_round00.json
  agent_round00.json
  states/
    round00_input.json
    round00.json
  round00_renders/
    <sample-id>.png
    metadata.json
    render_summary.json
    blender.log
  init_search/
    candidate_00/
      <sample-id>_init00.png
      metadata.json
      placement_state.json
      init_vlm_review.json   # only when --use-vlm-init-review is enabled
    init_search_summary.json
  placement_state.json
  reviews/
    <sample-id>_r00_agg.json
    <sample-id>_r00_trace.json
```

## Action Model

The control state is intentionally separate from the single-object pipeline:

```json
{
  "object_a": {"scale": 1.0, "yaw_deg": 0.0, "offset_xy": [0, 0], "offset_z": 0.0},
  "object_b": {"scale": 1.0, "yaw_deg": 0.0, "offset_xy": [0, 0], "offset_z": 0.0},
  "pair": {"distance_scale": 1.0, "height_offset_scale": 1.0, "side_sign": 0},
  "camera": {"distance_scale": 1.0, "azimuth_offset_deg": 0.0, "elevation_deg": 20.0},
  "lighting": {"spot_energy_scale": 1.0, "fill_energy_scale": 1.0}
}
```

Typical actions:

- `A_SCALE_DOWN_10`, `B_SCALE_UP_10`
- `PAIR_INCREASE_DISTANCE`, `PAIR_DECREASE_DISTANCE`
- `A_LOWER_SMALL`, `B_LIFT_SMALL`
- `CAMERA_WIDER`, `CAMERA_AZ_POS_15`
- `L_SPOT_UP`, `L_FILL_DOWN`

## Integration Point

For the existing `pipeline/run_dual_object_pipeline.py`, call `dual_gate_loop.py` after Stage 3 meshes are available and before accepting the Stage 4 render. The minimal integration is:

1. generate or supply two meshes
2. call `dual_gate_loop.py`
3. read `<root>/<sample-id>/gate_status.json`
4. accept only `status == "accepted"`, or use `best_render_dir` for manual review
