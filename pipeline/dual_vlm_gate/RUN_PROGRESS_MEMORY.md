# Dual VLM Gate Run Progress Memory

Last updated: 2026-05-09 00:27:14 Asia/Shanghai

## Active Run

- Remote host: `<ssh-alias>`
- tmux session: `dual_vlm_obj1709_obj1708_r05_20260509_000802`
- Runtime root: `<runtime-root>/dual_vlm_obj1709_obj1708_outdoorpool_round05_20260509_000802`
- Command entry: `pipeline/dual_vlm_gate/run_scene_pool.py`
- Max rounds: `5`
- Threshold: `0.72`
- Engine: `CYCLES`
- Resolution: `768`
- Device: `cuda:0`
- Seed base: `9100`

## Objects

- Object A id: `obj_1709`
- Object B id: `obj_1708`
- Mesh A: `<runtime-root>/new_asset_force_rot8_20260508_204833/stage1_assets/meshes/obj_1709.glb`
- Mesh B: `<runtime-root>/new_asset_force_rot8_20260508_140934/stage1_assets/meshes/obj_1708.glb`

## Scene Pool

Current scene order:

1. `outdoor13`
2. `outdoor16`
3. `outdoor11`
4. `outdoor9`
5. `outdoor7`
6. `outdoor6`
7. `outdoor5`
8. `outdoor4`
9. `outdoor3`

## Latest Known Progress

At remote time `2026-05-09 00:27:14`:

- `obj1709_obj1708_00_outdoor13`: done, `rejected`, best score `0.5`, best round `3`, rounds `5`.
- `obj1709_obj1708_01_outdoor16`: running/incomplete, latest file `agent_round00.json`, score `0.5`, success `true`.
- `obj1709_obj1708_02_outdoor11`: pending.
- `obj1709_obj1708_03_outdoor9`: pending.
- `obj1709_obj1708_04_outdoor7`: pending.
- `obj1709_obj1708_05_outdoor6`: pending.
- `obj1709_obj1708_06_outdoor5`: pending.
- `obj1709_obj1708_07_outdoor4`: pending.
- `obj1709_obj1708_08_outdoor3`: pending.

Completed scenes: `1 / 9`.
Started scenes: `2 / 9`.

Previous ETA estimate at `2026-05-09 00:24:07` was about `2026-05-09 01:52:51`, based only on the first completed scene. Recompute ETA after `outdoor16` completes.

## Useful Commands

Monitor tmux:

```powershell
ssh <ssh-alias> "tmux capture-pane -pt dual_vlm_obj1709_obj1708_r05_20260509_000802 -S -120"
```

Check process:

```powershell
ssh <ssh-alias> "pgrep -afu <user> 'run_scene_pool|dual_gate_loop|dual_agent_step|dual_scene_render|blender' | grep -v grep || true"
```

Read scene-pool summary:

```powershell
ssh <ssh-alias> "cat <runtime-root>/dual_vlm_obj1709_obj1708_outdoorpool_round05_20260509_000802/scene_pool_run_summary.json 2>/dev/null || true"
```

List gate statuses:

```powershell
ssh <ssh-alias> "find <runtime-root>/dual_vlm_obj1709_obj1708_outdoorpool_round05_20260509_000802/gate_loop -maxdepth 2 -name gate_status.json -print"
```
