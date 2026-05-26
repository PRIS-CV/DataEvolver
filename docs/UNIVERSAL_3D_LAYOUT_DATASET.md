# Universal 3D Layout Dataset

This document records the DataEvolver universal 3D layout dataset workflow used
for the multi-paper 3D-control demo. It is a project update and reproducibility
entry, not a paper source release.

## Goal

The workflow starts from a paper-search task anchored by:

```text
SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation
```

DataEvolver uses websearch results from related 3D-control papers to derive one
shared artifact contract. The current contract covers common requirements from:

- SeeThrough3D
- POCI-Diff
- 3D Space as a Scratchpad
- Camera Control via Viewpoint Tokens
- Ctrl&Shift
- 3D-ARD+

The goal is not to claim full reproduction of all six papers. The goal is to
generate a reusable Blender-backed dataset contract that can be exported into
paper-specific views.

## WebSearch Entry

The universal workflow is entered through the Stage0 WebSearch module:

```bash
python scripts/stage0_web_research.py init \
  --query "SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation" \
  --output-dir runtime/research_priors/<batch>/universal_3d_layout \
  --dataset-mode universal \
  --tag universal-3d-layout
```

After the agent records evidence with `add-evidence`, finalize the handoff:

```bash
python scripts/stage0_web_research.py finalize \
  --session-dir runtime/research_priors/<batch>/universal_3d_layout \
  --max-papers 5
```

The key output for this module is:

```text
DATASET_WORKFLOW_HANDOFF.md
```

For a single-paper dataset reproduction, use the same WebSearch module with
`--dataset-mode single-paper` and usually finalize with `--max-papers 1`.

## Schema

Tracked schema:

```text
configs/schemas/universal_3d_layout_dataset_schema.json
```

Each record contains:

- `target`: Blender RGB render
- per-object binary `mask`
- `structure` / OSCR-style 3D box visualization
- `depth`: depth-order proxy
- `normal`: orientation proxy
- camera pose, intrinsics and viewpoint token
- object mesh path, 3D bbox, scale and rotation
- scene graph, spatial relation and occlusion pair
- validation fields, loop trace and promotion decision

Important wording: current `depth` and `normal` are proxy artifacts, not dense
depth maps or dense surface normal maps.

## Generation Scripts

Tracked scripts:

```text
scripts/universal_3d_layout/render_dataevolver_universal_contract_record.py
scripts/universal_3d_layout/run_dataevolver_universal_contract_batch.py
scripts/universal_3d_layout/run_universal_vlm_inner_loop.py
scripts/universal_3d_layout/run_vggt_omega_geometry_review.py
```

These scripts record the generation workflow. They do not include generated
images, mesh assets, scene `.blend` files, model weights, or paper source.

## Remote Batch

Completed 200-record batch on `jzz`:

```text
/aaaidata/jiazhuangzhuang/ARIS_runtime/universal_contract_batch_200_20260518
```

`ARIS_runtime` is a legacy server directory name. The project name is
DataEvolver.

## Server Validation

The checked-in WebSearch and universal dataset scripts were validated on `jzz`
after being copied into a temporary verification directory. The validation
covered:

- Python compilation for the Stage0 WebSearch module and universal dataset
  helpers.
- Stage0 smoke runs for both `--dataset-mode single-paper` and
  `--dataset-mode universal`, including `DATASET_WORKFLOW_HANDOFF.md` output.
- VGGT-Omega proxy review against the existing 200-record remote batch with
  `--limit 2`.
- A one-record Blender smoke batch using the DataEvolver server runtime,
  Blender 4.2.4, real scene pool and existing mesh assets.

The one-record smoke generated the expected artifact contract:

```text
target/*.png
structure/*_oscr.png
mask/*_obj_a.png
mask/*_obj_b.png
depth/*_depth_order.png
normal/*_orientation_proxy.png
metadata/records.jsonl
metadata/st3d_compat.jsonl
metadata/render_manifest.json
```

Batch summary:

- 200 accepted records
- 0 failures
- 200 target renders
- 200 structure / OSCR renders
- 400 per-object masks
- 200 depth-order proxy renders
- 200 orientation proxy renders
- 200 universal-schema JSONL records
- 200 SeeThrough3D-compatible JSONL records
- total size: 567M

The batch used the same Blender 4.2.4 LTS binary used by the DataEvolver
rendering pipeline on `jzz`:

```text
/aaaidata/zhangqisong/blender-4.24/blender
```

## Example Batch Command

Run long jobs in `tmux`:

```bash
tmux new -s dataevolver_universal200
cd /path/to/DataEvolver
python scripts/universal_3d_layout/run_dataevolver_universal_contract_batch.py \
  --dataevolver-root /path/to/DataEvolver \
  --runtime-root /path/to/runtime \
  --blender /path/to/blender \
  --out /path/to/universal_contract_batch_200 \
  --num-samples 200 \
  --seed 20260518 \
  --resolution 512 \
  --engine EEVEE \
  --exclude-scenes indoor_4blend,outdoor16,outdoor7
```

The checked-in script still accepts the legacy `--aris-root` argument for
backward compatibility, but new commands should use `--dataevolver-root`.

## VLM Inner Loop

Completed selected subset run:

```text
/aaaidata/jiazhuangzhuang/ARIS_runtime/universal_contract_selected108_vlm_scorecalib_t075_r5_20260519
```

Configuration:

- selected records: 108
- threshold: 0.75
- max rounds: 5
- VLM: Qwen3.6-27B
- scoring: calibrated hybrid score with real CV geometry features plus VLM score
- default free-form repair influence disabled for score stability

Observed summary from the run:

- 108 / 108 completed
- 43 accepted
- 65 rejected
- 29 / 43 accepted samples improved through VLM loop rather than round-0 accept

Example command:

```bash
tmux new -s dataevolver_universal108_vlm
cd /path/to/DataEvolver
python scripts/universal_3d_layout/run_universal_vlm_inner_loop.py \
  --dataset-root /path/to/universal_contract_batch_200 \
  --output-root /path/to/universal_contract_selected108_vlm \
  --dataevolver-root /path/to/DataEvolver \
  --blender /path/to/blender \
  --vlm-model-path /path/to/Qwen3.6-27B \
  --total 108 \
  --max-rounds 5 \
  --threshold 0.75 \
  --device cuda:0 \
  --resume
```

## VGGT-Omega Hook

Server model cache inspected:

```text
/huggingface/model_hub/VGGT-Omega
```

Current script:

```text
scripts/universal_3d_layout/run_vggt_omega_geometry_review.py
```

Current status is proxy review only. It checks artifact completeness, mask
coverage, and proxy depth-order consistency. It does not claim real VGGT-Omega
inference until the official inference code is installed alongside the weights.

## What Should Not Be Committed

Do not commit:

- full generated dataset folders
- server runtime outputs
- mesh assets
- `.blend` scene files
- model weights
- paper source folders such as `paper/` or `KDD/`
- generated presentation / zip / HTML output packages unless explicitly curated
