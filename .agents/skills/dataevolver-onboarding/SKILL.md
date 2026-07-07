---
name: dataevolver-onboarding
description: Guide DataEvolver server setup and safe dry-run onboarding. Use when a user wants a Codex or Claude Code agent to configure a local or remote DataEvolver checkout, interview for SSH/runtime paths, inspect Python/uv/conda/Hugging Face/Blender/GPU readiness, plan dependency and model setup, choose a dynamic or fixed GPU shard policy, prepare HYWorld, Blender MCP, or paper-validation execution handoff, or generate non-sensitive .dataevolver/local/ENVIRONMENT.md and env.config.json profiles without launching long jobs.
---

# DataEvolver Onboarding

Use this skill as the standard first step when an agent takes over a DataEvolver environment. Keep the first pass bounded: interview, inspect, write non-sensitive setup profiles when requested, and hand off a dry-run execution plan. Do not store secrets or start heavy jobs from this skill.

## Workflow

1. Ask at most five base setup questions:
   - Target route: `quick_demo`, `t2i`, `edit`, `t2v`, `blender_3d`, `vlm_review`, `full_pipeline`, `world_model_scene`, `blender_mcp`, `paper_validation`, or `custom`.
   - Runtime location: local Linux path, remote SSH alias plus project path, or not cloned yet.
   - Install policy: inspect only, generate plan, install Python deps later, download models later, or full setup later.
   - Model strategy: default DataEvolver models, existing paths, custom replacement models, or skip for now.
   - Work/output paths: default `.dataevolver/`, user path, or remote path.
2. For `world_model_scene` or `paper_validation`, ask only the GPU follow-up values that are missing:
   - GPU policy: `inspect_only`, `dynamic_backfill`, or `fixed_range`.
   - Included GPUs, for example `all` or a user-provided GPU range.
   - Reserved GPUs, for example a user-provided GPU range that hosts VLM/vLLM or another service.
3. Let the dry-run script probe tool versions and paths. Do not ask tool-version questions manually.
4. Summarize known values, blockers, selected profile, GPU policy, and whether the run is paper validation.
5. Generate or update `.dataevolver/local/ENVIRONMENT.md` and `.dataevolver/local/env.config.json` only when the user asks to save the profile.
6. Hand off with:
   ```bash
   python -m pip install -e .
   bash src/dataevolver/cli/bootstrap_dataevolver_default.sh --profile <profile> --dry-run --write-local-config
   ```
   For HYWorld or paper validation, add `--profile world_model --gpu-policy <policy> --include-gpus <user-gpu-spec>` and, only if the user has service GPUs to protect, `--reserve-gpus <user-reserve-spec>`. Add `--paper-validation` when applicable.
   For Blender MCP operator setup, use `--profile blender_mcp`; this only prints and writes local operator configuration.
7. State explicitly that onboarding is dry-run only: it may print install/download/config/GPU shard plans, but it must not install dependencies, download weights, kill services, or launch long GPU jobs without explicit confirmation in the main conversation.

## Profiles

- `quick`: environment probes and a dry-run route only.
- `default`: current core pipeline defaults: Qwen-Image-2512, SAM3, Hunyuan3D-2.1, DINOv2 Giant, Qwen3.5-35B-A3B, and Blender.
- `full`: `default` plus Qwen-Image-Edit-2511 and Wan2.1-T2V.
- `blender_mcp`: optional operator/debug profile for remote Blender MCP control. It configures Codex-to-Blender inspection tools, viewport screenshots, and temporary Blender Python execution; it is not the production Stage 4 render path and must not change default pipeline behavior.
- `world_model`: optional HYWorld / WorldMirror scene reconstruction route. Use only when the target route is `world_model_scene` or `paper_validation`; it adds HY-World-2.0, MoGe, ZIM, GroundingDINO, SAM3, WorldStereo, Wan I2V base, and WorldMirror/3DGS checks.
- `custom`: record user-supplied model replacements as `needs_check`; do not perform compatibility review during onboarding.

Do not put world-model setup into `default`. When a user wants input image -> HY-Pano -> WorldMirror/metric depth mesh -> Blender scene contract -> multi-view validation -> object insertion -> final report, select route `world_model_scene` and profile `world_model`.

## GPU Policy

- `inspect_only`: inspect and report GPU inventory, memory, utilization, and obvious service processes; do not propose leases.
- `dynamic_backfill`: recommended after target-host inspection for multi-GPU inference/rendering runs. Use `src/dataevolver/runtime/gpu_scheduler.py` to plan one scheduling cycle from current `nvidia-smi` state and pending independent shards.
- `fixed_range`: restrict planning to user-provided `include_gpus` and `reserve_gpus`; use this when the user has pre-allocated cards or wants to protect a service card.

Follow these scheduling rules:

- Treat each independent inference, reconstruction, render, review, SR, or audit shard as an isolated process with an exclusive `CUDA_VISIBLE_DEVICES` lease unless it is CPU-only or an already-running service request.
- Poll `nvidia-smi` before each scheduling cycle. Reserve GPUs that are explicitly reserved, high-memory busy, high-utilization busy, or known to host vLLM/VLM/model services.
- Do not blindly use all GPUs for HYWorld pano/worldgen. Keep fragile high-memory or distributed stages on their smallest stable GPU set, then backfill idle GPUs with Blender render shards, scene-view gates, rotation8 shards, SR/contact-sheet batches, VLM service requests, and CPU/file audits.
- For NCCL/process-group failures, do not increase `nproc` blindly. First verify one process per GPU, clean `MASTER_ADDR`/`MASTER_PORT`/rank environment, unique rendezvous files, and consistent collective ordering.
- Preserve failed shard logs and outputs as evidence. For paper validation, failure samples are part of the report and must not be silently replaced.

## Environment Plan

- Prefer:
  ```bash
  uv venv .venv --python 3.10 --system-site-packages
  source .venv/bin/activate
  python -m pip install -e .
  ```
- Install `python -m pip install -e ".[hyworld]"` only for `world_model_scene` or paper-validation routes. The extra covers pip-managed packages, not model weights, source checkouts, Blender, or CUDA/native extension builds.
- Install SAM3, Hunyuan3D, HYWorld, CUDA/C++ extensions, and model weights only after target-host preflight confirms Python, CUDA/PyTorch ABI, `nvcc`, disk space, and source/weight paths.
- Reuse a validated system CUDA PyTorch through `--system-site-packages` when possible; install a CUDA wheel only if `import torch` fails or reports no usable CUDA.

## Safety Rules

- Never write Hugging Face, OpenAI, Anthropic, SSH, cookie, API, or service tokens to project files.
- Mention `HF_TOKEN` only as a shell environment variable for future real downloads.
- Treat SAM3 and other gated models as access-dependent; ask the user to obtain access instead of trying to bypass it.
- Treat missing `uv`, `uvx`, `conda`, `hf`, `nvidia-smi`, or `blender` as non-blocking in dry-run. Surface them as next actions for real setup.
- Do not modify `CLAUDE.md`; the main agent may decide later whether to sync selected non-sensitive facts.
- Do not commit user-specific GPU choices such as concrete card ranges, service reservations, or host process state to the public repo. Store them only in ignored `.dataevolver/local/` profiles or user memory.
- Do not kill existing vLLM/VLM services during onboarding. Record them as reservations unless the user explicitly authorizes restart outside the dry-run flow.
- If the user asks for real installation, downloads, service restarts, or GPU execution, first use the dry-run output as the plan and ask for explicit confirmation.
- Prefer `uvx --from huggingface_hub hf download ...` for printed Hugging Face download plans, with `hf download ...` as fallback. Do not execute either command in dry-run.
- For `blender_mcp`, do not write the user's global Codex config automatically. Generate `.dataevolver/local/blender_mcp.codex.toml` and ask the user to review/copy it.
- For `blender_mcp`, do not start remote Blender or install remote system packages during onboarding. Print preflight/start commands only unless the user explicitly asks for real setup outside the dry-run flow.

## Known V1 Requirements

- Source or export environment-variable overrides before real one-click setup runs the `dataevolver.workflows.stages` modules:
  `QWEN_IMAGE_MODEL_PATH`, `SAM3_CKPT`, `SAM3_DIR`, `HUNYUAN3D_REPO`, `MODEL_HUB`, `PAINT_MODEL_HUB`, `DINO_MODEL_PATH`, `REALESRGAN_CKPT`, and `VLM_MODEL_PATH`.
- For `world_model_scene`, also source or export:
  `HYWORLD_SRC`, `HYWORLD_WEIGHTS`, `HYWORLD_PYTHON`, `HYWORLD_MOGE_MODEL_PATH`, `HYWORLD_ZIM_MODEL_PATH`, `HYWORLD_GROUNDING_DINO_MODEL_PATH`, `HYWORLD_SAM3_MODEL_PATH`, `HYWORLD_WORLDSTEREO_PATH`, `HYWORLD_WAN_BASE_MODEL`, and `WORLDSTEREO_BASE_MODEL_PATH`.
- For paper validation, require `timing_ledger.json`, `gpu_shards.json`, `RUN_STATE.json`, source/image/GLB/final-output manifests, failure evidence, and quality reports before calling the run complete.
- Treat VLM scores as auxiliary. Completion evidence must come from source paths, hashes, scene contracts, pure-scene renders, per-angle masks/bboxes/visibility, and final manifests.
- Install SAM3 and Hunyuan3D repo-specific dependencies only after target-host preflight.
- Compile Hunyuan3D CUDA/C++ extensions only after CUDA/nvcc compatibility is confirmed.

## Local Profile Contents

`ENVIRONMENT.md` should include:

- Selected profile, target workflow, and whether this is paper validation.
- Runtime location, workspace/model roots, and work/output paths.
- Install policy and dependency plan.
- Model choices and missing values.
- Tooling status, server preflight summary, and missing commands.
- GPU policy, included/reserved GPUs, and dry-run GPU shard plan.
- Hugging Face gated/access-dependent repos.
- Runtime path override plan for v1.
- Timing/GPU/failure ledger requirements when applicable.
- Optional operator configuration such as `operators.blender_mcp` when the selected route is `blender_mcp`.
- Next actions for the main agent.

`env.config.json` should include:

- `profile`, `target_workflow`, `runtime_location`, `install_policy`
- `model_root`, `workspace_root`
- `tooling_status`, `server_preflight`, `models`, `custom_models`, `access_requirements`
- `gpu_policy`, `include_gpus`, `reserve_gpus`, `gpu_plan`
- `execution_mode`, `paper_validation`, `ledger_requirements`
- `operators.blender_mcp` for the `blender_mcp` profile, including `enabled`, `connection_mode`, `ssh_alias`, `remote_root`, `remote_blender_bin`, `remote_addon_dir`, `remote_port`, `remote_uvx`, `tmux_session`, and `codex_mcp_server_name`
- `path_override_plan`, `v1_blockers`, `next_actions`, `generated_at`

Keep both files concise. Use `unknown`, `not_requested`, or `needs_check` instead of blocking onboarding when the user is unsure.
