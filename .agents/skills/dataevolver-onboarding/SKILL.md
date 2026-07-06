---
name: dataevolver-onboarding
description: Guide a lightweight DataEvolver setup interview and dry-run deployment handoff. Use when a user wants quick DataEvolver onboarding, server/environment discovery, default or custom model setup planning, Hugging Face token-safe download planning, or generation of local ENVIRONMENT.md and env.config.json setup profiles.
---

# DataEvolver Onboarding

Use this skill to help a user quickly become ready to run DataEvolver without overwhelming them with the full environment surface. Keep the first pass light: interview, write a local profile, and hand off a dry-run deployment plan. Do not store secrets.

## Workflow

1. Ask at most five setup questions:
   - Target route: `quick_demo`, `t2i`, `edit`, `t2v`, `blender_3d`, `vlm_review`, `full_pipeline`, `world_model_scene`, or `custom`.
   - Runtime location: local Linux path, remote SSH alias plus project path, or not cloned yet.
   - Install policy: inspect only, generate plan, install Python deps, download models, or full setup later.
   - Model strategy: default DataEvolver models, existing paths, custom replacement models, or skip for now.
   - Work/output paths: default `runtime/`, user path, or remote path.
2. Do not ask tool-version questions in the interview. Let the demo script probe Python, `uv`, `uvx`, `conda`, `hf`, GPU, and Blender availability.
3. Summarize known values, missing blockers, and the selected profile: `quick`, `default`, `full`, `world_model`, or `custom`.
4. Generate or update `.dataevolver/local/ENVIRONMENT.md` and `.dataevolver/local/env.config.json` only when the user asks to save the profile.
5. Hand off to the main agent with the package install and demo command:
   `python -m pip install -e .`
   `bash src/dataevolver/cli/bootstrap_dataevolver_default.sh --profile <profile> --dry-run --write-local-config`
6. Tell the user that v0 is dry-run only: it prints install/download/config plans but does not install dependencies, download model weights, or launch long jobs.

## Profiles

- `quick`: environment probes and a dry-run route only.
- `default`: current core pipeline defaults: Qwen-Image-2512, SAM3, Hunyuan3D-2.1, DINOv2 Giant, Qwen3.5-35B-A3B, and Blender.
- `full`: `default` plus Qwen-Image-Edit-2511 and Wan2.1-T2V.
- `world_model`: optional HYWorld / WorldMirror scene reconstruction route. Use only when the target route is `world_model_scene`; it adds HY-World-2.0, MoGe, ZIM, GroundingDINO, SAM3, WorldStereo, Wan I2V base, and WorldMirror/3DGS checks.
- `custom`: record user-supplied model replacements as `needs_check`; do not perform compatibility review during onboarding.

Do not put world-model setup into `default`. When a user wants standard DataEvolver generation, keep `default` focused on the core pipeline. When a user wants input image -> HY-Pano -> WorldMirror/metric depth mesh -> Blender scene contract -> multi-view validation -> object insertion -> final report, select route `world_model_scene` and profile `world_model`.

## Safety Rules

- Never write Hugging Face, OpenAI, Anthropic, SSH, cookie, or API tokens to project files.
- Mention `HF_TOKEN` only as a shell environment variable for future real downloads.
- Treat SAM3 and other gated models as access-dependent; ask the user to obtain access instead of trying to bypass it.
- Treat missing `uv`, `uvx`, `conda`, `hf`, `nvidia-smi`, or `blender` as non-blocking in v0. Surface them as next actions for real setup.
- Do not modify `CLAUDE.md`; the main agent may decide later whether to sync selected non-sensitive facts.
- If the user asks for real installation or real downloads, first use the demo script output as the plan and ask for explicit confirmation in the main conversation.
- Prefer `uvx --from huggingface_hub hf download ...` for printed Hugging Face download plans, with `hf download ...` as fallback. Do not execute either command in v0.
- Use `python -m pip install -e ".[hyworld]"` only for the optional `world_model_scene` route; keep the default install lightweight.

## Known V1 Requirements

- Source or export environment-variable overrides before real one-click setup runs the `dataevolver.workflows.stages` modules:
  `QWEN_IMAGE_MODEL_PATH`, `SAM3_CKPT`, `SAM3_DIR`, `HUNYUAN3D_REPO`, `MODEL_HUB`, `PAINT_MODEL_HUB`, `DINO_MODEL_PATH`, `REALESRGAN_CKPT`, and `VLM_MODEL_PATH`.
- For `world_model_scene`, also source or export:
  `HYWORLD_SRC`, `HYWORLD_WEIGHTS`, `HYWORLD_PYTHON`, `HYWORLD_MOGE_MODEL_PATH`, `HYWORLD_ZIM_MODEL_PATH`, `HYWORLD_GROUNDING_DINO_MODEL_PATH`, `HYWORLD_SAM3_MODEL_PATH`, `HYWORLD_WORLDSTEREO_PATH`, `HYWORLD_WAN_BASE_MODEL`, and `WORLDSTEREO_BASE_MODEL_PATH`.
- For `world_model_scene`, treat a VLM pass as advisory only. Review evidence should come from the HYWorld scene contract, multi-view pure-scene renders, manifest/lineage hashes, and the final scene/object report.
- Install SAM3 and Hunyuan3D repo-specific dependencies only after target-host preflight.
- Compile Hunyuan3D CUDA/C++ extensions only after CUDA/nvcc compatibility is confirmed.

## Local Profile Contents

`ENVIRONMENT.md` should be human-readable and include:

- Selected profile and target workflow.
- Runtime location and workspace/model roots.
- Install policy.
- Model choices and missing values.
- Tooling status and missing commands.
- Hugging Face gated/access-dependent repos.
- Runtime path override plan for v1.
- Next actions for the main agent.

`env.config.json` should include:

- `profile`, `target_workflow`, `runtime_location`, `install_policy`
- `model_root`, `workspace_root`
- `tooling_status`, `models`, `custom_models`, `access_requirements`
- `path_override_plan`, `v1_blockers`, `next_actions`, `generated_at`

Keep both files concise. Use `unknown`, `not_requested`, or `needs_check` instead of blocking onboarding when the user is unsure.
