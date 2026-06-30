#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROFILE="quick"
MODEL_ROOT="${DATAEVOLVER_MODEL_ROOT:-${ROOT}/runtime/model_hub}"
WORKSPACE_ROOT="${DATAEVOLVER_WORKSPACE_ROOT:-${ROOT}}"
PYTHON_BACKEND="auto"
DRY_RUN=1
WRITE_LOCAL_CONFIG=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/bootstrap_dataevolver_default.sh [options]

Dry-run only v0 demo. This script prints DataEvolver setup, install, model
download, and config plans. It does not install dependencies, download model
weights, write tokens, or launch long-running jobs.

Options:
  --dry-run                    Required behavior; enabled by default.
  --profile quick|default|full|custom
  --model-root PATH            Target model root for the printed plan.
  --workspace-root PATH        DataEvolver checkout path for the printed plan.
  --python-backend auto|uv|conda
  --write-local-config         Write .dataevolver/local demo profile files.
  -h, --help                   Show this help.
USAGE
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

quote() {
  printf '%q' "$1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --profile)
      [[ $# -ge 2 ]] || die "--profile requires a value"
      PROFILE="$2"
      shift 2
      ;;
    --model-root)
      [[ $# -ge 2 ]] || die "--model-root requires a value"
      MODEL_ROOT="$2"
      shift 2
      ;;
    --workspace-root)
      [[ $# -ge 2 ]] || die "--workspace-root requires a value"
      WORKSPACE_ROOT="$2"
      shift 2
      ;;
    --python-backend)
      [[ $# -ge 2 ]] || die "--python-backend requires a value"
      PYTHON_BACKEND="$2"
      shift 2
      ;;
    --write-local-config)
      WRITE_LOCAL_CONFIG=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

case "$PROFILE" in
  quick|default|full|custom) ;;
  *) die "--profile must be one of: quick, default, full, custom" ;;
esac

case "$PYTHON_BACKEND" in
  auto|uv|conda) ;;
  *) die "--python-backend must be one of: auto, uv, conda" ;;
esac

if [[ "$DRY_RUN" != "1" ]]; then
  die "v0 is dry-run only"
fi

WORKSPACE_ROOT="$(cd "$WORKSPACE_ROOT" 2>/dev/null && pwd || printf '%s' "$WORKSPACE_ROOT")"
MODEL_ROOT="${MODEL_ROOT%/}"

section() {
  printf '\n== %s ==\n' "$1"
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

cmd_status() {
  if has_cmd "$1"; then
    printf 'present: %s -> %s\n' "$1" "$(command -v "$1")"
  else
    printf 'missing: %s\n' "$1"
  fi
}

print_actionable_gaps() {
  local gaps=0
  printf '\nActionable gaps for real setup (non-blocking in dry-run):\n'
  if ! has_cmd python3 && ! has_cmd python; then
    printf '  - python: missing; real setup and config writing need Python 3.10+.\n'
    gaps=1
  fi
  if ! has_cmd uv && ! has_cmd conda; then
    printf '  - python environment backend: missing uv and conda; install uv for the fastest default path or provide conda.\n'
    gaps=1
  elif ! has_cmd uv; then
    printf '  - uv: missing; real setup will need the conda fallback unless uv is installed.\n'
    gaps=1
  fi
  if ! has_cmd uvx && ! has_cmd hf; then
    printf '  - Hugging Face CLI: missing uvx and hf; real model downloads need one of them.\n'
    gaps=1
  elif ! has_cmd hf; then
    printf '  - hf CLI: missing; real downloads can still use uvx --from huggingface_hub hf if uvx is available.\n'
    gaps=1
  fi
  if ! has_cmd blender; then
    printf '  - blender: missing; 3D render/export routes need Blender before real pipeline runs.\n'
    gaps=1
  fi
  if ! has_cmd nvidia-smi; then
    printf '  - nvidia-smi: missing; GPU availability is unknown and must be confirmed on the target host.\n'
    gaps=1
  fi
  if ! has_cmd git; then
    printf '  - git: missing; repo-based model/runtime dependencies need git.\n'
    gaps=1
  fi
  if ! has_cmd curl; then
    printf '  - curl: missing; installer/bootstrap commands may need curl.\n'
    gaps=1
  fi
  if [[ "$gaps" == "0" ]]; then
    printf '  - none detected for the dry-run host. Still confirm CUDA, disk, and HF access before real setup.\n'
  fi
}

print_probe_plan() {
  section "preflight"
  printf 'mode: dry-run only; no install/download/long job will run\n'
  printf 'profile: %s\n' "$PROFILE"
  printf 'workspace_root: %s\n' "$WORKSPACE_ROOT"
  printf 'model_root: %s\n' "$MODEL_ROOT"
  printf 'python_backend: %s\n' "$PYTHON_BACKEND"
  printf '\nDetected commands:\n'
  for name in uname git curl python3 python uv uvx conda nvidia-smi blender hf; do
    cmd_status "$name"
  done
  print_actionable_gaps
  printf '\nCommands to run before real setup:\n'
  printf '  uname -a\n'
  printf '  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv\n'
  printf '  python3 --version || python --version\n'
  printf '  uv --version || conda --version\n'
  printf '  uvx --from huggingface_hub hf --help || hf --help\n'
  printf '  blender --version\n'
  printf '  hf auth whoami\n'
}

print_env_plan() {
  section "env plan"
  if [[ "$PYTHON_BACKEND" == "uv" || "$PYTHON_BACKEND" == "auto" ]]; then
    printf 'Preferred uv plan:\n'
    printf '  cd %s\n' "$(quote "$WORKSPACE_ROOT")"
    printf '  uv venv .venv --python 3.10 --system-site-packages\n'
    printf '  source .venv/bin/activate\n'
    printf '  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"\n'
    printf '  # Fallback only if the torch import above fails or has no usable CUDA:\n'
    printf '  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n'
    printf '  uv pip install "numpy<2" "tokenizers==0.22.1" diffsynth transformers accelerate diffusers safetensors pillow opencv-python scipy scikit-image imageio trimesh rembg[gpu] anthropic qwen-vl-utils lpips basicsr realesrgan iopath timm ftfy opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http\n'
  fi
  if [[ "$PYTHON_BACKEND" == "conda" || "$PYTHON_BACKEND" == "auto" ]]; then
    printf '\nConda fallback plan:\n'
    printf '  conda create -n dataevolver python=3.10 -y\n'
    printf '  conda activate dataevolver\n'
    printf '  python -c "import torch; print(torch.__version__, torch.cuda.is_available())" || true\n'
    printf '  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n'
    printf '  pip install "numpy<2" "tokenizers==0.22.1" diffsynth transformers accelerate diffusers safetensors pillow opencv-python scipy scikit-image imageio trimesh rembg[gpu] anthropic qwen-vl-utils lpips basicsr realesrgan iopath timm ftfy opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http\n'
  fi
  printf '\nTool bootstrap plan if commands are missing (printed only):\n'
  printf '  # Install uv for the fastest Python environment path:\n'
  printf '  curl -LsSf https://astral.sh/uv/install.sh | sh\n'
  printf '  # Verify the Hugging Face CLI without persisting a token:\n'
  printf '  uvx --from huggingface_hub hf --help || hf --help\n'
  printf '  # Optional standalone Hugging Face CLI installer:\n'
  printf '  curl -LsSf https://hf.co/cli/install.sh | bash\n'
  printf '\nSource and native-extension plan after target-host preflight:\n'
  printf '  # Keep source checkouts separate from weight directories.\n'
  printf '  git clone --depth 1 https://github.com/facebookresearch/sam3.git .dataevolver/local/vendor/sam3-src\n'
  printf '  git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git .dataevolver/local/vendor/Hunyuan3D-2.1-src\n'
  printf '  test -f "$HUNYUAN3D_REPO/requirements.txt" && uv pip install -r "$HUNYUAN3D_REPO/requirements.txt"\n'
  printf '  uv pip install --no-build-isolation -e "$HUNYUAN3D_REPO/hy3dpaint/custom_rasterizer"\n'
  printf '  cd "$HUNYUAN3D_REPO/hy3dpaint/DifferentiableRenderer" && bash compile_mesh_painter.sh\n'
}

model_line() {
  local role="$1"
  local name="$2"
  local repo="$3"
  local subdir="$4"
  local gated="$5"
  local env_name="$6"
  local target="${MODEL_ROOT}/${subdir}"
  printf '%s|%s|%s|%s|%s|%s\n' "$role" "$name" "$repo" "$target" "$gated" "$env_name"
}

model_manifest() {
  if [[ "$PROFILE" == "quick" || "$PROFILE" == "custom" ]]; then
    return 0
  fi

  model_line "t2i_generator" "Qwen-Image-2512" "Qwen/Qwen-Image-2512" "Qwen-Image-2512" "no" "QWEN_IMAGE_MODEL_PATH"
  model_line "segmenter" "SAM3" "facebook/sam3" "sam3" "yes" "SAM3_CKPT"
  model_line "image_to_3d" "Hunyuan3D-2.1" "tencent/Hunyuan3D-2.1" "Hunyuan3D-2.1" "no" "MODEL_HUB"
  model_line "image_to_3d_dino" "DINOv2 Giant" "facebook/dinov2-giant" "dinov2-giant" "no" "DINO_MODEL_PATH"
  model_line "vlm_reviewer" "Qwen3.5-35B-A3B" "Qwen/Qwen3.5-35B-A3B" "Qwen3.5-35B-A3B" "access-dependent" "VLM_MODEL_PATH"

  if [[ "$PROFILE" == "full" ]]; then
    model_line "image_edit_generator" "Qwen-Image-Edit-2511" "Qwen/Qwen-Image-Edit-2511" "Qwen-Image-Edit-2511" "no" "QWEN_IMAGE_EDIT_MODEL_PATH"
    model_line "t2v_generator" "Wan2.1-T2V-1.3B-Diffusers" "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" "Wan2.1-T2V-1.3B-Diffusers" "no" "WAN_T2V_MODEL_PATH"
  fi
}

print_model_plan() {
  section "model plan"
  if [[ "$PROFILE" == "quick" ]]; then
    printf 'quick profile: no model downloads planned. Use dry-run routes only.\n'
    return
  fi
  if [[ "$PROFILE" == "custom" ]]; then
    printf 'custom profile: no default model downloads planned in v0.\n'
    printf 'Record replacement models as custom_models with status=needs_check in env.config.json.\n'
    return
  fi

  printf 'All commands below are printed only; they are not executed in v0.\n'
  printf 'HF token policy: set HF_TOKEN in the shell for real downloads; never write it to project files.\n\n'
  printf 'Access checks before real downloads (printed only):\n'
  printf '  test -n "${HF_TOKEN:-}" || echo "HF_TOKEN is not set"\n'
  printf '  uvx --from huggingface_hub hf auth whoami || hf auth whoami\n'
  printf '  # For gated/access-dependent repos, request access on Hugging Face before downloading.\n\n'
  while IFS='|' read -r role name repo target gated env_name; do
    [[ -n "$role" ]] || continue
    printf '%s\n' "- role: $role"
    printf '  model: %s\n' "$name"
    printf '  hf_repo: %s\n' "$repo"
    printf '  target: %s\n' "$target"
    printf '  gated_or_access_dependent: %s\n' "$gated"
    printf '  env: %s\n' "$env_name"
    printf '  uvx command: HF_TOKEN="${HF_TOKEN:?set HF_TOKEN for real downloads}" uvx --from huggingface_hub hf download %s --local-dir %s\n' "$(quote "$repo")" "$(quote "$target")"
    printf '  fallback command: HF_TOKEN="${HF_TOKEN:?set HF_TOKEN for real downloads}" hf download %s --local-dir %s\n' "$(quote "$repo")" "$(quote "$target")"
  done < <(model_manifest)

}

print_config_plan() {
  section "config plan"
  printf 'Local files when --write-local-config is used:\n'
  printf '  .dataevolver/local/ENVIRONMENT.md\n'
  printf '  .dataevolver/local/env.config.json\n'
  printf '  .dataevolver/local/env.sh.example\n'
  printf '\nNon-sensitive variables to export later:\n'
  printf '  DATAEVOLVER_MODEL_ROOT=%s\n' "$MODEL_ROOT"
  printf '  DATAEVOLVER_WORKSPACE_ROOT=%s\n' "$WORKSPACE_ROOT"
  printf '  BLENDER_BIN=/path/to/blender\n'
  if [[ "$PROFILE" == "default" || "$PROFILE" == "full" ]]; then
    printf '  QWEN_IMAGE_MODEL_PATH=%s/Qwen-Image-2512\n' "$MODEL_ROOT"
    printf '  QWEN_IMAGE_EDIT_MODEL_PATH=%s/Qwen-Image-Edit-2511\n' "$MODEL_ROOT"
    printf '  SAM3_CKPT=%s/sam3/sam3.pt\n' "$MODEL_ROOT"
    printf '  SAM3_DIR=<source checkout or package path for SAM3>\n'
    printf '  HUNYUAN3D_REPO=<clone path for Tencent-Hunyuan/Hunyuan3D-2.1>\n'
    printf '  MODEL_HUB=%s/Hunyuan3D-2.1\n' "$MODEL_ROOT"
    printf '  PAINT_MODEL_HUB=%s/Hunyuan3D-2.1\n' "$MODEL_ROOT"
    printf '  DINO_MODEL_PATH=%s/dinov2-giant\n' "$MODEL_ROOT"
    printf '  REALESRGAN_CKPT=<clone path for Tencent-Hunyuan/Hunyuan3D-2.1>/hy3dpaint/ckpt/RealESRGAN_x4plus.pth\n'
    printf '  VLM_MODEL_PATH=%s/Qwen3.5-35B-A3B\n' "$MODEL_ROOT"
    printf '  STAGE3_PAINT_MAX_FACES=5000\n'
    printf '  STAGE3_PAINT_REMESH_MODES=false\n'
    printf '  STAGE3_PAINT_ATTEMPT_TIMEOUT_SEC=300\n'
  fi
  if [[ "$PROFILE" == "full" ]]; then
    printf '  WAN_T2V_MODEL_PATH=%s/Wan2.1-T2V-1.3B-Diffusers\n' "$MODEL_ROOT"
  fi
  printf '\nRuntime env overrides to source before real one-click setup:\n'
  printf '  pipeline/stage2_t2i_generate.py: MODEL_PATH <- QWEN_IMAGE_MODEL_PATH\n'
  printf '  pipeline/stage2_5_sam2_segment.py: SAM3_CKPT/SAM3_DIR <- env overrides\n'
  printf '  pipeline/stage3_image_to_3d.py: HUNYUAN3D_REPO/MODEL_HUB/PAINT_MODEL_HUB/DINO_MODEL_PATH/REALESRGAN_CKPT <- env overrides\n'
  printf '  pipeline/stage5_5_vlm_review.py: VLM_MODEL_PATH <- env override\n'
  printf '  scene/render scripts: BLENDER_BIN <- env override\n'
}

write_local_config() {
  local out_dir="${WORKSPACE_ROOT}/.dataevolver/local"
  local python_bin=""
  if has_cmd python3; then
    python_bin="$(command -v python3)"
  elif has_cmd python; then
    python_bin="$(command -v python)"
  else
    die "--write-local-config requires python3 or python for safe JSON writing"
  fi

  MODEL_MANIFEST="$(model_manifest)" "$python_bin" - "$PROFILE" "$WORKSPACE_ROOT" "$MODEL_ROOT" "$PYTHON_BACKEND" "$out_dir" <<'PY'
import json
import os
import shutil
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

profile, workspace_root, model_root, python_backend, out_dir = sys.argv[1:6]
out = Path(out_dir)
out.mkdir(parents=True, exist_ok=True)

models = []
for line in os.environ.get("MODEL_MANIFEST", "").splitlines():
    if not line.strip():
        continue
    role, name, repo, target, gated, env_name = line.split("|", 5)
    models.append({
        "role": role,
        "name": name,
        "hf_repo": repo,
        "target_path": target,
        "gated_or_access_dependent": gated,
        "env": env_name,
        "status": "planned_dry_run",
    })

tool_names = ["git", "curl", "python3", "python", "uv", "uvx", "conda", "hf", "nvidia-smi", "blender"]
tooling_status = {
    name: {
        "status": "present" if shutil.which(name) else "missing",
        "path": shutil.which(name),
    }
    for name in tool_names
}

path_override_plan = [
    {
        "file": "pipeline/stage2_t2i_generate.py",
        "constant": "MODEL_PATH",
        "env": "QWEN_IMAGE_MODEL_PATH",
    },
    {
        "file": "pipeline/stage2_5_sam2_segment.py",
        "constant": "SAM3_CKPT",
        "env": "SAM3_CKPT",
    },
    {
        "file": "pipeline/stage2_5_sam2_segment.py",
        "constant": "SAM3_DIR",
        "env": "SAM3_DIR",
    },
    {
        "file": "pipeline/stage3_image_to_3d.py",
        "constant": "HUNYUAN3D_REPO",
        "env": "HUNYUAN3D_REPO",
    },
    {
        "file": "pipeline/stage3_image_to_3d.py",
        "constant": "MODEL_HUB",
        "env": "MODEL_HUB",
    },
    {
        "file": "pipeline/stage3_image_to_3d.py",
        "constant": "PAINT_MODEL_HUB",
        "env": "PAINT_MODEL_HUB",
    },
    {
        "file": "pipeline/stage3_image_to_3d.py",
        "constant": "DINO_MODEL_PATH",
        "env": "DINO_MODEL_PATH",
    },
    {
        "file": "pipeline/stage3_image_to_3d.py",
        "constant": "REALESRGAN_CKPT",
        "env": "REALESRGAN_CKPT",
    },
    {
        "file": "pipeline/stage5_5_vlm_review.py",
        "constant": "VLM_MODEL_PATH",
        "env": "VLM_MODEL_PATH",
    },
    {
        "file": "scene/render scripts",
        "constant": "BLENDER_BIN",
        "env": "BLENDER_BIN",
    },
]

access_requirements = [
    {
        "name": model["name"],
        "hf_repo": model["hf_repo"],
        "requirement": model["gated_or_access_dependent"],
        "action": "Confirm Hugging Face access and provide HF_TOKEN only at real download time.",
    }
    for model in models
    if model["gated_or_access_dependent"] != "no"
]

next_actions = [
    "Review this dry-run plan with the user.",
    "Ask for HF_TOKEN only at real download time; never write it to project files.",
]
if tooling_status["uv"]["status"] == "missing":
    next_actions.append("Install uv for the fastest default Python environment path, or choose conda fallback.")
if tooling_status["uvx"]["status"] == "missing" and tooling_status["hf"]["status"] == "missing":
    next_actions.append("Install uvx/uv or the standalone hf CLI before real Hugging Face downloads.")
if tooling_status["blender"]["status"] == "missing":
    next_actions.append("Install or expose Blender before real 3D render/export routes.")
if tooling_status["nvidia-smi"]["status"] == "missing":
    next_actions.append("Confirm GPU/CUDA on the target host before real model setup.")
if access_requirements:
    next_actions.append("Confirm Hugging Face access for gated/access-dependent model repos.")
next_actions.append("Before real setup, source reviewed path overrides derived from env.sh.example.")
next_actions.append("Run smoke tests in order: preflight, import smoke, Stage 2, Stage 2.5, Stage 3 shape-only, Stage 3 textured, Stage 5.5 VLM loader.")

target_workflow = {
    "quick": "quick_demo",
    "default": "full_core_pipeline",
    "full": "all_default_routes",
    "custom": "custom_model_planning",
}[profile]

payload = {
    "profile": profile,
    "target_workflow": target_workflow,
    "runtime_location": {
        "kind": "local_or_remote_path",
        "workspace_root": workspace_root,
    },
    "install_policy": {
        "mode": "dry_run_only",
        "python_backend": python_backend,
        "allow_real_install": False,
        "allow_real_model_download": False,
    },
    "model_root": model_root,
    "workspace_root": workspace_root,
    "tooling_status": tooling_status,
    "models": models,
    "custom_models": [] if profile != "custom" else [{
        "role": "unspecified",
        "choice": "custom",
        "status": "needs_check",
        "notes": "Fill from onboarding interview before real deployment.",
    }],
    "access_requirements": access_requirements,
    "path_override_plan": path_override_plan,
    "v1_blockers": [
        "Source or export the path override variables before running the pipeline on a new host.",
        "Keep SAM3 and Hunyuan3D source checkouts separate from downloaded weight directories.",
        "Install SAM3 and Hunyuan3D repo-specific dependencies after target-host preflight.",
        "Compile Hunyuan3D CUDA/C++ extensions only after CUDA/nvcc compatibility is known.",
        "Run Stage 3 textured smoke with bounded paint settings before treating the host as production-ready.",
    ],
    "next_actions": next_actions,
    "generated_at": datetime.now(timezone.utc).isoformat(),
}

(out / "env.config.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

model_rows = "\n".join(
    f"- {m['role']}: {m['name']} -> `{m['target_path']}` ({m['status']}, gated/access: {m['gated_or_access_dependent']})"
    for m in models
) or "- No default model downloads planned for this profile."

missing_tools = []
if tooling_status["python3"]["status"] == "missing" and tooling_status["python"]["status"] == "missing":
    missing_tools.append("python3/python")
for name in ["git", "curl", "uv", "uvx", "conda", "hf", "nvidia-smi", "blender"]:
    if tooling_status[name]["status"] == "missing":
        missing_tools.append(name)
missing_tool_rows = "\n".join(f"- {name}: missing" for name in missing_tools) or "- None detected on the dry-run host."

access_rows = "\n".join(
    f"- {item['name']}: `{item['hf_repo']}` ({item['requirement']})"
    for item in access_requirements
) or "- No gated/access-dependent model repos in this profile."

override_rows = "\n".join(
    f"- `{item['file']}`: `{item['constant']}` should read `{item['env']}`"
    for item in path_override_plan
)

next_action_rows = "\n".join(f"- {item}" for item in next_actions)

dependency_plan = """Use `uv venv .venv --python 3.10 --system-site-packages` first on shared GPU servers so a validated system NVIDIA PyTorch build can be reused. Install the CUDA wheel only when `import torch` fails inside the venv or reports no usable CUDA.

Runtime pins from production validation:

- `numpy<2`
- `tokenizers==0.22.1`
- SAM3 extras: `iopath`, `timm`, `ftfy`
- Hunyuan paint/native build: `uv pip install --no-build-isolation -e "$HUNYUAN3D_REPO/hy3dpaint/custom_rasterizer"` after CUDA/nvcc preflight
- DifferentiableRenderer: compile only after CUDA/nvcc compatibility is confirmed"""

path_separation = """Keep code and weights separate:

- `SAM3_CKPT` points to `sam3.pt`.
- `SAM3_DIR` points to the SAM3 source checkout or importable package.
- `HUNYUAN3D_REPO` points to the Tencent-Hunyuan/Hunyuan3D-2.1 source checkout.
- `MODEL_HUB` and `PAINT_MODEL_HUB` point to Hunyuan3D weight directories."""

smoke_plan = """Production readiness smoke order:

1. Preflight: Python, uv, GPU, disk, Blender, and model paths.
2. Import smoke: torch CUDA/bf16, Qwen image dependencies, SAM3, Hunyuan shape/paint, transformers, and telemetry packages.
3. Runtime smoke: Stage 2 -> Stage 2.5 -> Stage 3 shape-only -> Stage 3 textured -> Stage 5.5 VLM loader.
4. Stage 3 textured smoke should write to a local smoke directory such as `.dataevolver/local/smoke/meshes_textured` and use bounded paint controls such as `--paint-max-faces 5000 --paint-remesh-modes false --paint-attempt-timeout-sec 300`.
5. After the agent records the smoke evidence in the deployment note, delete generated smoke artifacts with `rm -rf .dataevolver/local/smoke` so the working directory stays clean.

`--shape-only` is a fallback for missing DINO, RealESRGAN, or paint extensions; it is not a full textured production-readiness check."""

(out / "ENVIRONMENT.md").write_text(
    f"""# DataEvolver Local Environment

Generated by `scripts/bootstrap_dataevolver_default.sh` in dry-run mode.

## Summary

- Profile: `{profile}`
- Target workflow: `{target_workflow}`
- Workspace root: `{workspace_root}`
- Model root: `{model_root}`
- Python backend preference: `{python_backend}`
- Install policy: dry-run only; no real install or model download has been performed.

## Models

{model_rows}

## Preflight Gaps

{missing_tool_rows}

## Hugging Face Access

{access_rows}

## V1 Path Overrides

{override_rows}

## Production Dependency Plan

{dependency_plan}

## Source and Weight Path Separation

{path_separation}

## Production Smoke Tests

{smoke_plan}

## Next Actions

{next_action_rows}
""",
    encoding="utf-8",
)

env_lines = [
    "# Source this file after reviewing paths. It intentionally contains no tokens.",
    f"export DATAEVOLVER_WORKSPACE_ROOT={shlex.quote(workspace_root)}",
    f"export DATAEVOLVER_MODEL_ROOT={shlex.quote(model_root)}",
    f"export BLENDER_BIN={shlex.quote(shutil.which('blender') or '/path/to/blender')}",
]
for model in models:
    env_name = model.get("env")
    if not env_name:
        continue
    value = model["target_path"]
    if env_name == "SAM3_CKPT":
        value = str(Path(value) / "sam3.pt")
    env_lines.append(f"export {env_name}={shlex.quote(value)}")
if profile in {"default", "full"}:
    if profile == "default":
        env_lines.append(f"export QWEN_IMAGE_EDIT_MODEL_PATH={shlex.quote(str(Path(model_root) / 'Qwen-Image-Edit-2511'))}")
    env_lines.append(f"export PAINT_MODEL_HUB={shlex.quote(str(Path(model_root) / 'Hunyuan3D-2.1'))}")
    env_lines.append(f"export SAM3_DIR={shlex.quote(str(Path(workspace_root) / '.dataevolver/local/vendor/sam3-src'))}")
    env_lines.append(f"export HUNYUAN3D_REPO={shlex.quote(str(Path(workspace_root) / '.dataevolver/local/vendor/Hunyuan3D-2.1-src'))}")
    env_lines.append('export REALESRGAN_CKPT="$HUNYUAN3D_REPO/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"')
    env_lines.append("export STAGE3_PAINT_MAX_FACES=5000")
    env_lines.append("export STAGE3_PAINT_REMESH_MODES=false")
    env_lines.append("export STAGE3_PAINT_ATTEMPT_TIMEOUT_SEC=300")
env_lines.append("# Set HF_TOKEN in your shell only when performing real downloads.")
(out / "env.sh.example").write_text("\n".join(env_lines) + "\n", encoding="utf-8")

print(f"[write-local-config] wrote {out / 'ENVIRONMENT.md'}")
print(f"[write-local-config] wrote {out / 'env.config.json'}")
print(f"[write-local-config] wrote {out / 'env.sh.example'}")
PY
}

print_probe_plan
print_env_plan
print_model_plan
print_config_plan

if [[ "$WRITE_LOCAL_CONFIG" == "1" ]]; then
  write_local_config
else
  section "write-local-config"
  printf 'Skipped. Re-run with --write-local-config to write .dataevolver/local demo files.\n'
fi

section "done"
printf 'Dry-run demo complete. No install, download, token write, or long-running job was performed.\n'
