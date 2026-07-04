<p align="center">
  <img src="assets/showcase.png" alt="DataEvolver Showcase" width="800"/>
</p>

<h1 align="center">DataEvolver</h1>

<p align="center">
  <strong>Autonomous Synthetic Data Construction via VLM-Guided Iterative Rendering</strong><br/>
  Build photorealistic, scene-aware training data with a closed loop of 3D rendering, VLM review, and targeted parameter repair.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.01789"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white" alt="Paper: arXiv"/></a>
  <a href="https://pris-cv.github.io/DataEvolver/"><img src="https://img.shields.io/badge/Website-Project%20Page-1f6feb?logo=githubpages&logoColor=white" alt="Project website"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="License: Apache-2.0"/></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/Blender-4.2%2B-f5792a?logo=blender&logoColor=white" alt="Blender 4.2+"/>
</p>

<p align="center">
  <a href="README_zh.md">🇨🇳 中文</a> •
  <a href="https://pris-cv.github.io/DataEvolver/">🌐 Website</a> •
  <a href="https://arxiv.org/abs/2605.01789">📄 Paper</a> •
  <a href="#honors--recognition">🏆 Recognition</a> •
  <a href="#dataevolver-rotate">🧩 Dataset: DataEvolver-Rotate</a>
</p>

<p align="center">
  <strong>🧠 VLM free-form feedback</strong> •
  <strong>🎬 3D render-review-fix loop</strong> •
  <strong>📦 Training-ready multimodal data</strong>
</p>

---

## Why DataEvolver?

DataEvolver turns synthetic data construction into a **goal-driven optimization loop**: a VLM reviews rendered scenes in natural language, an agent diagnoses concrete rendering issues, and the scene is re-rendered with targeted parameter updates until the result is worth keeping.

- **Perceive beyond rigid scores** — free-form VLM feedback catches scene context, object placement, lighting, and material issues that fixed rules miss.
- **Repair with structured actions** — 24 bounded atomic actions adjust lighting, object pose, scene environment, and material appearance without uncontrolled drift.
- **Export training-ready data** — RGB, masks, depth, normals, geometry metadata, and object-disjoint splits are produced for downstream model training.

## Key Features

- **Goal-Driven Loop Agents** — VLM reviewer provides semantic feedback ("flat lighting", "floating object") &rarr; AI agent selects targeted actions &rarr; re-render &rarr; repeat until quality goals are met
- **24 Atomic Actions** — Structured action space across 5 groups: lighting, object placement, scene environment, and material properties — with anti-oscillation control and step-scale scheduling
- **Scene-Aware Rendering** — Objects placed in real Blender scenes with HDRI environments, raycast ground detection, and preserved scene lighting
- **Multi-Modal Output** — RGB, mask, depth, normal maps, and geometry metadata
- **End-to-End Automation** — From natural language seed concept to training-ready dataset, zero human intervention

---

## Pipeline Overview

![Pipeline Overview](assets/pipeline_show.svg)

| Stage | What it does | Model / Tool |
|-------|-------------|-------------|
| **1. Text Expansion** | LLM expands seed concept into detailed T2I prompt | Claude API (Anthropic) |
| **2. T2I Generation** | Generate 1024&times;1024 object image | Qwen-Image-2512 |
| **2.5. Segmentation** | Extract RGBA foreground, remove background | SAM3 |
| **3. 3D Reconstruction** | Reconstruct textured mesh from single image | Hunyuan3D-2.1 |
| **4. Scene Rendering** | Scene-aware Blender insertion with configurable EEVEE/Cycles rendering | Blender 4.2+ |
| **5. VLM Review Loop** | Free-form review &rarr; agent action &rarr; re-render until *keep* | Qwen3.5-35B-A3B |

### The VLM Review Loop (Stage 5)

The core innovation: a **goal-driven loop agent** that iteratively improves rendering quality.

![VLM Review Loop](assets/stage5.svg)

**Anti-oscillation control** prevents parameter thrashing:
- Sign-flip tracking: freeze a parameter after 3 direction reversals
- Step-scale scheduling: Round 0 → 100%, Round 1 → 70%, Round 2 → 50%, Round 3+ → 40%
- Score-adaptive boost: &times;1.2 when hybrid_score < 0.65

---

## Prerequisites

- **OS**: Linux (tested on Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with &ge;24 GB VRAM for rendering; &ge;80 GB for VLM inference
- **Python**: 3.10+
- **Blender**: 4.2+
- **CUDA**: Compatible with your PyTorch version

### Required Models

| Model | Purpose | Approx. Size |
|-------|---------|-------------|
| [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) | T2I generation | ~56 GB |
| [SAM3](https://github.com/facebookresearch/sam3) | Foreground segmentation | ~2 GB |
| [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | Image-to-3D reconstruction | ~20 GB |
| [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | VLM quality reviewer | ~35 GB |
| [Blender 4.2+](https://www.blender.org/download/) | 3D rendering engine | ~300 MB |

---

## Quick Start

### Production Setup

Start with the dry-run profile. It probes the host and writes local, non-secret setup files without installing packages, downloading models, or launching long jobs.

```bash
git clone https://github.com/PRIS-CV/DataEvolver.git
cd DataEvolver

bash scripts/bootstrap_dataevolver_default.sh \
  --profile default \
  --dry-run \
  --write-local-config
```

For HYWorld / WorldMirror scene reconstruction, use the optional world-model profile instead of `default`:

```bash
bash scripts/bootstrap_dataevolver_default.sh \
  --profile world_model \
  --dry-run \
  --write-local-config
```

Create the runtime environment. On shared GPU servers, prefer `--system-site-packages` so an already validated NVIDIA PyTorch build can be reused; install the CUDA wheel only if `import torch` fails inside the venv.

```bash
uv venv .venv --python 3.10 --system-site-packages
source .venv/bin/activate

python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available())
PY

# Fallback only when the torch import above fails or reports no usable CUDA.
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

uv pip install \
  "numpy<2" "tokenizers==0.22.1" \
  diffsynth transformers accelerate diffusers safetensors \
  pillow opencv-python scipy scikit-image imageio trimesh \
  rembg[gpu] anthropic qwen-vl-utils lpips basicsr realesrgan \
  iopath timm ftfy \
  opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

Review `.dataevolver/local/env.sh.example`, copy it to a machine-local file such as `.dataevolver/local/env.remote.sh`, then source it before every real run.

```bash
cp .dataevolver/local/env.sh.example .dataevolver/local/env.remote.sh
# Edit paths in .dataevolver/local/env.remote.sh. Do not put tokens in this file.
source .dataevolver/local/env.remote.sh
source .venv/bin/activate
```

Production deployments should use environment variables instead of editing pipeline scripts:

| Variable | Points to |
|----------|-----------|
| `DATAEVOLVER_WORKSPACE_ROOT` | DataEvolver checkout |
| `QWEN_IMAGE_MODEL_PATH` | Qwen-Image-2512 weights |
| `QWEN_IMAGE_EDIT_MODEL_PATH` | Qwen-Image-Edit-2511 weights, optional for edit routes |
| `SAM3_CKPT` | `sam3.pt` checkpoint |
| `SAM3_DIR` | SAM3 source checkout or importable package path |
| `HUNYUAN3D_REPO` | Tencent-Hunyuan/Hunyuan3D-2.1 source checkout |
| `MODEL_HUB` | Hunyuan3D-2.1 weights |
| `PAINT_MODEL_HUB` | Hunyuan3D paint weights, usually the same as `MODEL_HUB` |
| `DINO_MODEL_PATH` | DINOv2 Giant checkpoint directory |
| `REALESRGAN_CKPT` | `RealESRGAN_x4plus.pth` used by Hunyuan paint |
| `VLM_MODEL_PATH` | Qwen3.5-35B-A3B, or a verified compatible Qwen3.x replacement |
| `BLENDER_BIN` | Blender executable |

Keep source checkouts and model weights separate. `SAM3_CKPT` is the checkpoint file, while `SAM3_DIR` is the SAM3 code path. `HUNYUAN3D_REPO` is the Hunyuan3D source checkout, while `MODEL_HUB` and `PAINT_MODEL_HUB` are weight directories.

Run the generated production profile through the doctor before starting production jobs. A new host with missing paths should return structured JSON listing the blockers instead of crashing.

```bash
python scripts/dataevolver_production.py doctor \
  --profile .dataevolver/local/production_profile.json \
  --no-imports
```

#### Optional HYWorld / WorldMirror Scene Reconstruction

Use `--profile world_model` only when the target route is HYWorld / WorldMirror scene reconstruction. This profile adds the world-model dependencies and generated environment variables:

| Variable | Points to |
|----------|-----------|
| `HYWORLD_SRC` | HY-World-2.0 source checkout |
| `HYWORLD_WEIGHTS` | HY-World-2.0 weights, including `HY-WorldMirror-2.0` |
| `HYWORLD_PYTHON` | Python environment used for HYWorld dependencies |
| `HYWORLD_MOGE_MODEL_PATH` | Local MoGe model used for real panorama depth |
| `HYWORLD_ZIM_MODEL_PATH` | Local ZIM model used for sky masking |
| `HYWORLD_GROUNDING_DINO_MODEL_PATH` | Local GroundingDINO model used by HYWorld navigation |
| `HYWORLD_SAM3_MODEL_PATH` | Local SAM3 model/source path used by HYWorld |
| `HYWORLD_WORLDSTEREO_PATH` | Local WorldStereo weights root |
| `HYWORLD_WAN_BASE_MODEL` | Local Wan I2V base Diffusers model used by WorldStereo/video generation |

HYWorld production scene generation requires local MoGe, ZIM, GroundingDINO, SAM3, WorldStereo, Wan I2V base, and HY-WorldMirror weights. Offline mode forbids downloads but must still load these local models; do not set `HYWORLD_NO_MODEL_DOWNLOADS=1` for real 3D scene generation because that skips metric depth and can produce a fixed-radius panorama shell.

`scripts/run_hyworld_scene_pano.py` and `scripts/run_hyworld_full_worldgen.py` now preserve every new or changed stage artifact under `<scene-dir>/intermediates/<run-id>/` by default. The HY-Pano stage snapshots the generated 360-degree equirectangular panorama, and full worldgen snapshots that input again under `00_source_panorama/` before retaining trajectory, depth/sky-mask, WorldStereo, GS, and WorldMirror outputs. `manifest.json` records source paths, snapshot paths, sizes, and SHA-256 digests. Use `--intermediate-root` for durable storage; only `--no-preserve-intermediates` disables snapshots.

##### Automated Scene Construction

The world-model route can build a reviewable scene package automatically from scene prompts or an input scene image. The pipeline first generates a 360-degree HY-Pano context, runs HYWorld / WorldMirror reconstruction, converts metric geometry into a Blender scene contract, renders pure-scene multi-view evidence, and then inserts objects with fixed scene-camera alignment. The final report records contract checks, per-view scene renders, object rotation views, masks, depth, normals, metadata gates, and lineage hashes.

![HYWorld automated scene construction pipeline placeholder](assets/hyworld_scene_pipeline.png)

The image above is a placeholder for the generated 2:1 pipeline figure. Replace `assets/hyworld_scene_pipeline.png` with the final diagram asset before release.

```bash
# 1. Generate or ingest the scene panorama.
python scripts/run_hyworld_scene_pano.py \
  --scene-prompts-path <scene-prompts.json> \
  --output-root <hyworld-scene-root>

# 2. Run full world generation and preserve stage artifacts.
python scripts/run_hyworld_full_worldgen.py \
  --profile .dataevolver/local/production_profile.json \
  --scene-dir <hyworld-scene-root>/<scene-id> \
  --intermediate-root <intermediate-root>/<scene-id>

# 3. Build per-view Blender scene shards and pure-scene validation renders.
python scripts/build_hyworld_scene_view_shards.py \
  --scene-id <scene-id> \
  --hyworld-scene-dir <hyworld-scene-root>/<scene-id> \
  --scene-output-dir <scene-output-root>/<scene-id>/reconstruction \
  --template-output-dir <template-output-root> \
  --report-scene-dir <report-root>/<scene-id> \
  --dry-run

# 4. Finalize the object-scene package after object insertion/rendering.
python scripts/finalize_hyworld_object_scene_report.py \
  --dataset-base <dataset-base> \
  --strict-scene-views
```

Run the generated world-model production profile through the doctor before starting a world-model job.

```bash
python scripts/dataevolver_production.py doctor \
  --profile .dataevolver/local/production_profile.json \
  --no-imports

python scripts/run_hyworld_scene_pano.py \
  --scene-prompts-path /data/scenes/scene_prompts.json \
  --output-root /data/hyworld_scenes

python scripts/run_hyworld_full_worldgen.py \
  --profile .dataevolver/local/production_profile.json \
  --scene-dir /data/hyworld_scenes/scene_001 \
  --intermediate-root /data/hyworld-intermediates/scene_001

python scripts/build_hyworld_scene_view_shards.py \
  --scene-id scene_001 \
  --hyworld-scene-dir /data/hyworld_scenes/scene_001 \
  --scene-output-dir /data/hyworld_scenes/scene_001/reconstruction \
  --template-output-dir /data/hyworld_templates \
  --report-scene-dir /data/hyworld_reports/scene_001 \
  --dry-run

python scripts/finalize_hyworld_object_scene_report.py \
  --dataset-base /data/hyworld_dataset \
  --strict-scene-views
```

HYWorld / WorldMirror follows the staged world-generation route HY-Pano -> WorldNav -> WorldStereo -> WorldMirror/3DGS. Do not treat a VLM pass as authoritative for world-model completion; use contract-backed geometry, multi-view pure-scene renders, and final manifest/lineage evidence.

After CUDA and `nvcc` are confirmed, build Hunyuan3D's native pieces:

```bash
uv pip install --no-build-isolation -e "$HUNYUAN3D_REPO/hy3dpaint/custom_rasterizer"
cd "$HUNYUAN3D_REPO/hy3dpaint/DifferentiableRenderer"
bash compile_mesh_painter.sh
```

### Production Smoke Tests

Run smoke tests in this order on a new host:

```bash
python3 --version
uv --version
nvidia-smi
df -h .
"$BLENDER_BIN" --version

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), "bf16:", torch.cuda.is_bf16_supported())
from transformers import AutoProcessor
from opentelemetry import trace
PY
```

Then validate the runtime pipeline on one object:

```bash
SMOKE_ROOT=.dataevolver/local/smoke

python pipeline/stage2_t2i_generate.py --ids obj_001 --steps 1 --height 512 --width 512 --device cuda:0
python pipeline/stage2_5_sam2_segment.py --ids obj_001 --device cuda:0
python pipeline/stage3_image_to_3d.py --ids obj_001 --shape-only --device cuda:0 --output-dir "$SMOKE_ROOT/meshes_shape_only"

python pipeline/stage3_image_to_3d.py \
  --no-skip \
  --ids obj_001 \
  --device cuda:0 \
  --output-dir "$SMOKE_ROOT/meshes_textured" \
  --paint-max-faces 5000 \
  --paint-remesh-modes false \
  --paint-attempt-timeout-sec 300
```

`--shape-only` is a fallback for missing DINO, RealESRGAN, or paint extensions; it is not a complete textured production-readiness check. A production-ready Stage 3 smoke should finish the textured command and produce a GLB with an image texture.

After the agent has summarized the smoke evidence into a deployment note, remove the generated smoke artifacts so the working directory stays clean:

```bash
rm -rf .dataevolver/local/smoke
```

The base pipeline prepares assets and render outputs after the setup above is verified:

```bash
bash pipeline/run_all.sh
```

The scene-aware VLM optimization loop is launched separately through the scene-agent workflow, for example with `scripts/run_scene_agent_monitor.py` after model paths, `BLENDER_BIN`, and `configs/scene_template.json` are configured for your environment.

---

## Project Structure

```
DataEvolver/
├── pipeline/                          # Core pipeline stages
│   ├── stage1_text_expansion.py             # LLM prompt generation
│   ├── stage2_t2i_generate.py               # Text-to-image (Qwen-Image-2512)
│   ├── stage2_5_sam2_segment.py             # SAM3 foreground extraction
│   ├── stage3_image_to_3d.py                # 3D mesh reconstruction (Hunyuan3D-2.1)
│   ├── stage4_scene_render.py               # Blender scene-aware rendering
│   ├── stage5_5_vlm_review.py               # VLM quality review (Qwen3.5-35B-A3B)
│   ├── stage5_6_feedback_apply.py           # Action selection & anti-oscillation
│   ├── asset_lifecycle.py                   # Asset lifecycle management
│   └── rotation_geomodal_dataset.py         # Training dataset loader
├── configs/
│   ├── scene_action_space.json              # 24 atomic actions definition
│   ├── scene_template.json                  # Blender scene template config
│   ├── vlm_review_schema.json               # VLM review output schema
│   ├── dataset_profiles/                    # Dataset configuration profiles
│   └── seed_concepts/                       # Seed object definitions (20/50 objects)
├── scripts/                           # Utility & build scripts
│   ├── run_scene_agent_monitor.py           # VLM loop agent monitor
│   ├── run_scene_agent_step.py              # Single-step agent execution
│   ├── export_rotation8_from_best_object_state.py  # Consistent rotation export
│   ├── build_rotation8_trainready_dataset.py       # Build training pairs
│   ├── build_object_split_for_rotation_dataset.py  # Object-disjoint split
│   ├── run_full_pipeline.py                 # Full pipeline orchestrator
│   ├── run_vlm_quality_gate_loop.py         # VLM quality gate loop
│   ├── feedback_loop/                       # Feedback loop utilities
│   └── ...                                  # Additional build & eval scripts
├── assets/
│   ├── hdri/                                # HDRI environment maps
│   └── scene/                               # Blender scene files (.blend)
├── paper/                             # Technical report (LaTeX source)
└── web/                               # Project website (GitHub Pages)
```

---

## Action Space

The AI agent selects from **24 structured atomic actions** organized in 5 groups:

| Group | Actions | Parameters |
|-------|---------|-----------|
| **Lighting** (4) | Key light intensity &uarr;&darr;, key light yaw &plusmn;15&deg; | Multiplicative &times;1.2/&times;0.8 or additive, bounded |
| **Object** (6) | Elevation &plusmn;0.02, yaw &plusmn;15&deg;, scale &times;1.1/&times;0.9 | Bounded within safe ranges |
| **Scene** (5) | Env rotation &plusmn;30&deg;, env intensity &uarr;&darr;, contact shadow | HDRI and environment controls |
| **Material** (9) | Saturation, value/brightness, hue offset, roughness, specular/sheen | Fine-grained material tuning |
| **Camera** (0) | Reserved for future use | — |

Full action definitions: [`configs/scene_action_space.json`](configs/scene_action_space.json)

---

## DataEvolver-Rotate

The first benchmark dataset produced by DataEvolver — for rotation-conditioned image editing.

| Metric | Value |
|--------|-------|
| Unique Objects | 50 |
| Rotation Angles | 8 (0&deg;, 45&deg;, 90&deg;, 135&deg;, 180&deg;, 225&deg;, 270&deg;, 315&deg;) |
| Training Pairs | 350 (front &rarr; 7 target views) |
| Train / Val / Test | 245 / 49 / 56 pairs (object-disjoint, seed=42) |
| Modalities | RGB, mask, depth, normal |

### Methodology

Each object uses a single **canonical yaw-0&deg; best state** as the base. The object is then rotated while the scene, camera, lighting, and material remain fixed — ensuring cross-angle consistency.

### Loading the Dataset

```python
import json
from pathlib import Path
from PIL import Image

root = Path("path/to/dataset_split")
rows = []
with (root / "pairs" / "train_pairs.jsonl").open("r") as f:
    for line in f:
        rows.append(json.loads(line))

row = rows[0]
source = Image.open(root / row["source_image"]).convert("RGB")
target = Image.open(root / row["target_image"]).convert("RGB")
instruction = row["instruction"]  # e.g., "Rotate the object 45 degrees clockwise"
```

---

## Using with Claude Code

DataEvolver is designed to work with [Claude Code](https://docs.anthropic.com/en/docs/claude-code) as an AI-powered development and operations assistant. Claude Code reads a project-level `CLAUDE.md` file to understand your environment, then helps you run pipelines, analyze results, and build datasets through natural language.

### Step 1: Install Claude Code

```bash
npm install -g @anthropic-ai/claude-code
```

### Step 2: Create your `CLAUDE.md`

Create a `CLAUDE.md` in the project root (it's gitignored — each user maintains their own). This file tells Claude Code about your specific environment:

```markdown
# CLAUDE.md

## Remote Server
- SSH alias: `my-server`
- GPU: 3x A800 80GB (or your setup)
- Python: `/path/to/python3` (3.10+, with PyTorch)
- Blender: `/path/to/blender` (4.2+)
- Code directory: `/path/to/DataEvolver`

## Model Paths
- Qwen-Image-2512: `/path/to/Qwen-Image-2512`
- SAM3 checkpoint: `/path/to/sam3/sam3.pt`
- Hunyuan3D-2.1 repo: `/path/to/Hunyuan3D-2.1`
- Hunyuan3D-2.1 weights: `/path/to/model_hub/Hunyuan3D-2.1`
- Qwen3.5-35B-A3B: `/path/to/Qwen3.5-35B-A3B`

## Scene Config
- Blender scene file: `/path/to/scene.blend`
- HDRI directory: `/path/to/hdri/`
- Render engine: CYCLES, 512 samples, 1024x1024

## Key Configs
- Action space: `configs/scene_action_space.json` (24 atomic actions)
- Scene template: `configs/scene_template.json`
- Dataset profiles: `configs/dataset_profiles/`

## Pipeline
Stage 1 (Text Expansion) → Stage 2 (T2I) → Stage 2.5 (SAM3)
→ Stage 3 (3D Reconstruction) → Stage 4 (Blender Render) → Stage 5 (VLM Loop)

## Working Rules
- Always test on the remote server, not locally
- Use tmux for long-running tasks (screen not available on all servers)
- Read trace.json free-form text for VLM results — not just agg.json scores
- Only stop VLM loop when reviewer explicitly says "keep"
- Check GPU usage before launching new jobs
```

### Step 3: Create Claude Code Skills (Optional)

You can create reusable skills in a `skills/` directory for common workflows:

```bash
mkdir -p skills/scene-agent-loop
```

```markdown
# skills/scene-agent-loop/SKILL.md
---
name: scene-agent-loop
description: Manage the VLM review loop for scene rendering
---
# Scene Agent Loop
Monitor and continue the VLM review → render → agent decision loop.
Read trace.json to understand current state, then decide next action.
```

### Step 4: Launch Claude Code

```bash
cd DataEvolver
claude
```

Claude Code will read your `CLAUDE.md` and understand the full project context. Example commands:

```
> Check GPU usage on the server
> Run the full pipeline for 10 new furniture objects
> Export rotation8 dataset from the latest best states
> Build train-ready dataset with object-disjoint split
```

---

## Citation

```bibtex
@misc{zhang2026dataevolverletdatabuild,
  title={DataEvolver: Let Your Data Build and Improve Itself via Goal-Driven Loop Agents},
  author={Qisong Zhang and Wenzhuo Wu and Zhuangzhuang Jia and Yunhao Yang and Huayu Zhang and Xianghao Zang and Zhixiang He and Zhongjiang He and Kongming Liang and Zhanyu Ma},
  year={2026},
  eprint={2605.01789},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2605.01789},
}
```

## Honors & Recognition

- 🏆 **[AIDataSci 2026, KDD 2026 Workshop](https://usail-hkust.github.io/aidatasci/)** — DataEvolver was accepted as an **Oral Presentation**. [View on OpenReview](https://openreview.net/forum?id=8ppusILCIH).
- 🎖️ **2026 BAAI Conference Agent for Science Competition** — DataEvolver received a **Certificate of Poster Presentation** and was selected for on-site poster presentation.

<p align="center">
  <img src="assets/baai_agent4s_poster_certificate.jpg" alt="DataEvolver BAAI Agent for Science poster presentation certificate" width="720"/>
</p>

## Roadmap

### Internal Enhancement — perception, understanding, and decision-making

- [ ] **Structured VLM knowledge base** — build a prior-knowledge store to anchor VLM quality assessment with objective criteria, reducing reliance on subjective judgment when evaluating data quality, task suitability, and assessment direction
- [ ] **WebSearch-integrated AI Agent** — add pre-task web retrieval: when the agent receives a user request, it searches for relevant datasets, papers, and existing methods to understand the data construction landscape before acting. Combine with local deep-research tools to establish background knowledge, preventing the model from working in an information vacuum

### External Expansion — richer dataset types and construction modalities

- [ ] **Multi-object scenes** — extend the rendering system to support multiple objects per scene, enabling composite geometry editing (rotation, insertion, removal, translation) within a single render
- [ ] **Video object datasets** — expand from image-based datasets to video-level object editing: removal, translation, and attribute editing across frames
- [ ] **Auto-generated reasoning-evaluation datasets** — after user training, automatically produce evaluation benchmarks with training–eval information alignment, customized rendering, and automated metric computation
- [ ] **Real-world dataset ingestion** — use WebSearch to discover real-world datasets, then crawl, clean, process, and construct structured datasets, broadening data source coverage and downstream applicability

## License

[Apache-2.0](LICENSE)
