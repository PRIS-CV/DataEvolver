# DataEvolver

**Autonomous Synthetic Data Construction via VLM-Guided Iterative Rendering**

DataEvolver is a goal-driven data synthesis pipeline that generates high-quality training datasets through an automated loop of 3D rendering, VLM (Vision-Language Model) quality review, and intelligent parameter adjustment. Unlike traditional pipelines with rigid scoring rules, DataEvolver uses free-form VLM feedback to perceive, diagnose, and fix rendering issues — producing photorealistic, scene-aware training data without human intervention.

[Website](https://kamisato520.github.io/DataEvolver/) &middot; [Paper](paper/main.tex) &middot; [Dataset: DataEvolver-Rotate](#dataevolver-rotate)

---

## Key Features

- **Goal-Driven Loop Agents** — VLM reviewer provides semantic feedback ("flat lighting", "floating object") &rarr; AI agent selects targeted actions &rarr; re-render &rarr; repeat until quality goals are met
- **24 Atomic Actions** — Structured action space across 5 groups: lighting, object placement, scene environment, and material properties — with anti-oscillation control and step-scale scheduling
- **Scene-Aware Rendering** — Objects placed in real Blender scenes with HDRI environments, raycast ground detection, and preserved scene lighting
- **Multi-Modal Output** — RGB, mask, depth, normal maps, and geometry metadata
- **End-to-End Automation** — From natural language seed concept to training-ready dataset, zero human intervention

---

## Pipeline Overview

```
Seed Concept ─→ T2I Generation ─→ Segmentation ─→ 3D Reconstruction ─→ Scene Rendering ─→ VLM Review Loop
  (Stage 1)       (Stage 2)       (Stage 2.5)        (Stage 3)           (Stage 4)          (Stage 5)
```

| Stage | What it does | Model / Tool |
|-------|-------------|-------------|
| **1. Text Expansion** | LLM expands seed concept into detailed T2I prompt | Claude API (Anthropic) |
| **2. T2I Generation** | Generate 1024&times;1024 object image | Qwen-Image-2512 |
| **2.5. Segmentation** | Extract RGBA foreground, remove background | SAM3 |
| **3. 3D Reconstruction** | Reconstruct textured mesh from single image | Hunyuan3D-2.1 |
| **4. Scene Rendering** | Blender Cycles 512spp scene-aware insertion | Blender 4.24 |
| **5. VLM Review Loop** | Free-form review &rarr; agent action &rarr; re-render until *keep* | Qwen3.5-35B-A3B |

### The VLM Review Loop (Stage 5)

The core innovation: a **goal-driven loop agent** that iteratively improves rendering quality.

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│   Blender   │────→│  VLM Review  │────→│  AI Agent        │────→│ Quality Gate │
│   Render    │     │  (free-form  │     │  Decision        │     │ keep/revise  │
│             │     │   critique)  │     │  (select action) │     │              │
└─────────────┘     └──────────────┘     └──────────────────┘     └──────┬───────┘
       ↑                                                                  │
       └──────────────── loop until reviewer says "keep" ─────────────────┘
```

**Anti-oscillation control** prevents parameter thrashing:
- Sign-flip tracking: freeze a parameter after 3 direction reversals
- Step-scale scheduling: Round 0 → 100%, Round 1 → 70%, Round 2 → 50%, Round 3+ → 40%
- Score-adaptive boost: &times;1.2 when hybrid_score < 0.65

---

## Prerequisites

- **OS**: Linux (tested on Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with &ge;24 GB VRAM for rendering; &ge;80 GB for VLM inference
- **Python**: 3.10+
- **Blender**: 4.24
- **CUDA**: Compatible with your PyTorch version

### Required Models

| Model | Purpose | Approx. Size |
|-------|---------|-------------|
| [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) | T2I generation | ~56 GB |
| [SAM3](https://github.com/pvc-tube/sam3) | Foreground segmentation | ~2 GB |
| [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | Image-to-3D reconstruction | ~20 GB |
| [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | VLM quality reviewer | ~35 GB |
| [Blender 4.24](https://www.blender.org/download/) | 3D rendering engine | ~300 MB |

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Kamisato520/DataEvolver.git
cd DataEvolver

# Configure model paths in each pipeline stage:
#   pipeline/stage1_text_expansion.py  → Anthropic API key
#   pipeline/stage2_t2i_generate.py    → MODEL_PATH (Qwen-Image-2512)
#   pipeline/stage2_5_sam2_segment.py  → SAM3_CKPT
#   pipeline/stage3_image_to_3d.py     → HUNYUAN3D_REPO, MODEL_HUB
#   pipeline/stage5_5_vlm_review.py    → Qwen3.5-35B model path
#   configs/scene_template.json        → blend_path, blender_binary

# Place your .blend scene file in assets/scene/
# Place HDRI environment maps in assets/hdri/

# Run the full pipeline
bash pipeline/run_all.sh
```

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
- Blender: `/path/to/blender` (4.24)
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
@misc{dataevolver2026,
  title   = {DataEvolver: Autonomous Synthetic Data Construction
             via VLM-Guided Iterative Rendering},
  year    = {2026},
  url     = {https://github.com/Kamisato520/DataEvolver}
}
```

## License

[MIT](LICENSE)
