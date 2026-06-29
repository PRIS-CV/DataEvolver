<p align="center">
  <img src="assets/showcase.png" alt="DataEvolver Showcase" width="800"/>
</p>

<h1 align="center">DataEvolver</h1>

<p align="center">
  <strong>基于 VLM 引导迭代渲染的自主合成数据构建系统</strong><br/>
  通过 3D 渲染、VLM 审查和参数修复闭环，构建逼真、场景感知、训练就绪的合成数据。
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.01789"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white" alt="Paper: arXiv"/></a>
  <a href="https://pris-cv.github.io/DataEvolver/"><img src="https://img.shields.io/badge/Website-Project%20Page-1f6feb?logo=githubpages&logoColor=white" alt="Project website"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="License: Apache-2.0"/></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/Blender-4.2%2B-f5792a?logo=blender&logoColor=white" alt="Blender 4.2+"/>
</p>

<p align="center">
  <a href="README.md">🇬🇧 English</a> •
  <a href="https://pris-cv.github.io/DataEvolver/">🌐 Website</a> •
  <a href="https://arxiv.org/abs/2605.01789">📄 Paper</a> •
  <a href="#荣誉展示">🏆 荣誉展示</a> •
  <a href="#dataevolver-rotate">🧩 数据集: DataEvolver-Rotate</a>
</p>

<p align="center">
  <strong>🧠 VLM 自由反馈</strong> •
  <strong>🎬 3D 渲染-审查-修复闭环</strong> •
  <strong>📦 训练就绪多模态数据</strong>
</p>

---

## 为什么是 DataEvolver？

DataEvolver 将合成数据构建转化为一个 **目标驱动的优化闭环**：VLM 用自然语言审查渲染结果，Agent 诊断具体问题，并通过有界参数更新重新渲染，直到样本达到可保留质量。

- **超越固定评分规则** — 自由形式 VLM 反馈能发现规则难以覆盖的场景关系、物体摆放、光照和材质问题。
- **用结构化动作修复问题** — 24 种有界原子操作覆盖光照、物体位姿、场景环境和材质外观，减少无约束漂移。
- **直接导出训练数据** — 输出 RGB、mask、depth、normal、几何元数据和物体隔离划分，便于下游模型训练。

## 核心特性

- **目标驱动闭环智能体** — VLM 审查器提供语义反馈（"光照平淡"、"物体悬浮"）&rarr; AI Agent 选择目标行为 &rarr; 重新渲染 &rarr; 重复直至达到质量标准
- **24 种原子操作** — 覆盖光照、物体位姿、场景环境和材质属性 5 大类别的结构化操作空间，配备抗振荡控制和步长衰减调度
- **场景感知渲染** — 物体置于真实 Blender 场景中，支持 HDRI 环境光、射线检测地面定位，保留原场景光照
- **多模态输出** — 同步输出 RGB、mask、depth、normal 以及几何元数据
- **端到端自动化** — 从自然语言种子概念到训练就绪数据集，全过程零人工干预

---

## 流水线概览

![Pipeline Overview](assets/pipeline_show.svg)

| 阶段 | 功能 | 模型 / 工具 |
|------|------|-------------|
| **1. 文本扩展** | LLM 将种子概念扩展为详细 T2I prompt | Claude API (Anthropic) |
| **2. T2I 生成** | 生成 1024&times;1024 物体图像 | Qwen-Image-2512 |
| **2.5. 前景分割** | 提取 RGBA 前景，移除背景 | SAM3 |
| **3. 3D 重建** | 单图重建带纹理网格模型 | Hunyuan3D-2.1 |
| **4. 场景渲染** | 场景感知 Blender 插入，支持可配置 EEVEE/Cycles 渲染 | Blender 4.2+ |
| **5. VLM 审查闭环** | 自由形式评审 &rarr; Agent 选择操作 &rarr; 重新渲染直至 *keep* | Qwen3.5-35B-A3B |

### VLM 审查闭环（Stage 5）

核心创新：**目标驱动的闭环智能体**迭代优化渲染质量。

![VLM Review Loop](assets/stage5.svg)

**抗振荡控制**防止参数反复震荡：
- 回摆追踪：同一参数方向反转 3 次后冻结
- 步长衰减调度：Round 0 → 100%，Round 1 → 70%，Round 2 → 50%，Round 3+ → 40%
- 分数自适应增强：hybrid_score < 0.65 时步长 &times;1.2

---

## 环境要求

- **操作系统**：Linux（已在 Ubuntu 20.04+ 测试）
- **GPU**：NVIDIA GPU，渲染 ≥24 GB 显存；VLM 推理 ≥80 GB 显存
- **Python**：3.10+
- **Blender**：4.2+
- **CUDA**：与 PyTorch 版本兼容

### 所需模型

| 模型 | 用途 | 大小 |
|------|------|------|
| [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) | T2I 图像生成 | ~56 GB |
| [SAM3](https://github.com/facebookresearch/sam3) | 前景分割 | ~2 GB |
| [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | 图像转 3D 重建 | ~20 GB |
| [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | VLM 质量审查 | ~35 GB |
| [Blender 4.2+](https://www.blender.org/download/) | 3D 渲染引擎 | ~300 MB |

---

## 快速开始

建议先从轻量 onboarding dry-run 开始。它会检查你的环境形态，打印安装、
模型下载和配置写入计划，并在需要时生成本地非敏感配置文件。它**不会**
安装依赖、下载模型权重、写入 token，或启动 GPU 长任务。

### 1. 克隆仓库

```bash
git clone https://github.com/PRIS-CV/DataEvolver.git
cd DataEvolver
```

### 2. 运行安全的 onboarding dry-run

如果只是想最快了解当前环境是否可用，先使用 `quick` profile：

```bash
bash scripts/bootstrap_dataevolver_default.sh \
  --profile quick \
  --dry-run \
  --write-local-config
```

如果想生成当前默认 DataEvolver pipeline 的完整部署计划，指定模型根目录：

```bash
bash scripts/bootstrap_dataevolver_default.sh \
  --profile default \
  --model-root /path/to/dataevolver-models \
  --workspace-root "$PWD" \
  --python-backend auto \
  --dry-run \
  --write-local-config
```

脚本会输出四段内容：

| 段落 | 说明 |
|------|------|
| `preflight` | Linux、GPU、Python、`uv`/`conda`、Blender、Hugging Face CLI 可用性 |
| `env plan` | 优先 `uv` 的环境计划，以及 `conda` fallback |
| `model plan` | Hugging Face repo、目标路径、gated access 提示和打印出的下载命令 |
| `config plan` | pipeline 可读取的非敏感环境变量 |

可选 profile：

| Profile | 适用场景 |
|---------|----------|
| `quick` | 只做环境探测和 dry-run 路线 |
| `default` | 使用当前核心 pipeline：Qwen-Image-2512、SAM3、Hunyuan3D-2.1、DINOv2 Giant、Qwen3.5-35B-A3B 和 Blender |
| `full` | 在 `default` 基础上额外覆盖 Edit/T2V 默认计划：Qwen-Image-Edit-2511 和 Wan2.1-T2V |
| `custom` | 你已有替代模型，希望先记录下来，后续再做兼容性检查 |

使用 `--write-local-config` 后，会生成：

- `.dataevolver/local/ENVIRONMENT.md`：人类和 agent 可读的环境摘要
- `.dataevolver/local/env.config.json`：结构化配置，便于 agent 和脚本读取
- `.dataevolver/local/env.sh.example`：可 source 的非敏感路径变量示例

`.dataevolver/local/` 已被 Git 忽略。不要把 Hugging Face、Anthropic、
OpenAI、SSH、cookie 或 API token 写入这些文件。

如果你在支持 skills 的 agent 环境中使用 DataEvolver，可以让 agent 使用
`dataevolver-onboarding` skill。它应该只围绕五类信息进行访谈：目标路线、
运行位置、安装策略、模型策略、工作/输出路径，然后运行上面的 dry-run 脚本。

### 3. Production Setup（生产部署）

创建运行环境。在共享 GPU 服务器上，优先使用
`--system-site-packages` 复用已经验证过的 NVIDIA PyTorch。只有当 venv 内
`import torch` 失败或没有可用 CUDA 时，再安装 CUDA wheel。

```bash
uv venv .venv --python 3.10 --system-site-packages
source .venv/bin/activate

python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available())
PY

# 仅当上面的 torch 检查失败时使用该 fallback。
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

uv pip install \
  "numpy<2" "tokenizers==0.22.1" \
  diffsynth transformers accelerate diffusers safetensors \
  pillow opencv-python scipy scikit-image imageio trimesh \
  rembg[gpu] anthropic qwen-vl-utils lpips basicsr realesrgan \
  iopath timm ftfy \
  opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

检查 `.dataevolver/local/env.sh.example`，复制为机器本地配置文件，并在每次
真实运行前 source：

```bash
cp .dataevolver/local/env.sh.example .dataevolver/local/env.remote.sh
# 只编辑路径，不要把 token 写入这个文件。
source .dataevolver/local/env.remote.sh
source .venv/bin/activate
```

生产部署应使用环境变量覆盖路径，而不是直接修改 pipeline 脚本：

| 变量 | 指向 |
|------|------|
| `DATAEVOLVER_WORKSPACE_ROOT` | DataEvolver 代码目录 |
| `QWEN_IMAGE_MODEL_PATH` | Qwen-Image-2512 权重 |
| `QWEN_IMAGE_EDIT_MODEL_PATH` | Qwen-Image-Edit-2511 权重，编辑路线可选 |
| `SAM3_CKPT` | `sam3.pt` checkpoint |
| `SAM3_DIR` | SAM3 源码目录或可 import 的 package 路径 |
| `HUNYUAN3D_REPO` | Tencent-Hunyuan/Hunyuan3D-2.1 源码 checkout |
| `MODEL_HUB` | Hunyuan3D-2.1 权重目录 |
| `PAINT_MODEL_HUB` | Hunyuan3D paint 权重目录，通常与 `MODEL_HUB` 相同 |
| `DINO_MODEL_PATH` | DINOv2 Giant checkpoint 目录 |
| `REALESRGAN_CKPT` | Hunyuan paint 使用的 `RealESRGAN_x4plus.pth` |
| `VLM_MODEL_PATH` | Qwen3.5-35B-A3B，或验证兼容的 Qwen3.x 替代模型 |
| `BLENDER_BIN` | Blender 可执行文件 |

源码目录和权重目录必须分开。`SAM3_CKPT` 是 checkpoint 文件，
`SAM3_DIR` 是 SAM3 代码路径。`HUNYUAN3D_REPO` 是 Hunyuan3D 源码
checkout，`MODEL_HUB` 和 `PAINT_MODEL_HUB` 是权重目录。

确认 CUDA 和 `nvcc` 可用后，再编译 Hunyuan3D 原生扩展：

```bash
uv pip install --no-build-isolation -e "$HUNYUAN3D_REPO/hy3dpaint/custom_rasterizer"
cd "$HUNYUAN3D_REPO/hy3dpaint/DifferentiableRenderer"
bash compile_mesh_painter.sh
```

### 4. 生产 Smoke Tests

先运行 preflight 和 import 检查：

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

然后用一个物体验证运行链路：

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

`--shape-only` 只是在 DINO、RealESRGAN 或 paint 扩展缺失时的 fallback，
不等于完整 textured production readiness。启动完整 review loop 前，还需
完成 Stage 5.5 VLM loader smoke。

部署证据整理完成后，应删除生成的 smoke 产物，保持工作目录干净：

```bash
rm -rf .dataevolver/local/smoke
```

以上检查通过后，再准备场景资产并运行 `bash pipeline/run_all.sh`。场景感知
VLM 优化闭环应在模型路径、`BLENDER_BIN` 和
`configs/scene_template.json` 配置完成后，通过
`scripts/run_scene_agent_monitor.py` 单独启动。

---

## 项目结构

```
DataEvolver/
├── pipeline/                          # 核心流水线阶段
│   ├── stage1_text_expansion.py             # LLM prompt 生成
│   ├── stage2_t2i_generate.py               # 文生图 (Qwen-Image-2512)
│   ├── stage2_5_sam2_segment.py             # SAM3 前景提取
│   ├── stage3_image_to_3d.py                # 3D 网格重建 (Hunyuan3D-2.1)
│   ├── stage4_scene_render.py               # Blender 场景感知渲染
│   ├── stage5_5_vlm_review.py               # VLM 质量审查 (Qwen3.5-35B-A3B)
│   ├── stage5_6_feedback_apply.py           # 操作选择与抗振荡控制
│   ├── asset_lifecycle.py                   # 资产生命周期管理
│   └── rotation_geomodal_dataset.py         # 训练数据集加载器
├── configs/
│   ├── scene_action_space.json              # 24 种原子操作定义
│   ├── scene_template.json                  # Blender 场景模板配置
│   ├── vlm_review_schema.json               # VLM 审查输出结构
│   ├── dataset_profiles/                    # 数据集配置模板
│   └── seed_concepts/                       # 种子物体定义 (20/50 物体)
├── scripts/                           # 工具与构建脚本
│   ├── run_scene_agent_monitor.py           # VLM 闭环 Agent 监控
│   ├── run_scene_agent_step.py              # 单步 Agent 执行
│   ├── export_rotation8_from_best_object_state.py  # 一致性旋转导出
│   ├── build_rotation8_trainready_dataset.py       # 训练对构建
│   ├── build_object_split_for_rotation_dataset.py  # 物体隔离划分
│   ├── run_full_pipeline.py                 # 完整流水线编排
│   ├── run_vlm_quality_gate_loop.py         # VLM 质量门控闭环
│   ├── feedback_loop/                       # 反馈闭环工具集
│   └── ...                                  # 其他构建与评测脚本
├── assets/
│   ├── hdri/                                # HDRI 环境贴图
│   └── scene/                               # Blender 场景文件 (.blend)
├── paper/                             # 技术报告 (LaTeX 源码)
└── web/                               # 项目展示页面 (GitHub Pages)
```

---

## 操作空间

AI Agent 从 5 大类 **24 种结构化原子操作**中选择：

| 类别 | 操作数 | 参数说明 |
|------|--------|----------|
| **光照** (4) | 主光强度 &uarr;&darr;，主光方位 &plusmn;15&deg; | 乘性 &times;1.2/&times;0.8 或加性操作，有界约束 |
| **物体** (6) | 高度 &plusmn;0.02，方位 &plusmn;15&deg;，缩放 &times;1.1/&times;0.9 | 安全范围内约束 |
| **场景** (5) | 环境旋转 &plusmn;30&deg;，环境强度 &uarr;&darr;，接触阴影 | HDRI 与环境光照控制 |
| **材质** (9) | 饱和度、明度、色相偏移、粗糙度、高光/光泽 | 细粒度材质微调 |
| **相机** (0) | 预留 | — |

完整操作定义：[`configs/scene_action_space.json`](configs/scene_action_space.json)

---

## DataEvolver-Rotate

DataEvolver 产出的首个基准数据集 — 面向旋转条件图像编辑。

| 指标 | 数值 |
|------|------|
| 物体数量 | 50 |
| 旋转角度 | 8 (0&deg;, 45&deg;, 90&deg;, 135&deg;, 180&deg;, 225&deg;, 270&deg;, 315&deg;) |
| 训练对数量 | 350 (正面 → 7 个目标视角) |
| 训练 / 验证 / 测试 | 245 / 49 / 56 对 (物体隔离，seed=42) |
| 模态 | RGB、mask、depth、normal |

### 构建方法

每个物体基于唯一的 **canonical yaw-0&deg; best state** 作为基准，旋转过程中场景、相机、光照和材质保持固定，确保跨角度一致性。

### 加载数据集

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
instruction = row["instruction"]  # 例如："将物体顺时针旋转 45 度"
```

---

## 与 Claude Code 集成

DataEvolver 设计为与 [Claude Code](https://docs.anthropic.com/en/docs/claude-code) 协同工作，作为 AI 驱动的开发与运维助手。Claude Code 读取项目级 `CLAUDE.md` 文件理解当前环境，然后通过自然语言帮助你运行流水线、分析结果和构建数据集。

### 第一步：安装 Claude Code

```bash
npm install -g @anthropic-ai/claude-code
```

### 第二步：创建你的 `CLAUDE.md`

在项目根目录创建 `CLAUDE.md`（已加入 .gitignore，每位用户独立维护）：

```markdown
# CLAUDE.md

## Remote Server
- SSH 别名: `my-server`
- GPU: 3x A800 80GB (或你的配置)
- Python: `/path/to/python3` (3.10+, 含 PyTorch)
- Blender: `/path/to/blender` (4.2+)
- 代码目录: `/path/to/DataEvolver`

## Model Paths
- Qwen-Image-2512: `/path/to/Qwen-Image-2512`
- SAM3 checkpoint: `/path/to/sam3/sam3.pt`
- Hunyuan3D-2.1 repo: `/path/to/Hunyuan3D-2.1`
- Hunyuan3D-2.1 weights: `/path/to/model_hub/Hunyuan3D-2.1`
- Qwen3.5-35B-A3B: `/path/to/Qwen3.5-35B-A3B`

## Scene Config
- Blender 场景文件: `/path/to/scene.blend`
- HDRI 目录: `/path/to/hdri/`
- 渲染引擎: CYCLES, 512 samples, 1024x1024

## Key Configs
- 操作空间: `configs/scene_action_space.json` (24 种原子操作)
- 场景模板: `configs/scene_template.json`
- 数据集模板: `configs/dataset_profiles/`

## Pipeline
Stage 1 (文本扩展) → Stage 2 (T2I) → Stage 2.5 (SAM3)
→ Stage 3 (3D 重建) → Stage 4 (Blender 渲染) → Stage 5 (VLM 闭环)

## Working Rules
- 始终在远程服务器上测试，而非本地
- 长时间任务使用 tmux
- VLM 结果先看 trace.json 自由文本，而非仅看 agg.json 分数
- 仅当审查器明确给出 "keep" 时停止 VLM 闭环
- 启动新任务前检查 GPU 使用情况
```

### 第三步：创建 Claude Code Skills（可选）

```bash
mkdir -p skills/scene-agent-loop
```

```markdown
# skills/scene-agent-loop/SKILL.md
---
name: scene-agent-loop
description: 管理场景渲染的 VLM 审查闭环
---
# Scene Agent Loop
监控并推进 VLM 审查 → 渲染 → Agent 决策闭环。
读取 trace.json 了解当前状态，然后决定下一步操作。
```

### 第四步：启动 Claude Code

```bash
cd DataEvolver
claude
```

Claude Code 读取 `CLAUDE.md` 后将理解完整项目上下文。示例指令：

```
> 检查服务器 GPU 使用情况
> 为 10 个新家具物体运行完整流水线
> 从最新 best state 导出 rotation8 数据集
> 构建物体隔离的训练就绪数据集
```

---

## 引用

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

## 荣誉展示

- 🏆 **[AIDataSci 2026, KDD 2026 Workshop](https://usail-hkust.github.io/aidatasci/)** — DataEvolver 被接收为 **Oral Presentation**。[查看 OpenReview 页面](https://openreview.net/forum?id=8ppusILCIH)。
- 🎖️ **2026 智源大会「Agent for Science」大赛** — DataEvolver 获得 **Poster Presentation** 证书，并取得线下海报展示资格。

<p align="center">
  <img src="assets/baai_agent4s_poster_certificate.jpg" alt="DataEvolver 智源大会 Agent for Science 海报展示证书" width="720"/>
</p>

## 路线图

### 内部提升 — 增强感知、理解与决策能力

- [ ] **VLM 结构化知识库** — 构建先验知识存储，为 VLM 质量评估提供客观依据，减少在判断数据质量、任务合理性和评估方向时的主观偏差
- [ ] **集成 WebSearch 的 AI Agent** — 在 Agent 接收用户请求后，主动检索相关数据集、论文和已有方法，理解当前任务的数据构建思路，并结合本地深度研究工具建立知识体系，避免模型在缺乏背景信息时完全自由发挥

### 外部扩展 — 丰富数据集类型与构建方式

- [ ] **多物体场景** — 扩展渲染系统支持单场景多物体，在单次渲染中实现旋转、插入、消除、移动等复合几何编辑操作
- [ ] **视频物体数据集** — 从图像数据集扩展到视频级别物体编辑，支持跨帧的物体消除、移动和属性编辑
- [ ] **自动生成推理评测数据集** — 根据用户训练需求自动生成评测基准，实现训练信息对齐、定制化渲染和自动化评测流程
- [ ] **真实世界数据集采集** — 通过 WebSearch 发现真实世界数据源，完成原始数据爬取、清洗、处理与结构化构建，扩展数据源覆盖和应用场景

## 许可证

[Apache-2.0](LICENSE)
