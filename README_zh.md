<p align="center">
  <img src="assets/showcase.png" alt="DataEvolver Showcase" width="800"/>
</p>

<h1 align="center">DataEvolver</h1>

<p align="center">
  <strong>基于 VLM 引导迭代渲染的自主合成数据构建系统</strong><br/>
  通过 3D 渲染、VLM 审查和参数修复闭环，构建逼真、场景感知、训练就绪的合成数据。
</p>

<p align="center">
  <strong>Qisong Zhang, Wenzhuo Wu, Zhuangzhuang Jia, Yunhao Yang, Shuo Zhang</strong><br/>
  北京邮电大学人工智能学院
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

## 当前能力

DataEvolver 目前由两条相互连接的数据构建主线组成：

- **Agent 编排的多模态生成与 3D 重建** — 从用户需求出发，Agent 可以在 text-to-image、image edit、text-to-video 和 image-to-3D reconstruction 四类接口之间选择策略。这些接口可以并行、串行，也可以串行与并行混合使用。3D 路线当前采用串行流程：先生成物体图像，再重建带纹理 3D 资产，插入 Blender 场景，并通过 VLM 审查和有界渲染参数修复持续迭代；HYWorld / WorldMirror 场景生成是这条 pipeline 的可选补强，用于在物体插入前构建 contract-backed 场景上下文。
- **WebSearch 驱动的可复现研究 pipeline** — 针对某一篇论文或一类相关论文，Stage0 WebSearch 记录论文证据，提取文章的数据集构建方法，把官方开源数据集和评测标准作为 GT 参考，并产出 handoff，用于构建稳定可复现且高质量的数据集 pipeline。

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
模型下载、GPU 策略和配置写入计划，并在需要时生成本地非敏感配置文件。
它**不会**安装依赖、下载模型权重、写入 token、kill 服务，或启动 GPU 长任务。

### 1. 克隆仓库

```bash
git clone https://github.com/PRIS-CV/DataEvolver.git
cd DataEvolver
```

### 2. 安装包入口

```bash
python -m pip install -e .
```

只在确实要运行对应路线时安装可选依赖，例如 HYWorld / WorldMirror 使用
`python -m pip install -e ".[hyworld]"`。
`hyworld` extra 只覆盖可由 pip 管理的运行依赖；模型权重、源码 checkout、
Blender、CUDA/native extension 构建仍需在目标机器 preflight 后显式配置。

### 3. 运行安全的 onboarding dry-run

如果只是想最快了解当前环境是否可用，先使用 `quick` profile：

```bash
bash src/dataevolver/cli/bootstrap_dataevolver_default.sh \
  --profile quick \
  --dry-run \
  --write-local-config
```

如果想生成当前默认 DataEvolver pipeline 的完整部署计划，指定模型根目录：

```bash
bash src/dataevolver/cli/bootstrap_dataevolver_default.sh \
  --profile default \
  --model-root /path/to/dataevolver-models \
  --workspace-root "$PWD" \
  --python-backend auto \
  --dry-run \
  --write-local-config
```

如果是 HYWorld / WorldMirror 或论文验证实验，首次 dry-run 先显式记录
GPU 探测策略，便于后续 agent 按目标机器实际状态选择动态分片或保护服务卡：

```bash
bash src/dataevolver/cli/bootstrap_dataevolver_default.sh \
  --profile world_model \
  --model-root /path/to/dataevolver-models \
  --workspace-root "$PWD" \
  --gpu-policy inspect_only \
  --include-gpus all \
  --paper-validation \
  --dry-run \
  --write-local-config
```

可选的 Blender MCP operator 配置，用于 Codex 控制远端 Blender 调试：

```bash
bash src/dataevolver/cli/bootstrap_dataevolver_default.sh \
  --profile blender_mcp \
  --dry-run \
  --write-local-config
```

这会写入 `.dataevolver/local/blender_mcp.codex.toml`，并生成远端 Blender 预检/启动说明。Blender MCP 用于交互式场景诊断、viewport 截图、临时执行 Blender Python，以及快速调试灯光/相机/材质；正式数据集生成仍走 DataEvolver pipeline 脚本和已审查的 `BLENDER_BIN`/profile 配置。详见 [`docs/BLENDER_MCP_REMOTE.md`](docs/BLENDER_MCP_REMOTE.md)。

完成目标机器探测后，再按用户选择改用 `--gpu-policy dynamic_backfill` 或
`--gpu-policy fixed_range`。如果某张 GPU 已经在运行 vLLM、VLM review 或其他服务，
按该机器实际情况追加 `--reserve-gpus <user-reserved-gpu-spec>`。不要把具体机器的
GPU 范围、服务卡配置或策略选择提交到公共仓库；这些内容只应写入被忽略的
`.dataevolver/local/` profile。

脚本会输出五段内容：

| 段落 | 说明 |
|------|------|
| `preflight` | Linux、GPU、Python、`uv`/`conda`、Blender、Hugging Face CLI 可用性 |
| `gpu plan` | 面向后续 shard 的 inspect-only、dynamic backfill 或 fixed-range GPU 规划 |
| `env plan` | 优先 `uv` 的环境计划，以及 `conda` fallback |
| `model plan` | Hugging Face repo、目标路径、gated access 提示和打印出的下载命令 |
| `config plan` | pipeline 可读取的非敏感环境变量 |

可选 profile：

| Profile | 适用场景 |
|---------|----------|
| `quick` | 只做环境探测和 dry-run 路线 |
| `default` | 使用当前核心 pipeline：Qwen-Image-2512、SAM3、Hunyuan3D-2.1、DINOv2 Giant、Qwen3.5-35B-A3B 和 Blender |
| `full` | 在 `default` 基础上额外覆盖 Edit/T2V 默认计划：Qwen-Image-Edit-2511 和 Wan2.1-T2V |
| `world_model` | 使用可选 HYWorld / WorldMirror 场景重建计划 |
| `blender_mcp` | 使用可选的 Codex 远端 Blender MCP 调试 operator |
| `custom` | 你已有替代模型，希望先记录下来，后续再做兼容性检查 |

使用 `--write-local-config` 后，会生成：

- `.dataevolver/local/ENVIRONMENT.md`：人类和 agent 可读的环境摘要
- `.dataevolver/local/env.config.json`：结构化配置，便于 agent 和脚本读取
- `.dataevolver/local/env.sh.example`：可 source 的非敏感路径变量示例

`.dataevolver/local/` 已被 Git 忽略。不要把 Hugging Face、Anthropic、
OpenAI、SSH、cookie 或 API token 写入这些文件。

如果你在支持 skills 的 agent 环境中使用 DataEvolver，可以让 agent 使用
`dataevolver-onboarding` skill。它应该围绕五类基础信息进行访谈：目标路线、
运行位置、安装策略、模型策略、工作/输出路径。若选择 HYWorld 或论文验证路线，
还应记录 GPU 策略（`inspect_only`、`dynamic_backfill` 或 `fixed_range`）、
可用 GPU 范围和保留 GPU，然后运行上面的 dry-run 脚本。

### 4. Production Setup（生产部署）

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
  iopath timm ftfy moviepy==1.0.3 nerfview rtree \
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

#### 可选 HYWorld / WorldMirror 场景重建

只有当目标路线是 HYWorld / WorldMirror 场景重建时，才使用 `--profile world_model`。这会让世界模型配置保持在可选 route 中，不进入默认核心 pipeline：

```bash
bash src/dataevolver/cli/bootstrap_dataevolver_default.sh \
  --profile world_model \
  --model-root /path/to/dataevolver-models \
  --workspace-root "$PWD" \
  --python-backend auto \
  --dry-run \
  --write-local-config
```

world-model profile 会额外写入这些本地环境变量：

| 变量 | 指向 |
|------|------|
| `HYWORLD_SRC` | HY-World-2.0 源码 checkout |
| `HYWORLD_WEIGHTS` | HY-World-2.0 权重目录，需包含 `HY-WorldMirror-2.0` |
| `HYWORLD_PYTHON` | HYWorld 依赖使用的 Python 环境 |
| `HYWORLD_MOGE_MODEL_PATH` | 本地 MoGe 权重，用于真实 panorama depth |
| `HYWORLD_ZIM_MODEL_PATH` | 本地 ZIM 权重，用于天空 mask |
| `HYWORLD_GROUNDING_DINO_MODEL_PATH` | 本地 GroundingDINO 权重，用于 HYWorld 导航 |
| `HYWORLD_SAM3_MODEL_PATH` | HYWorld 使用的本地 SAM3 模型/源码路径 |
| `HYWORLD_WORLDSTEREO_PATH` | 本地 WorldStereo 权重根目录 |
| `HYWORLD_WAN_BASE_MODEL` | WorldStereo / 视频生成使用的本地 Wan I2V base Diffusers 模型 |

HYWorld 生产级场景生成必须有本地 MoGe、ZIM、GroundingDINO、SAM3、WorldStereo、Wan I2V base 和 HY-WorldMirror 权重。离线模式表示禁止下载，但仍要加载这些本地模型；真实 3D 场景生成时不要设置 `HYWORLD_NO_MODEL_DOWNLOADS=1`，否则会跳过 metric depth，容易产出固定半径 panorama shell。

`python -m dataevolver.workflows.hyworld.scene_pano` 和 `python -m dataevolver.workflows.hyworld.full_worldgen` 默认把变化的阶段产物保存到 `<scene-dir>/intermediates/<run-id>/`。HY-Pano 阶段会保存 360° 等距柱状全景图，完整 worldgen 会在 `00_source_panorama/` 再保存进入 3D 重建的原始输入，并保留 trajectory、depth/sky-mask、WorldStereo、GS 和 WorldMirror 输出。

##### 自动化场景构建

世界模型路线可以从场景 prompt 或输入场景图自动构建可 review 的场景包。流程会先生成 360° HY-Pano 上下文，再运行 HYWorld / WorldMirror 重建，将 metric geometry 转换为 Blender scene contract，渲染多视角纯场景证据，随后在固定 scene-camera 对齐下插入物体。最终 report 会记录 contract 检查、多视角场景图、物体旋转视图、mask、depth、normal、metadata gates 和 lineage hashes。

![HYWorld 自动化场景构建流程](assets/3D-build.png)

```bash
python -m dataevolver.cli.production doctor \
  --profile .dataevolver/local/production_profile.json \
  --no-imports

python -m dataevolver.workflows.hyworld.scene_pano \
  --scene-prompts-path <scene-prompts.json> \
  --output-root <hyworld-scene-root>

python -m dataevolver.workflows.hyworld.full_worldgen \
  --profile .dataevolver/local/production_profile.json \
  --scene-dir <hyworld-scene-root>/<scene-id> \
  --intermediate-root <intermediate-root>/<scene-id>

python -m dataevolver.workflows.hyworld.finalize_object_scene_report \
  --dataset-base <dataset-base> \
  --strict-scene-views
```

HYWorld / WorldMirror 使用 HY-Pano -> WorldNav -> WorldStereo -> WorldMirror/3DGS 的分阶段 world generation 路线。不要把 VLM pass 当作世界模型完成的权威证据；应以 contract 支撑的几何、多视角纯场景渲染和最终 manifest/lineage 作为验收依据。

确认 CUDA 和 `nvcc` 可用后，再编译 Hunyuan3D 原生扩展：

```bash
uv pip install --no-build-isolation -e "$HUNYUAN3D_REPO/hy3dpaint/custom_rasterizer"
cd "$HUNYUAN3D_REPO/hy3dpaint/DifferentiableRenderer"
bash compile_mesh_painter.sh
```

`cupy-cuda12x`、`flash-attn`、`pytorch3d`、`fused_ssim` 这类 HYWorld
vendor CUDA/source-build 包，只应在目标 HYWorld Python 环境通过 CUDA、
`nvcc`、编译器和 PyTorch ABI preflight 后安装。

### 5. 生产 Smoke Tests

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

python -m dataevolver.workflows.stages.t2i_generate --ids obj_001 --steps 1 --height 512 --width 512 --device cuda:0
python -m dataevolver.workflows.stages.sam_segment --ids obj_001 --device cuda:0
python -m dataevolver.workflows.stages.image_to_3d --ids obj_001 --shape-only --device cuda:0 --output-dir "$SMOKE_ROOT/meshes_shape_only"

python -m dataevolver.workflows.stages.image_to_3d \
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

以上检查通过后，再准备场景资产并运行 `bash src/dataevolver/cli/run_all.sh`。场景感知
review 与动作应用使用 `python -m dataevolver.annotation.vlm_review_stage`
和 `python -m dataevolver.agents.feedback_apply`，并要求模型路径、
`BLENDER_BIN` 和 `configs/scene_template.json` 已配置完成。

---

## 可复现 Research Prior 工作流

公共仓库保留轻量 Stage0 WebSearch prior 入口和 universal 3D layout schema，
但不发布旧的本地 Blender 批处理脚本、私有场景池、生成网格、`.blend` 场景文件、
模型权重、API key 或论文源码。

Stage0 是 agent-run research harness：当前 agent 负责 WebSearch 和论文阅读，
DataEvolver 负责记录 evidence、trace event、handoff 文档和 `research_prior.json`。
该 prior 只作为下游生成的软指导；prior 缺失或损坏时应 warning 后继续，而不是阻断 pipeline。

核心入口：

- Stage0 WebSearch prior：[`python -m dataevolver.tools.stage0_web_research`](python -m dataevolver.tools.stage0_web_research)
- Universal dataset schema：[`configs/schemas/universal_3d_layout_dataset_schema.json`](configs/schemas/universal_3d_layout_dataset_schema.json)

### 单论文复现模式

当用户提供一篇论文并希望形成可复现的数据集构建 handoff 时，使用
`--dataset-mode single-paper`。例如 SeeThrough3D session 应先区分官方资源和
自生成 proxy，记录论文中的数据集构建方法和评测信号，再进入大规模执行。

```bash
python -m dataevolver.tools.stage0_web_research init \
  --query "SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation" \
  --output-dir .dataevolver/runtime/research_priors/demo_seethrough3d \
  --dataset-mode single-paper \
  --tag seethrough3d-reproduction
```

### 多论文通用模式

当用户提供 topic 或 idea 时，使用 `--dataset-mode universal`。Agent 应将搜索范围
收敛到 3-5 篇高度相关论文，提取它们的数据集构建方法、质量门和失败模式，并将
可复用需求合成为 universal 3D layout contract。

```bash
python -m dataevolver.tools.stage0_web_research init \
  --query "occlusion-aware 3D control datasets for text-to-image generation" \
  --output-dir .dataevolver/runtime/research_priors/demo_universal \
  --dataset-mode universal \
  --tag universal-3d-layout
```

---

## 项目结构

```
DataEvolver/
├── src/dataevolver/
│   ├── agents/                              # feedback / action 应用
│   ├── annotation/                          # VLM 与约束审查
│   ├── cli/                                 # 公共 onboarding 与 runtime 入口
│   ├── dataset/                             # 数据集加载与 metadata 导出
│   ├── runtime/                             # runtime profile、资产生命周期、HYWorld contract
│   ├── testing/                             # 公共 smoke 与契约测试
│   ├── tools/                               # research prior 与诊断工具
│   └── workflows/
│       ├── hyworld/                         # 可选 HYWorld 桥接模块
│       ├── multimodal/                      # T2I/Edit/T2V 数据集 workflow
│       └── stages/                          # 核心 stage 模块
├── configs/
│   ├── action_space.json                    # 核心 action-space 契约
│   ├── scene_action_space.json              # 24 种原子操作定义
│   ├── scene_template.json                  # Blender 场景模板配置
│   ├── production_profile.example.json      # 示例生产 profile
│   ├── vlm_review_schema.json               # VLM 审查输出结构
│   └── dataset_profiles/                    # 公共示例配置模板
├── .agents/                                 # Agent skills 与配置说明
├── .github/                                 # GitHub workflow 与仓库元信息
├── assets/
│   ├── hdri/                                # HDRI 环境贴图
│   └── scene/                               # Blender 场景文件 (.blend)
└── web/                                     # 项目展示页面 (GitHub Pages)
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

### 已完成

- [x] **集成 WebSearch 的 AI Agent** — Stage0 WebSearch 已支持单论文复现和多论文 universal dataset 两种模式，能记录论文证据、导出 handoff 文档，并接入后续 `research_prior.json` 软指导。
- [x] **多论文 universal 3D 数据集综合** — 工作流可从 anchor paper 扩展到相关 3D-control 论文，抽取 target image、mask、结构视图、相机 metadata、物体布局、遮挡关系、scene graph 和 validation trace 等共享需求。
- [x] **Universal 3D layout dataset contract** — 当前 schema 覆盖 Blender-backed target renders、per-object masks、structure views、depth-order / orientation proxies、camera pose / intrinsics、mesh metadata、3D boxes、scene graphs、空间关系和 promotion decisions。
- [x] **Blender-backed 双物体场景** — 当前 universal 3D workflow 可基于 DataEvolver 场景和 mesh 资产生成真实场景双物体记录；通用 N-object 组合场景仍是后续扩展。
- [x] **VLM/CV 自进化闭环** — 部分 universal 3D records 可通过 calibrated hybrid VLM/CV geometry scores、有界修复动作和显式 accept/reject promotion trace 进行审查。
- [x] **论文特定与 universal export 模式** — WebSearch 和 universal layout 脚本支持 single-paper dataset reproduction 与 multi-paper universal schema export。

### 进行中

- [ ] **VLM 结构化知识库** — 构建先验知识存储，为 VLM 质量评估提供客观依据，减少在判断数据质量、任务合理性和评估方向时的主观偏差
- [ ] **真实世界数据集 ingestion** — WebSearch 已可发现公开数据集和论文资源，SeeThrough3D 等公开产物也可作为 reference；任意真实数据集的 crawl-clean-process-ingest 全流程仍待实现。
- [ ] **自动化评测集生成** — 下一步重点是把 validation log、hybrid VLM/CV score、promotion trace 和 research prior 转成可复用 evaluation set，包含任务规格、answer key、object / scene split、provenance 检查和 regression gates。
- [ ] **加快 3D 重建闭环** — 优化 image-to-3D 和 mesh-to-scene 迭代耗时，通过 profile 化 dry-run、缓存、批量渲染和轻量 review pass，减少进入重型 VLM 检查前的等待。
- [ ] **3D 重建模型 LoRA 微调** — 尝试用已接受的 reconstruction trace 和失败样本进行参数高效微调，提升几何保真、纹理一致性，以及插入 Blender 场景后的可用性。

### 待扩展

- [ ] **视频物体数据集** — 从图像数据集扩展到视频级别物体编辑，支持跨帧的物体消除、移动和属性编辑
- [ ] **Temporal review and filtering** — 增加序列级 flicker、轨迹平滑、mask 稳定性、depth 连续性和 action consistency 检查。
- [ ] **通用 N-object 组合场景** — 在当前双物体 workflow 之外，还需要 relation-aware sampling、collision handling、occlusion planning 和更强的多物体 VLM/CV review。
- [ ] **大规模公共数据集 ingestion** — 将 WebSearch 发现的数据源扩展为可复用的 crawling、cleaning、licensing、provenance 和 structured export 流程。

## 许可证

[Apache-2.0](LICENSE)
