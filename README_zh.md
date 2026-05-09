# DataEvolver

**基于 VLM 引导迭代渲染的自主合成数据构建系统**

[English](README.md) &middot; [Website](https://pris-cv.github.io/DataEvolver/) &middot; [Paper](https://arxiv.org/abs/2605.01789) &middot; [数据集: DataEvolver-Rotate](#dataevolver-rotate)

DataEvolver 是一个目标驱动的数据合成流水线，通过 3D 渲染、VLM（视觉-语言模型）质量审查和智能参数调整的自动化闭环，生成高质量训练数据集。与传统基于固定评分规则的流水线不同，DataEvolver 利用 VLM 的自由形式反馈来感知、诊断和修复渲染问题，从而在无需人工干预的情况下产出逼真、场景感知的训练数据。

---

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
| **4. 场景渲染** | Blender Cycles 512spp 场景感知渲染 | Blender 4.24 |
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
- **Blender**：4.24
- **CUDA**：与 PyTorch 版本兼容

### 所需模型

| 模型 | 用途 | 大小 |
|------|------|------|
| [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) | T2I 图像生成 | ~56 GB |
| [SAM3](https://github.com/pvc-tube/sam3) | 前景分割 | ~2 GB |
| [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | 图像转 3D 重建 | ~20 GB |
| [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | VLM 质量审查 | ~35 GB |
| [Blender 4.24](https://www.blender.org/download/) | 3D 渲染引擎 | ~300 MB |

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/PRIS-CV/DataEvolver.git
cd DataEvolver

# 在各阶段脚本中配置模型路径：
#   pipeline/stage1_text_expansion.py  → Anthropic API key
#   pipeline/stage2_t2i_generate.py    → MODEL_PATH (Qwen-Image-2512)
#   pipeline/stage2_5_sam2_segment.py  → SAM3_CKPT
#   pipeline/stage3_image_to_3d.py     → HUNYUAN3D_REPO, MODEL_HUB
#   pipeline/stage5_5_vlm_review.py    → Qwen3.5-35B 模型路径
#   configs/scene_template.json        → blend_path, blender_binary

# 将 .blend 场景文件放入 assets/scene/
# 将 HDRI 环境贴图放入 assets/hdri/

# 运行完整流水线
bash pipeline/run_all.sh
```

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
- Blender: `/path/to/blender` (4.24)
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
