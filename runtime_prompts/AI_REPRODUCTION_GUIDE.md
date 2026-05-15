# AI 论文复现提示词工作流

本文档用于指导一个“无上下文记忆”的 AI 代码代理，从论文链接、PDF 或少量用户说明出发，尽可能完整地复现一篇论文。它来自 SeeThrough3D 与 GEditBench v2 两次实际复现经验，目标不是只写总结，而是让代理产出可运行代码、可检查数据、可视化证据和明确的复现等级判断。

核心原则：

```text
论文链接优先，PDF 次之。
先拆解论文方法线，再写代码。
官方最终产物只能用于验证，不能直接算作方法复现。
先跑通小规模闭环，再考虑扩大规模。
所有产物必须标注来源和复现等级。
大文件和模型权重必须遵守用户指定的数据目录和下载限制。
```

## 1. 初始输入契约

AI 代理首先确认用户给出的输入类型，优先级如下：

1. arXiv URL。
2. 本地 PDF。
3. 用户提供的论文文本、截图或说明。
4. 官方 GitHub、Hugging Face、项目主页、服务器路径、已有模型路径。

代理必须完成：

- 解析论文身份，例如 arXiv ID；
- 下载或复用 PDF；
- 抽取可搜索的 `.txt`；
- 找到官方公开资源；
- 记录来源 URL、本地路径、服务器路径。

如果用户提供本地 PDF，仍应尝试确认 arXiv 链接，作为论文身份标识。

## 2. 方法线拆解是强制步骤

在写任何复现代码之前，必须创建：

```text
docs/METHOD_CONSTRAINT_CARD.md
```

该文件必须把论文拆成若干条“方法线”。一篇论文经常不止一条线，例如 GEditBench v2 就至少包含：

```text
benchmark construction
candidate generation
preference pair synthesis
judge training
judge inference
leaderboard recomputation
visualization/reporting
```

每条方法线必须记录：

- 论文原始目标；
- 所需输入；
- 所需输出；
- 中间 schema；
- 依赖的模型、数据、人工标注或专家过滤；
- 官方已释放的文件；
- 本次可自生成的文件；
- 复用了哪些官方产物；
- 跳过原因；
- 最小可运行复现方案。

不要把“找到官方结果文件并重新统计”误写成“复现了数据构建方法”。

## 3. 产物来源标签

每个主要文件、目录、表格、图像、指标都必须标注来源。

使用以下标签：

| 标签 | 含义 |
|---|---|
| `self_generated` | 由本次复现管线从上游输入生成 |
| `official_downloaded` | 从作者、官方仓库或官方镜像下载 |
| `official_result_recomputed` | 基于官方中间结果重新计算 |
| `mock_or_proxy` | 用简化替代物测试结构、接口或流程 |
| `synthetic_proxy` | 用生图、3D 渲染或合成资产替代真实资产 |
| `not_completed` | 论文要求但尚未复现 |

报告中必须明确说明：

```text
官方下载产物 ≠ 方法复现
官方评测结果重算 ≠ 数据构建复现
synthetic proxy ≠ 真实论文数据来源
```

## 4. 复现等级

最终必须给出复现等级：

```text
Level 0: 解析论文，发现官方资源。
Level 1: 官方产物加载、验证、统计或重新计算。
Level 2: 小规模方法复现，有自生成中间产物。
Level 3: 中规模复现，使用接近论文的真实模型和真实数据。
Level 4: 完整论文级复现，规模、训练/推理、评测协议和主要结果可比。
```

每个等级都必须列出证据文件。

推荐最终表述：

```text
official-artifact verification only
evaluation-chain smoke reproduction
small-scale method reproduction
real-small method reproduction
synthetic-proxy construction branch
partial paper-level reproduction
full paper-level reproduction
```

## 5. 下载策略和大文件策略

默认不要下载全量大数据。

优先顺序：

1. 论文、dataset card、API 元数据。
2. 官方仓库文件列表。
3. 小样本数据。
4. 单个 shard 或少量图片。
5. 只有用户明确同意时才下载全量数据。

每次下载必须记录：

```text
source
file_count
bytes
local_path
server_path
download_reason
whether_required
artifact_label
```

如果用户规定大文件目录，例如：

```text
<large-file-root>/<project>/
```

则超过 1G 的数据、生成图片、候选图、模型输出、缓存都必须放到该目录。代码应放在用户指定的代码目录，不要把模型权重复制进项目。

如果用户要求“不下载新模型”，只能引用已有模型路径，并在环境检查报告中记录未找到的模型。

如果遇到以下任一情况，AI 必须暂停并请求用户确认，不能自行继续：

- 需要用户授权的大下载或全量数据下载；
- 需要使用闭源模型、付费 API 或需要额外账号/密钥的服务；
- 预计会超过用户指定的算力、存储或费用预算；
- 单个训练、推理、渲染或评测任务预计运行超过用户指定的 N 小时；
- 需要改变已有项目、已有 conda 环境或服务器共享资源。

## 6. 环境与服务器检查

进入工作区后先检查目录，而不是假设已有代码。

必须记录：

- 本地项目根目录；
- 服务器代码根目录；
- 服务器数据根目录；
- conda 环境；
- 可用模型；
- 可用数据集；
- GPU 情况；
- 网络和下载限制。

如果需要从已有项目借鉴，例如 ARIS：

```text
只读检查已有项目。
不要修改已有项目目录。
不要修改已有 conda 环境。
如需环境，创建平行环境。
如需代码，复制思路到新项目，不要直接污染原项目。
```

示例状态记录：

```json
{
  "code_root": "/home/user/NewPaper_repro",
  "data_root": "<large-file-root>/NewPaper_repro_data",
  "conda_environment": "/home/user/miniconda3/envs/newpaper_repro",
  "new_model_downloads": false,
  "large_model_copies": false
}
```

## 7. 官方资源发现

必须寻找并记录：

1. arXiv 页面和 PDF；
2. 官方 GitHub；
3. Hugging Face dataset；
4. Hugging Face model；
5. project page；
6. supplementary material；
7. leaderboard 或 released evaluation files。

对于每个资源，标注用途：

```text
schema discovery
official validation
candidate gallery
evaluation recomputation
training data source
model checkpoint
not used
```

如果只找到了官方最终结果，必须写清楚它只能支持 Level 1 或 evaluation smoke reproduction。

## 8. 小规模方法复现要求

如果完整复现太贵，不能只停在官方产物统计。必须构建小规模方法复现。

最低要求：

- 创建或抽取小规模 raw input pool；
- 程序化实现论文中的过滤、选样或构建约束；
- 输出论文相似的中间 schema；
- 至少执行一个生成、编辑、渲染、转换或推理步骤；
- 至少计算一个论文相似指标；
- 生成最终 paper-like 输出；
- 做 schema、文件路径、数量、可视化校验。

如果用 proxy，文件名和报告中必须显式包含 `proxy`、`synthetic` 或 `mock`。

## 9. Real-Small 复现分支

当论文需要真实模型或真实数据，但全量数据过大时，应优先建立 real-small 分支。

real-small 的定义：

```text
使用真实数据来源的一小部分
使用真实模型或已有本地模型
产生真实候选输出
计算简化但可解释的指标
合成少量 paper-like pairs 或结果
```

GEditBench v2 的经验示例：

```text
MagicBrush real source images
+ existing InstructPix2Pix model
+ identity baseline
+ semireal region metrics
+ semireal preference pairs
```

这种分支可以称为：

```text
real-small method reproduction smoke_pass
```

但不能称为完整论文级复现，除非真实数据池、真实模型集合、真实指标和训练/评测流程都达到论文要求。

## 10. Synthetic Proxy 资产分支

如果缺少真实 source images、真实 target images 或可控编辑资产，可以建立 synthetic proxy 分支。

可用方案：

- 生图模型生成 source image；
- 3D 模型生成 mesh；
- Blender 渲染多视角、多环境、多 mask；
- 现有 3D 项目或资产库只读复用；
- 用目标渲染作为 oracle candidate；
- 用 identity、flip、blur、brightness 等弱 baseline 构造 pairs。

该分支适合验证：

- schema；
- source/target/candidate/pair 结构；
- target-aware metrics；
- mask-aware metrics；
- pair synthesis；
- contact sheet 可视化。

该分支不适合声称：

```text
复现了真实用户查询；
复现了官方 benchmark source image 收集；
复现了真实人类偏好标注；
复现了论文完整数据分布。
```

报告中应写成：

```text
synthetic asset construction branch: proxy_pass
authentic paper-level dataset construction: not_completed
```

## 11. 候选生成和偏好数据构建

如果论文涉及候选生成和偏好数据，必须区分：

```text
benchmark source construction
candidate model generation
metric computation
top/bottom thresholding
Pareto or auxiliary filtering
preference pair synthesis
judge training
judge inference
leaderboard aggregation
```

每一步都要有输入输出 schema。

例如 preference pair 至少应包含：

```json
{
  "key": "...",
  "task": "...",
  "instruction": "...",
  "source_image": "...",
  "image_a": "...",
  "image_b": "...",
  "model_a": "...",
  "model_b": "...",
  "winner": "Image A",
  "primary_metric_evidence": {},
  "auxiliary_filter_evidence": {},
  "artifact_label": "self_generated | synthetic_proxy | mock_or_proxy"
}
```

如果没有训练 judge，只能标注为 `not_completed` 或 `inference_only`。

## 12. 校验门

声明成功前必须通过以下 gate。

### Gate A: Schema

- 必需字段存在；
- JSONL 可解析；
- 图片路径存在；
- 任务标签合法；
- 计数与报告一致。

### Gate B: Method

- 每条方法线都有代码或 `not_completed`；
- 官方产物没有被冒充为自生成；
- mock/proxy/synthetic 明确标注。

### Gate C: Quantitative

- 记录样本数；
- 记录文件数和字节数；
- 统计任务分布；
- 至少计算一个论文相似指标；
- 能与官方统计做 sanity check。

### Gate D: Visual

- 生成 contact sheet；
- 展示 source、target、candidate、baseline、official examples；
- 明确说明是否同 source / 同 instruction。

### Gate E: Final Claim

最终结论必须保守，不能把 smoke test 写成 full reproduction。

## 13. 必需报告

每个项目至少输出：

```text
docs/METHOD_CONSTRAINT_CARD.md
docs/REPRODUCTION_PLAN.md
REPRO_STATUS.json
reports/<paper>_repro_summary.md
reports/artifact_lineage.json
reports/contact_sheet.jpg
```

`REPRO_STATUS.json` 建议包含：

```json
{
  "project": "...",
  "code_root": "...",
  "data_root": "...",
  "large_file_policy": "...",
  "status": {
    "official_resources_discovered": true,
    "evaluation_smoke": "pass",
    "real_small_reproduction": "smoke_pass",
    "synthetic_asset_branch": "proxy_pass",
    "paper_level_reproduction": "not_completed"
  },
  "outputs": {},
  "counts": {},
  "known_blockers": [],
  "method_notes": []
}
```

每完成一个阶段都必须更新 `REPRO_STATUS.json`，例如资源发现、方法线拆解、环境检查、小样本生成、指标计算、可视化校验和最终报告阶段。不要只在任务最后一次性补写状态。

如果任务被权限、数据、模型、算力或时间限制中断，`REPRO_STATUS.json` 必须记录当前已完成阶段、阻塞原因、需要用户确认的事项，以及下一步可恢复执行的命令或入口。

`artifact_lineage.json` 建议格式：

```json
{
  "path": "...",
  "label": "self_generated | official_downloaded | official_result_recomputed | mock_or_proxy | synthetic_proxy | not_completed",
  "method_line": "...",
  "inputs": ["..."],
  "notes": "..."
}
```

## 14. 对比官方样例

如果要“和官方对比”，必须先确认是否满足同源条件：

```text
same source image?
same instruction?
same task?
same model?
same metric?
```

如果不是同源，只能称为 qualitative comparison 或 quality reference，不要称为严格复现对比。

对比报告必须写清楚：

- 哪些是 ours；
- 哪些是 official；
- 是否同一个 case；
- 官方是否缺 source image；
- 分辨率、文件大小、视觉质量差异；
- 差异来自模型、数据源、任务还是指标。

## 15. 常见失败模式

必须避免：

- 把官方数据下载当作数据构建复现；
- 把官方 evaluation JSONL + 本地 Elo 当作完整复现；
- 论文有多条构建线却只复现 leaderboard；
- 使用 PIL 规则编辑器却不标注 proxy；
- synthetic 资产冒充真实用户数据；
- 没有真实模型生成却声称 candidate generation 完成；
- 未训练 judge 却声称复现 preference model；
- 大文件乱放进代码目录；
- 修改用户已有项目或 conda 环境；
- 不生成 contact sheet 就声称视觉流程跑通。

## 16. 推荐最终措辞

只验证官方产物时：

> This project verifies official artifacts and recomputes part of the evaluation chain, but it does not reproduce the paper's data construction or training method.

小规模方法复现时：

> This project implements a small-scale method reproduction with self-generated intermediate artifacts and schema-compatible outputs. It is not a full paper-level reproduction because scale, model fidelity, human annotation, or training still differs from the paper.

real-small 真实模型小闭环时：

> This project runs a real-small reproduction using real input data and an existing real model to generate candidates and paper-like outputs. It is a smoke pass for the method chain, not a full paper-level reproduction.

synthetic proxy 分支时：

> This project adds a synthetic proxy construction branch using generated or rendered assets to test schema, metrics, and pair synthesis. It is useful engineering evidence but cannot replace the paper-authentic data source.

完整复现时：

> This project reproduces the paper-level method with comparable data scale, training/inference setup, evaluation protocol, and reported metrics, subject to the documented differences.

## 17. 任务结束前检查清单

最终回复用户前，AI 必须确认：

- 代码是否在指定项目目录；
- 大文件是否在指定数据目录；
- 是否没有下载未授权新模型；
- 是否没有修改不该修改的已有项目；
- 每条方法线状态是否明确；
- 是否有报告、状态文件、contact sheet；
- 是否清楚区分 official、self-generated、proxy、not completed；
- 最终结论是否没有夸大；
- 是否说明如果要升级到更高复现等级，还需要哪些资源、预计成本和主要风险。
