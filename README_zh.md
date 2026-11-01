<p align="center">
  <img src="apps/project-page/frontend/logo.png" alt="DataEvolver Logo" width="120"/>
</p>

<h1 align="center">DataEvolver</h1>

<p align="center">
  <strong>面向 Agent 数据工作流、支持持久化运行的 Harness。</strong><br/>
  Agent 制订计划，注册算子负责执行，运行证据支持恢复。
</p>

<p align="center">
  <a href="https://pris-cv.github.io/DataEvolver/"><img src="https://img.shields.io/badge/Website-Project%20Page-1f6feb?logo=githubpages&logoColor=white" alt="项目网站"/></a>
  <a href="https://arxiv.org/abs/2605.01789"><img src="https://img.shields.io/badge/arXiv-Technical%20Report-b31b1b?logo=arxiv&logoColor=white" alt="首份技术报告"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="许可证：Apache-2.0"/></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong> ·
  <a href="AGENT_SETUP.md">Agent 部署说明</a> ·
  <a href="docs/FRAMEWORK_INTEGRATION.md">框架接入指南</a>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> · <a href="#流水线概览">Harness</a> ·
  <a href="#替换模型和工具">模型与工具</a> ·
  <a href="#查看暂停与恢复">恢复机制</a>
</p>

`main` 是可继续开发、可迁移的数据工作流基础版本，用来组合数据处理、模型调用和原生工具。
Codex 或其他应用提供计划与候选代码，Harness 管理运行过程。模型、可执行文件、路径和资源权限
由部署配置明确指定。私有配置及凭证放在已忽略的 `.dataevolver/local/` 中，或通过宿主环境提供。

![DataEvolver 历史项目海报](apps/project-page/frontend/images/showcase-poster.png)

*原版项目海报展示早期渲染与审查设计。*

## DataEvolver 与 DataRSI

**DataEvolver 是工程项目，DataRSI 是后续研究工作。**
首份[技术报告](https://arxiv.org/abs/2605.01789)介绍了早期目标驱动的循环 Agent。
[DataRSI](https://github.com/PRIS-CV/DataRSI) 的标题是 *A 3D Data Harness for Failure-to-Data Evolution*，
研究固定预算和规则下的数据修订与模型准入。其公开仓库当前公告称，实现和复现材料仍在整理。

本仓库的 `main` 提供当前 Harness 基础版本。早期 VLM 双循环保留在
[历史分支](https://github.com/PRIS-CV/DataEvolver/tree/codex/legacy-vlm-dual-loop)，
范围说明见[项目与论文指南](docs/DATARSI_RELATION.md)。

## 为什么是 DataEvolver？

数据构建需要兼容的算子、明确的进程输入输出，以及中断后可核对的运行证据。
DataEvolver 为这些环节提供共用契约和统一 Controller。
Agent 可以提出不同的数据工作流，执行过程仍复用同一套状态管理。

## 项目网页与修复轨迹

[项目网站](https://pris-cv.github.io/DataEvolver/)展示工程路线、研究图示和渲染画廊。
[修复轨迹页](https://pris-cv.github.io/DataEvolver/traces/?lang=zh)保留接地与瓶子计数的历史案例，
包括退化和未解决结果。单独的 Qwen/FLUX 对比使用不同生成器；这些材料不能证明连续改进过程或新增训练收益。

网站是独立的[应用](apps/project-page/README.md)，源码开发需要 Node.js 22+ 和 npm。
只读归档无需 Harness 或 GPU。

```bash
cd apps/project-page
npm ci
npm run dev
```

打开 `http://127.0.0.1:4173/`。`npm run build` 生成静态构建，`npm run preview` 提供预览服务。
Pages 与可选归档 API 的配置见[部署指南](apps/project-page/deploy/README.md)。

## DataRSI 研究更新

**已投稿至 ICASSP 2027。** DataRSI 连接数据修订循环与模型准入循环。VLM 检查提供缺陷证据，已注册的策略在预算内修复，确定性样本合同决定训练样本准入。候选模型更新须通过目标效用与受保护区域的回归检查。

![DataRSI 方法：数据修订、失败到数据反馈与模型准入](apps/project-page/frontend/images/paper/datarsi-method.jpg)

论文比较 Uniform-72 与 Targeted-72，**每组使用 72 次请求**，训练设置和候选池相同，种子为 **42、43、44**。评测采用 **288 对开发样本，分辨率为 1024 × 1024**。Weak-4 是诊断出的四个相机控制单元。下表为三个种子的均值 ± 标准差。

| 指标 | Uniform-72 | Targeted-72 |
|---|---|---|
| Weak-4 旋转误差（°）↓ | 50.45 ± 2.15 | 35.50 ± 3.23 |
| Weak-4 PSNR（dB）↑ | 13.88 ± 0.07 | 14.43 ± 0.09 |
| 整体 LPIPS ↓ | 0.2058 ± 0.0028 | 0.2014 ± 0.0030 |

相对 Uniform，非 Weak-4 区域旋转误差增加 **2.370 ± 0.046°**，整体旋转误差增加 **0.205 ± 0.125°**。相机准入回放将三个定向候选与冻结的基准模型比较，均返回 `Promote`。物体移动案例虽然 18 个样本全部准入、目标效用达到要求，但受保护空间单元出现回退，因此返回 `Rollback`。

这些结果覆盖**一次预算内的修订**。论文尚未证明多轮递归改进。另一组历史 DataRSI-Bench 评测采用 384 对样本、512 × 512 分辨率；下文的原始 DataEvolver-Rotate 数据集包含 350 对训练样本。几组评测各有适用范围。修复图与历史检查点对比图见[网页结果区](https://pris-cv.github.io/DataEvolver/#results)。此处数字来自论文稿件，本次文档更新没有重跑实验。

## 当前能力

| 范围 | 当前代码接口 |
|---|---|
| 数据工作流 | 注册的图像/目录数据图步骤、产物存储与验证。 |
| Agent 规划 | 结构化计划、显式模型角色绑定和预算内提案。 |
| 多模态工具 | 可选生成、编辑、提示增强、观察、审查和分割适配器。 |
| 3D 工作流 | 可选 Blender 建模/仿真/导出及 UE5 渲染 Worker。 |
| 研究准备 | 单论文或多论文证据会话与 handoff 导出。 |

这些是实现入口。所选主机与配置的部署可用性、输出质量需要分别验证。

## 核心特性

- **统一执行职责：** Controller 管理状态、调度、预算和恢复。
- **显式组合：** 编译器检查算子版本、端口及模型角色预算。
- **可替换的模型与工具：** Profile 绑定宿主提供的实现，没有全局默认模型。
- **可核对的执行记录：** 产物哈希和原生回执支持验证与复用。
- **分别检查验收条件：** 执行、视觉质量及下游效用各有证据要求。

## 流水线概览

**Campaign** 是一次持久化运行，由 `GoalSpec` 绑定目标、允许的算子、预算和评测规则。
`EvolutionController` 统一管理状态转换、执行调度和恢复。算子返回产物与执行回执，
评测器检查已声明的约束，Controller 再决定是否接受结果。

[![DataEvolver Harness 动态架构图](apps/project-page/frontend/architecture/preview-zh.gif)](https://pris-cv.github.io/DataEvolver/architecture/zh.html)

[打开完整交互 HTML](https://pris-cv.github.io/DataEvolver/architecture/zh.html) · [可编辑图规格](apps/project-page/frontend/architecture/zh.spec.json)

实线表示计划、调用与结果记录，虚线表示恢复与查看。模型和原生工具的具体实现由配置选择。

采用 [arch-diagram](https://github.com/bassnova42/arch-diagram-skill) 的默认设计。点击动画可打开完整页面。

**Codex 有两种接入角色。** 外层具备执行能力的 Agent 可以在授权范围内编写计划、Worker 或建模配方，
编译目标，再调用公共 CLI。它读取 JSON 状态和回执，决定是否请求下一步。
可选的 [`CodexCLIPlannerProvider`](src/dataevolver/adapters/codex_planner.py) 是内部模型接口：
它只返回符合 schema 的提案，禁用 Shell、执行及多 Agent 工具，不执行数据工作流。
其他受支持的 Provider 也可通过相同模型契约提供提案。

例如，Codex 可以准备 Blender 组件建模配方，通过原生 Worker 提交执行，检查产物，
再准备单独的 UE5 渲染任务。建模、场景导出和渲染各有输入输出契约。
Agent 生成的计划或脚本通过配置的算子绑定后才能执行，Campaign 状态由 Controller 管理。


### VLM 审查闭环（Stage 5）

早期场景流程用 VLM 反馈选择预算内的渲染修复，保留的
[Stage 5 图](apps/project-page/frontend/images/readme/stage5.svg)和
[流水线图](apps/project-page/frontend/images/readme/pipeline_show.svg)描述该历史流程。
当前基础版本将审查作为可选模型步骤，修复需要注册动作、评测规则和预算，不额外引入一套 Controller。

## 环境要求

| 组件 | 要求 |
|---|---|
| 基础开发 | 可编辑源码检出；宜使用源码 CI 所用的 Python 3.12。 |
| Harness 依赖 | `.[harness]`；查看接口无需模型权重或引擎。 |
| 原生工具/模型执行 | 所选工具要求的系统、引擎、模型依赖和宿主权限。 |
| GPU 工作负载 | 明确选定的适配器要求的显存、CUDA 和运行时兼容性。 |
| 网站开发 | Node.js 22+ 和 npm，与模型执行独立。 |

### 所需模型

基础版本没有统一必装的模型组合。按任务绑定所需角色：规划需要受支持的结构化输出 Provider，
图像/视频生成、审查和分割需要对应的注册实现。[视觉适配器文档](docs/LOCAL_VISUAL_MODELS.md)
列出可选的现有实现及精确身份。其他实现需要自己的适配器，纯工具 Worker 可以不调用模型。

### 可选 HYWorld / WorldMirror 场景重建

[HYWorld 工作流](src/dataevolver/workflows/hyworld/)和
[场景契约](src/dataevolver/runtime/hyworld_scene_contract.py)保留早期场景上下文路线。
第三方源码、权重和原生依赖由执行主机提供。该路线可选，当前
[Blender/UE5 Worker](docs/NATIVE_SCENE_WORKFLOW.md)有单独的契约。

## 快速开始

### 方式 A：让 AI Agent 准备配置

可以给 Codex 或其他具备执行能力的 Agent 以下指令：

```text
Read AGENT_SETUP.md and the selected workflow contract.
Inspect the authorized execution host and existing jobs.
Bind the chosen models/tools explicitly in private profiles.
Prepare a plan or worker profile, compile its goal, then use the Harness CLI.
Inspect state and receipts between steps; preserve unknown effects.
Only execute workloads that the user has authorized.
```

[部署技能](.agents/skills/dataevolver-onboarding/SKILL.md)遵循这条流程，由用户选择模型和工具。

### 方式 B：手动安装

#### 1. 克隆仓库

```bash
git clone https://github.com/PRIS-CV/DataEvolver.git
cd DataEvolver
```

#### 2. 安装包入口


宜使用源码契约 CI 所用的 Python 3.12，并以可编辑模式安装。
CLI 会读取仓库中与 `src/` 配套的 `configs/`。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[harness]'
dataevolver --help
dataevolver operator-sdk catalog
```

`dataevolver`、`dataevolver-harness` 和 `python -m dataevolver` 共用
[`harness.cli:main`](src/dataevolver/harness/cli.py)。查看 CLI 和算子目录无需模型权重或渲染引擎。
宿主环境准备与私有配置见 [Agent 部署说明](AGENT_SETUP.md)。


#### 3. 查看接口

`--help` 和 `operator-sdk catalog` 查看接口，不做推理。
旧 bootstrap/onboarding dry-run 已由部署技能和显式配置准备替代。

#### 4. Production Setup（生产部署）

先按[模型与工具配置](#替换模型和工具)绑定实现，再按[任务生命周期](#任务生命周期)准备运行。
私有路径与凭证保存在忽略的本地配置中，创建 Campaign 前核对编译结果的绑定。

#### 5. 生产 Smoke Tests

运行验收需要目标主机和工作负载授权。[项目验收技能](.agents/skills/verify-dataevolver/SKILL.md)
说明验收场景，当前执行状态见[迁移与验证](#迁移与验证)。历史测试数量不能用于验收当前整合版本。

## 任务生命周期

选择一种编译入口，再使用共用的 Campaign 命令。以下命令供已配置的执行主机使用；
Worker 模板应先填入实际可执行文件和输入，才能编译。

### 原生 Worker：脚本、模型工具或渲染进程

从[通用 Worker 模板](configs/worker.example.json)开始：

```bash
mkdir -p .dataevolver/local/task
cp configs/worker.example.json .dataevolver/local/task/worker.json
```

修改 `stage`、`argv`、`files`、`inputs`、具名 `outputs`、超时和资源权限。
`files` 与 `inputs` 的相对路径按配置文件所在目录解析。将需要绑定哈希的 Worker 代码和依赖列入配置。
可执行文件应接收配置的请求及输出参数，并实现[请求/结果契约](docs/NATIVE_SCENE_WORKFLOW.md#worker-contract)，
包括 `result.json` 及其中的请求摘要。普通命令需要适配后才能使用这个契约。

```bash
dataevolver worker-compile \
  --profile .dataevolver/local/task/worker.json \
  --output .dataevolver/local/task/prepared \
  --campaign task-v1 --campaign-root .dataevolver/campaigns
```

`worker-compile` 绑定可执行文件、代码和输入的哈希，生成 `goal.json`，不做推理或渲染。
`scene-compile` 是兼容别名。准备结果的输出目录应为新目录。

### 数据集图：组合注册步骤

也可准备结构化 `plan.json`、绑定配置 `profile.json` 和评测配置 `evaluator.json`，
再编译到同一个准备目录：

```bash
dataevolver compile-plan \
  --plan .dataevolver/local/task/plan.json \
  --profile .dataevolver/local/task/profile.json \
  --evaluator .dataevolver/local/task/evaluator.json \
  --output .dataevolver/local/task/prepared --campaign task-v1
```

Python 接口是 `compile_plan(plan, CompileContext(...))`。它检查注册域、算子版本、端口、参数、
模型角色和预算，返回 `CompiledRun` 或 `CompilationGap`。编译成功后，CLI 写入
`compiled_run.json` 和 `goal.json`；存在缺失绑定时，返回 `compiled: false` 和退出码 `2`。
先补齐绑定，再执行。编译器接收结构化计划，用户需求由外层 Agent 转成该结构。
契约见[框架接入指南](docs/FRAMEWORK_INTEGRATION.md)和[编译器源码](src/dataevolver/harness/plan_compiler.py)。

### 共用的创建和推进接口

```bash
dataevolver create --goal .dataevolver/local/task/prepared/goal.json \
  --campaign-root .dataevolver/campaigns --idempotency-key task-v1
dataevolver step --campaign task-v1 \
  --campaign-root .dataevolver/campaigns --max-actions 8
dataevolver status --campaign task-v1 --campaign-root .dataevolver/campaigns
```

`create` 初始化并冻结 Campaign，不派发 Worker。`step` 在 Campaign 租约内调用
`EvolutionController.tick(max_actions)`。这个上限计算的是 **Controller 状态动作数**，
不代表图片数、模型调用数或秒数；一个动作可能等待原生 Worker 完成。任务预算和 Worker 超时仍然生效。
继续按步推进，直到记录的状态进入终态或暂停。
`evolve --goal ... --idempotency-key ...` 则合并创建与 `Controller.run()`，按同一冻结契约持续执行。

## 查看、暂停与恢复

以下命令均接收 `--campaign` 和 `--campaign-root`：

| 接口 | 行为 |
|---|---|
| `status` | 读取当前快照，不加载执行运行时。 |
| `explain --limit 20` | 读取快照和近期事件，运行时发生漂移后仍可查看。 |
| `replay` | 从事件重建状态并报告 `matches_snapshot`，不执行工作负载。 |
| `pause` | 请求在 Controller 动作边界暂停，不强制停止运行中的 Worker。 |
| `resume` | 恢复暂停前的状态，并 **立即调用 `run()` 持续执行**。 |

限制本次推进量时使用 `step --max-actions ...`。冻结契约变化后会进入 `PAUSED_CONTRACT_DRIFT`；
漂移仍存在时，`resume` 会保留该暂停。恢复原绑定后可继续原运行；有意修改配置时，创建新目标和新 Campaign。

Campaign 目录保留 `goal.json`、`frozen_contract.json`、`events.jsonl`、`state.json`，
结束时写入 `final_report.json`，并保留算子证据。进程中断后，先查看证据，再推进同一个 Campaign：
已计划的调用会先走 `Operator.recover()`，再走 `verify_effect()`。已验证的回执可以复用；
重新执行需要明确返回安全恢复结果。原生 Worker 还会核对保留的请求、执行意图、日志、退出状态、
结果和输出哈希。已有执行意图但缺少可验证完成证据时，自动重跑会被阻止，需要检查并核对原生证据。
排查时保留这些文件。

恢复机制依据执行证据。每个适配器需要实现自己的副作用查询，无法据此承诺所有外部服务
都恰好执行一次。实现见 [`EvolutionController`](src/dataevolver/harness/controller.py)、
[`StateStore`](src/dataevolver/harness/state_store.py) 和
[`ArtifactWorkloadOperator`](src/dataevolver/harness/artifact_operator.py)。

## 替换模型和工具

### 模型：角色 → 配置 → Provider

| 契约 | 配置或实现内容 |
|---|---|
| `ModelPolicy` | 角色到 Profile 的 `bindings`、允许的端点来源、调用预算。 |
| `ModelProfile` | 明确的模型 ID/版本、后端、能力和环境变量绑定。 |
| `ModelProvider` | `health()`、`invoke(request, request_hash)` 和 `lookup(request, request_hash)`。 |
| `ModelRequest` / `ModelReceipt` | 结构化输入输出、schema、用量和证据绑定。 |

替换协议兼容的对话或视觉模型服务时，修改[配置模板](configs/harness/model_profiles.example.json)：
将角色绑定到所选 Profile，设置 `model` 或 `model_env`、`endpoint_env`，并在
`allowed_local_origins` 中授权端点来源。`local_endpoint` 使用所配置服务的 `/models` 和
`/chat/completions` 接口。填写配置后，可用 `dataevolver models --config <your-config.json>`
查看绑定；加上 `--check-endpoint` 会请求健康状态，不做推理。
`model_env` 覆盖的是服务端模型 ID。原生检查点目录使用所选适配器的路径绑定。

替换原生模型实现时，需要通过 `ModelGateway` 注册 Provider 工厂，提供兼容的输出和独立的副作用查询。
修改模型名称不会适配其架构、权重或执行 API。预留的 `external_api` 后端处于禁用状态，
修改标志不会启用云端请求。可选的已认证 Codex 接口有单独的显式配置。
实现见[模型契约](src/dataevolver/adapters/model_contracts.py)和[网关 Provider](src/dataevolver/adapters/model_gateway.py)。
[网关指南](docs/MODEL_GATEWAY.md)也说明了可选的原生用量审计接口。
原生日志格式及附加统计由对应适配器负责。

可选的图像生成/编辑、提示增强、审查和分割实现见[视觉适配器](docs/LOCAL_VISUAL_MODELS.md)。
选用后仍需满足该实现的模型、权重和运行时检查。底层多模态路由接收显式参数或
`DATAEVOLVER_<ROUTE>_MODEL` / `DATAEVOLVER_<ROUTE>_MODEL_PATH`；未绑定时保持未配置状态。

### 工具：注册步骤或原生 Worker

需要组合的数据/模型步骤使用 `declare_step(...)` 和 `OperatorRegistry.register_step(...)`，
再绑定到编译器及运行时的 Registry。内置 CLI 目录使用显式注册的实现。
`operator-sdk init/check/package` 只生成和检查声明，不负责安装、注册或资格验收。
外部进程通过 Worker ABI 适配，再修改供 `worker-compile` 使用的私有配置。

Blender 建模、仿真、冻结场景导出和 UE5 渲染属于[可选原生适配器](docs/NATIVE_SCENE_WORKFLOW.md)。
Blender 建模结果需要转换成规定的冻结场景包，才能交给 UE5 使用。引擎、工程、资产和资源隔离由宿主环境提供。
基础版本没有全局指定的模型或工具。修改代码、模型绑定或输入时，需要新的冻结目标和 Campaign，原运行保留为证据。

## 可复现 Research Prior 工作流

Stage0 为证据收集和数据集规划创建研究会话。Agent 使用选定的搜索工具查找来源并记录证据。
`init` 只创建会话，不自动联网搜索，也不启动数据生产 Campaign。

### 单论文复现模式

```bash
dataevolver-stage0-web-research init \
  --query "paper title or dataset construction method" \
  --output-dir .dataevolver/local/research/single \
  --dataset-mode single-paper
```

记录论文的数据构建方法、官方资源、评测标准和证据缺口，区分原始资源与自行生成的替代数据。

### 多论文通用模式

```bash
dataevolver-stage0-web-research init \
  --query "shared dataset requirements across related papers" \
  --output-dir .dataevolver/local/research/universal \
  --dataset-mode universal
```

使用 `add-evidence` 记录来源，`finalize` 导出研究先验和报告，`validate` 检查先验。
这些准备接口位于 [stage0_web_research.py](src/dataevolver/tools/stage0_web_research.py)。
形成的数据需求仍需绑定注册算子、模型和工具，再通过 `compile-plan` 及 Campaign 接口执行。

## 项目结构

```text
DataEvolver/
├── src/dataevolver/
│   ├── harness/          # Controller、编译器、算子 SDK、状态与恢复
│   ├── adapters/         # 模型契约、接口与原生执行记录
│   ├── dataset/          # 产物、数据图步骤与数据集验证
│   ├── workflows/        # 可选场景、多模态及 HYWorld 实现
│   ├── runtime/          # 资产生命周期与场景契约
│   ├── agents/           # 场景反馈和动作辅助模块
│   ├── annotation/       # 审查辅助模块
│   └── tools/            # 研究准备与诊断
├── configs/              # 公开模板与 schema
├── docs/                 # 接口、部署与迁移说明
├── .agents/skills/       # 部署与验收指导
├── .github/workflows/    # 源码契约与网站工作流
└── apps/project-page/    # 独立网站及只读归档 API
```

本地部署文件和生成的 Campaign 放在已忽略的 `.dataevolver/` 下。

## 扩展边界

| 职责归属 | 扩展位置 |
|---|---|
| [CLI 与 Controller](src/dataevolver/harness/) | Campaign 状态、调度、预算、暂停与恢复。 |
| [计划编译器](src/dataevolver/harness/plan_compiler.py) / [算子 SDK](src/dataevolver/harness/operator_sdk.py) | 域/步骤声明和绑定检查。 |
| [数据契约](src/dataevolver/dataset/contracts.py) / [产物算子](src/dataevolver/harness/artifact_operator.py) | 产物身份、回执和执行效果核验。 |
| [模型适配器](src/dataevolver/adapters/) / [场景 Worker](src/dataevolver/workflows/scene/) | Provider 接口和工具特有行为。 |

优先新增步骤或适配器。模型提案与获准执行分开处理，预算和状态变更统一留在 Controller。
声明版本、端口和副作用，让编译器拒绝无效组合。实现执行逻辑时同时实现恢复。
复用已有产物和回执契约，减少重复的运行器、状态格式和重试循环。

## 操作空间

早期场景审查适配器保留 **24 种原子操作**，定义见
[`scene_action_space.json`](configs/scene_action_space.json)：

| 类别 | 数量 | 修改内容 |
|---|---|---|
| 光照 | 4 | 主光强度和方位。 |
| 物体 | 6 | 高度、方位和缩放。 |
| 场景 | 5 | 环境旋转、强度和接触阴影。 |
| 材质 | 9 | 颜色、粗糙度及高光/光泽属性。 |
| 相机 | 0 | 该动作文件中预留。 |

这些是该适配器的有界参数修改。当前 Harness 的动作由所选算子契约和目标声明，
整体框架不固定为这 24 种操作。模型输出不能替换 Worker 的固定命令。

## DataEvolver-Rotate

原项目材料记录的旋转条件图像编辑数据集。下表属于该历史数据版本，不是当前 Harness 新导出的结果。

| 指标 | 数值 |
|------|------|
| 物体数量 | 50 |
| 旋转角度 | 8 (0&deg;, 45&deg;, 90&deg;, 135&deg;, 180&deg;, 225&deg;, 270&deg;, 315&deg;) |
| 训练对数量 | 350 (正面 → 7 个目标视角) |
| 训练 / 验证 / 测试 | 245 / 49 / 56 对 (物体隔离，seed=42) |
| 模态 | RGB、mask、depth、normal |

### 构建方法

每个物体以唯一的 **canonical yaw-0&deg; best state** 为基准。
旋转物体时，场景、相机、光照和材质保持固定。

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

Claude Code 可选作外层 Agent，编写计划和代码，并调用与 Codex 相同的 Harness CLI。
框架不要求将其指定为规划模型或工具。

### 第一步：安装 Claude Code

按[官方部署说明](https://code.claude.com/docs/en/overview)选择适合宿主环境的安装与认证方式。

### 第二步：创建你的 `CLAUDE.md`

根目录 `CLAUDE.md` 已忽略，可引用私有主机/Profile 配置，并指向 `AGENT_SETUP.md`、
对应工作流契约和公共 CLI。凭证通过私有环境提供，不写入指令正文。

### 第三步：使用项目技能（可选）

按 Agent 支持的指令格式复用[部署](.agents/skills/dataevolver-onboarding/SKILL.md)和
[验收](.agents/skills/verify-dataevolver/SKILL.md)指导。
新增技能用于解释工作流，执行状态仍复用已有 Controller。

### 第四步：启动 Claude Code

在源码检出目录启动已配置的 `claude` 客户端，让其通过 `compile-plan` 或 `worker-compile`
准备任务。执行使用 `create`、有限步 `step` 和观察接口。模型及原生工作负载仍需遵守用户授权。

## 迁移与验证

旧 VLM 渲染/审查双循环保留在
[`codex/legacy-vlm-dual-loop`](https://github.com/PRIS-CV/DataEvolver/tree/codex/legacy-vlm-dual-loop)。
旧 production CLI、独立多模态 CLI、Shell 批处理和 bootstrap 脚本已退役。
统一使用 Harness CLI，迁移说明见[迁移文档](docs/MAINLINE_MIGRATION.md)。

公共 schema 和 Operator 版本用于兼容性检查；模型发行名标识可选实现。
原生 Worker 现已用摘要绑定允许传入的环境值。旧 Worker 配置应重新编译为新 Campaign，
详见 [Worker 契约](docs/NATIVE_SCENE_WORKFLOW.md)。

[源码契约 CI](.github/workflows/source-contracts.yml)检查源码、配置和未定义的 Python 名称，
不执行工作负载。本地源码检查器为：

```bash
python .agents/skills/verify-dataevolver/scripts/check_repository.py
```

当前基础版本完成了静态检查，测试、模型推理和渲染尚未执行。
[项目验收技能](.agents/skills/verify-dataevolver/SKILL.md)中的运行场景需要另行授权，并在目标主机执行。
执行成功和产物完整性本身不能证明视觉真实性、物理标定正确性或训练收益。

## 项目沿革与展示

[项目网站](https://pris-cv.github.io/DataEvolver/)保留了渲染示例、研究材料和修复轨迹。
首份 [DataEvolver 技术报告](https://arxiv.org/abs/2605.01789)描述较早版本；
[DataRSI](https://github.com/PRIS-CV/DataRSI) 是后续研究仓库，关系说明见[项目与论文指南](docs/DATARSI_RELATION.md)。

**历史展示：早期 VLM 渲染/审查流程。** 保留旧 README 插图，用于介绍项目来源。
当前 Harness 架构见[流水线概览](#流水线概览)。

![早期 DataEvolver 工作流：物体生成、重建、渲染和 VLM 审查](apps/project-page/frontend/images/readme/showcase.png)

<details>
<summary>早期物体入场景路线</summary>

下面的物体入场景示意图展示早期 HYWorld / Hunyuan3D 路线及其预期数据输出。
当前 Blender/UE5 接口以原生工作流指南为准。

![早期物体入场景路线：场景重建与物体插入](apps/project-page/frontend/images/readme/3D-build.png)

</details>

## 引用

下方条目引用的是**第一版 arXiv DataEvolver 技术报告**。DataRSI 是后续投稿论文，发布与引用信息请查阅其[论文仓库](https://github.com/PRIS-CV/DataRSI)。

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

- **AIDataSci 2026，KDD 2026 Workshop：** 当前
  [官方录用列表](https://usail-hkust.github.io/aidatasci/accepted-papers.html)
  将 DataEvolver 列为 **Poster Presentation**，据此更正旧 README 的 Oral 标注。
- **2026 智源大会 Agent for Science 大赛：** 获海报展示证书，见下图。这些记录属于早期项目。

<p align="center">
  <img src="apps/project-page/frontend/images/readme/baai_agent4s_poster_certificate.jpg" alt="DataEvolver 智源大会海报展示证书" width="720"/>
</p>

## 路线图

### 已完成

- [x] 统一 Campaign CLI、编译器、算子契约及持久化恢复代码。
- [x] 显式模型配置与 Provider/Worker 边界。
- [x] 可选多模态步骤、原生 Blender/UE5 Worker 和研究先验工具。
- [x] 源码契约 CI 与双语框架说明。

勾选项表示基础版本已有这些代码或文档，不代表已经完成原生运行验收或证明学习收益。

### 进行中

- [ ] 获得授权后，在目标执行主机验收当前整合版本。
- [ ] 持续对齐适配器接口、冻结绑定与开发文档。

### 待扩展

早期研究方向包括可复用的审查知识、真实数据接入、重建加速、重建模型训练、时序审查/过滤及通用多物体场景。
这些方向的端到端验收仍待完成。Benchmark 探针及 Benchmark 缺陷发现不属于本次工程更新。

## 许可证

[Apache-2.0](LICENSE)
