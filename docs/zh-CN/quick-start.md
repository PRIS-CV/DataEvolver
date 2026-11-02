# 快速开始

[首页](../../README_zh.md) · [文档目录](../../README_zh.md#文档导航) · [English](../en/quick-start.md)

DataEvolver 围绕多模态大模型的训练目标组织图像与 3D 数据合成和修订。
可通过可选的[工作台](workbench.md)在网页配置并对话，也可独立使用 Harness CLI 准备工作流。
两种入口共用 Harness 的计划编译、执行记录和恢复接口。

除明确要求切换目录的步骤外，命令示例均从仓库根目录执行。

使用 DeepSeek Harness 时，可安装可选的 [DSH 插件](../../plugins/dsh/README_zh.md)。
插件通过 DSH 的工具注册表接入同一套 Campaign 生命周期。

## 环境要求

| 组件 | 要求 |
|---|---|
| 基础开发 | 可编辑源码检出；宜使用源码 CI 所用的 Python 3.12。 |
| Harness 依赖 | `.[harness]`；查看接口无需模型权重或引擎。 |
| 网页工作台 | 工作台主机安装 Node.js 22.19+、npm、Git 和 Python 3.10+；首次安装需要网络。 |
| 原生工具/模型执行 | 所选工具要求的系统、引擎、模型依赖和宿主权限。 |
| GPU 工作负载 | 明确选定的适配器要求的显存、CUDA 和运行时兼容性。 |
| 网站开发 | Node.js 22+ 和 npm，与模型执行独立。 |
| 可选 DSH 插件 | Node.js 24.9+、兼容的 DSH，以及目标执行主机上的 DataEvolver。 |

### 所需模型

基础版本没有统一必装的模型组合。按任务绑定所需角色：规划需要受支持的结构化输出 Provider，
图像/视频生成、审查和分割需要对应的注册实现。[视觉适配器文档（英文参考）](../LOCAL_VISUAL_MODELS.md)
列出可选的现有实现及精确身份。其他实现需要自己的适配器，纯工具 Worker 可以不调用模型。

### 可选 HYWorld / WorldMirror 场景重建

[HYWorld 工作流](../../src/dataevolver/workflows/hyworld/)和
[场景契约](../../src/dataevolver/runtime/hyworld_scene_contract.py)保留早期场景上下文路线。
第三方源码、权重和原生依赖由执行主机提供。该路线可选，当前
[Blender/UE5 Worker（英文参考）](../NATIVE_SCENE_WORKFLOW.md)有单独的契约。

## 快速开始

先克隆仓库，再选择网页或 CLI 安装方式：

```bash
git clone https://github.com/PRIS-CV/DataEvolver.git
cd DataEvolver
```

### 网页工作台

在仓库根目录执行：

```bash
./apps/workbench/launch.sh
```

启动器默认创建 `.venv`、安装 `.[harness]` 和应用锁文件中的 Node 依赖、取得固定提交的
Agent 扩展、构建前端并启动本地工作台。

1. 如果浏览器未打开，打开终端输出的启动链接。
2. 在 **设置** 添加 Provider、服务地址和模型 ID，再选择主 Agent 与子 Agent。
3. 填写工作流需要的图像接口与原生工具路径。
4. 在 **私有凭据** 填写各接口对应的密钥环境变量名和 API Key，再点击 **暂存密钥**。
5. 点击 **保存配置**。
6. 新建对话。

计划编译和 Campaign 命令可在 **Harness** 页面提交。

Blender、UE5、模型权重、推理服务和远端环境由执行主机提供。
服务器或混合部署见 [OpenSSH 部署](workbench.md#openssh-部署)。
根目录 `./launch.sh` 保留为兼容入口。
配置、启动参数和恢复说明见[工作台指南](workbench.md)。

### 方式 A：让 AI Agent 准备配置

可以给 Codex 或其他具备执行能力的 Agent 以下指令：

```text
阅读 AGENT_SETUP.md 和所选工作流的契约。
查看已获授权的执行主机及现有任务。
在私有配置中明确绑定所选模型和工具。
准备计划或 Worker 配置，编译目标，再使用 Harness CLI。
每步之间查看状态和回执，保留完成状态未知的执行证据。
只执行用户已授权的工作负载。
```

[部署技能](../../.agents/skills/dataevolver-onboarding/SKILL.md)遵循这条流程，由用户选择模型和工具。

### 方式 B：手动安装

#### 1. 克隆仓库

使用上方已克隆的源码，以仓库根目录作为工作目录。

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
[`harness.cli:main`](../../src/dataevolver/harness/cli.py)。查看 CLI 和算子目录无需模型权重或渲染引擎。
宿主环境准备与私有配置见 [Agent 接入说明](agents.md)。

#### 3. 查看接口

`--help` 和 `operator-sdk catalog` 查看接口，不做推理。
旧 bootstrap/onboarding dry-run 已由部署技能和显式配置准备替代。

#### 4. 配置工作流

先按[模型与工具配置](deployment.md#替换模型和工具)绑定实现，再按[任务生命周期](harness.md#任务生命周期)准备运行。
私有路径与凭证保存在忽略的本地配置中，创建 Campaign 前核对编译结果的绑定。

#### 5. 运行验收

运行验收需要目标主机和工作负载授权。[项目验收技能](../../.agents/skills/verify-dataevolver/SKILL.md)
说明验收场景，当前执行状态见[迁移与验证](development.md#迁移与验证)。历史测试数量不能用于验收当前整合版本。
