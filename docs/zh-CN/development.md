# 开发与扩展

[首页](../../README_zh.md) · [文档目录](../../README_zh.md#文档导航) · [English](../en/development.md)

在已有边界扩展算子、Provider 或 Domain。状态转换和执行权限由 Controller 统一管理。

除明确要求切换目录的步骤外，命令示例均从仓库根目录执行。

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
│   ├── en/              # 英文任务指南
│   └── zh-CN/           # 简体中文任务指南
├── .agents/skills/       # 部署与验收指导
├── .github/workflows/    # 源码契约与网站工作流
└── apps/project-page/    # 独立网站及只读归档 API
```

本地部署文件和生成的 Campaign 放在已忽略的 `.dataevolver/` 下。
模型绑定、原生工具路径和凭证也放在该目录中。`configs/` 提供公开模板；
编译 Goal 前，先选择工作流需要的实现。

## 扩展边界

| 职责归属 | 扩展位置 |
|---|---|
| [CLI](../../src/dataevolver/harness/cli.py) | 解析命令并输出结果。两个公开命令继续使用同一入口。 |
| [组装工厂](../../src/dataevolver/harness/composition.py) | 绑定 Goal 的算子、规划器、评价器及冻结源码，创建或重新打开 Controller。 |
| [Controller](../../src/dataevolver/harness/controller.py) | Campaign 状态、调度、预算、效果恢复和提交。 |
| [最终报告](../../src/dataevolver/harness/final_report.py) / [域适配器](../../src/dataevolver/harness/adapters/) | 从证据生成报告，并解释对应路线的结果。 |
| [计划编译器](../../src/dataevolver/harness/plan_compiler.py) / [算子 SDK](../../src/dataevolver/harness/operator_sdk.py) | 域/步骤声明和绑定检查。 |
| [数据契约](../../src/dataevolver/dataset/contracts.py) / [产物算子](../../src/dataevolver/harness/artifact_operator.py) | 产物身份、回执和执行效果核验。 |
| [模型适配器](../../src/dataevolver/adapters/) / [场景 Worker](../../src/dataevolver/workflows/scene/) | Provider 接口和工具特有行为。 |

优先新增步骤或适配器。模型提案与获准执行分开处理，预算和状态变更统一留在 Controller。
声明版本、端口和副作用，让编译器拒绝无效组合。实现执行逻辑时同时实现恢复。
复用已有产物和回执契约，减少重复的运行器、状态格式和重试循环。

新增图步骤时，沿用[数据集步骤](../../src/dataevolver/dataset/catalog_steps.py)的
`declare_step` 和 `OperatorRegistry.register_step`。新增完整任务时，实现
`execute`、`verify_effect` 和 `recover`，再注册 `OperatorSpec` 及可选的
[OperatorContract（英文参考）](../OPERATOR_CONTRACTS.md)。自定义工作流通过
[DomainBinding](../../src/dataevolver/harness/domains.py)和[计划编译器](../../src/dataevolver/harness/plan_compiler.py)接入，
评价策略放在声明的评价器和 Goal 中。已有进程工具可沿用[原生 Worker 示例](harness.md#原生-worker脚本模型工具或渲染进程)。

[框架接入说明（英文参考）](../FRAMEWORK_INTEGRATION.md#source-boundaries)列出这些入口、模型后端及序列化契约。
修改源码会使旧 Campaign 检测到契约漂移；恢复旧 Campaign 时应使用原冻结源码，保留原回执。

## 操作空间

早期场景审查适配器保留 **24 种原子操作**，定义见
[`scene_action_space.json`](../../configs/scene_action_space.json)：

| 类别 | 数量 | 修改内容 |
|---|---|---|
| 光照 | 4 | 主光强度和方位。 |
| 物体 | 6 | 高度、方位和缩放。 |
| 场景 | 5 | 环境旋转、强度和接触阴影。 |
| 材质 | 9 | 颜色、粗糙度及高光/光泽属性。 |
| 相机 | 0 | 该动作文件中预留。 |

这些是该适配器的有界参数修改。当前 Harness 的动作由所选算子契约和目标声明，
整体框架不固定为这 24 种操作。模型输出不能替换 Worker 的固定命令。

## 迁移与验证

旧 VLM 渲染/审查双循环保留在
[`codex/legacy-vlm-dual-loop`](https://github.com/PRIS-CV/DataEvolver/tree/codex/legacy-vlm-dual-loop)。
旧 production CLI、独立多模态 CLI、Shell 批处理和 bootstrap 脚本已退役。
统一使用 Harness CLI，迁移说明见[迁移文档（英文参考）](../MAINLINE_MIGRATION.md)。

公共 schema 和 Operator 版本用于兼容性检查；模型发行名标识可选实现。
原生 Worker 现已用摘要绑定允许传入的环境值。旧 Worker 配置应重新编译为新 Campaign，
详见 [Worker 契约（英文参考）](../NATIVE_SCENE_WORKFLOW.md)。

[源码契约 CI](../../.github/workflows/source-contracts.yml)检查源码、配置和未定义的 Python 名称，
不执行工作负载。本地源码检查器为：

```bash
python .agents/skills/verify-dataevolver/scripts/check_repository.py
```

当前基础版本完成了静态检查，测试、模型推理和渲染尚未执行。
[项目验收技能](../../.agents/skills/verify-dataevolver/SKILL.md)中的运行场景需要另行授权，并在目标主机执行。
执行成功和产物完整性本身不能证明视觉真实性、物理标定正确性或训练收益。

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
