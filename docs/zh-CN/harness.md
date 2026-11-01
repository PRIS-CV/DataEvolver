# 计划、执行与恢复

[首页](../../README_zh.md) · [文档目录](../../README_zh.md#文档导航) · [English](../en/harness.md)

选择编译入口并创建 Campaign，再通过共用 Controller 接口推进和查看运行。

除明确要求切换目录的步骤外，命令示例均从仓库根目录执行。

## 执行模型

**Campaign** 是一次持久化运行，由 `GoalSpec` 绑定目标、允许的算子、预算和评测规则。
`EvolutionController` 统一管理状态转换、执行调度和恢复。算子返回产物与执行回执，
评测器检查已声明的约束，Controller 再决定是否接受结果。

架构图见[动态架构展示](../../README_zh.md#流水线概览)。实线表示计划、调用与结果记录，虚线表示恢复与查看。

## 核心特性

- **统一执行职责：** Controller 管理状态、调度、预算和恢复。
- **显式组合：** 编译器检查算子版本、端口及模型角色预算。
- **可替换的模型与工具：** Profile 绑定宿主提供的实现，没有全局默认模型。
- **可核对的执行记录：** 产物哈希和原生回执支持验证与复用。
- **分别检查验收条件：** 执行、视觉质量及下游效用各有证据要求。

## 任务生命周期

选择一种编译入口，再使用共用的 Campaign 命令。以下命令供已配置的执行主机使用；
Worker 模板应先填入实际可执行文件和输入，才能编译。

### 原生 Worker：脚本、模型工具或渲染进程

从[通用 Worker 模板](../../configs/worker.example.json)开始：

```bash
mkdir -p .dataevolver/local/task
cp configs/worker.example.json .dataevolver/local/task/worker.json
```

修改 `stage`、`argv`、`files`、`inputs`、具名 `outputs`、超时和资源权限。
`files` 与 `inputs` 的相对路径按配置文件所在目录解析。将需要绑定哈希的 Worker 代码和依赖列入配置。
可执行文件应接收配置的请求及输出参数，并实现[请求/结果契约（英文参考）](../NATIVE_SCENE_WORKFLOW.md#worker-contract)，
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
契约见[框架接入指南（英文参考）](../FRAMEWORK_INTEGRATION.md)和[编译器源码](../../src/dataevolver/harness/plan_compiler.py)。

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
都恰好执行一次。实现见 [`EvolutionController`](../../src/dataevolver/harness/controller.py)、
[`StateStore`](../../src/dataevolver/harness/state_store.py) 和
[`ArtifactWorkloadOperator`](../../src/dataevolver/harness/artifact_operator.py)。
