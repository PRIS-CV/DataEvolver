# Agent 接入

[首页](../../README_zh.md) · [文档目录](../../README_zh.md#文档导航) · [English](../en/agents.md)

外层 Agent 准备计划与代码，并调用公共 Harness 接口。内部规划 Provider 只返回提案。

除明确要求切换目录的步骤外，命令示例均从仓库根目录执行。

## Codex 与其他外层 Agent

**Codex 有两种接入角色。** 外层具备执行能力的 Agent 可以在授权范围内编写计划、Worker 或建模配方，
编译目标，再调用公共 CLI。它读取 JSON 状态和回执，决定是否请求下一步。
可选的 [`CodexCLIPlannerProvider`](../../src/dataevolver/adapters/codex_planner.py) 是内部模型接口：
它只返回符合 schema 的提案，禁用 Shell、执行及多 Agent 工具，不执行数据工作流。
其他受支持的 Provider 也可通过相同模型契约提供提案。

例如，Codex 可以准备 Blender 组件建模配方，通过原生 Worker 提交执行，检查产物，
再准备单独的 UE5 渲染任务。建模、场景导出和渲染各有输入输出契约。
Agent 生成的计划或脚本通过配置的算子绑定后才能执行，Campaign 状态由 Controller 管理。

## 与 Claude Code 集成

Claude Code 可选作外层 Agent，编写计划和代码，并调用与 Codex 相同的 Harness CLI。
框架不要求将其指定为规划模型或工具。

### 第一步：安装 Claude Code

按[官方部署说明](https://code.claude.com/docs/en/overview)选择适合宿主环境的安装与认证方式。

### 第二步：创建你的 `CLAUDE.md`

根目录 `CLAUDE.md` 已忽略，可引用私有主机/Profile 配置，并指向 `AGENT_SETUP.md`、
对应工作流契约和公共 CLI。凭证通过私有环境提供，不写入指令正文。

### 第三步：使用项目技能（可选）

按 Agent 支持的指令格式复用[部署](../../.agents/skills/dataevolver-onboarding/SKILL.md)和
[验收](../../.agents/skills/verify-dataevolver/SKILL.md)指导。
新增技能用于解释工作流，执行状态仍复用已有 Controller。

### 第四步：启动 Claude Code

在源码检出目录启动已配置的 `claude` 客户端，让其通过 `compile-plan` 或 `worker-compile`
准备任务。执行使用 `create`、有限步 `step` 和观察接口。模型及原生工作负载仍需遵守用户授权。
