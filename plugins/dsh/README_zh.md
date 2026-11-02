# DataEvolver 的 DeepSeek Harness 插件

[首页](../../README_zh.md) · [English](README.md)

DataEvolver 是面向训练目标的多模态大模型图像与 3D 数据自动化自进化合成系统。
这个可选 Bundle 让 DeepSeek Harness（DSH）的 Agent 通过 `dataevolver_harness`
调用现有 DataEvolver Harness。DSH 负责对话、模型选择和子 Agent 调度；
DataEvolver 负责 Campaign 状态、算子、预算、回执和恢复。

插件位于 `plugins/dsh/`，使用 DSH 的工具注册表、进程服务和会话沙箱策略。
插件无需独立 Workbench、Pi 或 SoL-Pi，也不另建 Controller 或服务器。

## 从源码安装

使用 Node.js 24.9+ 和兼容的 DSH。插件按
[DSH 提交 ce7073d](https://github.com/deepseek-ai/deepseek-harness/tree/ce7073d51254c26cd63b22a60835b8f948deb447)
中的公开接口编写，对应版本为 `0.2.2-alpha.2`。DSH 处于开发预览阶段；
声明的兼容范围表示接口适配目标，尚未经运行验收。

1. 按 [DSH 官方说明](https://github.com/deepseek-ai/deepseek-harness#run)安装 DSH。
2. 在目标执行主机按 [CLI 安装指南](../../docs/zh-CN/quick-start.md#方式-b手动安装)安装 DataEvolver。
3. 在本仓库根目录初始化或查看内置 Web Profile，并安装插件：

   ```sh
   dsh --profile web --dump-config
   dsh plugin --profile web add ./plugins/dsh
   ```

   第一个命令仅在 `web` 缺失时从内置模板初始化。插件安装使用 Profile 的包管理器。
   插件直接提供 ESM 文件，无需构建或安装脚本。

4. 把[配置示例](config.example.patch.yml)复制到同目录的 `config.local.patch.yml`。
   插件的 `.gitignore` 会排除这个私有文件。
   `command` 可填写已安装的 `dataevolver` 可执行文件，或 Python 可执行文件加 `-m`、`dataevolver`。
5. 使用私有配置启动 DSH：

   ```sh
   dsh --profile web --patch /absolute/path/to/config.local.patch.yml
   ```

打开 DSH 输出的启动链接，在 DSH 中配置 Agent 的 Provider 与模型，再让 Agent
调用 `dataevolver_harness`。调用和结果显示在 DSH 既有会话工具卡中，插件不另建仪表盘。

移除时执行 `dsh plugin --profile web remove dsh-dataevolver`，并移除对应的私有配置。
如需继续查看或恢复任务，应保留 Campaign 文件。

## 配置

| 字段 | 默认值 | 用途 |
|---|---|---|
| `command` | `[dataevolver]` | 可执行文件与固定参数，以 argv 传递。可执行文件填写绝对路径或 PATH 名称。 |
| `workingDirectory` | 会话初始目录或策略根目录 | 进程执行环境中的绝对目录。无 Agent 的调用回退到策略根目录。沙箱边界仍取自会话策略。 |
| `campaignRoot` | `.dataevolver/campaigns` | Campaign 目录可使用绝对路径；相对路径以进程工作目录为基准。 |
| `maxActions` | `1` | 每次 `step` 的默认动作数与上限。 |
| `timeoutMs` | `300000` | 前台调用时限，包含可执行文件查找和沙箱准备。 |
| `graceMs` | `5000` | 传给进程服务的终止宽限时间。 |
| `maxOutputBytes` | `262144` | stdout 和 stderr 各自保留的尾部字节上限。 |
| `forwardEnv` | `[]` | 从 DSH 宿主环境显式传给子进程的变量名。 |

通过 DSH 的 Profile 编辑器或 `--patch` 配置这些字段。覆盖行会替换整份配置，
省略的字段使用 Schema 默认值。插件把 `workingDirectory` 原样传给执行服务，
不会用本机文件系统解析远端路径。

DataEvolver 的模型绑定、评价规则、Blender/UE5 可执行文件、资产和工作流环境
仍在私有计划与 Worker 配置中声明，见[部署](../../docs/zh-CN/deployment.md#替换模型和工具)和
[任务编译](../../docs/zh-CN/harness.md#任务生命周期)。插件没有固定模型 ID、原生工具路径或服务器地址。

DSH 的进程服务默认过滤名称类似凭据的环境变量。如果 DataEvolver 模型配置使用
`api_key_env`，应把对应变量名加入 `forwardEnv`，并在 DSH 宿主环境提供变量值。
配置引用服务地址或模型变量时，也应显式传递这些变量。已列出的变量未设置时，插件报错。
工具参数不接受密钥值，插件也不会自动传递 DSH Agent 的凭据。SSH 私钥仍由 OpenSSH 管理。

## 工具接口

每次请求包含 `operation`，并且只能携带对应操作的字段：

| 操作 | 必填字段 | 可选字段或行为 |
|---|---|---|
| `catalog` | 无 | 调用 `operator-sdk catalog`，可在创建 Campaign 前查看算子。 |
| `compile-plan` | `plan`、`profile`、`evaluator`、`output`、`campaign` | 编译注册的数据图。 |
| `worker-compile` | `profile`、`output`、`campaign` | 编译既有原生 Worker 配置，涵盖 Blender/UE5 路线。 |
| `create` | `goal`、`idempotencyKey` | 可选 `dryRun`；创建并冻结 Campaign。 |
| `step` | `campaign` | 可选 `maxActions`、`dryRun`；动作数受部署配置上限约束。 |
| `status` | `campaign` | 读取已保存的快照。 |
| `explain` | `campaign` | 可选 `limit`；读取近期事件。 |
| `replay` | `campaign` | 从事件重建状态，不执行工作负载。 |
| `operators` | `campaign` | 查看 Campaign 的算子绑定和漂移情况。 |
| `pause` | `campaign` | 在 Controller 动作边界请求暂停。 |
| `resume` | `campaign`、`resumeExecution: true` | 立即调用 `Controller.run()`，可能连续派发工作。 |

`resumeExecution` 表示调用方理解命令语义，不等于获得用户授权。
只有用户已授权连续执行时才能调用 `resume`。
`step` 计算 Controller 动作数，不是图像数量或秒数；一个动作也可能等待原生 Worker 完成。
插件不暴露合并创建与连续执行的 `evolve`，也不通过 `step --goal` 隐式创建任务。

准备好已获授权的 Worker 配置后，可对 Agent 说：

```text
用 dataevolver_harness 的 worker-compile，把我的配置编译到新的准备目录。
创建 Campaign 前先展示编译后的目标。等我授权执行后，使用稳定的 idempotencyKey
创建任务，通过 step 推进一个 Controller 动作，再展示 status 和回执，之后再继续。
```

查看请求示例：

```json
{"operation":"status","campaign":"scene-task"}
```

结构化结果包含 `operation`、`cwd`、`exitCode`、`signal`、`timedOut`、
带截断标记的 stdout/stderr，以及 `receipt`。stdout 完整且为 JSON 时，插件把它解析到
`receipt`；截断或不是 JSON 时返回 `receipt: null`。应同时查看退出信息与回执中的领域字段。
进程返回结果不代表 Campaign 已通过准入或数据质量已改善。DSH 的程序化工具调用可读取同一结构化值。

## 远端执行与恢复

插件使用 DSH 已配置的进程和沙箱服务。可执行文件、工作目录和文件路径均属于这些服务的
执行环境。通过 SSH 执行时，应配置 DSH 配套的远端文件、进程和沙箱 Provider，
并在远端安装 DataEvolver，见
[DSH SSH 文档](https://github.com/deepseek-ai/deepseek-harness/tree/ce7073d51254c26cd63b22a60835b8f948deb447/packages/ssh)。
插件不自行创建 SSH 连接或隧道。在服务器运行 DSH 时，也可直接使用服务器上的本地进程服务。

调用遵循会话沙箱策略。私有 `workingDirectory` 配置不会扩大可写目录，沙箱限制失败时
也不会自动改为无沙箱重试。工作流需要额外可写目录时，应通过 DSH 既有策略配置。
网络和原生工具权限仍属于部署条件。

取消、超时和插件卸载会传给 Provider 管理的进程。终止不能证明远端操作没有副作用，
也不会撤销已经完成的 Worker。中断后先查看 `status`、`explain` 和 `replay`，
再按 DataEvolver 的[恢复契约](../../docs/zh-CN/harness.md#查看暂停与恢复)继续。
插件不自动重试命令。

## 开发与验证状态

`commands.mjs` 负责操作白名单、参数校验和 CLI 绑定。
`index.mjs` 负责注册、配置、沙箱派发和结果收集。
新增业务能力宜扩展现有 DataEvolver Operator 或 Worker，让专用规则留在对应实现中。

[test/commands.test.mjs](test/commands.test.mjs)保留了无效请求、创建身份、动作上限和
连续恢复确认的针对性用例，尚未运行。本插件只完成静态审查；安装、DSH 加载、
本地与 SSH 执行、取消、真实模型和渲染行为均未验收。本轮未运行测试、项目任务或 CI。

插件遵循 DSH 的[Bundle 打包方式](https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/docs/user/develop/basic/publish.md)和
[工具契约](https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/docs/cookbook/adding-a-tool.md)。
DSH 使用 MIT 许可；本独立实现的适配器使用仓库的 Apache-2.0 许可，未收录 DSH 源码或素材。
