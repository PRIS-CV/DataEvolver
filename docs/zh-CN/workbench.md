# 私有工作台

[首页](../../README_zh.md) · [快速开始](quick-start.md) · [English](../en/workbench.md)

DataEvolver Workbench 是单用户应用。用户可在网页配置模型与工具、与 Pi Agent
对话，并向现有 Harness 提交命令。工作台与公开项目网站独立运行。

## 启动

1. 在工作台主机安装 Node.js 22.19+、npm、Git 和 Python 3.10+。
2. 在源码目录执行 `./launch.sh`。
3. 如果浏览器未打开，打开终端输出的启动链接。
4. 在 **设置** 注册 Provider 和模型，再选择主 Agent 与子 Agent。
5. 保存配置并新建对话。

启动器创建 `.venv`、安装 `.[harness]`、安装锁文件中的 Node 依赖、取得固定提交的
SoL-Pi 源码并构建前端。依赖的安装脚本均被禁用。首次安装需要网络。
模型权重、GPU 驱动、Blender、UE5 和远端环境由用户准备。
`./launch.sh --no-install --no-open` 复用已准备的依赖，仅输出网页地址。
`--port 3080` 指定监听端口。

默认 Pi 依赖为 `@earendil-works/pi-coding-agent` 1.3.0。锁文件记录兼容的依赖版本。
SoL-Pi 使用 Pi 的公开扩展接口，Pi 源码保持原样。升级运行时属于需要审阅的依赖变更。

| 环境变量 | 用途 |
|---|---|
| `DATAEVOLVER_BOOTSTRAP_PYTHON` | 创建虚拟环境所用的 Python。 |
| `DATAEVOLVER_VENV` | Python 虚拟环境目录。 |
| `DATAEVOLVER_SOL_ROOT` / `DATAEVOLVER_SOL_REVISION` | SoL-Pi 独立源码目录及 Git 提交。 |
| `DATAEVOLVER_WORKBENCH_HOME` | 私有配置、会话与任务日志目录。 |
| `DATAEVOLVER_WORKBENCH_PORT` | 本机 HTTP 端口。 |
| `DATAEVOLVER_WORKBENCH_BROWSER_ORIGIN` | 浏览器的精确 Origin，可用于不同的 SSH 转发端口。 |

服务监听 `127.0.0.1`。一次性启动链接建立 HttpOnly、SameSite Cookie；API 核对 Host
和 Origin。应用使用宿主机账号的操作系统权限，Agent 可以编辑文件并执行主机命令。
工作台不提供操作系统沙箱，也不适合作为公开多用户服务。

## 配置 Agent 与模型

Provider 定义 Pi 协议、服务地址、密钥环境变量名及模型列表。协议可选择 OpenAI Chat
Completions、OpenAI Responses、Anthropic Messages 和 Google Generative AI。
模型 ID、上下文长度、输出上限、推理能力及图像输入均由部署配置指定。
公开 Base 不预选 Agent 模型。

主 Agent 与子 Agent 分别选择 Provider、模型和 thinking。会话内的模型切换在轮次之间
调用 Pi 的 `set_model` 和 `set_thinking_level`。新增 Provider 后需要创建新会话，
因为每个会话保留配置快照。已有 Harness 的冻结目标不随设置变化。

在私有凭据面板输入环境变量名与密钥，暂存后保存。页面仅返回已配置的名称，不返回
已保存的密钥值。空白输入保留旧值；移除按钮在下次保存时删除旧值。
凭据也可来自宿主机环境。删除保存的覆盖值后，宿主机环境中的值仍会生效。
修改凭据后应新建对话；已有 Pi 进程保留原环境。

默认私有目录位于 `~/.dataevolver/workbench/` 下，并按源码目录区分。
`config.json` 保存部署设置，`credentials.json` 保存密钥。配置与回执 JSON 文件的权限为 `0600`，
私有根目录为 `0700`。生成的 Pi `models.json` 引用环境变量。
私有目录、会话日志、Campaign 回执与生成的目标可能包含提示词、数据和主机路径，
不应上传到代码仓库。

完整的[配置模板](../../apps/workbench/config.example.json)使用占位地址和模型 ID。
可在工作台停止时编辑私有 `config.json`，也可使用设置页面。
Pi 的 `cli` 字段接收 Node.js CLI 入口文件路径，不接收 Shell 命令。

## 图像与原生工具

图像生成调用 `/images/generations`；编辑通过 multipart 将图像上传到
`/images/edits`。这两类适配器要求 `b64_json` 响应。提示增强调用
`/chat/completions`，可附带一张图像。
如果 Qwen-Image-2.1 或配套 PE 的部署实现这些协议，可直接配置对应模型 ID。
其他模型也可使用相同接口。仅修改模型名称不会转换原生或专用协议。

调用在输出目录保留意图与回执。网络失败记录为 `unknown`，不会自动重试。
图像工作区的输入路径和输出路径均指工作台主机，混合部署也遵循此规则。
需要让整个数据集流程留在服务器时，应在服务器运行工作台或使用远端 Harness Worker。

原生工具设置包含可执行文件、UE 项目、资产/数据集/输出目录和 Python。
这些设置进入 Agent 的部署上下文。Harness 子进程读取以下环境变量：
`DATAEVOLVER_BLENDER_BIN`、`DATAEVOLVER_UE5_BIN`、`DATAEVOLVER_UE_PROJECT`、
`DATAEVOLVER_ASSET_ROOT`、`DATAEVOLVER_DATASET_ROOT` 和 `DATAEVOLVER_OUTPUT_ROOT`。
Worker Profile 仍需声明实际 argv、输入输出契约与允许的环境变量。
设置不会隐式改写已经冻结的 Worker。

SoL-Pi 默认启用 Action Fusion 和 ObservationPack。Reducer 与在线压缩需要显式启用。
Reducer 可能将日志发给配置的模型；压缩可能调用模型并继续当前任务。
SoL 的对话计划与 Harness 编译计划分别记录。本 Base 不加载上游可选的
SoL-Pi Visual 示例；Agent 可通过 Pi 图像读取工具和所配置的视觉模型查看产物。

## 计划、执行与恢复

Agent 调用 `dataevolver_harness`，传入命令和具名参数。
后端将参数转换成 `python -m dataevolver` 的参数数组。
Harness 页面提供相同命令：

| 命令 | 必需参数 |
|---|---|
| `compile-plan` | `plan`、`profile`、`evaluator`、`output`、`campaign` |
| `worker-compile` | `profile`、`output`、`campaign`、`campaignRoot` |
| `create` | `goal`、`campaignRoot`、`idempotencyKey` |
| `step` | `campaign`、`campaignRoot`、`maxActions`（1–100） |
| `status`、`explain`、`replay`、`pause`、`resume` | `campaign`、`campaignRoot` |

`compile-plan` 消费 Agent 编写的结构化计划。`create` 冻结目标，不派发 Worker。
`step` 限制 Controller 动作数，不代表秒数或图像数。
`resume` 会立即调用 Controller 的连续 `run()`。
暂停在动作边界生效，不会强制终止运行中的 Worker。
修改运行中的 Campaign 前，应阅读 [Harness 生命周期](harness.md)。

Pi 使用长驻 RPC 会话，接受 `prompt`、`steer`、`follow_up`、队列清理与 abort。
完成状态以 `agent_settled` 为准，覆盖压缩后的自动继续行为。
`dataevolver_delegate` 使用子 Agent 配置和显式任务上下文创建真实 Pi 子会话。
`dataevolver_child` 可以查看、等待或停止子会话。子会话不能再次委派。
主会话 abort 会停止活动子会话，但不会撤销原生效果或暂停 Campaign。

浏览器重连会重新读取消息和保留的任务输出。服务重启后，**恢复对话** 重新打开持久化
Pi 会话，不发送新提示词，也不恢复 Campaign。中断任务的传输状态显示为 `unknown`。
继续前应核对 Campaign 状态和原生回执。Pi 对话历史不能保证任意 Shell 命令只执行一次。

## OpenSSH 部署

混合部署在本机运行工作台与 Pi。将执行目标设置为 `ssh`，填写已有的 OpenSSH 别名、
远端绝对源码目录、Python 及工具路径。先在远端源码目录安装 Harness 依赖。
SSH 使用宿主机配置与 ssh-agent，不通过网页上传私钥。
后端启用 BatchMode、严格主机密钥检查，并禁止 Agent 转发。
提交任务前，通过普通 OpenSSH 建立主机信任。
别名列表仅提示主配置文件中的条目；Include 文件中的别名可手动填写。

Harness 命令在远端源码目录执行。目标、Profile、输入输出及 Campaign 路径均指远端主机。
完整产物留在远端，工作台接收命令输出与回执。SSH 断开后远端 Worker 可能继续运行；
系统不会自动重发命令，应通过远端回执核对 Campaign。

全远端部署在服务器运行：

```bash
./launch.sh --no-open
```

在本机建立隧道，替换为自己的 SSH 别名：

```bash
ssh -N -L 3080:127.0.0.1:3080 your-workstation
```

在本机浏览器打开服务器输出的启动链接。两端宜使用相同端口；如端口不同，
设置 `DATAEVOLVER_WORKBENCH_BROWSER_ORIGIN` 为精确的转发地址。
Provider 地址需要能从工作台主机访问。混合部署中的远端模型若只监听服务器回环地址，
需要单独建立模型端口转发。工作台不自动部署推理服务或创建网络隧道。

## 扩展与审阅

后端分别保存配置、Pi 会话、任务、图像 API 与 HTTP 路由职责。
Pi 扩展将这些接口注册为工具。React 前端消费 JSON 与 SSE，不持有服务端凭据或模型客户端。
Campaign 的扩展继续使用现有 Python Operator、Provider 和 Domain 接口。
公开项目网站保留独立的只读契约。

CLI 到网页的布局与凭据设计参考
[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)。
DataEvolver 保留独立界面，不引入 DSH 的 Cordis 插件运行时。
来源见[第三方说明](../../apps/workbench/THIRD_PARTY_NOTICES.md)。

本次集成只经过静态审阅。为检查类型，本轮安装了禁用生命周期脚本的 Node 依赖，
但没有执行启动器。本轮没有执行完整安装流程、网页交互、Agent 调用、SSH 任务、
原生渲染、测试或 CI。随代码附带的契约测试留待后续授权时运行；
源码与类型检查不能替代运行验收。
