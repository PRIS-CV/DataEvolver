# DataEvolver

[English](README.md) · [Agent 配置](AGENT_SETUP.md) · [框架说明](docs/FRAMEWORK_INTEGRATION.md)

DataEvolver 是面向数据工作流的开发底座。Agent 提交计划和候选实现。
一个 `EvolutionController` 负责执行、预算、产物校验、事件日志和故障恢复。
`ModelGateway` 通过显式配置绑定模型角色。已注册算子负责调用执行器。

公开版本不选择默认模型或 Agent 工具。部署方提供模型 ID、版本、服务地址、
执行器和资源路径。私有配置与凭证存放在忽略目录 `.dataevolver/local/`。
旧版 VLM 渲染与评审双循环保存在 `codex/legacy-vlm-dual-loop` 分支。

## 开发入口

使用 Python 3.12 和可编辑源码安装。CLI 从仓库读取 `configs/`。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[harness]'
python -m dataevolver --help
python -m dataevolver operator-sdk catalog
python .agents/skills/verify-dataevolver/scripts/check_repository.py
```

基础安装不需要模型权重、模型服务或渲染引擎。源码检查只读取文件，
不导入应用代码，也不执行工作负载。

## 模型和执行器

通过 `ModelProfile`、`ModelPolicy` 和 `ModelGateway` 配置模型、角色、传输方式及预算。
HTTP 传输使用部署方提供的服务模型 ID。本地实现通过已注册适配器接入。
参见[模型配置模板](configs/harness/model_profiles.example.json)。

数据计划通过 `compile-plan` 绑定算子。原生进程通过 `worker-compile` 绑定执行器。
部署方填写[执行器模板](configs/worker.example.json)中的命令、输入、输出和环境变量白名单。
`files` 与 `inputs` 中的相对路径以配置文件目录为基准。
`scene-compile` 保留为兼容别名。
两类任务都由 `create`、`step`、`status` 和 `replay` 驱动同一个 Controller。

图像生成、提示增强、评审、分割、Blender 和 UE5 实现均为可选适配器。
选用适配器时，部署方应配置其依赖、资源和版本。
不同 API 的模型应提供对应适配器。
参见[可选视觉适配器](docs/LOCAL_VISUAL_MODELS.md)和[原生执行协议](docs/NATIVE_SCENE_WORKFLOW.md)。

底层多模态请求通过参数，或 `DATAEVOLVER_<ROUTE>_MODEL` 与
`DATAEVOLVER_<ROUTE>_MODEL_PATH` 环境变量绑定模型。未配置时保持空值。

## 迁移与验证

旧生产 CLI、多模态独立 CLI、批处理脚本及旧启动脚本已退役。
更改代码、模型或资源绑定后，应创建新任务。旧运行记录保留原始哈希。
参见[迁移说明](docs/MAINLINE_MIGRATION.md)。

本次整合已完成静态检查。按用户要求，未执行测试、模型推理或渲染。
算子完成与产物校验不代表视觉质量、物理真实性或训练收益。
Benchmark 探针和缺陷发现不在本次更新范围内。

[项目网站](https://pris-cv.github.io/DataEvolver/) · [早期技术报告](https://arxiv.org/abs/2605.01789)
