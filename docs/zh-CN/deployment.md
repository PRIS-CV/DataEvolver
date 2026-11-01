# 部署、模型与工具

[首页](../../README_zh.md) · [文档目录](../../README_zh.md#文档导航) · [English](../en/deployment.md)

把部署选择放入私有配置。编译目标前，先选择兼容的模型与工具实现。

除明确要求切换目录的步骤外，命令示例均从仓库根目录执行。

## 替换模型和工具

### 模型：角色 → 配置 → Provider

| 契约 | 配置或实现内容 |
|---|---|
| `ModelPolicy` | 角色到 Profile 的 `bindings`、允许的端点来源、调用预算。 |
| `ModelProfile` | 明确的模型 ID/版本、后端、能力和环境变量绑定。 |
| `ModelProvider` | `health()`、`invoke(request, request_hash)` 和 `lookup(request, request_hash)`。 |
| `ModelRequest` / `ModelReceipt` | 结构化输入输出、schema、用量和证据绑定。 |

替换协议兼容的对话或视觉模型服务时，修改[配置模板](../../configs/harness/model_profiles.example.json)：
将角色绑定到所选 Profile，设置 `model` 或 `model_env`、`endpoint_env`，并在
`allowed_local_origins` 中授权端点来源。`local_endpoint` 使用所配置服务的 `/models` 和
`/chat/completions` 接口。填写配置后，可用 `dataevolver models --config <your-config.json>`
查看绑定；加上 `--check-endpoint` 会请求健康状态，不做推理。
`model_env` 覆盖的是服务端模型 ID。原生检查点目录使用所选适配器的路径绑定。

替换原生模型实现时，需要通过 `ModelGateway` 注册 Provider 工厂，提供兼容的输出和独立的副作用查询。
修改模型名称不会适配其架构、权重或执行 API。预留的 `external_api` 后端处于禁用状态，
修改标志不会启用云端请求。可选的已认证 Codex 接口有单独的显式配置。
实现见[模型契约](../../src/dataevolver/adapters/model_contracts.py)和[网关 Provider](../../src/dataevolver/adapters/model_gateway.py)。
[网关指南（英文参考）](../MODEL_GATEWAY.md)也说明了可选的原生用量审计接口。
原生日志格式及附加统计由对应适配器负责。

可选的图像生成/编辑、提示增强、审查和分割实现见[视觉适配器（英文参考）](../LOCAL_VISUAL_MODELS.md)。
选用后仍需满足该实现的模型、权重和运行时检查。底层多模态路由接收显式参数或
`DATAEVOLVER_<ROUTE>_MODEL` / `DATAEVOLVER_<ROUTE>_MODEL_PATH`；未绑定时保持未配置状态。

### 工具：注册步骤或原生 Worker

需要组合的数据/模型步骤使用 `declare_step(...)` 和 `OperatorRegistry.register_step(...)`，
再绑定到编译器及运行时的 Registry。内置 CLI 目录使用显式注册的实现。
`operator-sdk init/check/package` 只生成和检查声明，不负责安装、注册或资格验收。
外部进程通过 Worker ABI 适配，再修改供 `worker-compile` 使用的私有配置。

Blender 建模、仿真、冻结场景导出和 UE5 渲染属于[可选原生适配器（英文参考）](../NATIVE_SCENE_WORKFLOW.md)。
Blender 建模结果需要转换成规定的冻结场景包，才能交给 UE5 使用。引擎、工程、资产和资源隔离由宿主环境提供。
基础版本没有全局指定的模型或工具。修改代码、模型绑定或输入时，需要新的冻结目标和 Campaign，原运行保留为证据。

## 项目网站部署

网站是独立的[应用（英文参考）](../../apps/project-page/README.md)，源码开发需要 Node.js 22+ 和 npm。
只读归档无需 Harness 或 GPU。

```bash
cd apps/project-page
npm ci
npm run dev
```

打开 `http://127.0.0.1:4173/`。`npm run build` 生成静态构建，`npm run preview` 提供预览服务。
Pages 与可选归档 API 的配置见[部署指南（英文参考）](../../apps/project-page/deploy/README.md)。

## 本地配置与适配器限制

模型配置、原生工具路径、数据集根目录和宿主设置放在已忽略的 `.dataevolver/local/` 下。
凭证通过私有宿主环境提供。公开的 `configs/` 文件用于提供模板；
生成的目标与回执可能包含解析后的宿主路径，应保留在代码 Git 之外。

通用模型 Gateway 接受配置的 Profile 和注册 Provider。部分内置训练工作流还会在
[训练运行时](../../src/dataevolver/harness/training_rsi_run.py)和
[训练监督适配器](../../src/dataevolver/harness/adapters/training_supervision.py)中检查特定的规划器或原生 Worker 身份。
替换这些路线的实现时，可能需要同时修改适配器和配置。只修改 Profile 不能完成原生模型架构适配。

环境准备与首个任务见[快速开始](quick-start.md)和[任务生命周期](harness.md)。
Provider 与算子注册见[开发与扩展](development.md)。
