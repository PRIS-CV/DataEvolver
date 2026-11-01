# Deployment, models and tools

[Home](../../README.md) · [Documentation index](../../README.md#documentation) · [简体中文](../zh-CN/deployment.md)

Keep deployment choices in private profiles. Select compatible implementations before compiling a goal.

Run command examples from the repository root unless a step changes directory.

## Replace models and tools

### Models: role → profile → provider

| Contract | What you configure or implement |
|---|---|
| `ModelPolicy` | Role-to-profile `bindings`, allowed endpoint origins and call budgets. |
| `ModelProfile` | Explicit model ID/revision, backend, capabilities and environment bindings. |
| `ModelProvider` | `health()`, `invoke(request, request_hash)` and `lookup(request, request_hash)`. |
| `ModelRequest` / `ModelReceipt` | Structured input/output, schema, usage and evidence binding. |

For a compatible served chat/vision model, edit the
[profile template](../../configs/harness/model_profiles.example.json): bind the role to
your profile, set `model` or `model_env`, set `endpoint_env`, and authorize its
origin in `allowed_local_origins`. The `local_endpoint` transport uses the
configured service's `/models` and `/chat/completions` interfaces. Inspect a
completed configuration with `dataevolver models --config <your-config.json>`;
`--check-endpoint` adds a health request without inference.
`model_env` overrides a served model ID. Native checkpoint directories use the
selected adapter's path bindings.

For a different native model implementation, supply a registered provider factory
through `ModelGateway`, with compatible outputs and independent effect lookup.
Changing a model name does not adapt its architecture, weights or execution API.
The reserved `external_api` backend is disabled; its flag does not enable cloud
requests. The optional authenticated Codex transport has its own explicit setup.
See [model contracts](../../src/dataevolver/adapters/model_contracts.py) and
[gateway providers](../../src/dataevolver/adapters/model_gateway.py).
The [gateway guide](../MODEL_GATEWAY.md) also describes optional provider-owned
native accounting. Its journal format and extra measurements stay in the adapter.

Optional image generation/editing, prompt enhancement, review and segmentation
implementations are described in [visual adapters](../LOCAL_VISUAL_MODELS.md).
Their exact model/weight and runtime checks apply when selected. Low-level
multimodal routes accept explicit values or `DATAEVOLVER_<ROUTE>_MODEL` /
`DATAEVOLVER_<ROUTE>_MODEL_PATH`; an unset binding stays unset.

### Tools: declared step or native worker

For a composable data/model step, use `declare_step(...)` and
`OperatorRegistry.register_step(...)`, then bind it in the compiler/runtime
registry. The built-in CLI catalog uses explicit registered implementations.
`operator-sdk init/check/package` scaffolds and checks declarations; it does not
install, register or qualify an operator. For an external process, adapt the
worker ABI and change the private profile used by `worker-compile`.

Blender modeling, simulation, frozen-scene export and UE5 rendering are
[optional native adapters](../NATIVE_SCENE_WORKFLOW.md). A Blender modeling
result must be converted to the declared frozen scene package before UE5 can
consume it. Supply the engine, project, assets and resource isolation on the host.
The base has no globally selected model or tool. Changed code, model bindings or
inputs require a new frozen goal and campaign; retain earlier runs as evidence.

## Project website deployment

The website is an independent [app](../../apps/project-page/README.md). Its source
requires Node.js 22+ and npm; its read-only archive needs no Harness or GPU.

```bash
cd apps/project-page
npm ci
npm run dev
```

Open `http://127.0.0.1:4173/`. `npm run build` creates the static build;
`npm run preview` serves it. See the [deployment guide](../../apps/project-page/deploy/README.md)
for Pages and the optional archive API.

## Local configuration and adapter limits

Use ignored `.dataevolver/local/` files for model profiles, native executable paths,
dataset roots and host-specific settings. Pass credentials through the private
host environment. Public `configs/` files are templates; generated goals and
receipts can contain resolved host paths and belong outside code Git.

The generic model gateway accepts configured profiles and registered providers.
Some bundled training workflows also check specific planner or native worker
identities in [training runtime](../../src/dataevolver/harness/training_rsi_run.py)
and [training supervision](../../src/dataevolver/harness/adapters/training_supervision.py).
For these routes, replacing an implementation can require adapter changes as well
as configuration. A profile edit alone does not adapt a different native model.

For setup and the first task, use [quick start](quick-start.md) and the
[campaign lifecycle](harness.md). For provider and operator registration, use
[development and extension](development.md).
