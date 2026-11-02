# Development and extension

[Home](../../README.md) · [Documentation index](../../README.md#documentation) · [简体中文](../zh-CN/development.md)

Extend an operator, provider or domain at its existing boundary. Keep state transitions and execution authority in the Controller.

Run command examples from the repository root unless a step changes directory.

## Project Structure

```text
DataEvolver/
├── src/dataevolver/
│   ├── harness/          # Controller, compiler, operator SDK, state and recovery
│   ├── adapters/         # Model contracts, transports and native effect journals
│   ├── dataset/          # Artifacts, graph steps and dataset validation
│   ├── workflows/        # Optional scene, multimodal and HYWorld implementations
│   ├── runtime/          # Asset lifecycle and scene contracts
│   ├── agents/           # Scene feedback/action helpers
│   ├── annotation/       # Review helpers
│   └── tools/            # Research preparation and diagnostics
├── configs/              # Public templates and schemas
├── docs/                 # Interface, deployment and migration guides
│   ├── en/              # English task guides
│   └── zh-CN/           # Simplified Chinese task guides
├── .agents/skills/       # Onboarding and verification guidance
├── .github/workflows/    # Source contracts and website workflows
└── apps/
    ├── workbench/       # Optional launcher, private UI and agent/tool bridge
    └── project-page/    # Independent website and read-only archive API
```

The Workbench launcher is `apps/workbench/launch.sh`; the root `launch.sh` is a
compatibility forwarder. UI and Node dependencies belong to that optional app.
The Harness uses the same public CLI and requires no Workbench installation.

Local deployment files and generated campaigns live under ignored `.dataevolver/`.
Keep model bindings, native executable paths and credentials in that local area.
The files under `configs/` are public templates; select the implementations your
workflow needs before compiling a goal.

## Extension boundaries

| Owner | Extend here |
|---|---|
| [CLI](../../src/dataevolver/harness/cli.py) | Parse commands and print results. Both public commands keep this entry point. |
| [Composition factories](../../src/dataevolver/harness/composition.py) | Bind the goal's operator, planner, evaluator and frozen sources; create or reopen a Controller. |
| [Controller](../../src/dataevolver/harness/controller.py) | Campaign transitions, dispatch, budgets, effect recovery and commits. |
| [Final report](../../src/dataevolver/harness/final_report.py) / [domain adapters](../../src/dataevolver/harness/adapters/) | Project evidence into reports and apply route-specific interpretation. |
| [Plan compiler](../../src/dataevolver/harness/plan_compiler.py) / [operator SDK](../../src/dataevolver/harness/operator_sdk.py) | Domain/step declarations and validated bindings. |
| [Dataset contracts](../../src/dataevolver/dataset/contracts.py) / [artifact operator](../../src/dataevolver/harness/artifact_operator.py) | Artifact identity, receipts and effect verification. |
| [Model adapters](../../src/dataevolver/adapters/) / [scene workers](../../src/dataevolver/workflows/scene/) | Provider transports and tool-specific behavior. |

Add a step or adapter before adding a new runner. Keep model proposals separate
from accepted execution, and keep budget/state changes in the Controller. Declare
versions, ports and side effects so the compiler can reject invalid composition.
Implement recovery alongside execution. Reuse the artifact and receipt contracts
instead of inventing another status format or retry loop.

For a new graph step, use `declare_step` and `OperatorRegistry.register_step` as
the existing [dataset steps](../../src/dataevolver/dataset/catalog_steps.py) do. For a
whole task, implement `execute`, `verify_effect` and `recover`, then register its
`OperatorSpec` and optional [OperatorContract](../OPERATOR_CONTRACTS.md).
Bind a custom workflow through [DomainBinding](../../src/dataevolver/harness/domains.py)
and the existing [plan compiler](../../src/dataevolver/harness/plan_compiler.py); keep its
evaluation policy in its declared evaluator and goal. The
[native-worker example](harness.md#native-worker-a-script-model-tool-or-rendering-process)
is the path for tools that already run as a process.

[Framework integration](../FRAMEWORK_INTEGRATION.md#source-boundaries) maps
these entry points, model backends and serialization contracts. Source changes
produce contract drift for existing campaigns; reopen those campaigns with their
original frozen source rather than rewriting their receipts.

## Action Space

The earlier scene-review adapter retains **24 atomic actions**, defined in
[`scene_action_space.json`](../../configs/scene_action_space.json):

| Group | Count | Changes |
|---|---|---|
| Lighting | 4 | Key light intensity and yaw. |
| Object | 6 | Height, yaw and scale. |
| Scene | 5 | Environment rotation, intensity and contact shadows. |
| Material | 9 | Color, roughness and specular/sheen properties. |
| Camera | 0 | Reserved in this action file. |

These are that adapter's bounded parameter changes. Current Harness actions are
defined by the selected operator contracts and goal; they are not universally
limited to this list. A worker's fixed command is not replaced by model output.

## Migration and verification

The previous VLM render/review dual loop is preserved on
[`codex/legacy-vlm-dual-loop`](https://github.com/PRIS-CV/DataEvolver/tree/codex/legacy-vlm-dual-loop).
The old production and standalone multimodal CLIs, shell batch runners and
bootstrap script are retired. Use the unified Harness CLI. See
[migration notes](../MAINLINE_MIGRATION.md).

Public schema and operator versions identify compatibility; model release names
identify optional implementations. Native worker contracts now bind allowed
environment values by hash. Recompile earlier worker profiles into a new campaign;
see the [worker contract](../NATIVE_SCENE_WORKFLOW.md).

The [source-contract CI](../../.github/workflows/source-contracts.yml) checks source,
configuration and undefined Python names without running workloads. Its local
source checker is:

```bash
python .agents/skills/verify-dataevolver/scripts/check_repository.py
```

The current base received static checks. Its tests, model inference and rendering
have not been executed. Runtime scenarios in the
[verification skill](../../.agents/skills/verify-dataevolver/SKILL.md) require a separate,
authorized run on the intended host. Execution and artifact integrity alone do
not establish visual realism, physical calibration or training benefit.

## Roadmap

### Completed

- [x] Unified campaign CLI, compiler, operator contracts and durable recovery code.
- [x] Explicit model profiles and provider/worker boundaries.
- [x] Optional multimodal steps, native Blender/UE5 workers and research-prior tooling.
- [x] Source-contract CI and bilingual framework documentation.

These checkboxes identify code/documentation present in the base, not completed
native validation or demonstrated learning gains.

### In Progress

- [ ] Validate the integrated base on its intended execution host when authorized.
- [ ] Keep adapter interfaces, frozen bindings and developer documentation aligned.

### Pending

Earlier research directions include reusable review knowledge, real-world data
ingestion, faster reconstruction, reconstruction-model tuning, temporal
review/filtering and general N-object scenes. Their end-to-end acceptance remains
future work. Benchmark probes and benchmark flaw discovery are outside the
current engineering update.
