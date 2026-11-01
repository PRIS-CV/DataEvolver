# Planning, execution and recovery

[Home](../../README.md) · [Documentation index](../../README.md#documentation) · [简体中文](../zh-CN/harness.md)

Choose a compile entry point, create a campaign, and advance or inspect it through the common Controller interfaces.

Run command examples from the repository root unless a step changes directory.

## Execution model

A **campaign** is one durable run bound to a `GoalSpec`: its objective, permitted
operators, budgets and evaluator. One `EvolutionController` owns its transitions,
dispatch and recovery. Operators return artifacts and receipts; the evaluator
checks the declared constraints before the Controller accepts a result.

See the [animated architecture](../../README.md#pipeline-overview). Solid arrows
show plans, calls and recorded results; dashed arrows show recovery and inspection.

## Key Features

- **One execution owner:** the Controller manages state, dispatch, budgets and recovery.
- **Explicit composition:** the compiler checks operator versions, ports and model-role budgets.
- **Replaceable providers and tools:** profiles bind host-owned implementations without a global model default.
- **Recorded effects:** artifact hashes and native receipts support verification and reuse.
- **Separate acceptance evidence:** execution, visual quality and downstream utility have distinct checks.

## Task lifecycle

Choose a compilation entry, then use the common campaign commands below. These
examples are instructions for a configured execution host; the worker template
requires your executable and inputs before it can be compiled.

### Native worker: a script, model tool or rendering process

Start with the [neutral worker template](../../configs/worker.example.json):

```bash
mkdir -p .dataevolver/local/task
cp configs/worker.example.json .dataevolver/local/task/worker.json
```

Edit `stage`, `argv`, `files`, `inputs`, named `outputs`, timeout and resource
permissions. `files` and `inputs` resolve relative to the profile file. Include
worker code and dependencies that must be hash-bound. The executable must accept
the configured request/output arguments and implement the
[request/result contract](../NATIVE_SCENE_WORKFLOW.md#worker-contract), including
`result.json` and its request digest. An arbitrary executable needs an adapter
before it can use this contract.

```bash
dataevolver worker-compile \
  --profile .dataevolver/local/task/worker.json \
  --output .dataevolver/local/task/prepared \
  --campaign task-v1 --campaign-root .dataevolver/campaigns
```

`worker-compile` binds the executable, code and inputs by hash and writes
`goal.json`. It performs no inference or rendering. `scene-compile` is its
compatible alias. The prepared output directory must be new.

### Dataset graph: compose registered steps

Alternatively, prepare a structured `plan.json`, a binding `profile.json` and an
`evaluator.json`, then compile them into the same prepared directory:

```bash
dataevolver compile-plan \
  --plan .dataevolver/local/task/plan.json \
  --profile .dataevolver/local/task/profile.json \
  --evaluator .dataevolver/local/task/evaluator.json \
  --output .dataevolver/local/task/prepared --campaign task-v1
```

The Python interface is `compile_plan(plan, CompileContext(...))`. It checks
registered domains, operator versions, ports, parameters, model roles and budgets,
and returns a `CompiledRun` or a `CompilationGap`. The CLI writes
`compiled_run.json` and `goal.json` only on success; a gap returns
`compiled: false` and exit code `2`. Resolve missing bindings before execution.
This compiler consumes structured plans; the outer agent translates the user's
request into that structure. See the [framework guide](../FRAMEWORK_INTEGRATION.md)
and [compiler source](../../src/dataevolver/harness/plan_compiler.py) for the contract.

### Create and advance either kind of task

```bash
dataevolver create --goal .dataevolver/local/task/prepared/goal.json \
  --campaign-root .dataevolver/campaigns --idempotency-key task-v1
dataevolver step --campaign task-v1 \
  --campaign-root .dataevolver/campaigns --max-actions 8
dataevolver status --campaign task-v1 --campaign-root .dataevolver/campaigns
```

`create` initializes and freezes the campaign without dispatching its worker.
`step` calls `EvolutionController.tick(max_actions)` under a campaign lease. Its
limit counts **Controller state actions**, not images, model calls or seconds;
one action can wait for a native worker. Task budgets and worker timeouts still
apply. Continue with bounded steps until the recorded state is terminal or paused.
`evolve --goal ... --idempotency-key ...` combines creation with `Controller.run()`
for continuous execution under the same frozen contract.

## Inspect, pause and recover

All campaign commands accept `--campaign` and `--campaign-root`:

| Interface | Behavior |
|---|---|
| `status` | Read the current snapshot without loading the execution runtime. |
| `explain --limit 20` | Read the snapshot and recent events, including after runtime drift. |
| `replay` | Reconstruct state from events and report `matches_snapshot`; no workload execution. |
| `pause` | Request a pause at a Controller action boundary; an in-flight worker is not forcibly stopped. |
| `resume` | Restore the paused state and **call `run()` immediately**, continuing execution. |

Use `step --max-actions ...` for bounded advancement. A changed frozen contract
causes `PAUSED_CONTRACT_DRIFT`; `resume` leaves that pause in place while the drift
persists. Restore the original binding to continue that run, or create a new goal
and campaign for an intentional change.

The campaign directory retains `goal.json`, `frozen_contract.json`,
`events.jsonl`, `state.json` and, when finalized, `final_report.json`, together
with operator evidence. After an interruption, inspect it and step the same
campaign: a planned call is routed through `Operator.recover()`, then
`verify_effect()`. A verified receipt can be reused; re-execution requires an
explicitly safe recovery result. Native workers also check their retained
request, intent, log, exit, result and output hashes. A native intent without
verified completion blocks automatic re-execution and requires inspection and
reconciliation. Preserve those files when diagnosing a failure.

Recovery depends on each adapter's independent effect lookup. The Harness cannot
guarantee exactly-once execution for every external service. See
[`EvolutionController`](../../src/dataevolver/harness/controller.py),
[`StateStore`](../../src/dataevolver/harness/state_store.py) and
[`ArtifactWorkloadOperator`](../../src/dataevolver/harness/artifact_operator.py).
