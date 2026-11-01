# Training-goal planning

Stage0 `plan-goal` translates a training request into a supervision contract and a research-backed production plan. The training contract is independent of Harness execution `GoalSpec`. A later compiler must preserve media roles, variable subject lists, video observations and source-family relationships before it can dispatch existing Controller operations.

## Request and outputs

Pass a JSON request with `request`, `constraints`, `materials`, `task_hint` and `task_config`, or a text request with separate `--materials` and `--task-config` files. The three examples under `configs/training_goal/` are acceptance configurations, not global templates. Change counts, scene requirements, materials and desired supervision there.

Each material has an immutable ID and an origin of `user_provided` or `user_declared`. A declared material must set `content_inspected=false`; a placeholder does not establish image readability or identity. `goal.json` retains the original request, materials and task configuration separately from the Planner's inferred assumptions.

The same validated canonical `plan.json` produces four files:

| File | Handoff |
| --- | --- |
| `goal.json` | Original request, material provenance, constraints, training input/target and assumptions |
| `research.md` | Fetched source versions, design changes and concrete external operator adaptations |
| `dataset_plan.yaml` | Dependency graph, typed media ports, bindings, model roles, checks, splits and a placeholder record |
| `todo.md` | Gaps, dependencies, implementation tasks, deliverables and acceptance criteria |

`dataset_plan.yaml` uses the JSON subset of YAML 1.2. It can be loaded by either parser. Sources and model output remain untrusted until semantic validation succeeds. External implementations are registered as `candidate`; their source inspection does not establish runtime readiness.

## Execution boundary and evidence

Use an already authenticated private CLI instance through `--planner-instance` or `DATAEVOLVER_PLANNER_INSTANCE`. The wrapper sets the child process's `CODEX_HOME` and per-invocation options. It leaves instance configuration and credentials unchanged. The model is fixed to `gpt-6.1-sol`.

The CLI runs with native search, `--json` events and `--output-schema`. Shell execution and delegation tools are disabled. A stdlib-only MCP bridge exposes only `read_source`, `query_operator_catalog` and `query_experiences`. A local protocol preflight checks bridge startup before a paid Planner turn; it is not a real-model acceptance.

Each output directory retains the request, canonical plan, schema, invocation, CLI event stream, source contents and SHA/version receipts, tool trace, experience snapshot, code provenance and status. Mutable GitHub raw source links resolve to a commit; if resolution fails, the failed read remains in the trace. A source successfully read at a full commit can still support the plan.

User constraints are preserved verbatim. Nodes bind `node:<id>.<port>`, `material:<id>`, `goal:<field>` or `config:<field>`. `config:task_config` binds the complete configuration. Validation rejects cycles, missing ports or fields, missing materials, unverified executable operators and capability gaps without TODOs.

Placeholder subject examples bind each reference ID to a unique target instance. Video examples include initial-frame and action inputs, a video target, expected events, observed events and evidence. Target masks and review results stay outside model input. These examples describe a record format; they are not generated training samples.

## Experience input

`--experience-store` reads existing `CapabilityStore` data without appending or promoting records. Each compatible entry must be `PROMOTED`, have validation evidence, and match the task, model identities, frozen `task_config.model_versions`, operator/version, allowed change fields and explicit user constraints. The planning scope is stored in `intervention.parameters.planning_scope`; it includes `task`, `models`, `model_versions`, `operator`, `operator_version`, `compatible_constraints` and `changes`.

Only `parameters.*` or `checks` changes are eligible. An adopted experience cites its ID/version, target node, applicability and before/after values, and must change that actual node or check. Candidate, retired, incompatible and constraint-conflicting entries are excluded with reasons. If model versions are not frozen, validated experience reuse remains unavailable. No configured store and no compatible record are recorded as different conditions.

## Recovery and acceptance

Failed login, tool reads, invalid output, timeout and cancellation retain partial evidence. The wrapper does not automatically resubmit. Inspect the recorded process and terminal state before using `--resume` in the same output directory. Resume reuses the existing CLI thread. A changed request, contract, catalog or experience snapshot requires a separate directory. Completed runs validate their canonical plan and artifact hashes, then return without invoking the model again.

`--revalidate` checks a retained response after a local validator repair and makes no model call. An explicit `--revise --revision-note-file review.txt` continues research in the original thread, archiving the previous plan and artifacts first. CLI token usage is retained where returned; unavailable billing and interrupted-turn usage remain unknown.

Three real acceptances exercise the same command: scene decomposition without materials; action-conditioned physical video; and independent-reference composition with declared background plus three references. Compare nodes, supervision sources, input/output connections and TODOs. A title change alone does not establish material-aware planning.

The next phase consumes the generated TODO for the multi-subject production chain, then video production. Its own acceptance requires actual media, request receipts, quality evidence and independent loading. Planning tests and source inspection do not establish dataset quality, RSI benefit or downstream training gains.

[Training-data production compiler](TRAINING_DATA_PRODUCTION.md)
