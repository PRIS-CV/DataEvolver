# Reusable tools and evidence review

[Home](../../README.md) · [Orchestration](orchestration.md) · [中文](../zh-CN/hyra2-design.md)

[Hyra 2.0's official article](https://hunyuan.tencent.com/research/hyra2), published
2026-11-02, proposes moving from individual answers toward reusable research
tools. Its Harness description and Figure 2 separate research ownership, worker
execution, original-evidence review, project state, agent memory and shared
knowledge. These are design references; DataEvolver does not bundle Hyra code,
models or runtime, or claim its reported research results.

## What fits this base

| Hyra 2.0 design | DataEvolver mapping and scope |
|---|---|
| Build tools that help subsequent tasks | Existing Operators, native Worker Profiles and compiled plans can represent reusable data workflows. A changed tool requires a new version/binding and frozen Goal. A successful example alone does not establish transfer. |
| Separate research decisions from execution | The Workbench main agent can coordinate children; the Campaign Controller owns execution, budgets and recovery. This is not a reproduction of Overall/Mission/Worker project management. |
| Review the original question and evidence in a fresh context | `dataevolver_delegate(mode="review")` freezes an explicit question and selected raw job observations without copying author messages. The reviewer reports findings and limitations. |
| Use different models for different workers | The optional delegation `profile` selects a registered provider/model/thinking from the main session's saved configuration. Omission preserves the child default. |
| Keep project state, working context and reusable knowledge separate | Campaign/job records track execution; Pi history and context snapshots support a session; existing domain qualification and `CapabilityStore` govern reusable capabilities. This review interface does not write or promote capabilities. |

The review mode reuses the child process limit, independent local directories,
context size limit, digest checks and inspection-only bridge. It adds no model
default, endpoint, dependency, Controller or knowledge database.

## Record methods with their conditions

When handing off a candidate Operator, recipe, debugging method or skill draft,
include the following in the child report. This is a reporting convention, not a
new knowledge schema or an automatic admission endpoint.

| Field | What the reviewer needs |
|---|---|
| Method | The changed code, workflow or recipe and its revision/digests. |
| Reason | The failure or cost that motivated the change. |
| Applicability | Task, model/tool versions, data assumptions and resource limits. |
| Evidence | Raw receipt/log/trace paths and observed failures as well as successes. |
| Checks and limits | What the worker inspected, executed or independently reproduced, and what remains unknown. |
| Reuse boundary | Which interface can reuse it and which frozen bindings the caller must regenerate. |

For example, a Blender export repair can produce a candidate Worker Profile and
receipt references. A new review session can inspect the source and artifacts.
Actual rendering, calibration, cross-source qualification and training gains
remain separate claims, each requiring its own authorized evidence.

Hyra's Books distinguish observations, independently reproduced findings and
human-approved conventions. DataEvolver does not add those names as capability
statuses. An execution snapshot is an observation; a model review is an opinion;
reusable `PROMOTED` capabilities require route-specific qualification.
Team conventions require the maintainer's normal review and approval. Model
agreement alone cannot establish any of those higher evidence levels.

Existing catalog qualification is in
[`catalog_knowledge.py`](../../src/dataevolver/harness/catalog_knowledge.py);
training workflow admission is in
[`training_workflow.py`](../../src/dataevolver/harness/training_workflow.py).
Their contracts stay authoritative. `CapabilityStore` storage alone does not
establish qualification. Do not bypass them by writing status fields directly.

Agents may propose reusable methods as skills, but the Workbench still disables
automatic host skill loading. An agent-authored skill is a candidate artifact
for normal source review, not an automatically installed instruction.

## Designs deliberately deferred

Hyra also describes persistent priority/dependency queues, timed wakeups,
Research Board messages, layered agent memory and shared Books. Those are useful
directions for long projects, but adding them would require a new durable
coordination contract. This change retains bounded FIFO dispatch and explicit
recovery: a restart never silently replays an unknown effect. It does not add a
three-tier agent organization, notification service, approval platform or
autonomous compute allocation. Evaluator co-evolution remains out of scope.

The expected benefit is a more traceable handoff and less exposure to author
conclusions before review. Fresh context does not guarantee unbiased reasoning;
the caller supplies the question, native tools retain host permissions, and
referenced files can change. Snapshot hashes identify the supplied context,
not the truth of the evidence or the immutability of referenced artifacts.
Measure candidate quality, cross-task reuse, compute cost and downstream
training benefit separately before making improvement claims.

This adaptation received static review only. We supplied contract tests but did
not run them. We did not run workloads, models, engines or CI, so this review
does not establish runtime readiness.
