# Candidate orchestration

[Home](../../README.md) · [Workbench](workbench.md) · [中文](../zh-CN/orchestration.md)

The optional Workbench coordinates candidate authoring and execution through the
existing Harness. Its main agent prepares context, delegates proposals, reviews
candidate files, and submits approved work. Each Campaign's `EvolutionController`
retains ownership of its state, budget, evaluation contract, dispatch and recovery.
The standalone CLI and DSH plugin retain their existing interfaces.

## Propose, inspect, execute

Call `dataevolver_delegate` from a main session with explicit task context:

```json
{
  "name": "Alternative data recipe",
  "task": "Read the selected execution observations. Write a candidate recipe and explain its hypothesis, expected effect and required checks. Do not execute it.",
  "mode": "proposal",
  "contextJobIds": ["previous-job-id"]
}
```

Replace the example ID with a retained Workbench job ID. An empty list is valid.
The child uses the configured child Provider and model. Proposal sessions receive
distinct, initially empty local workspaces and artifact directories under private
session storage. They can read the parent workspace as a source reference. The
Workbench does not copy or symlink the source checkout into these directories.

Use `dataevolver_child` to inspect, wait for, or stop the child. Children cannot
delegate again. The main agent reviews the proposed files before preparing a
plan or Worker Profile. Execution continues through `compile-plan` or
`worker-compile`, `create`, and bounded `step`. `resume` immediately executes work.
The Workbench does not automatically adopt files, promote knowledge, or change
the frozen evaluator and budget.

Proposal sessions can invoke only Harness `status`, `explain`, and `replay`
through the Workbench bridge. Image calls and mutating Harness commands are
rejected there. Native agent tools still have the host account's permissions.
**Workspace separation is not an operating-system or security sandbox.**
SSH configuration remains a reference to the configured remote checkout; a local
proposal directory does not create or synchronize a remote workspace.

Omitting `mode`, or setting it to `task`, preserves ordinary delegation with the
shared workspace and existing tool access. Use this mode for explicitly authorized
work that requires the shared workspace.

For local proposal sessions, relative Campaign inspection paths resolve against
the source workspace. Candidate files belong in the proposal workspace. SSH
inspections use the configured remote workspace.

## Review original evidence in a fresh context

Use `mode=review` to commission a separate source/evidence review. Supply the
original question and source or artifact references in `task`. Select raw job
receipts with `contextJobIds`; an empty list permits source-only review.
Do not copy the author's verdict or conversation into the review request.

```json
{
  "name": "Review the proposed data recipe",
  "mode": "review",
  "task": "Original question: does this recipe preserve the declared image constraints? Inspect the candidate file at the supplied path and the selected raw receipts. Report findings, evidence locations, checks performed and limitations. Do not execute it.",
  "contextJobIds": ["previous-job-id"],
  "profile": {
    "provider": "registered-provider",
    "model": "registered-review-model",
    "thinking": "high"
  }
}
```

Replace the IDs and supply the actual candidate path. `profile` is optional for
all child modes. When supplied, it must contain exactly `provider`, `model` and
`thinking`; the provider/model must exist in the main session's saved catalog.
It cannot add an endpoint, credential or model to that catalog. If omitted, the
configured child profile applies. Provider additions require a new main session.

The backend freezes the redacted question, source workspace and selected job
observations in `context.json` under `dataevolver.review_context.v1`. The combined
UTF-8 JSON must fit `maxContextBytes`; the backend rejects an oversized question
before creating a session. The saved file receives the same digest check as proposal
context. The initial prompt points to this file instead of repeating the question.
The backend cannot determine whether the caller supplied the true original
question, or whether the supplied text contains an author's conclusion.

The reviewer gets its own empty local workspace and Pi history, without parent
or sibling messages. Its bridge permits only `status`, `explain` and `replay`;
relative Campaign paths still use the source workspace. The bridge denies image
calls, execution, delegation and child control. Native tools retain
host permissions, and referenced source/artifact files remain live inputs rather
than immutable copies. Record the revision or digests actually inspected.

Inspect the review with `dataevolver_child`, then stop it to release its slot.
A review opinion does not certify independent reproduction, automatically accept
a candidate, or write `PROMOTED` knowledge. Existing task/proposal sessions and
configuration files need no migration. The UI labels review sessions separately.

## Evidence context

`contextJobIds` selects at most 16 distinct jobs. The backend captures their
transport statuses, exit codes, source paths and bounded log tails in a private
`context.json`. Logs are read by byte range; collecting context does not load an
entire engine log. The backend redacts context and checks `maxContextBytes`.
The session record includes its path and a SHA-256 of the saved file bytes;
the backend checks that digest before starting or reopening the agent process.

The snapshot contains execution observations, including failures and unknown
effects. It does not interpret arbitrary log text as a quality score or success
certificate. Records and log fragments can represent a still-running job; review
the current receipts before accepting a candidate. Changing an active session's
context file does not update the context already read by its agent.

The main agent can organize these observations into different hypotheses.
Reusable knowledge continues through `CapabilityStore` and the existing task,
model, version and constraint compatibility checks for `PROMOTED` experience.
Successful execution alone does not establish visual quality, downstream training
benefit, method improvement or generalization.

## Limits and queue

Set `orchestration` in private configuration or **Settings → Agent runtime →
Candidate orchestration**. Existing version-1 configurations acquire this section
with defaults on load; other unknown fields remain invalid.

| Setting | Default | Meaning |
|---|---:|---|
| `maxOpenChildren` | 4 | Child agent processes, including idle children; stop completed children to release slots. Range: 1–32. |
| `maxConcurrentJobs` | 1 | Concurrent Workbench execution jobs. Range: 1–32. |
| `maxQueuedJobs` | 16 | Waiting execution admissions, including requests with pending record writes. Range: 1–256. |
| `maxContextBytes` | 32768 | Serialized evidence context size in UTF-8 bytes. Range: 1024–1048576. |

Mutating Harness commands enter a bounded FIFO queue. `status`, `explain`,
`replay`, and `pause` bypass that queue so inspection and pause requests remain
available. The Controller still determines when a pause can take effect.
Queue overflow returns an error without spawning a process.

Queued requests retain their arguments and deployment configuration snapshot.
Referenced plans and profiles still follow the Harness compilation and freezing
rules. The current Workbench limits govern new admission and dispatch; lowering
a limit does not terminate existing processes. Raising a limit can dispatch
already queued work. These are per-Workbench-process limits, not host-wide GPU
quotas, model API rate limits, or caps on direct CLI and native agent commands.

## Cancellation and recovery

- `POST /api/jobs/{id}/cancel`, also available in the Harness receipt panel,
  cancels a queued job before dispatch. It does not stop a running worker.
- Agent abort or stop cancels that session's queued jobs. Aborting a main session
  also stops its children, as before. Native effects require separate inspection.
- Shutdown cancels pending work. Restarted Workbench inspection reports retained
  queued jobs as `cancelled` and unowned running jobs as `unknown`. The Workbench replays neither automatically.
- An unknown execution or pause effect cancels waiting execution jobs before automatic dispatch continues.
  Future submissions are explicit; inspect Campaign and native receipts first.
- Ending an SSH transport does not prove that its remote process stopped.

## Design basis and limits

[Hyra's official design](https://hy.tencent.com/research/hyra) describes context
synthesis, parallel proposals, fresh execution sandboxes, and resource limits.
The [AI4AI artifacts](https://github.com/Tencent-Hunyuan/Hyra-results/tree/8443f9a9aae1a0b72c79e0b2908c79026b8e1555/AI4AI)
publish final solutions rather than the orchestration implementation. This
integration applies those design ideas with DataEvolver's existing interfaces;
it introduces no Hyra runtime, model requirement, or second Controller.

The expected benefits are fewer candidate file collisions, bounded dispatch, and
traceable feedback. These mechanisms do not prove better candidate quality or
training results. Evaluator co-evolution and benchmark-probe development are
outside this change. Review is static only. We did not run the accompanying
contract tests; runtime behavior and performance remain unvalidated.

[Hyra 2.0](https://hunyuan.tencent.com/research/hyra2) adds a useful emphasis on
reusable tools, original-evidence review, and separate project state, working
context and reusable knowledge. See the [design mapping](hyra2-design.md) for
what this base adopts, what already exists, and what remains outside its scope.
