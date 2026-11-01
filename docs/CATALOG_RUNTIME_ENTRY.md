# Catalog and image runtime entry

The existing Harness CLI assembles the Catalog RSI domain and compiled data
graphs from a frozen Goal. The training RSI domain retains its separate model
and authority checks. This entry does not deploy services or grant experiment
permission, candidate slots, model calls, image generation or training.

## Create, execute and recover

Run on the configured execution host, using the matching source and environment:

```bash
PYTHONPATH=src "$HARNESS_PYTHON" -m dataevolver.harness.cli create \
  --goal "$APPROVED_GOAL" --campaign-root "$CAMPAIGNS" \
  --idempotency-key "$CAMPAIGN_KEY"
PYTHONPATH=src "$HARNESS_PYTHON" -m dataevolver.harness.cli resume \
  --campaign "$CAMPAIGN" --campaign-root "$CAMPAIGNS"
PYTHONPATH=src "$HARNESS_PYTHON" -m dataevolver.harness.cli status \
  --campaign "$CAMPAIGN" --campaign-root "$CAMPAIGNS"
PYTHONPATH=src "$HARNESS_PYTHON" -m dataevolver.harness.cli replay \
  --campaign "$CAMPAIGN" --campaign-root "$CAMPAIGNS"
```

`create` freezes the actual INIT consumer and performs version admission without
executing the graph or Planner. `evolve` creates and executes with the same
checks. `step` supports bounded execution. `resume` reconstructs the frozen
runtime from the saved Goal; it does not select the latest policy or operator.
Status, explain and replay remain readable after revocation or configuration
drift. The original registered training example keeps its deterministic mode.

## Catalog configuration

Use `initial_operator_id="data.catalog_trial"`,
`planner.kind="runtime_rsi"` and `planner.runtime.domain_kind="catalog"`.
`planner.runtime` contains:

| Field | Meaning |
| --- | --- |
| `domain` | Existing `CatalogRSIDomain` configuration: serialized `CompiledRun`, source ArtifactRef, baseline/allowed rankings, quotas, target and matched trial key |
| `roots` | Host-bound ArtifactStore roots, including each compiled store |
| `gateway` | Existing ModelProfile and ModelPolicy values, exact receipt root and optional registered worker configuration |
| `planner_role` | Explicit Gateway role; default `agent.runtime_rsi` |
| `max_tokens`, `timeout_seconds` | Bounded Planner request settings; defaults 4096 and 300 |
| `policy` | Optional exact PolicySpec; this declaration alone is not qualification |
| `policy_adoption` | Qualified policy binding described below |
| `step_bindings` | Optional qualified repair registration; absent means the original built-in registry |
| `training_feedback` | Optional already collected ExternalTrainingRunAdapter job and receipt |

Native planning requires explicit registered inference network authority in the
Goal and existing ModelAttemptLimits in `gateway.attempt_limits`, including all
linked historical ledger roots and the shared reservation lock where required.
New receipt directories therefore do not reset native subquotas. Profiles,
model revision, endpoint allowlist, budgets and store boundaries
are frozen. Worker providers are limited to the existing visual and benchmark
providers; configuration cannot import arbitrary Python. A local endpoint
without registered native-effect lookup still fails closed on ambiguous effects.

Catalog policies remain limited to observation order, candidate count and stop
rule. The current Catalog domain has no additional observation tools. A different
parameter value is not evidence of a different action or a better policy.

## Qualified versions and INIT admission

A `policy_adoption` contains `catalog_root`, `expected_head`, `adoption`, and
`qualification`. Campaign references have exactly the host campaign root and
campaign ID. The policy qualification descriptor names the engineering and
independent committed workload campaigns. JSON result summaries cannot replace
those Controllers, effect receipts and frozen Evaluator checks.

The order is fixed:

1. Assemble the declared policy, operators, Gateway and data domain without effects.
2. Create and freeze the actual INIT Controller.
3. Resolve the exact policy through PolicyResolver, including its qualified
   `catalog_policy.v1` context, adoption event and current catalog head.
4. Write `runtime-admission.json` and commit its hash and provenance in the
   existing Controller event log. Operator, Planner and experiment counts stay zero.
5. Execute only after admission. Recheck pinned evidence and current revocation
   before subsequent ticks. Recovery does not call the INIT-only resolver again.

An incomplete admission artifact can be reconciled only while INIT and before
effects. A missing or changed committed admission cannot be silently recreated.
Refusal writes `runtime-admission-refusal.json`; it does not qualify a version or
reset the Goal, counters, time or original experimental costs.

For graphs, use the existing `data.graph` Goal and serialized compiled graph.
`objective.artifact_roots` supplies explicit stores.
`objective.step_bindings.repair` supplies the exact catalog head, committed
adoption, typed repair qualification and candidate context. That context binds
the development plan, CandidatePackage, stores, request origin, actual committed
builder and LinuxPureDataRunner. The entry calls the existing
`qualified_repair_step` and `builtin_step_registry(repair_step=...)`. It does not
turn an unqualified package or hand-written instruction into an adopted operator.

Catalog policy and repair qualifications retain their distinct scopes. A
catalog quality-proxy qualification grants no image-quality or training benefit.

`objective.graph_runtime` optionally reconstructs two existing graph dependencies:

- `process_endpoints` maps a declared local Python profile ID to its exact
  `deployment` and `native_root`. The existing LocalProcessEndpointProvider
  validates model/revision and owns native request recovery. Its journal must
  stay within the graph's writable dataset store.
- `source_gateways` maps an original, disjoint read-only role to `gateway` and
  `contract_hash`. The gateway uses the original profiles, policy, receipt root
  and attempt limits. A changed contract is refused. Existing RoutedModelGateway
  prevents invoking source roles; their confirmed receipts can only be recovered.

These bindings are frozen in the Goal. They do not start a service, copy an old
receipt into a new journal, grant a request or loosen qualification.

## Evidence reader reconstruction

Existing stage operators can be reconstructed through the explicit
`objective.host_workload` descriptor, with a registered `kind` and its original
`configuration`. Supported kinds reuse policy comparison, version mutation,
candidate build, repair comparison and generation benchmark implementations.
Each reader reopens the original frozen campaign and verifies committed effects;
it does not run the source Controller. References to incomplete, unreconciled or
cyclic source campaigns fail closed before constructing their workload. Readers
require the matching frozen source; they do not migrate a historical campaign
to changed code or a different model environment.

Historical version-mutation constructors use an actual catalog prefix only to
reconstruct the original identity. Effect verification uses the complete current
catalog and therefore observes later revocations. These descriptors do not
authorize executing a previously closed experimental path.

## Collected feedback

`training_feedback` contains `roots`, the existing adapter `work_root`, the exact
serialized TrainingJobSpec, and the collected receipt ArtifactRef. Construction
validates the adapter's original job/receipt record. The next Planner context
contains feedback evidence IDs and metrics. There is no prepare/submit/train
operation in this path. Sealed results, missing collection, failed receipts and
artifact drift are rejected. ABSTAIN is recorded as an actual decision; it is not
a sampling change or a training improvement.

## Acceptance boundary

Integration checks use synthetic provider responses and existing qualification
fixtures through the entry, with actual Controllers, graphs, event logs and
recovery. They cover two batches, policy limits/stopping, pinned versions,
revocation, drift, feedback and no duplicate confirmed effects. Synthetic build
or qualification evidence is labelled as such and grants no native qualification.

The historical baseline has 1154 CPU tests at commit `5de4738`. New acceptance
must bind its own final source commit, exact tested files and JUnit artifact;
those historical results cannot be relabelled as the new implementation.
Native independent qualification, method benefit, recursive contribution,
generalization and downstream training benefit require separate evidence.
