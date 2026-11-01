# DataEvolver framework integration

This release shares operator discovery, graph compilation, artifact validation,
effect recovery and feedback across image and structured-catalog workflows.
EvolutionController remains the owner of campaign transitions and budgets.

## Source boundaries

`harness/cli.py` parses commands and prints results. The ordinary factories in
`harness/composition.py` bind the selected route, planner, evaluator and frozen
sources. `build_controller(..., create=True)` creates a campaign;
`open_controller(campaign_root, campaign_id)` reconstructs it from its stored
GoalSpec. Existing private CLI helper imports remain compatibility aliases.
New integrations should import the factories from `composition` directly.

`EvolutionController` owns execution order, pending calls, effect recovery,
budgets, state commits and final-report writes. `harness/final_report.py` projects
the existing evidence into the versioned payload. The recipe adapter interprets
reuse evidence and prepares an optional capability; the Controller still decides
when to execute, persist and complete. The legacy 3D adapter owns its resource
checks and production preflight. Optional implementations keep their own rules.

`DomainRuntime` declares frozen configuration, context and compilation methods.
`RSIDomain` extends that protocol with concrete RSI observation/intervention
return types. `DomainBinding` remains the workflow entry point;
`TransitionProposal` remains the existing mechanism for validated domain state
updates. An adapter does not gain permission to commit Controller state.

The two gateway bridges serve distinct contracts:

- `harness.model_backend.GatewayPlannerBackend` invokes a configured role and
  preserves its `planner-` request identity and configurable limits.
- `adapters.codex_planner.RecoveringPlannerBackend` recovers an RSI request before
  invoking the gateway. It preserves its `rsi-` identity, role and attempt limits.
  The old `GatewayPlannerBackend` import in that module aliases the same class.

Both obtain model execution from the gateway; the configured provider supplies
the transport. Renaming a bridge does not alter its backend ID or receipts.

Serialization helpers retain separate contracts. `harness.models.canonical_json`
normalizes Harness model values and preserves its legacy number handling.
`dataset.contracts.canonical_json` accepts plain JSON and rejects non-finite
numbers. Candidate signatures use escaped ASCII, while candidate recovery binds
strict UTF-8 dataset JSON. Interchanging these encoders changes request or receipt
identity. Streaming file SHA-256 is shared through `dataset.contracts.file_sha256`;
the evaluator retains its string-to-Path adapter and operators retain their old
helper alias. Campaign composition includes that shared source in every route's
frozen execution files.

Keep local profiles under ignored `.dataevolver/`. Bind model roles, tool commands,
evaluation policies and host paths explicitly in the selected profile and goal.
Use the existing native-worker or graph examples below to add a workflow. A
source refactor changes frozen source hashes: existing campaigns need their
original source revision. Receipts and snapshots are not migrated by this refactor.

## Compile and execute

`compile_plan(plan, CompileContext(...))` returns `CompiledRun` or
`CompilationGap`. The context supplies allowed Registry entries, domain bindings,
the execution contract, source bindings and artifact roots. Compilation checks
versions, ports, parameters, model roles and budgets. Binding checks exact
registered implementations again before execution.

```bash
python -m dataevolver.harness.cli operator-sdk catalog
python -m dataevolver.harness.cli compile-plan \
  --plan plan.json --profile profile.json --output compiled \
  --campaign example --evaluator evaluator.json
python -m dataevolver.harness.cli evolve \
  --goal compiled/goal.json --idempotency-key example
```

The existing `operators --campaign ...` command still inspects a campaign.
`operator-sdk init/check/package` creates or checks declarations without
inference, installation, registration or qualification.

`attach_development_context` packages a registered public contract and its exact
spec hash with JSON examples from the protocol's development split. It never
resolves regression, independent or heldout artifacts. The resulting frozen
`OperatorInterface` is passed intact by `CandidateBuildOperator`; an existing
context cannot be overwritten. Packaging checks do not establish real model
consumption. Adding examples requires a new frozen interface and new evidence.

`DomainBinding` combines a compiler with optional runtime, evaluator and delivery
implementations. `LegacyDomainAdapter` preserves existing optional RSI hooks.
Training-workflow completion returns `TransitionProposal`; the Controller checks
the current evaluation and submits it.

Node/graph V3 permits text-only model nodes. V2 retains its positive-image rule.
Graph execution accepts explicitly injected registered step/provider factories.
Factories do not grant production qualification or arbitrary import authority.

## Quality and evolution

Catalog and image workflows use the same `images.families` implementation:
pixel SHA, dHash distance and declared source families, followed by existing
family-aware selection and quotas. Semantic duplicate detection is not performed.

`CatalogRSIDomain` compiles permitted sampling changes. `CatalogTrialOperator`
executes the graph and binds measured quality, source identity, quota and family
coverage. Registered reviews retain separate ModelGateway receipts. Controlled
blur calibration does not establish semantic quality or downstream usefulness.

Catalog M candidates reuse `CandidateBuildOperator`, `CandidatePackage`,
`LinuxPureDataRunner` and `OperatorVersionStore`. Engineering and independent
Controller workloads qualify only `catalog_family.v1`. Resolution requires
committed adoption and current qualification; revocation prevents new dispatch.
Candidate Python never enters the host import path. Equivalent grouping proves
the lifecycle, not a method improvement or production-code promotion.

Catalog K uses `CapabilityStore`: adopted R comparison, conditional distillation,
independent source comparison, then scoped reuse. Insufficient evidence returns
`NO_ENTRY`. Different manifest hashes do not establish source independence:
promotion also requires nonoverlapping declared source families. K-off and K-on
are separate runs.

Planner context includes the exact patch parameter schemas accepted by the
compiler. Experience identity belongs to the top-level action, and capability
requests use the existing structured `capability_gap` contract. Invalid model
output remains a recorded rejection; it does not become an executable action.

## Feedback and strategy

`FeedbackSignal` binds capability, sample-quality, dataset-quality or training
feedback to evidence and scope. `ChangeTarget` identifies recipe, operator,
policy or experience. `PolicySpec` binds strategy configuration into a new Planner
contract. Strategy benefit requires a separate comparison.

`ExternalTrainingRunAdapter` registers an external job, reads its status and
collects a successful receipt bound to dataset, initialization, configuration,
model, seed, evaluation and checkpoint. Development feedback supplies context
for the next data plan. Sealed results cannot return to optimization. Test
fixtures prove receipt validation; actual training requires real external evidence.

## Verification and cost

Run baseline and changed full pytest suites in the same authorized execution environment. Use a
short temporary directory because Unix sockets have a path-length limit.
Preserve baseline failures and skipped-test reasons.

`ModelAttemptLimits` projects subquotas and a deadline from existing ModelGateway
journals, including linked builder ledgers. Cache recovery consumes no new
attempt. Every native failure remains counted. Unknown monetary cost stays unknown.

The CLI inference bridge retains its original planner-only default. Other roles
require an explicit deployment allowlist, included in the worker identity. Role
registration does not weaken request/profile/native-receipt hash binding.

Historical test results belong to their original source revisions. This public
integration has static checks; its tests and runtime workloads have not been run.

Report engineering, controlled quality, M lifecycle, K reuse and training evidence
separately. New training, semantic deduplication, general production adoption,
equal-budget search and strategy-benefit claims need their own evidence.

This integration requires its own test run; historical counts above do not validate the current branch. Native scene integration is described in `NATIVE_SCENE_WORKFLOW.md`.
