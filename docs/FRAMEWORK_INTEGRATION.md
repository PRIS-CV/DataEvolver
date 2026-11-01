# DataEvolver framework integration

This release shares operator discovery, graph compilation, artifact validation,
effect recovery and feedback across image and structured-catalog workflows.
EvolutionController remains the owner of campaign transitions and budgets.

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
