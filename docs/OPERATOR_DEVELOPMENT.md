# Operator development and qualification contracts

`harness/operator_development.py` represents development plans, inert candidate
packages and the evidence required to assess them. The registered
`evolution.inspect_operator_candidate:1` adapter executes through the existing
EvolutionController. It does not create a second execution controller, run
candidate code, install dependencies, change the registry or promote a version.

## Frozen inputs

An `OperatorDevelopmentPlan` binds the original runtime capability request,
its campaign and frozen execution identity, the intended OperatorContract hash,
an isolated candidate store, protected stores, a qualification protocol and a
finite total budget with prior usage. Its resource deadline is explicit and is
not refreshed by parsing, inspection or recovery. These declarations do not
grant budget or model authority. Actual execution still requires an authorized
Goal, the existing Controller's checks and ModelGateway admission/accounting.

The protocol declares disjoint development, regression, independent and heldout
source families and content hashes. Independent qualification needs at least
two predeclared families. Each qualification stage pins its approved validator
OperatorSpec hash, evaluator hash and required checks. A matched-context hash
binds the separately frozen model, R/K starting state, source pool, incumbent
and online-budget comparison design. Heldout sources are not qualification
inputs. Inventory inspection may hash their files; their contents must not be
given to candidate construction or tuning.

A `CandidatePackage` binds its development plan, a closed file inventory, one
implementation entrypoint, OperatorSpec, OperatorContract, pinned dependency
inventory and change description. The manifest lives outside the candidate
store so it does not hash itself. Every candidate file belongs to the isolated
store; extra files, changed content, symlinks, hard links and overlapping stores
are rejected. The implementation source path in the inert spec is relative to
the candidate store. It is not passed to CommandOperator or imported. Package
validation does not establish that the implementation meets the capability
request. That requires actual qualification.

These inventory checks enforce application-level file boundaries; they do not provide
OS isolation. An execution runner must enforce its own approved process,
filesystem, dependency, network, model and budget boundaries before executing
candidate code. This inspector must not be reused as that runner.

## Evidence authority

Evidence readers come from trusted host configuration, never candidate JSON.
`ControllerRequestAuthority` rechecks the original frozen contract and snapshot
replay, then requires the exact request hash/path in the original terminal
Controller state and `RSI_CAPABILITY_REQUESTED` event. The request itself remains
non-executing evidence. Snapshot inspection does not repair or overwrite a
stale snapshot.

`FrozenSourceRequestAuthority` supports historical campaigns whose original
Controller version differs from the current development source. It loads the
original config's trusted source and environment in a separate interpreter,
then loads only the current host's request-verification contracts. The original
Controller performs frozen/replay verification. It does not tick, resume or
invoke a model. Its interpreter, env.sh and reader hashes are pinned in the
inspection operator identity. Venv entrypoints retain their original path.
Candidate stores never enter this interpreter's import path.

`ControllerQualificationAuthority` accepts only a completed frozen campaign
with matching event replay. The Goal must already bind the plan, package,
protocol, stage and matched context before execution. A unique planned and
committed call must identify the approved validator and successful receipt.
The validator re-verifies the actual effect; artifact hashes must still match
the Controller commit. The frozen evaluator must pass, and check results,
predeclared families and finite usage must match. Independent qualification
requires a separate frozen campaign after engineering evidence. Both stages
and prior usage are charged against one declared lifecycle ceiling. Candidate
inventory bytes are counted once in addition to stage-local output usage.

The trust boundary is the host-approved validator implementation and its
effect verifier. It must verify real checks, protected metrics, matched
comparisons, source bindings, cost journals and native receipts. An arbitrary
CommandOperator that merely writes `verified: true` is not a suitable
qualification validator. The package cannot name a new evidence authority or
replace the protocol's approved validator/evaluator. Model inference, if needed
by such validators, continues through the existing ModelGateway.

## Results and recovery

Inspection without qualification receipts reports `STRUCTURE_VERIFIED`, with
engineering and independent qualification `NOT_ASSESSED`. It explicitly leaves
builder verification, promotion, reuse and method improvement false. A
qualification result can report only the stages whose original Controller
evidence was re-verified. Qualification does not activate the candidate.

The inspection adapter binds the exact call and spec to its durable output.
Recovery rechecks the package, original request and qualification evidence. A
crash after the output but before the receipt can recover the same effect.
Missing output with a prior receipt, changed evidence or output drift is
ambiguous and cannot be overwritten by a fresh result. Finished source
campaigns remain read-only.

Runtime candidate construction can produce inert packages as described below.
The narrow isolated check runner is described below. Production qualification protocols, new frozen-contract activation, promotion/revocation
and subsequent actual reuse remain separate lifecycle work. CPU fixtures or
an inspection of real request files establish neither operator evolution nor
image quality, K adoption, method superiority or downstream training benefit.

## Generating a candidate before implementation hashes exist

`OperatorDevelopmentPlan` v1 still binds an already existing exact contract.
`OperatorDevelopmentPlanV2` instead pins an `OperatorInterface` artifact before
source generation. The interface contains the requested public declaration and
fixed retry/timeout authority, and has no implementation hash. After generation,
the builder computes the actual source hash, creates an inert OperatorSpec and
binds the OperatorContract to that exact spec. V2 inspection checks both the
frozen interface and the generated implementation. V1 hashes and parsing keep
their original meaning.

The registered `evolution.build_operator_candidate:1` takes a v2 plan and a
host-configured ModelGateway with one bounded attempt. It sends only the
general capability request and frozen interface to the model. Task media and
independent/heldout source contents are not sent. The response schema allows
bounded relative source files, one Python `resolve(payload)` entrypoint, an
empty standard-library dependency inventory and a change description. The
builder supplies spec/contract/manifest metadata; model text cannot supply
runtime argv, environment keys, write roots, budgets or qualification claims.

One plan owns one durable builder-call identity. Changing a call ID cannot
reset its journal or the shared ModelGateway attempt ceiling. A completed model
effect can be materialized after a crash using the same output and receipt;
an unresolved native attempt is ambiguous and cannot be regenerated. A
completed response with invalid paths, dependencies or output sizes produces
terminal rejection evidence and preserves the model cost. No candidate source
is imported, tested, installed or executed by this builder.

The build result is `BUILT_UNQUALIFIED`. Its native audit, request, receipt,
inventory and inspection are independently rechecked by `verify_effect`.
Generated source bytes are bounded before writing. Operational model/native
audit files are retained as execution evidence. A build is not engineering or
independent qualification and does not establish that the code satisfies the
request. An execution-ready qualification protocol and approved isolated runner
remain prerequisites for running and adopting that implementation. A design-only
qualification protocol cannot provide executable validator evidence.

The entrypoint accepts `file.py` and the explicit `file.py:resolve` notation.
Only that fixed callable name is accepted; normalization changes metadata, not
source bytes. A historical response rejected by an older parser stays rejected
in its original frozen campaign. `ControllerBuildOutputAuthority` can verify its
committed effect, actual completed model receipt, native inventory and costs.
The registered CPU-only `evolution.recover_operator_candidate:1` can then
materialize those exact bytes in a new isolated store and frozen contract. The
new plan retains prior model/operator/wall usage, permits no new model, planner
or image calls, preserves the resource deadline and adds only the approved CPU
repair step. Verification reads existing evidence without restoring missing
candidate files. Recovery still yields an unqualified, unexecuted candidate.

## Isolated pure-data behavior checks

`candidate_sandbox.py` and the standalone trusted bootstrap implement the
`linux-chroot-uid-seccomp-pure-data-v1` profile. This profile requires Linux
x86_64, a root bootstrap capable of chroot, and a pinned libseccomp runtime.
It fails closed when these prerequisites are unavailable; it never falls back
to importing a candidate in the host process. No user namespaces or new model
services are required.

Only bounded source text and JSON enter a fresh `Python -I -S` process with a
minimal environment. The trusted bootstrap preloads its runtime, limits address
space/CPU/core/file/process resources, closes stdin and ambient descriptors,
chroots into an empty directory, drops all UID/GID identities and supplementary
groups, disables dumpability, and sets no-new-privileges. It then installs a
native-architecture default-kill seccomp filter before any candidate statements
execute. File opens, directory access, networking, process creation/execution,
ptrace/process-memory/IPC and policy changes are absent from the allowlist.
Writes are limited to bounded stdout/stderr pipes. A separate bootstrap pipe
records confinement before being closed; the candidate cannot provide its own
isolation attestation. The parent bounds wall time and captured output and
terminates only its own child process group.

The profile permits only the preloaded standard-library dependency set and
anonymous-memory, signal, clock and random primitives. It has no filesystem,
dependency installation, credentials, model gateway or GPU interface. It is a
limited pure-data execution boundary, not general Python application support,
a VM, or a claim to eliminate interpreter/kernel vulnerabilities. Kernel syscall
filtering is one layer of this combined boundary, not a complete sandbox alone.
See the [kernel seccomp documentation](https://docs.kernel.org/userspace-api/seccomp_filter.html)
and the [libseccomp API declarations](https://github.com/seccomp/libseccomp/blob/main/include/seccomp.h.in).

`evolution.check_operator_candidate:1` runs predeclared protected JSON cases
through this profile under the existing EvolutionController. A separately
frozen behavior-check protocol binds the unchanged original plan/package,
runner identity, expected output subsets and finite inherited CPU budget. It
cannot increase model/Planner/image allowance or extend the resource deadline.
Every case retains captured streams, termination status, runtime/source/input
hashes and wall cost. New call IDs cannot reset its single-execution journal.
Completed case outputs recover without re-execution; an interrupted case with
no durable result is ambiguous and cannot be resent. Verifiers only read
captured results and immutable inputs, never rerun candidate code.

These are engineering behavior checks. A negative-only implementation can pass
refusal checks while failing the frozen evaluator's positive-capability gate.
No production engineering/independent qualification, promotion, registry
activation, reuse, image-quality or method-benefit claim follows. A design-only
qualification protocol is still design-only; these checks do not rewrite it.
Missing deployment trust provisioning must be resolved through a new explicit
interface/admission, not by editing generated source or injecting undeclared
globals to make a positive check pass.

`operator_readiness.py` provides a read-only authority for an exact committed
check invocation, including terminal frozen/replay verification and the original
operator's effect verifier. `CandidateReadinessDomain` exposes the immutable
general capability request, generated source/contract/documentation and actual
captured behavior to the existing RuntimeRSIPlanner. Its action space is only
REQUEST_CAPABILITY or ABSTAIN. Every new request must cite the committed checks;
it cannot execute observations/patches, provision trust, claim qualification or
activate a candidate. The caller must explicitly retain prior lifecycle costs
and allocate any review decision from the remaining authorized text budget.

## Explicit host trust interface and scoped qualification

`candidate_trust.py` defines `CandidateTrustContext` v1. It binds the original
package/source, the committed runtime qualification request, an exact input
artifact and canonical payload digest, a synthetic fixture family, and an
independently controlled public RSA registrar/current-manifest snapshot. Its
only namespace is `TRUSTED_REGISTRARS`; its only scope is
`synthetic_authority_contract`. Algorithm constants, private signing keys,
arbitrary globals and production trust cannot be supplied through it.

The opt-in `TrustedContextRunner` v2 verifies these source/input bindings and
retains the same OS confinement and resource policy. The trusted bootstrap
loads unchanged candidate source, applies this explicitly declared public
configuration to the candidate module, and calls `resolve`. Candidate payloads
cannot select or override the host context. Configuration is part of the new
execution contract and captured hash binding; this is not a repair of an old
frozen execution or a change to generated source bytes. The ordinary v1 run
interface still has no context parameter.

A Python namespace is not an integrity boundary against the candidate program
itself. The candidate could ignore its configuration or fabricate outputs. The
trusted parent therefore grants no authority on that basis: it compares actual
outputs against an independently frozen host oracle, and downstream Controller
and production checks remain authoritative. No private signing material enters
the candidate process, and no fixture snapshot is eligible for production use.

`evolution.qualify_operator_contract:1` is a registered evaluation operator.
Its new protocol binds the existing inert package/plan, runtime identity, the
actual runtime qualification request, positive and adversarial input files,
separate expected results and host context snapshots, plus inherited finite
usage/deadline. `CapabilityRequestBinding` verifies request provenance through
the existing authority using a temporary request-only view; it does not rebind
the original package to another development plan.

The host oracle verifies canonical signatures with the separately pinned
`cryptography` implementation, exact support/crop/inverse projection, receipt
digests and operator identity, witness provenance, prior-support changes,
protection/uncertainty failures and preserved pending image-validation gates.
It does not import candidate code or derive expected support from its output.
The independently specified test-vector producer is in
`testing/authority_contract_vectors.py`. Vectors include sparse masks on a
standard 1024-square canvas as well as small aperture/surface cases; canvas
extent must not be confused with the number of explicitly supplied mask pixels.
The corpus is frozen before candidate execution. A failure never authorizes
editing expected results to match the candidate.

`pure_data_engineering_qualification` reports only this synthetic contract
scope. The original design-only image qualification protocol, its engineering
and independent statuses, production adoption, image quality, K and method
benefit remain separate and unverified. Passing a small positive example does
not waive any other frozen case or protection gate. The existing per-call
journal, bounded output reservation, captured costs, read-only verification and
ambiguous-effect recovery rules apply to this operator as well.

`FrozenCandidateReadinessRequestAuthority` reconstructs the historical checks
and readiness Controller with their original frozen core/domain in a separate
read-only interpreter. It rechecks the exact committed request, event replay,
native-completed Planner receipt and usage. It has no run/tick/invoke path and
never places candidate files on the import path. Its host-approved layout,
reader bytes, source commits, interpreter and frozen identities are pinned.

The legacy signature wire format is verified using the official
[cryptography RSA API](https://cryptography.io/en/latest/hazmat/primitives/asymmetric/rsa/).
This interoperability check does not introduce that legacy padding format as
a recommendation for unrelated new security protocols.

## Model-owned revisions from committed development feedback

`evolution.revise_operator_candidate:1` uses the existing bounded builder and
ModelGateway for one source-generation response. A host-selected
`ControllerContractFeedbackAuthority` requires a terminal, frozen, replayable
qualification Controller, its exact planned/committed invocation, unchanged
isolated captures and valid receipt. `FrozenContractFeedbackAuthority` reads
that evidence under the original frozen core without running the Controller,
candidate or a model. Arbitrary feedback JSON cannot replace those checks.

Only the explicit synthetic engineering scope can feed this builder. Failed
development cases disclose their frozen input, expectation and captured result;
passed cases disclose check summaries. Independent/heldout sources remain
unavailable. Such disclosed regressions cannot later serve as independent
qualification. The model receives predecessor source as untrusted text and
owns the new implementation; the host supplies no source patch or task answer.

The new plan retains the original request/interface and all committed usage
and the resource deadline. A separate candidate root preserves predecessor
bytes. The revision operator spec and exact model request bind the feedback,
predecessor, frozen runtime and cost lineage. Completed responses recover
without another call; unknown effects block regeneration. The new package
remains inert and unqualified until new frozen engineering and independent
protocols pass. Source revision alone never promotes, activates or proves
method/production benefit.

## Independent synthetic contract qualification

`FrozenContractExecutionAuthority` makes the historical reconstruction layout
explicit: original module and execution roots, environment, interpreter, plan,
package manifest, stores and sandbox limits. The frozen Controller, exact
committed qualifier receipt and captured effects are reverified under those
original module paths. Candidate stores cannot enter the import path. The
reader never invokes a model, candidate or Controller action. It returns the
engineering source families and input hashes so later validation can enforce
separation from disclosed development data.

`evolution.qualify_independent_contract:1` requires committed engineering
success for the same immutable package and an explicit
`dataevolver.independent_contract_qualification.v1` protocol. At least two new
source families must each contain positive and adversarial cases. Family IDs
and payload hashes cannot overlap the engineering corpus. Prior usage and the
original resource deadline carry forward; this CPU stage grants no model or
image budget. Inputs, independent expectations and registrar snapshots freeze
before executing the candidate through the same confined runner.

The synthetic independent producer declares lattice apertures and disconnected
objects on separate receiving surfaces, with authenticated positives and
preservation, unresolved attribution, surface crossing and signature failures.
It does not inspect candidate code/output to derive labels. Ephemeral private
signing keys remain in the producer process; only public snapshots and signed
payloads persist. The output bound is an explicit runtime contract, and actual
captured costs remain in the lifecycle ledger.

`pure_data_independent_qualification` concerns this declared synthetic contract
scope. The original image/production `independent_qualification` remains
`NOT_ASSESSED`, and production adoption stays false. Passing these cases cannot
replace the original independent image sources, their model/Agent quality
checks, qualification/promotion policy or a matched method comparison. An
engineering regression disclosed to the revision model never becomes
independent evidence by changing its name.

Historical nested readers retain their original frozen implementation identity
when reconstructing an independent stage from newer framework code. A separate
host environment snapshot may be bound explicitly, including its source commit
and file hash; candidate-owned environment files are rejected. Independent
summaries expose committed check results and provenance hashes, while keeping
their input payloads and captured answers out of revision feedback.
# Scope-bound operator versions and actual reuse

`operator_lifecycle.OperatorVersionStore` is a versioned capability artifact
store. It does not import candidate code or manage campaigns. The existing
EvolutionController registers `OperatorVersionMutation` as
`evolution.adopt_operator_version:1` or `evolution.revoke_operator_version:1`.
Adoption reopens both historical qualification Controllers using explicit
host-bound readers. It binds the package, implementation, interface, runtime,
two accepted qualification snapshots, scope, prior costs and original deadline.
An event written before Controller commit or rejected by the frozen Evaluator
cannot authorize resolution.

The current supported scope is `synthetic_authority_contract_independent`.
It grants no image/model permission or production qualification. The catalog
keeps immutable ID/version events in a hash-linked history; a revoked version
cannot be re-adopted. Revocation retains its reason and blocks new dispatch even
if its producing campaign is interrupted. Later catalog events do not invalidate
historical completed receipts.

`evolution.invoke_adopted_operator:1` resolves an accepted adoption and a pinned
catalog snapshot, checks the exact qualified runtime and new authenticated input,
then executes through the existing TrustedContextRunner. It holds the catalog
lease from resolution through dispatch to exclude a revocation race. The input
family and bytes must differ from qualification inputs. Its receipt binds the
adopted ID/version/snapshot, actual confined execution, captured output, trusted
oracle checks and lifecycle costs. A stored snapshot read is never counted as
reuse. Interrupted child effects remain ambiguous and are not resent; completed
effects recover without execution, including after revocation. Engineering,
independent synthetic qualification and this scope's reuse do not establish
production image quality, K/M benefit or training improvement.

Historical qualification keeps the reader hash from the original independent
core. A newer host reader must match all other origin bindings and prove that
original reader's file hash from the frozen independent source. Verified reader
results may be memoized only while every historical dependency remains byte
identical; added, deleted or changed files reject the cached result. This avoids
repeated reconstruction without replacing real qualification with a passed JSON.

`FrozenOperatorLifecycleAuthority` reopens completed adoption, invocation and
revocation in a fresh interpreter under the original frozen source. Constructor
binding uses read-only prefixes that actually occurred in the catalog; effect
verification always uses its full current history. A prefix never resolves an
active version or allows writes. The reader checks exact commits, artifact
hashes, frozen Evaluators and event replay for all three stages, and refuses
Controller/candidate execution. It returns sealed provenance, actual prior
costs, the current revoked status and original production qualification
requirements. It does not disclose independent payloads or grant production
permissions. This permits subsequent preparation to consume verified lifecycle
evidence after a process restart without rerunning the completed candidate.

# Production qualification execution and registered evidence

`production_qualification.ProductionQualificationExecution` is a separate,
inert execution contract. It retains the original qualification design, exact
matched-context artifact, source partitions and check IDs, while pinning actual
host validator specs/declarations and frozen Evaluators. Placeholder validators
cannot become execution bindings. The host runtime identity pins its confined
runner, existing ModelGateway, fixed models, initial R state, K snapshot, source
pool, online budget, trust profile and registered evidence schema. The namespace
must be `registered_production_runtime`; synthetic namespaces are rejected. A
genuine production validator must verify those hashes
against the actual runtime, requests and outcomes; this contract does not infer
them from candidate reports.

The predecessor is read through `FrozenOperatorLifecycleAuthority`, including
its exact current identity, summary, prior failed costs and original deadline.
The execution contract cannot lower prior costs, move the deadline or reuse a
revoked ID/version. `inspect_validator_bindings` reads real host registry and
Evaluator files. `inspect` also verifies source artifact hashes and reports an
expired resource deadline. Even `BOUND` means only bound identities: it grants
no execution, model budget, qualification or adoption. Writing this contract is
not permission to refill an exhausted development plan.

`ControllerProductionQualificationAuthority` reuses the existing
`ControllerQualificationAuthority` ABI. A producing EvolutionController must
freeze both its normal `operator_qualification` binding (the execution contract
serves as its plan hash) and `production_qualification_execution`. The registered
validator must commit `production_evidence` using schema
`dataevolver.registered_production_evidence.v1`: execution hash, stage, exact
runtime, original stage source artifact refs, and named host-owned refs for
`input_closure`, `matched_comparison`, `quality_checks`, `cost_ledger`,
`native_audit` and `independent_load`. Each ref must resolve by its exact hash
and occur in the originally committed host effect verification. Candidate-owned
refs, heldout substitutions, drift, partial checks and copied success JSON fail.
The actual validator remains responsible for native reconciliation, complete
costs, model/Agent quality and independent loading; reference closure alone does
not establish these results.

The reader verifies the original replay, committed receipt, accepted frozen
Evaluator, package metadata and complete package file hashes. It never calls
execute/recover. `assess_production_qualification` needs separate actual
engineering and independent Controllers and checks their combined usage against
the same prior lifecycle budget. Its results still grant no production adoption
or method/training benefit. Production validators and a production invocation
adapter must be implemented and genuinely qualified before production use; the
synthetic trust profile and already revoked synthetic version stay inactive.

## Registered production qualification producer

`adapters.production_qualification.ProductionQualificationOperator` registers
`evolution.qualify_production_operator:1`. Its composite action drives bounded
workload calls from a host-owned OperatorRegistry; each workload retains its
existing ModelGateway and effect verifier. It never imports a candidate program
as a host operator or starts another controller/queue. Concrete task parameters
and repair evidence must still come from their registered runtime producers.

Each predeclared family binds its original input ArtifactRef, exact workload
OperatorCall and finite upper usage bound. Workload owners/directories must be
distinct. The host workload must declare a `production_case` output port and
verify a closed result containing the exact execution/package/stage/source/
matched/runtime binding, every original check, six evidence roles, actual usage
and a captured candidate invocation. The producer verifies that capture's exact
implementation, production runtime profile, context digest, successful isolated
termination and stdout/output agreement, as well as all recursively referenced
host-owned evidence hashes. An `actual_execution: true` report alone cannot
establish candidate use. A concrete workload adapter must additionally reconcile
its real Gateway/native ledger and enforce the reserved model/image bounds;
declaring this port is not production qualification.

The existing Controller freezes `execution_contract_sha256` and
`workloads_sha256` in its initial call parameters. Execution data stays out of the
reusable producer spec to avoid a circular dependency between the validator spec
hash and the execution contract which binds it. Workload specs and source bytes,
the host store mapping, package and predecessor remain part of that spec.

Before any dispatch the producer checks the original lineage, immutable target,
full family coverage, workload declarations and total reserved budget/deadline.
It writes a single-owner intent, then retains every actual workload receipt and
emits role manifests compatible with the production evidence reader. Failed
checks stay false. Its output byte accounting closes over its data artifacts;
observed stage wall time includes preflight and sequential workload execution,
without adding native elapsed time twice.

Verification and recovery read completed effects only. If a child lacks a
durable receipt, the producer remains ambiguous and never resends it. If all
children completed before the parent report was written, it reconstructs the
report/receipt without dispatch. A different owner cannot reuse the same stage
to reset its allowance.

With `execution=None`, this same registered producer performs preflight only:
it reads actual lineage and registry/Evaluator gaps, writes a committed
`EXECUTION_CONTRACT_NOT_BOUND` result, and dispatches zero workloads. This mode
does not qualify a package. Existing training trial/observation operators do not
yet implement the `production_case` adapter, and the production trust runner,
actual qualification, production adoption and active invocation remain pending.

Production workloads also bind host `admission_probes` in the producer identity.
Missing probes refuse dispatch. The producer reserves their bounded evidence
output and verifies their durable reports during replay without polling again.
An expired resource deadline is not an unconditional rejection: the original
Gateway policy refuses unavailable resources after that unchanged deadline,
while an already available service still needs two stable observations. A
preflight does not waive per-request Gateway admission or add execution budget.

`TrainingResourceAdmission` delegates to the existing training operator's wait
method, retains each role's observations, pins the original model contract and
physical GPU mapping, and never invokes a Gateway or loads a model. Production
cannot replace the wait callback. An incomplete observation retains its owner;
completed evidence is verified without new health requests. This resource
adapter is not the pending `production_case` quality/evidence workload adapter.
