# Framework contracts and acceptance

The Controller owns the campaign, frozen source, Evaluator, budgets, pending
calls and committed effects. Basic operators expose named artifact ports through
the shared Registry. ModelGateway, native workers, external trainers and confined
candidate runners retain their own effects under that campaign's contract.

## Data and model effects

`compile_plan` checks registered versions, ports, parameters, roles, source
membership and declared budgets. `CompiledRun.bind` checks the registered specs
again. Execution freezes the source used by the Controller. Original data uses
read-only input stores; node outputs use the task's writable dataset store.

The standard Registry includes persistent families, image generation, review,
observation, subject segmentation, verified T2I supervision, append and export.
`supervision.text_to_image_verified` binds the exact image and generation caption
to the review, observed instances, SAM masks and original subject-count journal.
Catalog split aliases `dev` and `eval` become `validation` and `test` at the
supervision boundary. Provenance retains the original family-ledger split.

`FamilyLedger` retains original member/family/split identities and records later
mergers as aliases. Later cross-split mergers invalidate affected releases.
Balanced selection uses canonical family IDs. Export and training preparation
recheck a release against the current ledger and require unique families.
`supervision.export_t2i` delivers standalone supervision and imagefolder metadata
with exact IDs, image hashes and captions.

Gateway reservations include failed and uncertain requests. `ModelAttemptLimits`
links the approved ledger roots and uses a shared reservation lock across
processes. SAM image-subject units come from the frozen request, and observed
subject forwards come from its native journal. R/M candidate slots have separate
accounting through `CandidateSlotLimits`; a new call ID cannot reset a candidate.
Closed historical usage remains part of the round identity.

Resident hosts keep actual model objects across requests. Deployment identity
binds process start, physical devices, actual placement, handler source, model
assets, authorized job/output roots and input store mappings. Worker health and
in-process observation avoid recursively waiting on a serial inference socket.
A native completion can recover a lost Gateway receipt. An uncertain effect
cannot authorize a fresh dispatch.

## External training and feedback

`RegisteredTrainingRunner` accepts a host-approved command, source inventory,
environment, deadline and exact native receipt location. It records intent before
launch. A recorded or uncertain launch is reconciled; it cannot be submitted
again under a new process identity. Timeout cleanup targets the launched process
group. Failed exits, logs and incomplete receipts remain visible.

The LoRA contract checks actual logical/physical sample consumption, exact
captions/media, finite losses/gradients, optimizer steps, changed trainable
parameters, checkpoint tensor inventory and a fresh-process reload. Reload must
bind the actual active adapter, saved tensors and normal controlled inference.
The experiment-owned adapter instruments the pinned upstream trainer; Harness
does not implement another optimizer or training framework.

Only allowed development/validation feedback can enter a next-plan context.
Sealed evaluation results cannot return as optimization feedback. Any next data
plan requires a new frozen run. A receipt, finite loss or completed reload does
not establish training benefit.

See [the registered trainer entrypoint and configuration](../examples/harness/README.md).
The default command checks assembly; `--execute` uses the existing Controller.
Its arguments come from live host discovery and the authorized task contract.

## Benchmark evidence

`GenerationBenchmarkOperator` registers the real evaluation workload under the
Controller. `EvaluationJournal` freezes every expected image ID/hash. Missing,
failed or swapped IDs cannot leave the denominator. A confirmed replay reuses
the original requests.

Spatial scoring keeps ten ordered questions, five valid rollouts, a four-of-five
criterion and a 25-attempt ceiling. GenEval2 keeps the official answer variants
and first-token probabilities. Stable FP64 softmax prevents FP32 rounding from
producing probability mass above one. Native logits and their hashes permit an
independent probability recomputation. AM/GM reuse the same saved atoms, and zero
probability remains zero. Metrics are recomputed from their exact native answers;
changing a stored aggregate cannot create better evidence.

Pinned benchmark code and the evaluated question set have separate identities.
An engineering-authored integration set must be labeled accordingly; using the
official scorer does not make that set the published benchmark. Input capacity,
model precision, long-range reasoning quality and benchmark quality also retain
separate evidence.

## R, M and K

RuntimeRSIPlanner can propose only the actions allowed by its domain and frozen
policy. `PolicySpec` limits changes to observation order, candidate limit and
stopping rule. Evaluator, budget, permissions, model identity and protected gates
remain outside its editable fields.

Catalog row diagnostics require explicit development or reuse scope and a
successful verified trial. They expose the current ranking and at most 64
source measurements; independent and sealed inputs cannot enable this view.
Terminal planner actions carry empty operation fields in the strict schema.

The `catalog_selection.v1` scope permits a model-built pure selector. The package
stays inert until execution in the confined Linux runner. It returns existing
row IDs; the host reconstructs unchanged rows and checks eligibility, required
reviews, quotas, canonical families and category coverage.

Qualification requires the committed non-mock build origin, exact predeclared
development/regression/independent sources, at least two new independent
families, unread heldout outcomes, real native spatial/atomic outcomes, a bound
Evaluator, behavior change and at least 0.05 independent proxy gain. Different
operators, packages, source media, Evaluators or runtime identities cannot borrow
that qualification. A failed or unsupported case remains failed.

`OperatorVersionMutation` records adoption/revocation through the same Controller
and append-only version store. Active resolution additionally checks committed
adoption and the matching qualification authority. Revocation prevents future
use. The confined scope grants no additional model/image budget or general
production authority.

K uses scoped source/validation evidence and the existing capability store.
Catalog trials reopen every nested image, recompute its canonical pixel hash,
dHash and clarity features, and bind that inventory to the committed result.
When an inner graph fails, the outer trial retains a verified failed receipt
and reuses that failure on recovery; it does not read an absent export or report
a successful trial.
Independent K validation rejects reused image bytes, pixel hashes and exact
dHashes even when catalog paths or declared source-family labels differ.
Writing a distilled candidate, labeling it promoted or passing a synthetic
fixture does not authorize knowledge reuse. Actual independent validation and a
verified successor action are required for a reuse claim.

## Evidence levels

| Claim | Required evidence |
| --- | --- |
| Declaration/compilation | Exact schema, ports, versions and allowed bindings |
| CPU contract verification | Real tests with their fixture scope recorded |
| Native execution | Source/model/input identity, native effect, receipt, exit and readback |
| Data acceptance | Every frozen quality, review, family and membership requirement |
| Confined M qualification | Real build, isolated execution and the independent gate above |
| Training benefit | Frozen downstream comparison on the allowed evaluation split |
| Causal RSI gain | Controlled R/M/K comparison and measured independent successor results |

Registered workload capability hooks can report at most `real_execution_verified`.
They cannot grant training benefit or general production qualification. Runtime
acceptance reports retain failed attempts, unknown costs, source interventions,
unspent candidate slots and every unverified capability.
