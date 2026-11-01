# Harness runtime RSI

The existing `EvolutionController` owns the lifecycle. `runtime_rsi` adds a
domain interface to its `PROPOSE` and `UPDATE` states; it does not launch a
second controller. A runtime model returns one typed decision: observe, run a
matched experiment, request completion, or abstain.

The Controller commits the decision before invoking a registered Operator.
Observation effects and model calls use the existing leases, idempotency keys,
native lookup and ModelGateway journals. A disconnected or timed-out model
attempt must be reconciled before another submission. A resumed completed
campaign does not call the Planner again.

## Ownership and evidence

- The development agent implements the integration and tests. It does not supply
  source-specific repairs for runtime acceptance.
- The runtime Planner sees bound media, original requirements, real failed
  attempts, compatible knowledge and prior comparison results. It selects
  registered probes and proposes bounded parameters.
- The domain compiler checks source identity, current observation evidence,
  selected subject IDs and patch allowlists. It cannot modify the evaluator,
  models, task seed, original requirement or budget.
- `EvolutionController` runs the existing baseline/candidate gate, adopts or
  rolls back, then asks the runtime Planner to reconsider the actual result.
- Only an adopted comparison creates candidate K. One-source adoption does not
  promote knowledge or establish a better improvement method M.

## Training-domain adapter

`data.training_rsi_trial` imports verified native effects from one frozen source
family. The baseline recovers completed requests. A region-support candidate
reuses scene generation, PE-T2I, correspondence, SAM3 and the original effect
probe. A changed image condition causes new PE-I2I, image editing and review
calls. Completed source generation is never repeated to hide a failed edit.

`data.training_rsi_observe` provides two registered tools: actual VLM inspection
with a runtime question, and CPU original/raw/composite/blend measurements.
Pixel differences are observations, not shadow ground truth. A source-specific
R proposal must cite a committed observation of the current failed candidate.
The compiler rejects wrong coordinate units, non-finite values, unknown subjects,
stale evidence, threshold changes and method M labels on a recipe R patch.

Trials use separate frozen media namespaces and preserve old receipts exactly.
The original subject references and masks remain auxiliary evidence; training
inputs are background, subject references and instruction. Source-family splits
and inferred-background provenance remain unchanged.

An accepted trial needs the original model quality checks, pixel protection,
media/hash/load verification and a separate runtime Agent visual audit. Agent
review is recorded as such; it is not human review. The gate cannot lower these
requirements to obtain a passing record.

## Planner inference transport

The registered CLI provider uses its frozen `ModelProfile.model` and optional explicit worker `effort`, with JSON
schema output. It disables shell, unified execution and multi-agent tools. A
CLI attempt has durable native request, event, output and failure records.
Transport retries are explicitly disabled in the provider configuration.

If the authenticated instance is on another machine, a loopback SSH inference
transport carries only bound ModelRequests, images and native completions.
Authentication stays at that instance. It owns no task queue, controller,
operator execution, knowledge promotion or adoption decision. Worker and binary
hashes are frozen independently of the remote model's unavailable weight hash.

## Scope of validation

Fixture tests verify transitions, recovery, allowlists and accounting. Real
runtime evidence requires an authenticated Planner decision, its registered
probe effect, a resulting parameter/request difference and an independently
verified matched comparison. Only then may a report claim a runtime R iteration.

Independent new-source validation and later compatible reuse are separate K
requirements. M evolution requires a separate experiment with the same R/K
starting conditions, source pool and online budget. This adapter currently
accepts R region support; it deliberately rejects M patches. A passing image or
candidate K must never be reported as recursive method improvement or training
benefit. Additional domain compilers can implement those contracts without
replacing the Controller.

## Commands

Prepare a private deployment/Planner configuration, then freeze using the
committed checkout:

```sh
python -m dataevolver.harness.training_rsi_run freeze \
  --seed-config "$SEED_CONFIG" --out "$RUNTIME_CONFIG" \
  --campaign-id "$CAMPAIGN" --slot "$SOURCE_SLOT" \
  --planner-config "$PRIVATE_PLANNER_CONFIG"
```

Run in the resulting immutable source snapshot, with its private environment:

```sh
source "$RUNTIME_CONFIG/env.sh"
cd "$RUNTIME_CONFIG/source"
export PYTHONPATH="$PWD/src"
python -m dataevolver.harness.training_rsi_run run --config-dir "$RUNTIME_CONFIG"
```

The normal Harness `step`, `status`, `pause`, `resume` and `replay` commands remain
the controller interface. Original resource deadlines and all failed costs are
retained. `training_rsi_run validate` reports R, K and M evidence separately.

For an explicitly authorized hardware migration, `freeze --deployment-config`
binds the new resident process, GPU UUID allowlist, CPU tensor placements and
activation reserves without changing model assets, task requirements or prior
receipts. One serial resident transport may retain both media and PE objects.
The Controller and ModelGateway still own every effect. CPU reviewer requests
stay on CPU even when that process also owns a GPU image model; GPU admission
never counts memory from devices outside the frozen allowlist.

`--prior-runtime` imports the paused campaign's completed Planner ledger and
hash-verified registered observations. Probe evidence is usable only for the
same source and rejected candidate hashes, even if artifact stores were moved.
It does not regenerate an observation, supply a repair, reset cumulative Planner
usage, or modify the old frozen source. The resumed Planner chooses its next
action from that evidence itself.


## Production capability objective

The user-defined acceptance target covers production data processing comparable
to DataFlow/DataFlow-MM plus verified evolution of operators and the Harness.
See [the capability matrix and qualification protocol](PRODUCTION_CAPABILITY_ACCEPTANCE.md).
This specification does not establish current parity or an autonomous operator
development loop. Existing real delivery and frozen execution gates still apply.
