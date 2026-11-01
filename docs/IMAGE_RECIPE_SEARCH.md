# Frozen image pools and bounded recipe search

The pool operations and `data.image_recipe` route use the existing
`EvolutionController`, `CandidateGate`, event store and capability store.
Experiment configuration and runtime evidence remain outside framework source.
External model APIs remain disabled.

## Pool and label contracts

`images.labels` consumes a bound `dataevolver.label_mapping.v1` artifact:

```json
{
  "schema_version": "dataevolver.label_mapping.v1",
  "sources": {
    "source_a": {
      "0": {"label_id": "source-a-healthy", "name": "healthy", "evidence": "source taxonomy"}
    }
  }
}
```

Rows retain their original source labels. Unknown labels and conflicting
canonical meanings fail validation. The independent export loader checks the
mapping identity, name and mapping hash against every exported row.

`images.select_balanced` enforces positive quotas by source and canonical class.
It uses a family assignment that can resolve shared-family conflicts rather
than silently missing a feasible quota. Infeasible quotas fail with the affected
source/class. Ranking cannot relax quotas or family uniqueness.

`images.partition` freezes distinct family membership for `development`,
`calibration`, `canary`, `reuse` and `sealed` roles. Parameters and membership
must be fixed before recipe/model comparisons. The ordinary train/dev/eval
fields remain engineering splits; physical-family and global leakage claims
require evidence beyond observed hashes and declared source families.

`images.calibrate_quality` fits an objective clarity threshold on reference
images and controlled Gaussian-blur pairs. Separate control families validate
the fitted threshold. This establishes a limited reference-versus-blur test,
not human semantic quality, privacy/safety reliability or downstream benefit.

The exported pool includes mapping, calibration, control images, source evidence
and independent image-loading checks. Once frozen, later recipe calls reference
the exact pool metadata hash and create separate output versions.

## Local planner and reviewer

The registered `qwen_vl_cpu` worker supports local Qwen3.5 image/text checkpoints.
It validates every required shard/processor hash, runtime package versions and
its own implementation hash. Model path and interpreter are environment
references. No model code, shell command, import name or external URL comes from
the planner. Loading occurs in a deadline-controlled subprocess after a confirmed
cache miss. CUDA visibility is empty.

Worker configuration explicitly binds thread count, precision, image grid,
CPU capability and rotary implementation. The optional NumPy rotary implementation
reproduces the default text and vision RoPE math on CPU. It is a configured
compatibility implementation, not an automatic retry or backend substitution.
Text use is restricted to default RoPE and the bounded context. Tests compare
the float32 results and multiaxis layout with the reference implementation.

The reviewer first answers blinded sharpness-control comparisons, then supplies
advisory image scores. Reviewer-based rankings require its frozen control
criterion to pass. Neither model scores nor planner output grant protected-gate
authority. Planner requests contain development summaries and bounded comparison
history, never pool images or individual canary records. Sealed images are
rejected before model budget reservation or provider invocation.

## Recipe contract and adoption

`GoalSpec.planner.kind` is `recipe_rsi`; the objective supplies a frozen
`recipe_contract`, and the initial call is:

```json
{"recipe": {"ranking": "hash"}, "phase": "search"}
```

The contract binds the pool ArtifactRef, disjoint read/write roots, class/source
quotas, seed, baseline, shared model profiles/policy, review rubric and budgets.
An optional fixed-hash development limit bounds the reviewed candidate scope.
An inherited absolute deadline prevents recovery amendments from restarting
the original time allowance.

The planner can propose one field: `ranking`. Registered alternatives are
`clarity`, `review` and `hybrid`; `hash` is the fixed baseline. The compiler rejects
unknown fields, policy edits, unsupported reviewer rankings and repeated recipes.
It cannot change labels, pool membership, coverage requirements, sealed roles,
evaluation criteria, model profiles or execution permissions.

The CLI limits this route to at most three proposals/experiments. Baseline and
candidate use the same frozen canary key, output counts and protected metrics.
`CandidateGate` returns adoption, rollback or abstention. The Controller retains
the incumbent on rollback, records every comparison and stops at its declared
limits. A completed search may retain the baseline without claiming improvement.

An independent reuse role compares the final incumbent with the fixed baseline.
Positive canary and reuse evidence can register a campaign-scoped capability
candidate. This does not promote framework code or establish training validity.

## Budgets and recovery

- Shared model intents count attempts, including failed requests. Image counts
  are reserved before inference. Pure recipe selection/export has no model role.
- Operator attempts, unique source units, output bytes and the deadline are
  persisted. Missing ledgers fail closed. Recovery uses the bound contract.
- Native model receipts recover a completed effect without loading the model.
  A missing native result after an attempted request is ambiguous; it is not
  automatically resent.
- Failed Operators return verifiable failure evidence with `artifact_valid=false`.
  Failure evidence cannot satisfy positive quality or adoption requirements.
- Completed recipe effects verify exact selections, label/lineage bindings,
  model receipts, quotas and recomputed objective metrics. Resume reuses effects.

Experiments should materialize the Goal/config/launcher bundle before effects
and preserve original failed revisions. Any recovery amendment must bind prior
consumption and declare changed implementation/scope before new comparisons.

## Entry point and verification

Use an experiment-owned frozen Goal with explicitly injected store/model paths:

```bash
python -m dataevolver.harness.cli evolve \
  --goal "$FROZEN_GOAL" --campaign-root "$CAMPAIGNS_ROOT" \
  --idempotency-key "$CAMPAIGN_KEY"
python -m dataevolver.harness.cli resume \
  --campaign "$CAMPAIGN_KEY" --campaign-root "$CAMPAIGNS_ROOT"
```

This real route requires `objective.real_data_execution=true`; `--dry-run` is
rejected. The mock and optional 3D/T2I routes retain their separate contracts.

Relevant checks are `tests/test_image_recipe.py`, `tests/test_chat_worker_math.py`,
`tests/test_data_curation.py`, `tests/test_clip_worker.py` and existing Harness
controller/planner regressions. They cover infeasible quotas, source-label
binding, held control calibration, sealed scope, shared budget failures,
effect-before-receipt recovery, bounded adoption/rollback and event replay.
Real model, process interruption and downstream evidence must be reported
separately from these fixture checks.
