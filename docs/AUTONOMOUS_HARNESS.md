# DataEvolver Autonomous Harness V1

## Status

Phase 0–3A established the durable single-controller Harness, and Phase 3B verified one bounded real T2I side effect. Phase 3C adds a strategy-layer Agentic 3D loop and a recoverable composite 3D Operator. The multi-round Agentic path is `agentic_3d_mock_verified`; no real 3D Agentic canary has run because the required Agent endpoint is not configured.

The Harness remains opt-in and never writes the legacy `.dataevolver/runtime` tree. It does not modify code, evaluator thresholds, budgets, the Operator registry, training, papers, or sealed evaluation.

The current public entry is `python -m dataevolver`. Native scene workers use the same Controller; see `NATIVE_SCENE_WORKFLOW.md`. The former shell production loop is retired.

## Control plane

The Harness keeps the existing stages as a future data plane and adds a single explicit controller:

```text
INIT → PLAN → EXECUTE → OBSERVE → DIAGNOSE → PROPOSE
→ CANARY → EVALUATE → ADOPT_CANDIDATE / ROLLBACK
→ UPDATE → PLAN / COMPLETE / ABSTAIN
```

`PAUSED` and `PAUSED_CONTRACT_DRIFT` are non-terminal states. `controller.tick(max_actions=1)` advances at most one bounded controller action; `evolve` and `resume` repeatedly call `tick()`.

V1 uses JSONL append-only events, an atomic JSON state snapshot, one campaign-wide lease, and no database or external orchestration dependency.

## Safety and recovery contract

The Harness guarantees exactly-once logical `ACTION_COMMITTED`, idempotent event projection, and duplicate logical-call suppression. It does not claim universal exactly-once external effects.

Every Operator must implement:

```python
execute(call) -> OperatorReceipt
recover(call) -> RecoveryResult
verify_effect(call, receipt) -> EffectVerification
```

After an interrupted call, recovery verifies an Operator-native receipt or deterministic outputs. An unverified or ambiguous external effect fails closed unless the registered Operator is explicitly retry-safe and verification proves that no effect occurred.

`CommandOperator` accepts only a pre-registered argv list, uses `shell=False`, resolves the executable path, inherits only allowlisted environment keys, and never executes model-generated shell or Python.

## Evidence and matched canaries

Metric evidence is explicitly one of:

```text
UNOBSERVED | OBSERVED | VERIFIED | CONTRADICTED | STALE
```

All required hard, target, and protected evidence must be `VERIFIED` before candidate adoption. Missing values never imply success.

A failed observation is reused as the canary baseline only when its input, parent, seed, Operator version, model/config, preprocessing, evaluator, and environment hashes match the requested `MatchedTrialKey`. Otherwise the baseline is rematerialized. Candidate evidence records the single declared intervention delta.

`ADOPT_CANDIDATE` changes only the active strategy inside the current campaign. V1 creates a campaign-local `CapabilityCandidate`; it does not promote a global Capability.

For Agentic goals, candidate adoption and final goal completion are separate gates. An intermediate candidate may become the incumbent when its declared target metrics improve and hard/protected metrics do not regress, even if another target still needs work. `COMPLETE` still requires every frozen final threshold and the accepted-asset quota.

## Agentic 3D strategy loop

Agentic goals explicitly select `agentic_rsi`; old goals remain deterministic. The planner receives only structured goal, incumbent, evidence, experiment history, attempted novelty hashes, compatible Capability memory, and remaining budget. It returns one schema-validated `PlannerDecision`.

The only experiment kinds are:

```text
PROMPT_REGENERATE      → restart at T2I
RESEGMENT_RECONSTRUCT  → restart at segmentation
RECONSTRUCT_RETRY      → restart at reconstruction
SCENE_REFINE           → reuse mesh; rerun render and review
```

`ExperimentCompiler` validates the patch, scene-action whitelist, novelty and budget, then selects the single registered `dataevolver.3d_asset_evolution@1` Operator. The model cannot emit argv, paths, shell, Python, arbitrary kwargs, evaluator changes, or Operator chains.

The composite Operator writes RGB, RGBA, raw/sanitized GLB, mesh QC, four frozen representative views, VLM/CV review, per-stage receipts, bundle manifest and StageResult under the campaign. Recovery resumes after the last verified stage; malformed receipts, hash mismatches and path escapes fail closed. Agentic execution disables SAM fallback and Hunyuan shape-only fallback.

The production planner uses an OpenAI-compatible JSON endpoint configured only through environment variable names:

```text
DATAEVOLVER_AGENT_BASE_URL
DATAEVOLVER_AGENT_API_KEY
DATAEVOLVER_AGENT_MODEL
```

The API key is never frozen or written to campaign evidence. Missing backend or model resources prevent campaign creation; there is no deterministic fallback for an Agentic goal.

## Frozen execution contract

Campaign startup freezes the repository revision, execution-relevant diff and code inventory, schemas, Operator registry and source revisions, planner config, evaluator, promotion policy, budget, resource policy, and GoalSpec. Drift pauses the campaign before another effect or adoption. Restoring the exact contract permits resume; accepting changed code requires a new campaign.

## Dry-run

### T2I workflow adapter

This is the first non-Fake Harness Operator. It calls the existing multimodal `build_request()` and `run()` APIs with a fixed T2I, CPU, no-inference contract. Its accepted GoalSpec parameters are `user_request`, `num_samples`, `height`, `width`, `seed`, `sample_prefix`, `skip_existing`, and `validate_outputs`; route, generator, output root, model, device, VLM, and inference controls cannot be overridden.

```bash
python -m dataevolver.harness.cli evolve \
  --goal configs/harness/t2i_dryrun.json \
  --campaign-root .dataevolver/campaigns \
  --idempotency-key multimodal-t2i-dryrun-demo \
  --dry-run
```

The Operator verifies `request.json`, `dataset_summary.json`, `manifest.jsonl`, output containment, PNG signatures, and hashes. A missing receipt is rebuilt only from a complete verified dataset; missing outputs are retry-safe, while malformed or out-of-root outputs fail closed.

### Fake controller example

After installing the package entry points, run:

```bash
python -m dataevolver.harness.cli evolve \
  --goal configs/harness/fake_vertical_slice.json \
  --campaign-root .dataevolver/campaigns \
  --idempotency-key autonomous-harness-v1-demo \
  --dry-run
```

Run one bounded step for an existing campaign:

```bash
python -m dataevolver.harness.cli step \
  --campaign autonomous-harness-v1-demo \
  --campaign-root .dataevolver/campaigns \
  --max-actions 1 \
  --format json
```

Other commands are `pause`, `resume`, `status`, `explain`, and `replay`.

Each campaign contains `goal.json`, `frozen_contract.json`, `events.jsonl`, `state.json`, round and experiment evidence, `capabilities.jsonl`, and `final_report.json`.

## Next integration boundary

- Configure the three Agent backend variables plus the reviewed model, Blender, scene-template and GPU resource variables.
- Create a fresh single-asset campaign with four frozen views and at most four experiments.
- Run only the unified Harness command; do not edit intermediate prompt, control-state, receipt or manifest files.
- Require parseable RGB/RGBA/GLB, texture and geometry gates, complete VLM/CV review, replay consistency and terminal idempotency before claiming `agentic_3d_real_canary_verified`.

The present code and Fake-stage evidence do not establish real 3D self-evolution or a high-quality 3D dataset.

## Phase 3D bounded real canary contract

`configs/harness/agentic_3d_rsi_real_canary.json` freezes one asset, one initial prompt, four views, at most three candidate experiments, and at most four planner calls. The CLI performs a read-only production preflight before creating a campaign: it verifies the OpenAI-compatible Agent model, Python dependencies, Blender/scene contract, model paths, action space, and an idle explicitly selected GPU.

Each Agent call writes `planner/pNNN/trace.json` with the raw response(s), validation errors, repair status, backend/model identity and outcome. StrategyPatch and compiled experiment JSON are stored separately. `UPDATE` requires another planner call after every adopt or rollback, including when the deterministic goal gate is already satisfied.

The final report uses truthful terminal values:

```text
COMPLETE | BUDGET_EXHAUSTED | ABSTAINED | NO_VALID_EXPERIMENT | FAILED
```

The label `agentic_3d_canary_verified` is emitted only for a completed real-canary Goal with all of the following persisted: a valid real OpenAI-compatible planner response, a verified real 3D candidate effect, a matched adopt/rollback decision, a post-decision planner call, and a campaign-local Capability. Otherwise the label remains `agentic_3d_mock_verified`.

At the time of this implementation, the remote model and Blender resources were present but the required `DATAEVOLVER_AGENT_BASE_URL`, `DATAEVOLVER_AGENT_API_KEY`, and `DATAEVOLVER_AGENT_MODEL` variables were not configured. Therefore no Phase 3D real campaign had been created or run.
