# Visual Harness P1: contracts, local models and recovery

P1 extends the existing `EvolutionController`. The new visual route is inference-free and its evidence level is `mock_verified` only after the tests below pass. It does not process a real dataset or establish model/visual quality.

## Contracts and incremental local integration

- `dataset/contracts.py`: strict versioned parsing, canonical finite JSON, content hashes, `ArtifactRef` and explicitly injected artifact stores. Serialization and content-hash helpers were extracted from the local DataProcess primitives; v1 parsing now rejects unknown fields.
- `harness/task_contracts.py`: `DatasetIntent` (`curate`, `generate`, `hybrid`), `QualityPolicy`, `NodeSpec`, acyclic `NodeSpecGraph` and runtime `NodeRun`.
- `adapt_raw_image_pair_contracts(...)`: imports the intent and preservation/processing criteria from the local Raw Image Pair contracts. The caller supplies an immutable catalog reference; private legacy source paths are not read or copied into the new contract. Legacy promotion thresholds remain experiment-owned.
- The Raw Image Pair processing, champion loop and training feedback execution remain later integration work. They are not imported as another control plane.

Contracts use explicit schema versions and reject unknown top-level fields, invalid budgets, unsafe artifact paths, missing sources, graph cycles and unapproved roles/operators. Protected gates must remain declared. Actual data-gate evidence, labeling and dataset exports belong to P2 and later stages.

## Model gateway

`ModelProfile` declares backend, model, deployment revision, capabilities and environment-variable names. `ModelPolicy` binds roles, bounds model attempts and allowlists local endpoint origins. `ModelRequest` contains typed messages, artifact references, response schema, token bound, timeout, seed and retry bound. `ModelReceipt` records identity, output, finish reason and measured or unknown usage.

The configured deployment revision is not independently verified by `/models`; a model listing establishes an advertised identity, not weight integrity or inference quality. Missing usage and local endpoint cost remain `null`.

| Backend | P1 support |
| --- | --- |
| `local_endpoint` | OpenAI-compatible JSON/vision transport, optional env token, `/models` health, explicit origin allowlist, bounded responses and no redirects or inherited proxy |
| `local_python` | Trusted worker factory and optional independent effect lookup; lazy model loading, no model-generated imports/code |
| `mock` | Durable deterministic fixture effect and native request lookup |
| `external_api` | Reserved interface; invocation always raises `EXTERNAL_API_DISABLED` |

External API cannot be enabled by changing the P1 flag. No cloud SDK or automatic fallback is installed. Roles can reserve a disabled external profile while local roles continue to work.

Install only the lightweight schema dependency when needed:

```bash
python -m pip install -e ".[harness]"
```

Use `configs/harness/model_profiles.example.json` as a shape example. Its model and local origin are placeholders, not detected server settings. Set a private endpoint/model binding after inspecting the target server, record the deployment revision, and replace the origin allowlist. Credentials remain environment variables. Existing serving processes are not restarted or modified by profile inspection.

```bash
python -m dataevolver.harness.cli models --config <private-model-profile.json>
python -m dataevolver.harness.cli models --config <private-model-profile.json> --check-endpoint
```

The first command reads configuration only. The second performs read-only model-list health calls. Neither sends an inference request or loads local weights.

`GatewayPlannerBackend` implements the existing `PlannerBackend.complete_json(...)` protocol. An Agentic planner config explicitly selecting `backend: model_gateway` can provide `profiles` and `policy`; existing legacy planner configuration continues through its previous transport and production preflight. P1 verifies the bridge with fixtures.

## Receipts, budgets and recovery

Every new attempt first writes a request/profile/contract-bound budget intent. Attempts, including retry-safe failures, share the approved gateway call limit. Reusing a request ID with different messages, model configuration or policy fails closed.

Verified receipts are checked before provider construction. A complete cache hit causes zero model loads and zero inference requests. Receipt hash, identity, schema, completion state and matching budget intent are required. Missing accounting cannot silently become a zero-cost cache hit.

After interruption, a provider-native lookup may reconstruct the receipt. The durable mock backend supports this path. A local Python worker needs a separately registered receipt lookup. A generic HTTP endpoint has no native lookup in P1, so an interrupted/unknown effect is `AMBIGUOUS_EFFECT` and is never automatically resent. An explicit retry-safe response may be retried only within the request and shared budget.

HTTP timeouts bound transport waiting; an endpoint may still finish work after disconnection. Local Python workers must implement their own bounded execution/watchdog. P1 does not promise universal exactly-once external effects or a production GPU worker supervisor.

The visual mock composite executes a fixed topological node graph under one registered Operator. It writes model receipts and per-node outputs/NodeRun records under the injected campaign root, then returns the existing `OperatorReceipt` and `StageResult`. Partial graphs resume verified nodes; the existing Controller still owns action commit, adoption, terminal status and replay.

## Mock verification

```bash
python -m dataevolver.harness.cli evolve \
  --goal configs/harness/visual_contract_mock.json \
  --campaign-root <runtime-root> \
  --idempotency-key visual-contract-demo --dry-run

python -m dataevolver.harness.cli replay \
  --campaign visual-contract-mock-demo --campaign-root <runtime-root>
```

The route accepts only mock model profiles, no network/GPU permissions and no static interventions. It produces JSON fixtures rather than visual training samples. `--dry-run` is required.

Run checks on the target server, with artifacts outside source:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m pytest tests src/dataevolver/testing/test_harness*.py \
  -q -o cache_dir=<runtime-root>/pytest_cache --basetemp <runtime-root>/tmp
```

Tests cover contract rejection, artifact mutation/path escapes, graph permissions, Raw Image Pair adaptation, cache-before-load, interrupted effects, unknown effects, retries/budgets, invalid output, model mismatch/truncation, endpoint transport/redirects, credential exclusion, Controller resume/replay and execution-code drift. HTTP transport uses a local fixture server and is not a real model test.

真实模型推理、图像处理、数据冻结与训练分别需要对应任务和资源条件。P1 的 mock 通过只确认合同、编排和恢复路径。
