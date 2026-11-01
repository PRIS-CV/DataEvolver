# Agent setup

Start with the user's goal, execution host, model choices and resource budget.
Keep private paths, credentials and deployment profiles in ignored
`.dataevolver/local/` files. The public base has no required checkpoint or agent tool.

1. Inspect Git status, Python, disk and the selected host environment. Inspect GPU
   memory and processes before authorized GPU work; preserve existing jobs.
2. Install `.[harness]` in an isolated source-checkout environment. The CLI commands
   `--help` and `operator-sdk catalog` inspect interfaces without model inference.
3. Bind data plans with `compile-plan`; bind configured native processes with
   `worker-compile`. Both use the same `create`, `step`, `status` and `replay` flow.
4. Bind each model through `ModelProfile`, `ModelPolicy` and `ModelGateway`. Set the
   model ID, revision, endpoint or registered implementation explicitly. No role
   chooses a checkpoint or searches a host-specific model directory automatically.
5. Run source checks with Python 3.12:
   `python .agents/skills/verify-dataevolver/scripts/check_repository.py`.
6. Run tests or workloads only when authorized. A test setup uses `.[test]` and a
   short temporary directory for Unix socket fixtures. Preserve failure evidence
   and do not repeat an effect with an unknown completion status.

Use `configs/worker.example.json` as the neutral native-process template. Replace
its executable and input paths in a private profile. The worker implements the
request/result ABI in `docs/NATIVE_SCENE_WORKFLOW.md`; arbitrary executables do not
implement that ABI automatically. The compiler freezes binary, code, input and
allowed-environment hashes. Compile and execute with the same environment bindings.

## Extension boundaries

For a protocol-compatible served model, edit the model profile and policy in
`configs/harness/model_profiles.example.json`. `model_env` overrides the served
model ID; it does not select a checkpoint directory. Native worker adapters have
their own explicit checkpoint-path environment bindings. The configured endpoint
origin must be authorized in the policy. The `local_endpoint` name describes the
transport; it does not enforce host network isolation.

For a different native architecture, register a lazy `ModelProvider` factory in
the host's `ModelGateway` assembly, or expose the model through a compatible
endpoint. A new external tool can use `worker-compile` and the worker ABI. Graph
steps use `declare_step` and explicit compiler/runtime registration. The built-in
CLI catalog lists its installed graph steps; it is not the list of every composite
operator or runtime. See `docs/MODEL_GATEWAY.md` and `docs/OPERATOR_DEVELOPMENT.md`.

Public `schema_version`, operator versions and checkpoint release identifiers
identify compatibility and provenance. Preserve them when extending the base.
Internal development stages do not belong in public interfaces or deployment examples.

The repository supplies optional model and scene adapters. Their architecture,
runtime, GPU and artifact checks apply when explicitly selected. A CLI reasoning
provider is one optional transport; a host can instead bind a served model or a
registered native provider. Execution authority stays with the Controller.

Model weights, engine binaries, datasets, deployment manifests and native runtime
receipts belong outside code Git. Frozen goals contain resolved execution paths;
keep those generated files private. New code or model bindings require a new goal.

The old production and standalone multimodal CLIs and batch runners are retired.
The previous VLM dual loop is available on `codex/legacy-vlm-dual-loop`.
The project verification skill is `.agents/skills/verify-dataevolver/SKILL.md`;
its runtime scenarios remain drafts until executed with authorization.
