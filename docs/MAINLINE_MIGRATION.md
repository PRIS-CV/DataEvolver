# Public base migration

`main` is the portable development base. `codex/public-base` preserves the reviewed
publication commit. `codex/legacy-vlm-dual-loop` preserves the previous public main
at `952dfcb71203563ca3b2c0f99a56bf28a15316ea`.

## Source and scope

The base supplies the Controller, typed model gateway, operator lifecycle, graph
compiler and contract fixtures. The optional scene backend comes from
`WMBench-Synthesis-Harness`; `SCENE_BACKEND_SOURCES.json` records its source and
integration hashes. Benchmark audit and probe changes remain outside this update.

The public commit contains the reviewed source tree. Local deployment history,
cloud archives, credentials, generated goals and native execution journals stay
outside the published commit. `.dataevolver/` is ignored because frozen execution
records contain resolved host paths.

## Interfaces and compatibility

`dataevolver`, `dataevolver-harness` and `python -m dataevolver` share the existing
Controller. `compile-plan` binds registered graph operators. `worker-compile`
freezes an explicitly configured native process into an ordinary goal;
`scene-compile` remains a compatible alias. Worker stages use validated identifiers
instead of a fixed tool allowlist. Existing optional scene adapters retain their
format and native-render receipt validation.

Removed independent entry points:

- `dataevolver.cli.production` and `dataevolver-production`.
- `dataevolver-multimodal` and its standalone module CLI; registered operators
  retain the `run()` library API.
- `cli/run_all.sh`, `cli/stage4_batch_render.sh` and the old bootstrap script.
- `runtime/production_runtime.py` and its independent worker/manifest runner.

HYWorld's optional environment loading lives in `runtime/hyworld_environment.py`.
Its example profile contains only consumed environment fields. Low-level review,
repair, reconstruction, qualification and training modules with live callers
remain available as explicit adapters and versioned protocols.

## Explicit deployment bindings

No route selects a default checkpoint. Low-level multimodal requests take explicit
model values or `DATAEVOLVER_<ROUTE>_MODEL` / `DATAEVOLVER_<ROUTE>_MODEL_PATH`.
Specialized legacy adapters also accept their documented environment bindings;
unset checkpoint paths no longer resolve to a presumed local model directory.
The generic model gateway takes an explicit profile, policy and provider binding.

Optional image generation/editing, prompt enhancement and review implementations
remain available in `docs/LOCAL_VISUAL_MODELS.md`. Their implementation-specific
architecture and weight checks remain intact. They do not restrict the generic
model profile interface. A different model API needs its own registered adapter.
The optional CLI reasoning transport likewise requires an explicit model/tool
binding and retains planner-only limits, native journals and recovery checks.

Existing frozen runs keep their original code, configuration and evidence hashes.
Changing a worker, model or code binding requires a new goal and campaign. Restore
the matching source and deployment when reading or resuming an older frozen run.

Native worker configuration uses the public `dataevolver.native_scene.v2` schema
to include the allowed environment's value hash. Recompile a previous worker
profile for new execution. Worker request/result messages keep their existing
ABI. Environment value changes require a new compiled goal; changing the files
behind a path also requires updating the declared file inventory.

The model gateway's generic ledger is independent of native model formats.
Optional providers supply native accounting through `audit_native_usage`.
SAM-specific fields remain available with that provider; other providers no
longer receive placeholder SAM zero totals. The gateway rejects accounting
extensions that overwrite ledger fields or return invalid counter values.

Public schema/operator versions and released checkpoint identities are retained.
Old development-stage guide names have been replaced by `MODEL_GATEWAY.md`,
`IMAGE_CURATION.md` and `IMAGE_RECIPE_SEARCH.md`.

## Review and verification

Source review removed a stale test loader that referenced a deleted verifier and
retired duplicate CLI orchestration. Source checks cover syntax, internal module
paths, public entry points and optional Blender launch flags. The CI source job
runs no tests, models or native workers.

Historical validation belongs to its original source revisions. This integration
has received source/configuration checks and independent code review. Its tests,
model loading, Blender execution and UE5 rendering have not been run. Runtime
scenarios in `.agents/skills/verify-dataevolver/` remain drafts.
