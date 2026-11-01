# Native worker contract and optional scene adapters

The existing `EvolutionController` executes one configured native worker through
`NativeSceneOperator`. `worker-compile` accepts a validated stage identifier, an
explicit executable and the inputs/outputs described below. `scene-compile` is a
compatible alias. The neutral template is `configs/worker.example.json`.

Blender and UE5 are optional implementations of this worker contract.
Their model proposals, process exit status and artifact integrity serve different
purposes. The host decides what code is eligible to run; a model cannot replace
the worker command through its generated parameters.

## Prepare the host

Install the editable source checkout with `.[harness]`. Blender, UE5, Docker,
the UE project, local assets and the selected GPU belong to the execution host.
Set `PYTHONPATH` to this checkout's `src` for Blender's embedded Python. Keep
deployment paths and profiles in ignored `.dataevolver/local/` files.
Blender argv must include `--python-use-system-env` and `--python-exit-code 1`
before `--python`; the supplied profiles include both flags.

`configs/scene/blender.example.json` contains a complete component-modeling
profile. Its `argv` resolves `blender` from the active host. Change that entry if
Blender is installed elsewhere. Paths in `files` and `inputs` resolve relative
to the profile file. `argv` uses absolute bound code paths or literal arguments.
`{request}` and `{output}` must be separate arguments.

For UE5, edit `ue5.example.json` and `ue5-render-manifest.example.json`. Replace
the example host, GPU UUID, engine version, source/trajectory hashes and paths.
The process requires a GPU-isolated Docker runtime with the required UE project,
libraries and assets mounted at the same absolute paths. Choose a campaign root
under the manifest's `<root>/experiments/`, for example
`<render-root>/experiments/campaigns`.

The render manifest pins the engine, project, native renderer, resource helper,
scene configuration and dependencies. `scene-compile` also hashes the project's
Content, Config and Plugins files. The helper `workflows/scene/gpu_lease.py`
retains the original Blender/UE resource-lock convention without launching a
separate process controller. The backend checks the selected GPU again before
each render. Host resource admission remains required.

The optional `photo-v1` look development profile requires explicit
`lookdev_assets.texture_directory` and `lookdev_assets.detail_mesh` paths. The
host freezes these files with the render contract. The default frozen-scene
recipe does not require this optional kitchen-specific material profile.

## Submit a task

```bash
python -m dataevolver scene-compile --profile /path/to/scene-profile.json \
  --output /path/to/prepared/task --campaign task --campaign-root /path/to/campaigns
python -m dataevolver create --goal /path/to/prepared/task/goal.json \
  --campaign-root /path/to/campaigns --idempotency-key task
python -m dataevolver step --campaign task --campaign-root /path/to/campaigns --max-actions 8
python -m dataevolver status --campaign task --campaign-root /path/to/campaigns
python -m dataevolver replay --campaign task --campaign-root /path/to/campaigns
```

Compilation hashes inputs and code and writes a GoalSpec. It performs no model
inference or rendering. `create` freezes the ordinary Harness contract. `step`
executes the registered worker. A new modeling candidate or changed model profile
requires a new goal; existing frozen runs keep their original source and hashes.

An execution-capable Codex agent can author the recipe, prepare a candidate script
and invoke these commands within its authorized budget. The planner-only Codex
transport returns structured proposals and disables execution tools. Both paths
leave workload execution, accounting and recovery with the Controller.

## Worker contract

The profile declares `stage`, fixed `argv`, `files`, `inputs`, named relative
`outputs`, `timeout_seconds`, `allowed_env_keys`, `allow_gpu` and `allow_network`.
Compilation adds file hashes and the `dataevolver.native_scene.v2` contract.
It also hashes the actual allowed environment, distinguishing an unset value
from an empty value. Execution checks that digest before starting the process
and passes the checked snapshot. Raw environment values are not written into
the goal. List every model manifest, checkpoint or tool dependency that affects
results in `files` or `inputs`; hashing an environment value does not hash the
files to which it points.
Input and code files must remain outside the writable campaign directory.

Recompile earlier worker profiles and create a new campaign to use this contract.
The request/result ABI remains unchanged. Older frozen campaigns require their
matching source revision and evidence; the compiler does not migrate their receipts.
Verified completed effects still belong to their original frozen environment
and can be read back without launching a process under the current environment.

The worker receives a JSON request and its native output directory. Request
inputs contain absolute paths and SHA-256 values. The worker writes its declared
artifacts and `result.json`:

```json
{"status": "completed", "request_sha256": "canonical-request-sha256"}
```

Use `dataevolver.dataset.contracts.payload_sha256(request)` for that digest.
The host records the exact argv, request, log, exit code and independently
computed output hashes. It checks JSON parsing and `.blend`/`.glb` headers.
UE5 additionally requires a `render_index` and verifies every native frame,
render/decode receipt and optional pose-capture contract.

The supplied Blender workers are:

- `blender/model_scene.py`: component recipes → `.blend` and `.glb`.
- `blender/simulate.py`: an existing rigid-body scene plus `simulation_spec` →
  sampled trace, simulation metadata and frozen render scene. Its fixed argv
  includes the source `.blend`, `--mode nominal` or `--mode refined`, and the
  usual `--request`/`--output` arguments. It rejects deformable simulations and
  unsupported nonzero initial velocities.
- `ue5/export_frozen_scene.py`: export a prepared replay scene through the same
  request/output contract. `blender-export.example.json` binds `scene`, `trace`,
  `simulation` and `scene_spec` inputs. The spec supplies `family`; declared
  outputs cover `package/manifest.json`, GLB, trace and simulation metadata.

The UE5 worker consumes a `render_manifest`. Child UE scripts use the
`DATAEVOLVER_*` environment convention. A frozen scene package must include
the original source hash, GLB, cameras, trajectory and any material assets.
A component modeling result alone does not establish this complete render package.

## Recovery and validation

A completed native receipt can be reused after a Controller interruption.
Changed code, inputs, exit receipts or outputs invalidate that reuse. An intent
without verified completion requires reconciliation of the retained files; it
does not trigger automatic execution. Failure and timeout records stay in place.

`quality_verified` remains false. Successful execution does not establish mesh
quality, visual realism, physical calibration or benchmark validity. CPU
subprocess and render-contract fixtures exercise recovery and accounting without
Blender, UE5 or model weights. A native host smoke run is a separate check.

Imported implementation and fixture provenance is recorded in
`SCENE_BACKEND_SOURCES.json`. Independent WMBench runners, admission controllers
and benchmark probes are not used as DataEvolver execution owners.
