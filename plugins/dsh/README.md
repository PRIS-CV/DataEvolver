# DataEvolver plugin for DeepSeek Harness

[Home](../../README.md) · [简体中文](README_zh.md)

DataEvolver organizes image and 3D data synthesis and revision around multimodal
large language model training objectives. This optional bundle lets a DeepSeek
Harness (DSH) agent call the existing DataEvolver Harness through
`dataevolver_harness`. DSH owns conversations, model selection and agent delegation;
DataEvolver owns campaign state, operators, budgets, receipts and recovery.

The plugin lives in `plugins/dsh/`. It uses DSH's tool registry, subprocess provider
and session sandbox policy. It has no dependency on the standalone Workbench,
Pi or SoL-Pi, and adds no second Controller or server.

## Install from this checkout

Use Node.js 24.9+ and a compatible DSH installation. The bundle targets the public
interfaces in [DSH commit ce7073d](https://github.com/deepseek-ai/deepseek-harness/tree/ce7073d51254c26cd63b22a60835b8f948deb447),
version `0.2.2-alpha.2`. DSH is a developer preview; the declared compatibility
range is an interface target, not a runtime acceptance result.

1. Install DSH using its [official instructions](https://github.com/deepseek-ai/deepseek-harness#run).
2. Install DataEvolver on the intended execution host using the
   [CLI setup](../../docs/en/quick-start.md#option-b-manual-setup).
3. From this repository root, initialize or inspect the shipped Web profile and install the bundle:

   ```sh
   dsh --profile web --dump-config
   dsh plugin --profile web add ./plugins/dsh
   ```

   The first command initializes `web` from its shipped template only when missing.
   Plugin installation uses that profile's package manager. This bundle ships
   native ESM files, so it needs no build or lifecycle script.

4. Copy [config.example.patch.yml](config.example.patch.yml) to `config.local.patch.yml`
   beside the example. The plugin's `.gitignore` excludes that private file.
   Set `command` to your installed `dataevolver` executable or an argv prefix such
   as your Python executable followed by `-m`, `dataevolver`.
5. Start DSH with your private overlay:

   ```sh
   dsh --profile web --patch /absolute/path/to/config.local.patch.yml
   ```

Open DSH's printed startup URL. Configure the agent provider and model in DSH,
then ask the agent to use `dataevolver_harness`. Calls and results appear in DSH's
existing conversation tool cards. The plugin adds no separate dashboard.

To remove it, run `dsh plugin --profile web remove dsh-dataevolver` and remove its
private overlay. Keep campaign files if you intend to inspect or continue runs.

## Configuration

| Setting | Default | Meaning |
|---|---|---|
| `command` | `[dataevolver]` | Executable and fixed arguments; passed as argv without a shell. The executable must be an absolute path or a PATH name. |
| `workingDirectory` | Original session directory, or policy root | Absolute directory in the subprocess execution world. Agentless calls fall back to the policy root. The sandbox boundary still comes from the session policy. |
| `campaignRoot` | `.dataevolver/campaigns` | Absolute campaign directory, or a relative path resolved against the spawned command's working directory. |
| `maxActions` | `1` | Default and maximum action count for each `step` call. |
| `timeoutMs` | `300000` | Foreground call deadline, including executable lookup and sandbox preparation. |
| `graceMs` | `5000` | Termination grace passed to the subprocess provider. |
| `maxOutputBytes` | `262144` | Retained tail limit for each of stdout and stderr. |
| `forwardEnv` | `[]` | Explicit names to copy from the DSH host environment into the child. |

DSH's profile editor or `--patch` controls these settings. A patch replaces the
row's whole configuration; omitted fields receive the schema defaults.
The plugin passes `workingDirectory` unchanged to the execution provider. The plugin
does not resolve a remote path against the local filesystem.

DataEvolver model bindings, evaluator rules, Blender/UE5 executables, assets and
workflow environments remain in its private plan and worker profiles. See
[deployment](../../docs/en/deployment.md#replace-models-and-tools) and
[task compilation](../../docs/en/harness.md#task-lifecycle). The plugin leaves model IDs,
native tool executables and server addresses to deployment configuration.

DSH subprocess providers scrub credential-shaped environment names by default.
If a DataEvolver model profile uses `api_key_env`, list that variable in
`forwardEnv` and supply its value in the DSH host environment. Forward endpoint
and model variables too when a profile references them. The plugin rejects an
unset listed variable. It never accepts secret values in tool arguments or
forwards DSH's agent credentials automatically. SSH keys remain in OpenSSH.

## Tool operations

Each request has an `operation` and only that operation's fields:

| Operation | Required fields | Optional fields / behavior |
|---|---|---|
| `catalog` | None | Calls `operator-sdk catalog` before campaign creation. |
| `compile-plan` | `plan`, `profile`, `evaluator`, `output`, `campaign` | Compiles a registered data graph. |
| `worker-compile` | `profile`, `output`, `campaign` | Compiles an existing native worker profile, including Blender/UE5 routes. |
| `create` | `goal`, `idempotencyKey` | `dryRun`; creates and freezes a campaign. |
| `step` | `campaign` | `maxActions`, `dryRun`; deployment configuration bounds the count. |
| `status` | `campaign` | Reads the saved snapshot. |
| `explain` | `campaign` | `limit`; reads recent events. |
| `replay` | `campaign` | Reconstructs state from events without running a workload. |
| `operators` | `campaign` | Inspects the campaign's operator bindings and drift. |
| `pause` | `campaign` | Requests a pause at a Controller action boundary. |
| `resume` | `campaign`, `resumeExecution: true` | Immediately calls `Controller.run()`; can dispatch work continuously. |

`resumeExecution` acknowledges the command's semantics; it does not grant user
authorization. Use `resume` only when the user has authorized continuous execution.
`step` counts Controller actions, not generated images or seconds; one action
can wait for a native worker. The plugin does not expose the combined `evolve`
command or implicit creation through `step --goal`.

For example, after preparing an authorized worker profile, ask the agent:

```text
Use dataevolver_harness to worker-compile my profile into a new prepared
directory. Show the compiled goal before creating the campaign. After I
authorize execution, create it with a stable idempotencyKey, advance one
Controller action with step, and show status and receipts before continuing.
```

For an inspection request:

```json
{"operation":"status","campaign":"scene-task"}
```

The canonical result contains `operation`, `cwd`, `exitCode`, `signal`, `timedOut`,
stdout/stderr text with truncation flags, and `receipt`. The plugin decodes complete
JSON stdout into `receipt`; truncated or non-JSON stdout leaves `receipt: null`.
Read the exit facts and the receipt's domain fields. A returned process result
does not imply that a campaign was accepted or that data quality improved.
The same canonical value is available to DSH's programmatic tool calling.

## Remote execution and recovery

The plugin consumes DSH's configured subprocess and sandbox services. Their
execution world determines executable paths, the working directory and file
paths. For SSH operation, configure DSH's paired remote filesystem, subprocess
and sandbox providers, and install DataEvolver on that remote host. See the
[DSH SSH documentation](https://github.com/deepseek-ai/deepseek-harness/tree/ce7073d51254c26cd63b22a60835b8f948deb447/packages/ssh).
The plugin creates no SSH connection or tunnel itself. A server-hosted DSH can
use the local subprocess provider on that server.

Calls follow the session sandbox policy. A private `workingDirectory` setting
does not widen writable roots, and a confinement failure does not trigger an
unconfined retry. Configure writable roots through DSH's existing policy when
the workflow needs them. Network and native-tool permissions remain deployment
conditions.

Cancellation, deadlines and plugin unload propagate to the provider's managed
processes. They do not prove that a remote operation had no effects or undo a
completed worker. After an interruption, inspect `status`, `explain` and
`replay`; follow DataEvolver's [recovery contract](../../docs/en/harness.md#inspect-pause-and-recover)
before resuming. The plugin never retries commands automatically.

## Development and verification status

`commands.mjs` owns the operation whitelist, argument validation and CLI bindings.
`index.mjs` owns registration, configuration, sandboxed dispatch and result
collection. Extend an existing DataEvolver Operator or worker behind its public
contracts; keep workflow-specific rules out of this adapter.

The focused cases in [test/commands.test.mjs](test/commands.test.mjs) cover invalid
requests, creation identity, action bounds and continuous-resume acknowledgement.
They have not been run. This integration has only been reviewed statically;
installation, DSH loading, local/SSH execution, cancellation and real model or
renderer behavior have not been verified. No tests, project tasks or CI were run.

This adapter follows DSH's documented
[bundle packaging](https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/docs/user/develop/basic/publish.md)
and [tool contract](https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/docs/cookbook/adding-a-tool.md).
DSH uses the MIT license; this independently implemented adapter uses the
repository's Apache-2.0 license. It does not vendor DSH code or assets.
