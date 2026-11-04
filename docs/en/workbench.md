# Private Workbench

[Home](../../README.md) · [Quick start](quick-start.md) · [简体中文](../zh-CN/workbench.md)

DataEvolver Workbench is a single-user application for configuring models and
tools, talking to a Pi agent, and submitting commands to the existing Harness.
The optional application, launcher and Node dependencies live in
`apps/workbench/`; the core remains in `src/dataevolver/harness/` and can be used
without the UI or Node.js. The root `launch.sh` forwards to the app launcher.
The public project website is a separate app.

## Start

1. Install Node.js 22.19+, npm, Git and Python 3.10+ on the workbench host.
2. Run `./apps/workbench/launch.sh` from the checkout.
3. Open the startup URL printed in the terminal if your browser does not open.
4. Open **Settings**, register a provider and model, then select the main and child agents.
5. Save the configuration and open a new conversation.

The launcher creates `.venv`, installs `.[harness]`, installs the locked Node
dependencies with lifecycle scripts disabled, obtains the pinned SoL-Pi source,
and builds the frontend. It does not install checkpoints, GPU drivers, Blender,
UE5 or remote environments. Initial installation requires network access.
`./apps/workbench/launch.sh --no-install --no-open` reuses prepared dependencies and prints the
browser URL. `--port 3080` selects the listener port.

The default Pi dependency is `@earendil-works/pi-coding-agent` 1.3.0. Its lockfile
records the compatible runtime packages. The workbench loads SoL-Pi through public Pi APIs,
without a Pi fork. Runtime upgrades are dependency changes to review. The
following optional environment settings override installation locations:

| Setting | Purpose |
|---|---|
| `DATAEVOLVER_BOOTSTRAP_PYTHON` | Python used to create the environment. |
| `DATAEVOLVER_VENV` | Python environment directory. |
| `DATAEVOLVER_SOL_ROOT` / `DATAEVOLVER_SOL_REVISION` | Separate SoL-Pi checkout and Git revision. |
| `DATAEVOLVER_WORKBENCH_HOME` | Private configuration, sessions and job logs. |
| `DATAEVOLVER_WORKBENCH_PORT` | Loopback HTTP port. |
| `DATAEVOLVER_WORKBENCH_BROWSER_ORIGIN` | Exact browser origin, including a different SSH-forwarded port. |

The workbench binds to `127.0.0.1`. A one-use startup token establishes an
HttpOnly, SameSite cookie. API requests check Host and Origin. The application
uses the operating-system permissions of its host account. Use it as a private,
single-user application. Agent tools can edit files and run host commands.

## Interface components

The home screen provides entry points for conversations, Harness campaigns and
image work. Settings use six categories with a shared configuration draft and
one save action.

Interactive controls use [Base UI](https://base-ui.com/) and named
[Lucide](https://lucide.dev/) icons. Local component wrappers and their styles
live under `apps/workbench/frontend/`; backend request types remain in
`frontend/types.ts`. The home composition draws on
[Canvas UI](https://github.com/DavidHDev/canvas-ui), while keeping DataEvolver's
own colors and layout. Canvas UI source, shaders and artwork are not bundled.
See the [third-party notices](../../apps/workbench/THIRD_PARTY_NOTICES.md).

## Configure agents and models

Settings groups the configuration into Models and agents, Images, Tools and
paths, Execution and SSH, Agent runtime, and Credentials. Changes remain in the
form when switching categories. Use Save configuration to apply all categories.

Providers select a Pi protocol, endpoint, credential environment name, and model
definitions. Supported protocol choices are OpenAI Chat Completions, OpenAI
Responses, Anthropic Messages and Google Generative AI. Model IDs, context limits,
output limits, reasoning and image inputs are deployment choices. The public base leaves agent model selection to each deployment.

Main and child agents have independent provider/model/thinking settings. Model
switching within a session uses Pi's `set_model` and `set_thinking_level` between
runs. Additions to providers require a new session because each session retains
its configuration snapshot. Existing frozen Harness goals remain unchanged.
Delegation can select an optional registered `profile` per child. `mode=review`
starts a fresh-context evidence review; see [Candidate orchestration](orchestration.md).

Enter credentials in the private credentials card, stage them by environment
name, then save. The form reports only configured names. It never returns a
stored key. Empty key inputs retain stored values; the remove button schedules
deletion. Credentials can also come from the host environment. Host environment
values remain active after a stored override is removed. Start a new conversation after changing credentials;
existing Pi processes retain their original environment.

The default private directory is a checkout-specific directory below
`~/.dataevolver/workbench/`. `config.json` contains deployment settings;
`credentials.json` contains keys. Configuration and receipt JSON files use mode `0600`, and the private root
uses `0700`. Generated Pi `models.json` files reference environment values.
Do not publish this directory, session logs, campaign receipts or generated goals:
they can contain prompts, data and resolved host paths.

A complete [configuration template](../../apps/workbench/config.example.json) uses
placeholder endpoints and model IDs. Edit the private `config.json` while the
workbench is stopped, or use Settings. The Pi `cli` field accepts a Node.js CLI
entrypoint file, not a shell command.

## Images and native tools

Image generation calls `/images/generations`; editing uploads a multipart image
to `/images/edits`. These adapters require `b64_json` results. Prompt enhancement
uses `/chat/completions`, with an optional image. An endpoint exposing
Qwen-Image-2.1 or its text/image prompt enhancer can be configured here if it
implements these protocols. Other model IDs work through the same interfaces.
Changing an ID does not convert a native or proprietary protocol into an
OpenAI-compatible API.

Calls retain an intent and a receipt under the configured output directory. A
network failure is recorded as `unknown` and is not retried automatically. Images
and input paths in the Image Studio refer to the workbench host, including in
hybrid deployments. For an entirely remote dataset workflow, run the workbench
on that host or use a remote Harness worker.

The native-tool settings record executable files, UE project, asset/dataset/output
roots and Python. They are included in the agent's deployment context. Harness
subprocesses receive `DATAEVOLVER_BLENDER_BIN`, `DATAEVOLVER_UE5_BIN`,
`DATAEVOLVER_UE_PROJECT`, `DATAEVOLVER_ASSET_ROOT`, `DATAEVOLVER_DATASET_ROOT` and
`DATAEVOLVER_OUTPUT_ROOT`. Worker profiles must still declare their actual argv,
input/output contracts and allowed environment keys. These settings do not
silently rewrite a frozen worker.

SoL-Pi Action Fusion and ObservationPack are enabled by default. The reducer and
online compaction require explicit configuration. The reducer can send logs to
its configured model, and compaction can call a model and continue the task.
SoL's conversation plan is distinct from a compiled Harness plan. The optional
upstream SoL-Pi Visual example is not loaded by this base; visual inspection can
use Pi's image reader and configured vision model.

## Plan, execute and recover

The agent calls `dataevolver_harness`, which accepts a command and named values.
The backend converts those values to argument vectors for `python -m dataevolver`.
The Harness tab exposes the same command set:

| Command | Required values |
|---|---|
| `plan` | `goal` (existing image pipeline Goal) |
| `compile-plan` | `plan`, `profile`, `evaluator`, `output`, `campaign` |
| `worker-compile` | `profile`, `output`, `campaign`, `campaignRoot` |
| `create` | `goal`, `campaignRoot`, `idempotencyKey`; optional boolean `dryRun` |
| `step` | `campaign`, `campaignRoot`, `maxActions` (1–100) |
| `status`, `explain`, `replay`, `pause`, `resume` | `campaign`, `campaignRoot` |

`compile-plan` consumes the agent-authored structured plan. `create` freezes it
without dispatch. `step` counts Controller actions, not seconds or images.
`resume` immediately calls the Controller's continuous `run()`. Pause takes
effect at an action boundary and does not kill an in-flight worker. Read the
[Harness lifecycle](harness.md) before changing a running campaign.

Pi uses long-lived RPC sessions with `prompt`, `steer`, `follow_up`, queue clearing
and abort. Completion follows `agent_settled`, including automatic compaction
continuations. `dataevolver_delegate` creates a real child Pi session using the
child profile and explicit task context. `dataevolver_child` inspects, waits for,
or stops that child. Children cannot delegate again. Parent abort stops active
children; it does not undo native effects or pause Campaigns.

Browser reconnection reloads authoritative messages and retained job output. After
a service restart, **Reopen session** reopens the persistent Pi session without
sending a prompt. It does not resume a Campaign. Interrupted job transports are
shown as `unknown`; inspect Campaign state and native receipts before proceeding.
Pi history is not an exactly-once execution journal for arbitrary shell commands.

## Candidate orchestration

For isolated proposal sessions, explicit evidence context, dispatch limits and
queue recovery, see [Candidate orchestration](orchestration.md).

## OpenSSH deployments

For hybrid operation, keep the workbench and Pi local. Set execution target
`ssh`, an existing OpenSSH alias, an absolute remote checkout, and remote Python
and tool paths. Install DataEvolver's Harness dependencies in that remote checkout
first. The existing SSH config and ssh-agent supply authentication; private keys
are not uploaded through the page. The backend uses BatchMode, strict host-key
checking and no agent forwarding. Establish host trust through ordinary OpenSSH
before submitting work. Listed aliases are hints from the main SSH config;
aliases from Include files can be entered manually.

Harness commands execute in the remote checkout. Goal/profile/input/output paths
and Campaign roots then refer to the remote host. Full artifacts remain remote;
only command output and receipts return to the workbench. If the SSH connection
breaks, the remote worker might continue. No automatic command resend occurs.
Use remote receipts to reconcile the Campaign.

For full server deployment, run the launcher on the server:

```bash
./apps/workbench/launch.sh --no-open
```

On your laptop, establish a tunnel, substituting your existing alias:

```bash
ssh -N -L 3080:127.0.0.1:3080 your-workstation
```

Open the printed startup URL in the laptop browser. Use the same port at both
ends, or set `DATAEVOLVER_WORKBENCH_BROWSER_ORIGIN` to the exact forwarded origin.
Provider URLs must be reachable from the workbench host. A remote model's
loopback endpoint needs its own SSH forwarding in a hybrid deployment; the
workbench does not provision inference servers or network tunnels automatically.

## Extend and review

The backend separates configuration, Pi sessions, jobs, image API and HTTP
routing. The Pi extension exposes these operations as tools. The React frontend
consumes JSON and SSE; it has no provider credentials or model client.
Existing Python Operator/Provider/Domain interfaces remain the extension points
for campaign semantics. The public project page retains its independent
read-only contract.

The CLI-to-web layout and credential approach follow
[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness). DataEvolver
retains its own UI; it does not include DSH's Cordis plugin runtime. See the
[upstream notices](../../apps/workbench/THIRD_PARTY_NOTICES.md).

This integration was reviewed statically. Node dependencies were resolved with
lifecycle scripts disabled for type checking; the launcher was not executed. Installation, browser interaction,
agent calls, SSH execution, native rendering, tests and CI were not run for this
change. The included contract tests are available for later authorized execution;
source and type checks do not establish runtime acceptance.

## Data collection and curation

Open **Data workspace** for Goal preflight, Campaign control, and read-only exported sample review. See the [Data workspace guide](data-workspace.md) for host configuration and limits.
