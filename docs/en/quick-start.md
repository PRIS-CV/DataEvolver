# Quick start

[Home](../../README.md) · [Documentation index](../../README.md#documentation) · [简体中文](../zh-CN/quick-start.md)

DataEvolver organizes image and 3D data synthesis and revision around multimodal
large language model training objectives. Use the optional [Workbench](workbench.md)
for browser configuration and chat, or prepare a workflow through the standalone
Harness CLI. Both entry points use the same Harness for plan compilation,
execution records and recovery.

Run command examples from the repository root unless a step changes directory.

For DeepSeek Harness, use the optional [DSH plugin](../../plugins/dsh/README.md).
It exposes the same campaign lifecycle through DSH's tool registry.

## Prerequisites

| Component | Requirement |
|---|---|
| Base development | Editable checkout; Python 3.12 is recommended and used by source CI. |
| Harness dependencies | `.[harness]`; no model checkpoint or engine is required for interface inspection. |
| Browser Workbench | Node.js 22.19+, npm, Git and Python 3.10+ on the workbench host; network access for initial installation. |
| Native/model execution | The selected tool's OS, engine, model dependencies and host permissions. |
| GPU workloads | Capacity and CUDA/runtime compatibility for the explicitly selected adapter. |
| Website development | Node.js 22+ and npm; independent of model execution. |
| Optional DSH plugin | Node.js 24.9+, a compatible DSH installation and DataEvolver on the selected execution host. |

### Required Models

The base has no universal required model set. Bind the roles needed by your task:
planning requires a supported structured-output provider; image/video generation,
review and segmentation require the corresponding registered implementation.
[Visual adapter documentation](../LOCAL_VISUAL_MODELS.md) lists optional released
implementations and their exact identities. A different implementation needs its
own adapter; a tool-only worker may need no model.

### Optional HYWorld / WorldMirror Scene Reconstruction

The [HYWorld workflows](../../src/dataevolver/workflows/hyworld/) and
[scene contract](../../src/dataevolver/runtime/hyworld_scene_contract.py) retain the
scene-context route from earlier versions. Their vendor sources, weights and
native dependencies are host-supplied. This is an optional route alongside the
[current Blender/UE5 workers](../NATIVE_SCENE_WORKFLOW.md).

## Quick Start

Clone the repository, then choose a browser or CLI setup:

```bash
git clone https://github.com/PRIS-CV/DataEvolver.git
cd DataEvolver
```

### Browser Workbench

From the checkout root, run:

```bash
./apps/workbench/launch.sh
```

By default, the launcher creates `.venv`, installs `.[harness]` and the app's locked Node
dependencies, obtains the pinned agent extension, builds the frontend and starts
the local workbench.

1. If the browser does not open, open the startup URL printed in the terminal.
2. In **Settings**, add your provider, endpoint and model IDs, then select the main and child agents.
3. Configure the image routes and native tool paths your workflow needs.
4. In **Credentials**, enter each route's key environment name and API key, then click **Stage key for save**.
5. Click **Save configuration**.
6. Open a new conversation.

Use **Harness** for plan compilation and Campaign commands.

Blender, UE5, model weights, inference services and remote environments are
host-supplied. For server or hybrid use, follow [OpenSSH deployments](workbench.md#openssh-deployments).
The root `./launch.sh` remains a compatibility entry point.
See the [Workbench guide](workbench.md) for configuration, launcher options and recovery.

### Option A: Let your AI agent prepare the configuration

Give Codex or another execution-capable agent this starting instruction:

```text
Read AGENT_SETUP.md and the selected workflow contract.
Inspect the authorized execution host and existing jobs.
Bind the chosen models/tools explicitly in private profiles.
Prepare a plan or worker profile, compile its goal, then use the Harness CLI.
Inspect state and receipts between steps; preserve unknown effects.
Only execute workloads that the user has authorized.
```

The [onboarding skill](../../.agents/skills/dataevolver-onboarding/SKILL.md) follows this
flow and does not select a model or tool for the user.

### Option B: Manual setup

#### 1. Clone the repository

Use the checkout created above and keep the repository root as the working directory.

#### 2. Install the package entry points

Use Python 3.12, as used by the source-contract CI, and an editable checkout. The
CLI reads repository `configs/` alongside `src/`.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[harness]'
dataevolver --help
dataevolver operator-sdk catalog
```

`dataevolver`, `dataevolver-harness` and `python -m dataevolver` share
[`harness.cli:main`](../../src/dataevolver/harness/cli.py). Inspecting the CLI and catalog
requires no checkpoint or rendering engine. See [agent integration](agents.md) for
host preparation and private configuration.

#### 3. Inspect the interfaces

`--help` and `operator-sdk catalog` inspect the installed interfaces without
inference. The retired bootstrap/onboarding dry-run is replaced by the onboarding
skill and explicit profile preparation.

#### 4. Configure a workflow

Follow [model/tool configuration](deployment.md#replace-models-and-tools), then
[plan and execute a task](harness.md#task-lifecycle). Keep private paths and credentials in
the ignored local configuration. Review the compiled bindings before creation.

#### 5. Runtime acceptance

Runtime acceptance needs the intended host and authorized workloads. The
[verification skill](../../.agents/skills/verify-dataevolver/SKILL.md) describes the
scenarios; their current execution status is [documented in the development guide](development.md#migration-and-verification).
No historical test count qualifies the current integration.
