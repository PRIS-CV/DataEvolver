<p align="center">
  <img src="apps/project-page/frontend/logo.png" alt="DataEvolver logo" width="120"/>
</p>

<h1 align="center">DataEvolver</h1>

<p align="center">
  <strong>Automated, self-evolving image and 3D data synthesis for multimodal large language model training.</strong><br/>
  Guide generation, evaluation and data revision with training objectives.
</p>

<p align="center">
  <a href="https://pris-cv.github.io/DataEvolver/"><img src="https://img.shields.io/badge/Website-Project%20Page-1f6feb?logo=githubpages&logoColor=white" alt="Project website"/></a>
  <a href="https://arxiv.org/abs/2605.01789"><img src="https://img.shields.io/badge/arXiv-Technical%20Report-b31b1b?logo=arxiv&logoColor=white" alt="First technical report"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="License: Apache-2.0"/></a>
</p>

<p align="center">
  <strong>English</strong> · <a href="README_zh.md">简体中文</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="docs/en/quick-start.md">Quick start</a>
</p>

DataEvolver organizes image and 3D data synthesis and revision around multimodal
large language model training objectives. Its Harness connects data processing,
model interfaces and native tools. An external agent prepares plans and code; the Controller
manages campaign state, execution records and recovery. Deployment choices live
in private profiles.

![DataEvolver historical project poster](apps/project-page/frontend/images/showcase-poster.png)

*The original project poster presents the earlier rendering and review design.*

## DataEvolver and DataRSI

**DataEvolver is the engineering project; DataRSI is its subsequent research work.**
This repository provides a base for developers to fork and extend.
The first [technical report](https://arxiv.org/abs/2605.01789) describes the earlier
goal-driven agents. [DataRSI](https://github.com/PRIS-CV/DataRSI), *A 3D Data Harness
for Failure-to-Data Evolution*, studies bounded data revision and model admission.

The previous VLM dual loop remains on the
[legacy branch](https://github.com/PRIS-CV/DataEvolver/tree/codex/legacy-vlm-dual-loop).
See [research and project history](docs/en/research.md) for results and provenance.

## Why DataEvolver?

Data construction involves more than selecting a model: plans need compatible
operators, processes need explicit inputs and outputs, and interrupted runs need
recoverable evidence. DataEvolver gives those concerns shared contracts and one
Controller. An agent can propose a different workflow without creating another
execution state machine.

## Current Capabilities

| Area | Current code interface |
|---|---|
| Data workflows | Registered image/catalog graph steps, artifact stores and validation. |
| Agent planning | Structured plans, explicit model-role bindings and bounded proposals. |
| Multimodal tools | Optional generation, editing, prompt enhancement, observation, review and segmentation adapters. |
| 3D workflows | Optional Blender modeling/simulation/export and UE5 rendering workers. |
| Research preparation | Stage0 evidence sessions and handoff exports for single-paper or multi-paper work. |

These are implementation entry points. Deployment readiness and output quality
must be established for the selected host and configuration.

Image collection and multi-subject layout curation provide `SourceBundle.v1` and
portable `curated_images.v2` outputs. See the [pipeline guide](docs/IMAGE_DATA_PIPELINES.md)
and [server handoff](docs/IMAGE_DATA_SERVER_HANDOFF.md). The route is verified with
offline fixtures; real acquisition and model calibration remain server work.

The Controller owns state, scheduling, budgets and recovery. Models and native
tools connect through explicit contracts. See
[deployment and adapter limits](docs/en/deployment.md#local-configuration-and-adapter-limits).

## Pipeline Overview

A `GoalSpec` binds the task, operators, budgets and evaluation rules.
The Controller dispatches work, checks evidence and records campaign transitions.

[![Animated DataEvolver Harness architecture](apps/project-page/frontend/architecture/preview-en.gif)](https://pris-cv.github.io/DataEvolver/architecture/)

[Open the interactive HTML](https://pris-cv.github.io/DataEvolver/architecture/) ·
[Editable diagram specification](apps/project-page/frontend/architecture/en.spec.json) ·
[Execution and recovery guide](docs/en/harness.md)

The diagram uses the default [arch-diagram](https://github.com/bassnova42/arch-diagram-skill) design.

## Project Page trace showcase

The [project website](https://pris-cv.github.io/DataEvolver/) presents rendering
examples, research figures and a gallery.
The [trace archive](https://pris-cv.github.io/DataEvolver/traces/?lang=en) retains
historical ground-contact and bottle-count repairs, including regressions and
unresolved cases. These examples do not establish new training gains.

![Historical DataEvolver workflow: generation, reconstruction, rendering and review](apps/project-page/frontend/images/readme/showcase.png)

[Earlier scene routes and visuals](docs/en/research.md#project-history-and-visuals) ·
[Run the website locally](docs/en/deployment.md#project-website-deployment)

## DataRSI research update

DataRSI connects data revision with model admission under declared budgets and
regression checks. Its reported study covers one bounded revision.

![DataRSI method: data revision, failure-to-data feedback and model admission](apps/project-page/frontend/images/paper/datarsi-method.jpg)

[Manuscript results and limitations](docs/en/research.md#datarsi-research-update) ·
[Research repository](https://github.com/PRIS-CV/DataRSI)

## DataEvolver-Rotate

The original project includes a historical rotation-conditioned image-editing
dataset. Its version, splits and loading example are documented in the
[dataset guide](docs/en/research.md#dataevolver-rotate).

## Documentation

<!-- Moved sections remain reachable through their original README anchors. -->
<a id="key-features"></a>
<a id="prerequisites"></a>
<a id="quick-start"></a>
<a id="task-lifecycle"></a>
<a id="inspect-pause-and-recover"></a>
<a id="replace-models-and-tools"></a>
<a id="reproducible-research-prior-workflow"></a>
<a id="project-structure"></a>
<a id="extension-boundaries"></a>
<a id="action-space"></a>
<a id="using-with-claude-code"></a>
<a id="migration-and-verification"></a>
<a id="project-history-and-visuals"></a>

Start the optional browser Workbench with `./apps/workbench/launch.sh` from the
checkout root. See [browser setup](docs/en/quick-start.md#browser-workbench) for
requirements and first-run configuration, or use the standalone Harness CLI.
The launcher and UI live in `apps/workbench/`; the core lives in `src/dataevolver/harness/`.

| I want to… | Guide | Contents |
|---|---|---|
| Configure agents and chat in a browser | [Private Workbench](docs/en/workbench.md) | Optional app and launcher in `apps/workbench/`; agent profiles, image APIs, native tools and OpenSSH. |
| Inspect collection/curation Goals and review exported samples | [Data workspace](docs/en/data-workspace.md) | Existing image pipelines and Campaign controls; read-only local/SSH package viewer. |
| Explore and review candidate workflows | [Candidate orchestration](docs/en/orchestration.md) | Separate proposal workspaces, bounded execution context and configurable dispatch queues. |
| Use DataEvolver from DeepSeek Harness | [DSH plugin](plugins/dsh/README.md) | Native tool bundle, private configuration and existing campaign commands. |
| Set up the source checkout | [Quick start](docs/en/quick-start.md) | Browser launch, first-run configuration and standalone CLI installation. |
| Configure a deployment | [Models, tools and deployment](docs/en/deployment.md) | Profiles, providers, native workers, local settings and website setup. |
| Plan, execute or recover a task | [Harness lifecycle](docs/en/harness.md) | Compile, create, step, inspect, pause, resume and receipts. |
| Work through Codex or another agent | [Agent integration](docs/en/agents.md) | Outer agents, internal planner providers and optional Claude Code setup. |
| Fork and extend the code | [Development and extension](docs/en/development.md) | Directory map, operators, workflows, evaluation, action space and migration. |
| Read the research and datasets | [Research and project history](docs/en/research.md) | Results, research priors, Rotate, historical visuals and citations. |

Each guide has a Chinese counterpart and links back to this index.
Source-level API references remain under [docs/](docs/).

## Honors & Recognition

The earlier project is listed as a Poster Presentation at
[AIDataSci 2026 / KDD Workshop](https://usail-hkust.github.io/aidatasci/accepted-papers.html)
and received a poster presentation certificate at the 2026 BAAI Agent for Science competition.

<p align="center">
  <img src="apps/project-page/frontend/images/readme/baai_agent4s_poster_certificate.jpg" alt="DataEvolver BAAI poster presentation certificate" width="720"/>
</p>

[Recognition details](docs/en/research.md#honors--recognition)

## Roadmap

Implementation status, remaining host validation and research directions are
tracked in the [development guide](docs/en/development.md#roadmap).

## Citation

Use the [BibTeX entry for the first DataEvolver technical report](docs/en/research.md#citation).
For DataRSI releases and citations, consult its [research repository](https://github.com/PRIS-CV/DataRSI).

## License

[Apache-2.0](LICENSE)
