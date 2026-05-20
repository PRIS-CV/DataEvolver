# Stage0 Websearch Handoff

This handoff describes the DataEvolver Stage0 websearch framework and the local
test package produced on 2026-05-14.

## Purpose

Stage0 websearch converts a user research task into a small, high-precision set
of arXiv links plus a summary document. The next pipeline step should consume:

```text
ARXIV_LINKS_AND_SUMMARY.md
arxiv_links_and_summary.json
NEXT_AI_REPRO_PROMPT.md
DATASET_WORKFLOW_HANDOFF.md
```

`research_prior.json` is also generated for Stage1 soft guidance, but it is not
the primary handoff requested here.

## Core Files

```text
scripts/stage0_web_research.py
scripts/run_ldr_stage0_research.py
scripts/RESEARCH_PRIOR_STAGE0.md
pipeline/research_prior_utils.py
scripts/sync_research_prior_to_server.sh
scripts/prior_kb.py
docs/STAGE0_WEBSEARCH_TEST_REPORT.md
docs/STAGE0_WEBSEARCH_HANDOFF.md
runtime/AI_REPRODUCTION_GUIDE.md
runtime/NEW_AI_REPRO_PROMPT_TEMPLATE.md
```

## Framework Boundaries

The framework is DataEvolver-native and agent-run.

It does:

- create one research session folder per task
- generate search questions from a source-based template
- record evidence cards for selected papers
- produce a Markdown links-plus-summary file
- produce a JSON links-plus-summary file
- produce a downstream AI reproduction prompt aligned with runtime templates
- produce a dataset workflow handoff for either single-paper reproduction or
  multi-paper universal dataset construction
- produce a `research_prior.json` for Stage1 soft guidance

It does not:

- call `local-deep-research` runtime APIs
- call a Python-side websearch backend
- ask the user for extra API keys
- make prior guidance a hard pipeline gate

## Relation To local-deep-research

The implementation reuses design ideas from `local-deep-research`, especially:

- iterative source-based research
- original query plus generated follow-up questions
- accumulated source evidence
- citation-aware synthesis
- report structure separated from sources

The references are recorded in every `stage0_session.json` under
`framework_references`.

## Alignment With AI Reproduction Prompts

Stage0 is upstream of the reproduction workflow in
`runtime/AI_REPRODUCTION_GUIDE.md`. It does not attempt reproduction itself.
Instead, it narrows a research task to high-signal arXiv candidates and writes
`NEXT_AI_REPRO_PROMPT.md`, which follows the shape of
`runtime/NEW_AI_REPRO_PROMPT_TEMPLATE.md`.

Mapping:

- Stage0 `arxiv_links` become the downstream `论文` input.
- Stage0 `summary` becomes context for paper selection and method-line focus.
- Stage0 does not fill work directory, large-file directory, server, or compute;
  these remain placeholders because they are user/environment decisions.
- Downstream agent is explicitly instructed to create
  `docs/METHOD_CONSTRAINT_CARD.md`, update `REPRO_STATUS.json`, label artifact
  lineage, and report reproduction level.

## Standard Workflow

Create a session for a single-paper reproduction:

```bash
python scripts/stage0_web_research.py init \
  --query "<research task>" \
  --output-dir runtime/research_priors/<batch>/<task_slug> \
  --dataset-mode single-paper \
  --tag <tag>
```

Create a session for a multi-paper universal dataset contract:

```bash
python scripts/stage0_web_research.py init \
  --query "SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation" \
  --output-dir runtime/research_priors/<batch>/universal_3d_layout \
  --dataset-mode universal \
  --tag universal-3d-layout
```

Search and fetch papers with the active Codex or Claude Code agent. Then record
selected papers:

```bash
python scripts/stage0_web_research.py add-evidence \
  --session-dir runtime/research_priors/<batch>/<task_slug> \
  --url "https://arxiv.org/abs/..." \
  --title "..." \
  --source-type paper \
  --summary "..." \
  --methodology-note "..." \
  --dataset-workflow "..." \
  --quality-gate "..." \
  --stage1-seed-guidance "..." \
  --stage1-prompt-guidance "..."
```

Finalize:

```bash
python scripts/stage0_web_research.py finalize \
  --session-dir runtime/research_priors/<batch>/<task_slug> \
  --max-papers 5
```

The formal output is:

```text
runtime/research_priors/<batch>/<task_slug>/ARXIV_LINKS_AND_SUMMARY.md
runtime/research_priors/<batch>/<task_slug>/arxiv_links_and_summary.json
runtime/research_priors/<batch>/<task_slug>/NEXT_AI_REPRO_PROMPT.md
runtime/research_priors/<batch>/<task_slug>/DATASET_WORKFLOW_HANDOFF.md
```

`NEXT_AI_REPRO_PROMPT.md` is the bridge to
`runtime/AI_REPRODUCTION_GUIDE.md` and `runtime/NEW_AI_REPRO_PROMPT_TEMPLATE.md`.
It takes the selected arXiv links and Stage0 summary, then asks the downstream
agent to perform method-line decomposition, artifact labeling, small-scale
reproduction, `METHOD_CONSTRAINT_CARD.md`, `REPRO_STATUS.json`, and final
reproduction-level judgment.

`DATASET_WORKFLOW_HANDOFF.md` is the bridge to dataset construction:

- In `single-paper` mode, it keeps the handoff focused on one paper-specific
  small-scale reproduction and artifact lineage.
- In `universal` mode, it treats selected papers as a related paper family and
  maps their common data needs into the universal dataset contract documented in
  `docs/UNIVERSAL_3D_LAYOUT_DATASET.md`.

## Local Test Package

The local test produced two task folders:

```text
runtime/research_priors/local_websearch_smoke_20260514/instruction_editing_datasets/
runtime/research_priors/local_websearch_smoke_20260514/spatial_rotation_editing/
```

Each folder contains:

```text
stage0_session.json
AGENT_RESEARCH_BRIEF.md
research_prior.template.json
research_prior.schema.json
stage0_manifest.json
RESEARCH_PRIOR.md
research_prior.json
ARXIV_LINKS_AND_SUMMARY.md
arxiv_links_and_summary.json
NEXT_AI_REPRO_PROMPT.md
```

## Handoff Package Contents

The zip package should include:

```text
code/
docs/
test_outputs/
HANDOFF.md
```

Where:

- `code/` contains Stage0 framework code and integration helpers.
- `docs/` contains this handoff and the test report.
- `test_outputs/` contains the two local research session folders.
- `HANDOFF.md` points to the important files and expected next-step inputs.

## Validation

Minimum validation commands:

```bash
python -m py_compile scripts/stage0_web_research.py scripts/run_ldr_stage0_research.py
python scripts/stage0_web_research.py validate --prior-path <session>/research_prior.json
git diff --check
```

For the local smoke test, both `arxiv_links_and_summary.json` files contain
exactly five arXiv links.

## Next Owner Notes

Use the framework as a session recorder and synthesis harness. The agent should
still do the actual WebSearch/WebFetch calls. Keep paper selection precise:
fewer strong papers are better than many loosely related papers.

Do not convert this into a direct `local-deep-research` runtime dependency
unless the user explicitly changes the requirement. The current requirement is
to use the local-deep-research style and structure, not its API key dependent
backend.
