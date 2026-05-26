# Stage0 Websearch Test Report

Date: 2026-05-14

This document records the local smoke test for DataEvolver's Stage0 websearch
framework. The goal was to verify that a Codex/Claude Code style agent can use
the local websearch framework to collect a small, high-precision set of arXiv
papers, record intermediate evidence, and produce a next-step handoff containing
only arXiv links plus a summary.

## What Was Tested

Two preset user research tasks were used:

1. `instruction-guided image editing dataset and benchmark construction methods`
2. `spatial rotation and 3D-aware image editing benchmark construction methods`

Each task was initialized as a separate Stage0 session under:

```text
runtime/research_priors/local_websearch_smoke_20260514/
```

Each session produced a separate subfolder, like a conversation/session
category:

```text
instruction_editing_datasets/
spatial_rotation_editing/
```

The formal downstream handoff files are:

```text
ARXIV_LINKS_AND_SUMMARY.md
arxiv_links_and_summary.json
NEXT_AI_REPRO_PROMPT.md
```

`research_prior.json` and `RESEARCH_PRIOR.md` are still generated for Stage1
soft prompt injection and provenance, but the formal next-step output is the
links plus summary pair. `NEXT_AI_REPRO_PROMPT.md` adapts the links and summary
to `runtime/AI_REPRODUCTION_GUIDE.md` and
`runtime/NEW_AI_REPRO_PROMPT_TEMPLATE.md` for downstream paper reproduction.

## Framework Usage

The model did not freely improvise an unstructured search workflow. It used the
DataEvolver Stage0 websearch framework in `scripts/stage0_web_research.py`.

The framework is derived from selected `local-deep-research` concepts:

- `source_based_strategy.py`: source-based iteration, original query plus
  generated follow-up questions, accumulated source evidence, final synthesis.
- `standard_question.py`: first-pass search questions and context-aware
  unanswered follow-up questions.
- `citation_handler.py`: source-indexed evidence records.
- `report_generator.py`: report sections separated from source lists.

The framework did not call `local-deep-research` runtime APIs, LLM backends, or
search backends. It also did not require any extra user API key. The active
agent performed WebSearch/WebFetch in the conversation, then recorded evidence
through the local CLI.

## Preset Prompts And Search Keywords

### Task 1: Instruction Editing Datasets

Initial Stage0 query:

```text
instruction-guided image editing dataset and benchmark construction methods
```

Framework-generated initial questions:

```text
instruction-guided image editing dataset and benchmark construction methods
instruction-guided image editing dataset and benchmark construction methods methodology dataset construction benchmark
instruction-guided image editing dataset and benchmark construction methods arxiv HTML methodology dataset generation
```

Additional WebSearch prompts used by the agent:

```text
arxiv instruction-guided image editing dataset benchmark construction MagicBrush InstructPix2Pix Emu Edit GEdit-Bench
arxiv image editing benchmark dataset construction GEdit-Bench HQ-Edit image editing benchmark
arxiv InstructPix2Pix Learning to Follow Image Editing Instructions
arxiv MagicBrush manually annotated instruction guided image editing dataset
arxiv Emu Edit precise image editing recognition generation tasks
HQ-Edit arxiv image editing dataset
AnyEdit arxiv comprehensive multimodal instruction editing dataset
```

Source hints supplied to the framework:

```text
https://arxiv.org/abs/2211.09800
https://arxiv.org/abs/2306.10012
```

### Task 2: Spatial / Rotation / 3D-Aware Editing

Initial Stage0 query:

```text
spatial rotation and 3D-aware image editing benchmark construction methods
```

Framework-generated initial questions:

```text
spatial rotation and 3D-aware image editing benchmark construction methods
spatial rotation and 3D-aware image editing benchmark construction methods methodology dataset construction benchmark
spatial rotation and 3D-aware image editing benchmark construction methods arxiv HTML methodology dataset generation
```

Additional WebSearch prompts used by the agent:

```text
arxiv SpatialEdit Benchmarking Fine-Grained Image Spatial Editing object rotation
arxiv image spatial editing object rotation dataset benchmark
arxiv AnyEdit image editing dataset object rotation spatial editing
arxiv 3D-Aware Image Editing Benchmark object rotation dataset
arxiv Image Sculpting Precise Object Editing with 3D Geometry Control
arxiv VISOR spatial relationships text-to-image image editing benchmark
arxiv FineEdit Fine-Grained Image Edit Bounding Box Guidance
```

Source hints supplied to the framework:

```text
https://arxiv.org/abs/2604.04911
https://arxiv.org/abs/2401.01702
```

## Intermediate Process

For each task, the agent followed the same workflow:

1. Initialize a Stage0 session with `stage0_web_research.py init`.
2. Use the generated `AGENT_RESEARCH_BRIEF.md` and framework questions as the
   search plan.
3. Search the web for high-signal arXiv papers.
4. Prefer arXiv links over broad web pages because the formal output is arXiv
   links plus summary.
5. Keep the candidate set intentionally small and precise, with a hard maximum
   of five papers per task.
6. For each selected paper, record evidence with `stage0_web_research.py
   add-evidence`.
7. Finalize the session with `stage0_web_research.py finalize --max-papers 5`.
8. Validate the generated prior with `stage0_web_research.py validate`.

The evidence record for each paper included:

- core paper role
- dataset or benchmark construction method
- data workflow insight
- quality gate or evaluation insight
- Stage1 seed guidance
- Stage1 prompt guidance
- risks or warnings when applicable

## How Papers Were Analyzed

Each paper was evaluated against three questions.

### Core Problem

The agent asked what problem the paper solves for DataEvolver:

- Does it define an image editing dataset?
- Does it define a benchmark?
- Does it introduce a task taxonomy?
- Does it provide a method for instruction generation, annotation, filtering, or
  evaluation?
- Is it directly useful for dataset production, or only useful as an evaluation
  reference?

### Dataset Construction Method

The agent extracted the production workflow, such as:

- source image, edit instruction, target image triplets
- synthetic generation with LLMs or diffusion models
- human annotation
- task-type or edit-type taxonomy
- multimodal conditions
- region masks, bounding boxes, pose, viewpoint, geometry, or relation labels
- filtering and quality scoring

### Dataset Construction Boundary

The agent recorded boundaries and risks:

- synthetic data can scale but needs validation
- text-only spatial instructions are ambiguous
- relation correctness should be evaluated separately from image quality
- easy edit categories can hide failures in harder categories
- geometry or region constraints should be explicit when needed
- non-target preservation must be checked separately from target edit success

## Selected Papers

### Instruction Editing Dataset Task

The final selected arXiv papers were:

```text
https://arxiv.org/abs/2211.09800
https://arxiv.org/abs/2306.10012
https://arxiv.org/abs/2311.10089
https://arxiv.org/abs/2404.09990
https://arxiv.org/abs/2411.15738
```

Rationale:

- `InstructPix2Pix` is the scalable synthetic instruction-editing baseline.
- `MagicBrush` adds manually annotated real-image instruction editing.
- `Emu Edit` contributes a task taxonomy and benchmark framing.
- `HQ-Edit` focuses on high-quality synthetic pair generation and filtering.
- `AnyEdit` expands the construction pattern to a broad multimodal editing
  dataset.

### Spatial / Rotation / 3D-Aware Editing Task

The final selected arXiv papers were:

```text
https://arxiv.org/abs/2604.04911
https://arxiv.org/abs/2307.11073
https://arxiv.org/abs/2401.01702
https://arxiv.org/abs/2212.10015
https://arxiv.org/abs/2604.10954
```

Rationale:

- `SpatialEdit` is directly aligned with fine-grained spatial editing and
  rotation/pose-style operations.
- `3DIT` provides 3D-aware controls and viewpoint/object manipulation signals.
- `Image Sculpting` emphasizes precise object editing with 3D geometry control.
- `VISOR` contributes spatial relation taxonomies and relation-level evaluation
  ideas.
- `FineEdit` adds region/bounding-box guidance and target/non-target evaluation
  framing.

## How The Summary Was Produced

The final summary was generated from the structured evidence records, not from a
free-form memory of the search.

The summary synthesis used:

- per-paper `summary`
- per-paper `methodology_notes`
- per-paper `dataset_workflow`
- per-paper `quality_gates`
- per-paper Stage1 guidance fields

The framework then wrote:

```text
ARXIV_LINKS_AND_SUMMARY.md
arxiv_links_and_summary.json
NEXT_AI_REPRO_PROMPT.md
```

The JSON schema is:

```json
{
  "schema_version": "dataevolver_stage0_arxiv_summary_v1",
  "prior_id": "...",
  "query": "...",
  "max_papers": 5,
  "arxiv_links": [],
  "papers": [],
  "summary": "..."
}
```

The Markdown file is human-readable and contains:

- task name
- selected arXiv paper table
- links for next step
- summary for next step

The reproduction prompt file contains:

- selected arXiv links
- Stage0 summary
- instruction to follow `runtime/AI_REPRODUCTION_GUIDE.md`
- required outputs such as `METHOD_CONSTRAINT_CARD.md` and `REPRO_STATUS.json`
- placeholders for work directory, large-file directory, server, compute, and
  limits

## Test Commands

Representative commands:

```bash
python scripts/stage0_web_research.py init \
  --query "instruction-guided image editing dataset and benchmark construction methods" \
  --output-dir runtime/research_priors/local_websearch_smoke_20260514/instruction_editing_datasets \
  --tag image-editing \
  --tag dataset-construction \
  --tag instruction-editing

python scripts/stage0_web_research.py add-evidence \
  --session-dir runtime/research_priors/local_websearch_smoke_20260514/instruction_editing_datasets \
  --url https://arxiv.org/abs/2211.09800 \
  --title "InstructPix2Pix: Learning to Follow Image Editing Instructions" \
  --source-type paper \
  --summary "Uses GPT-3 and Stable Diffusion to synthesize a large instruction-following image editing dataset..."

python scripts/stage0_web_research.py finalize \
  --session-dir runtime/research_priors/local_websearch_smoke_20260514/instruction_editing_datasets \
  --max-papers 5
```

Validation commands:

```bash
python scripts/stage0_web_research.py validate \
  --prior-path runtime/research_priors/local_websearch_smoke_20260514/instruction_editing_datasets/research_prior.json

python -m py_compile scripts/stage0_web_research.py
```

## Acceptance Result

Passed:

- two separate research task subfolders were created
- each subfolder contains up to five arXiv links
- each subfolder contains a Markdown summary handoff
- each subfolder contains a JSON summary handoff
- `research_prior.json` still validates
- the framework did not require extra API keys
- the framework did not call `local-deep-research` runtime APIs

Known limitation:

- The current Stage0 framework is agent-run. It records and structures search
  work, but the WebSearch/WebFetch calls are performed by the active agent, not
  by the Python script.
