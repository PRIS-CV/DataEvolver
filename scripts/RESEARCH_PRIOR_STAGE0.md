# Research Prior Stage0

Stage0 is DataEvolver's own agent-run websearch framework. It is inspired by
selected `local-deep-research` framework ideas, but it does not run
`local-deep-research`'s LLM/search backend and does not require users to provide
extra API keys.

The active Codex or Claude Code agent performs WebSearch/WebFetch in the
conversation. The local framework manages:

- research session state
- source-based search questions
- follow-up question iterations
- source evidence cards
- citation-aware report structure
- `research_prior.json` generation
- local prior KB and server sync handoff

## Framework Source

The implementation is:

```text
scripts/stage0_web_research.py
```

It reuses these `local-deep-research` concepts as design references:

- `source_based_strategy.py`: original query plus generated questions,
  iterative follow-up, accumulated sources, final synthesis
- `standard_question.py`: first-pass and context-aware follow-up questions
- `citation_handler.py`: source-indexed evidence and citations
- `report_generator.py`: content-specific report sections and separate sources

## 1. Initialize A Research Session

```bash
python scripts/stage0_web_research.py init \
  --query "dataset construction methods for image editing rotation benchmarks" \
  --output-dir runtime/research_priors/rotation_benchmark \
  --tag rotation8 \
  --tag image-editing
```

Compatibility alias:

```bash
python scripts/run_ldr_stage0_research.py prepare \
  --query "dataset construction methods for image editing rotation benchmarks" \
  --output-dir runtime/research_priors/rotation_benchmark
```

This writes:

```text
stage0_session.json
AGENT_RESEARCH_BRIEF.md
research_prior.template.json
research_prior.schema.json
stage0_manifest.json
```

## 2. Search And Fetch With The Active Agent

The agent reads `AGENT_RESEARCH_BRIEF.md`, searches the generated questions,
and fetches primary sources. Prefer:

- arXiv HTML pages where available
- PDF links only when useful
- official dataset or benchmark docs
- project pages
- code repositories
- papers with inspectable methodology

The agent decides which sections are methodology-relevant. Do not rely on a
brittle generic PDF section extractor.

## 3. Record Source Evidence

After reading a source, the agent records a source card:

```bash
python scripts/stage0_web_research.py add-evidence \
  --session-dir runtime/research_priors/rotation_benchmark \
  --url "https://arxiv.org/abs/..." \
  --title "Paper or dataset title" \
  --source-type paper \
  --iteration 1 \
  --question "Which sources describe the construction methodology?" \
  --summary "Short evidence summary." \
  --methodology-note "Methodology detail relevant to DataEvolver." \
  --dataset-workflow "Concrete data production step." \
  --quality-gate "Filtering or validation step." \
  --stage1-seed-guidance "Short advisory seed-generation rule." \
  --stage1-prompt-guidance "Short advisory prompt-generation rule." \
  --risk "Risk or unknown."
```

Evidence fields can be repeated.

## 4. Generate Follow-Up Questions

```bash
python scripts/stage0_web_research.py next-iteration \
  --session-dir runtime/research_priors/rotation_benchmark
```

The framework inspects missing evidence fields and adds source-based follow-up
questions. The agent then searches/fetches those questions and records more
evidence.

## 5. Finalize The Prior

```bash
python scripts/stage0_web_research.py finalize \
  --session-dir runtime/research_priors/rotation_benchmark
```

This writes:

```text
RESEARCH_PRIOR.md
research_prior.json
ARXIV_LINKS_AND_SUMMARY.md
arxiv_links_and_summary.json
NEXT_AI_REPRO_PROMPT.md
```

The formal handoff to the next step is `ARXIV_LINKS_AND_SUMMARY.md` plus
`arxiv_links_and_summary.json`: at most 5 selected arXiv paper links and the
task summary. `research_prior.json` remains available for Stage1 soft prompt
injection and provenance metadata.

`NEXT_AI_REPRO_PROMPT.md` aligns the Stage0 output with
`runtime/AI_REPRODUCTION_GUIDE.md` and `runtime/NEW_AI_REPRO_PROMPT_TEMPLATE.md`.
It converts the selected arXiv links and summary into a downstream paper
reproduction prompt with placeholders for work directory, large-file directory,
server, compute, and limits.

Validate explicitly when needed:

```bash
python scripts/stage0_web_research.py validate \
  --prior-path runtime/research_priors/rotation_benchmark/research_prior.json
```

Validation is for local feedback. Stage1 treats the prior as soft
augmentation: missing or invalid prior files warn and continue.

## 6. Use The Prior In Stage1

```bash
python scripts/stage1_generate_ai_seed_concepts.py \
  --research-prior-path runtime/research_priors/rotation_benchmark/research_prior.json \
  ...

python pipeline/stage1_text_expansion.py \
  --research-prior-path runtime/research_priors/rotation_benchmark/research_prior.json \
  ...
```

The generated samples carry prior provenance metadata when a prior is loaded.

## 7. Store Successful Priors

```bash
python scripts/prior_kb.py add \
  --prior runtime/research_priors/rotation_benchmark/research_prior.json \
  --report runtime/research_priors/rotation_benchmark/RESEARCH_PRIOR.md \
  --tag rotation8 \
  --result-status untested

python scripts/prior_kb.py search rotation8
python scripts/prior_kb.py promote-draft <prior_id>
```

The KB writes to `runtime/research_kb/` by default, which is ignored.

## 8. Sync To Server

```bash
bash scripts/sync_research_prior_to_server.sh \
  --prior-path runtime/research_priors/rotation_benchmark/research_prior.json \
  --remote "$DATAEVOLVER_REMOTE" \
  --remote-dir "$DATAEVOLVER_REMOTE_RUN_ROOT"
```

Use the printed argument when starting the remote Linux pipeline:

```bash
--research-prior-path /remote/run/root/research_prior.json
```
