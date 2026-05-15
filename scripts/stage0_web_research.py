#!/usr/bin/env python3
"""DataEvolver Stage0 agent-run websearch framework.

This framework is inspired by selected local-deep-research ideas, but it does
not run local-deep-research's LLM or search backends. The active Codex/Claude
Code agent performs WebSearch/WebFetch, while this script manages the local
research session, source evidence, report generation, and research_prior.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SESSION_SCHEMA_VERSION = "dataevolver_stage0_websearch_v1"
PRIOR_SCHEMA_VERSION = "dataevolver_research_prior_v1"

LDR_FRAMEWORK_REFERENCES = [
    {
        "name": "source_based_strategy",
        "path": "local-deep-research/src/local_deep_research/advanced_search_system/strategies/source_based_strategy.py",
        "used_for": [
            "iteration loop",
            "original query plus generated follow-up questions",
            "accumulated source results",
            "final citation-aware synthesis",
        ],
    },
    {
        "name": "standard_question_generator",
        "path": "local-deep-research/src/local_deep_research/advanced_search_system/questions/standard_question.py",
        "used_for": [
            "first-pass search question generation",
            "context-aware unanswered follow-up questions",
        ],
    },
    {
        "name": "report_generator",
        "path": "local-deep-research/src/local_deep_research/report_generator.py",
        "used_for": [
            "content-specific report structure",
            "section-level synthesis",
            "separate sources section",
        ],
    },
    {
        "name": "citation_handler",
        "path": "local-deep-research/src/local_deep_research/citation_handler.py",
        "used_for": [
            "source-indexed evidence",
            "citation-aware findings",
        ],
    },
]

DATAEVOLVER_READING_GOALS = [
    "dataset production workflow",
    "input/output taxonomy",
    "prompt or seed concept constraints",
    "quality gates and common failure modes",
    "methodology choices that should affect Stage1 generation",
    "risks, unknowns, and assumptions",
]

GAP_FOLLOW_UPS = [
    (
        "methodology_notes",
        "Which sources describe the concrete dataset construction or benchmark methodology for: {query}?",
    ),
    (
        "dataset_workflow",
        "What step-by-step data generation pipeline is implied by the strongest sources for: {query}?",
    ),
    (
        "quality_gates",
        "What filtering, validation, or human/automatic quality gates are used for: {query}?",
    ),
    (
        "failure_modes",
        "What common artifacts, shortcuts, ambiguities, or failure modes should DataEvolver avoid for: {query}?",
    ),
    (
        "stage1_seed_guidance",
        "What seed concept constraints should Stage1 follow when generating examples for: {query}?",
    ),
    (
        "stage1_prompt_guidance",
        "What prompt-level instructions should Stage1 inject based on methodology evidence for: {query}?",
    ),
]


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9_-]+", "-", value)
    return value.strip("-")[:80] or "research-prior"


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            text = coerce_text(item)
            if text:
                out.append(text)
        return out
    text = coerce_text(value)
    return [text] if text else []


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = coerce_text(item)
        key = re.sub(r"\s+", " ", text.lower())
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def json_dump(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def session_path(session_dir: Path) -> Path:
    return session_dir / "stage0_session.json"


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return data


def load_session(session_dir: str | Path) -> dict[str, Any]:
    path = session_path(Path(session_dir).expanduser().resolve())
    if not path.exists():
        raise SystemExit(f"Stage0 session does not exist: {path}")
    return read_json(path)


def save_session(session_dir: str | Path, session: dict[str, Any]) -> None:
    out_dir = Path(session_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    session["updated_at"] = now_text()
    session_path(out_dir).write_text(json_dump(session) + "\n", encoding="utf-8")


def framework_references() -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for ref in LDR_FRAMEWORK_REFERENCES:
        item = dict(ref)
        item["available"] = (REPO_ROOT / item["path"]).exists()
        refs.append(item)
    return refs


def initial_questions(
    query: str,
    *,
    source_hints: list[str],
    questions_per_iteration: int,
) -> list[str]:
    hints = " ".join(source_hints)
    candidates = [
        query,
        f"{query} methodology dataset construction benchmark",
        f"{query} arxiv HTML methodology dataset generation",
        f"{query} official code dataset annotation filtering evaluation",
        f"{query} failure modes quality control data curation",
    ]
    if hints:
        candidates.append(f"{query} methodology {hints}")
    return dedupe(candidates)[: max(1, questions_per_iteration)]


def next_gap_questions(session: dict[str, Any], questions_per_iteration: int) -> list[str]:
    query = session["query"]
    evidence = session.get("evidence", [])
    existing = []
    for block in session.get("questions_by_iteration", []):
        existing.extend(coerce_list(block.get("questions")))

    candidates: list[str] = []
    for field, question in GAP_FOLLOW_UPS:
        has_field = any(coerce_list(item.get(field)) for item in evidence)
        if not has_field:
            candidates.append(question.format(query=query))

    candidates.extend(
        [
            f"Which primary sources most directly support Stage1 guidance for: {query}?",
            f"What assumptions remain weak or unverified after the current evidence for: {query}?",
            f"What reproducible implementation details can be transferred into DataEvolver for: {query}?",
        ]
    )

    existing_keys = {re.sub(r"\s+", " ", item.lower()) for item in existing}
    filtered = [
        item
        for item in dedupe(candidates)
        if re.sub(r"\s+", " ", item.lower()) not in existing_keys
    ]
    return filtered[: max(1, questions_per_iteration)]


def prior_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "DataEvolver research prior",
        "type": "object",
        "required": [
            "schema_version",
            "prior_id",
            "title",
            "tags",
            "source_links",
            "summary",
            "stage1_seed_guidance",
            "stage1_prompt_guidance",
        ],
        "properties": {
            "schema_version": {"const": PRIOR_SCHEMA_VERSION},
            "prior_id": {"type": "string"},
            "title": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "source_links": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
            "stage1_seed_guidance": {"type": "array", "items": {"type": "string"}},
            "stage1_prompt_guidance": {"type": "array", "items": {"type": "string"}},
            "risks_or_unknowns": {"type": "array", "items": {"type": "string"}},
            "promotion_candidate": {"type": "boolean"},
            "stage0_metadata": {"type": "object"},
            "evidence": {"type": "array"},
        },
        "additionalProperties": True,
    }


def default_prior_template(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PRIOR_SCHEMA_VERSION,
        "prior_id": session["prior_id"],
        "title": session["query"],
        "tags": session.get("tags", []),
        "source_links": [],
        "summary": "",
        "stage1_seed_guidance": [],
        "stage1_prompt_guidance": [],
        "risks_or_unknowns": [],
        "promotion_candidate": False,
        "stage0_metadata": {
            "research_mode": "agent_websearch_webfetch",
            "framework": "dataevolver_stage0_websearch",
            "inspired_by": "local-deep-research source-based framework",
            "framework_references": session["framework_references"],
        },
    }


def escape_md_cell(value: Any) -> str:
    text = coerce_text(value).replace("\n", " ")
    return text.replace("|", "\\|")


def compact_sentence_text(items: list[str], max_chars: int = 500) -> str:
    text = " ".join(dedupe(items))
    if len(text) <= max_chars:
        return text
    cutoff = text.rfind(".", 0, max_chars)
    if cutoff >= 160:
        return text[: cutoff + 1]
    return text[: max_chars - 3].rstrip() + "..."


def render_agent_brief(session_dir: Path, session: dict[str, Any]) -> str:
    refs = "\n".join(
        f"- {ref['name']}: {ref['path']} ({'available' if ref.get('available') else 'missing'})"
        for ref in session["framework_references"]
    )
    goals = "\n".join(f"- {goal}" for goal in DATAEVOLVER_READING_GOALS)
    questions = "\n".join(
        f"- {question}"
        for question in session["questions_by_iteration"][0]["questions"]
    )
    source_hints = "\n".join(f"- {item}" for item in session.get("source_hints", [])) or "- none"
    return f"""# DataEvolver Stage0 Websearch Brief

This is an agent-run Stage0 research session. Do not ask the user for API keys.
Do not call local-deep-research's LLM/search backends. Use the current
Codex/Claude Code WebSearch/WebFetch tools, then record evidence with this
session framework.

## Query

{session["query"]}

Prior id: `{session["prior_id"]}`
Tags: {", ".join(session.get("tags", [])) or "none"}

## Framework References

These local-deep-research files define the framework ideas reused here:

{refs}

## Reading Goals

{goals}

## Iteration 1 Questions

{questions}

Source hints:

{source_hints}

## Agent Workflow

1. Search the iteration questions with WebSearch.
2. Fetch primary sources with WebFetch. Prefer arXiv HTML pages, PDFs only when
   useful, official dataset docs, project pages, and code repositories.
3. For each useful source, record a source card:

```bash
python scripts/stage0_web_research.py add-evidence \\
  --session-dir "{session_dir}" \\
  --url "https://..." \\
  --title "..." \\
  --source-type paper \\
  --iteration 1 \\
  --question "..." \\
  --summary "..." \\
  --methodology-note "..." \\
  --stage1-seed-guidance "..." \\
  --stage1-prompt-guidance "..."
```

4. Create follow-up questions when needed:

```bash
python scripts/stage0_web_research.py next-iteration --session-dir "{session_dir}"
```

5. Finalize local artifacts:

```bash
python scripts/stage0_web_research.py finalize --session-dir "{session_dir}"
```

The prior is soft augmentation. Stage1 warns and continues if a prior is
missing or invalid.
"""


def write_session_outputs(session_dir: Path, session: dict[str, Any]) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "AGENT_RESEARCH_BRIEF.md").write_text(
        render_agent_brief(session_dir, session),
        encoding="utf-8",
    )
    (session_dir / "research_prior.template.json").write_text(
        json_dump(default_prior_template(session)) + "\n",
        encoding="utf-8",
    )
    (session_dir / "research_prior.schema.json").write_text(
        json_dump(prior_schema()) + "\n",
        encoding="utf-8",
    )
    (session_dir / "stage0_manifest.json").write_text(
        json_dump(
            {
                "query": session["query"],
                "prior_id": session["prior_id"],
                "research_mode": "agent_websearch_webfetch",
                "no_extra_api_keys_required": True,
                "framework": "dataevolver_stage0_websearch",
                "framework_references": session["framework_references"],
                "outputs": {
                    "session": str(session_path(session_dir)),
                    "brief": str(session_dir / "AGENT_RESEARCH_BRIEF.md"),
                    "template": str(session_dir / "research_prior.template.json"),
                    "schema": str(session_dir / "research_prior.schema.json"),
                    "prior": str(session_dir / "research_prior.json"),
                    "report": str(session_dir / "RESEARCH_PRIOR.md"),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def cmd_init(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    prior_id = slugify(args.prior_id or f"{time.strftime('%Y%m%d')}-{args.query}")
    source_hints = coerce_list(args.source_hint)
    questions = initial_questions(
        args.query,
        source_hints=source_hints,
        questions_per_iteration=args.questions_per_iteration,
    )
    session = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "query": args.query,
        "prior_id": prior_id,
        "tags": sorted(set(coerce_list(args.tag))),
        "source_hints": source_hints,
        "created_at": now_text(),
        "updated_at": now_text(),
        "research_mode": "agent_websearch_webfetch",
        "framework": "dataevolver_stage0_websearch",
        "framework_references": framework_references(),
        "iteration_config": {
            "iterations": int(args.iterations),
            "questions_per_iteration": int(args.questions_per_iteration),
        },
        "questions_by_iteration": [
            {
                "iteration": 1,
                "rationale": "Initial source-based pass: original query plus generated high-value searches.",
                "questions": questions,
            }
        ],
        "evidence": [],
        "warnings": [],
        "notes": [],
    }
    save_session(output_dir, session)
    write_session_outputs(output_dir, session)
    print(f"Wrote Stage0 websearch session: {session_path(output_dir)}")
    print(f"Wrote agent brief: {output_dir / 'AGENT_RESEARCH_BRIEF.md'}")
    print("Next: use WebSearch/WebFetch, then add evidence with add-evidence.")
    return 0


def record_list_arg(args: argparse.Namespace, name: str) -> list[str]:
    return coerce_list(getattr(args, name, []))


def cmd_add_evidence(args: argparse.Namespace) -> int:
    session_dir = Path(args.session_dir).expanduser().resolve()
    session = load_session(session_dir)
    citation_id = len(session.get("evidence", [])) + 1
    evidence = {
        "citation_id": citation_id,
        "url": coerce_text(args.url),
        "title": coerce_text(args.title),
        "source_type": coerce_text(args.source_type),
        "iteration": int(args.iteration),
        "question": coerce_text(args.question),
        "summary": coerce_text(args.summary),
        "methodology_notes": record_list_arg(args, "methodology_note"),
        "dataset_workflow": record_list_arg(args, "dataset_workflow"),
        "quality_gates": record_list_arg(args, "quality_gate"),
        "failure_modes": record_list_arg(args, "failure_mode"),
        "stage1_seed_guidance": record_list_arg(args, "stage1_seed_guidance"),
        "stage1_prompt_guidance": record_list_arg(args, "stage1_prompt_guidance"),
        "risks_or_unknowns": record_list_arg(args, "risk"),
        "warnings": record_list_arg(args, "warning"),
        "tags": sorted(set(record_list_arg(args, "tag"))),
        "added_at": now_text(),
    }
    if not evidence["url"]:
        raise SystemExit("--url is required")
    if not evidence["title"]:
        evidence["title"] = evidence["url"]
    session.setdefault("evidence", []).append(evidence)
    save_session(session_dir, session)
    print(f"Added evidence [{citation_id}]: {evidence['title']}")
    return 0


def cmd_next_iteration(args: argparse.Namespace) -> int:
    session_dir = Path(args.session_dir).expanduser().resolve()
    session = load_session(session_dir)
    existing_blocks = session.setdefault("questions_by_iteration", [])
    next_iteration = int(args.iteration or (len(existing_blocks) + 1))
    configured_max = int(session.get("iteration_config", {}).get("iterations", 2))
    questions_per_iteration = int(
        args.questions_per_iteration
        or session.get("iteration_config", {}).get("questions_per_iteration", 3)
    )
    if next_iteration > configured_max and not args.allow_extra:
        raise SystemExit(
            f"Next iteration {next_iteration} exceeds configured max {configured_max}. "
            "Pass --allow-extra to continue."
        )

    questions = dedupe(coerce_list(args.question))
    if not questions:
        questions = next_gap_questions(session, questions_per_iteration)
    if not questions:
        questions = [f"What evidence gap remains for: {session['query']}?"]

    existing_blocks.append(
        {
            "iteration": next_iteration,
            "rationale": "Context-aware follow-up questions generated from missing evidence fields.",
            "questions": questions[:questions_per_iteration],
        }
    )
    save_session(session_dir, session)
    print(f"Added iteration {next_iteration} questions:")
    for question in questions[:questions_per_iteration]:
        print(f"- {question}")
    return 0


def evidence_links(session: dict[str, Any]) -> list[str]:
    return dedupe([coerce_text(item.get("url")) for item in session.get("evidence", [])])


def canonical_arxiv_url(url: str) -> str:
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", url)
    if not match:
        return ""
    return f"https://arxiv.org/abs/{match.group(1)}"


def arxiv_papers(session: dict[str, Any], max_papers: int = 5) -> list[dict[str, Any]]:
    max_papers = min(max(1, int(max_papers)), 5)
    papers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in session.get("evidence", []):
        url = canonical_arxiv_url(coerce_text(item.get("url")))
        if not url or url in seen:
            continue
        seen.add(url)
        papers.append(
            {
                "citation_id": item.get("citation_id"),
                "title": coerce_text(item.get("title")) or url,
                "arxiv_url": url,
                "source_type": coerce_text(item.get("source_type")) or "paper",
                "summary": coerce_text(item.get("summary")),
                "methodology_notes": coerce_list(item.get("methodology_notes")),
                "stage1_seed_guidance": coerce_list(item.get("stage1_seed_guidance")),
                "stage1_prompt_guidance": coerce_list(item.get("stage1_prompt_guidance")),
            }
        )
        if len(papers) >= max_papers:
            break
    return papers


def collect_evidence_lists(session: dict[str, Any], field: str) -> list[str]:
    items: list[str] = []
    for evidence in session.get("evidence", []):
        items.extend(coerce_list(evidence.get(field)))
    return dedupe(items)


def build_summary(session: dict[str, Any], explicit_summary: str = "") -> str:
    if explicit_summary:
        return explicit_summary
    evidence = session.get("evidence", [])
    if not evidence:
        return ""
    parts = []
    for item in evidence[:8]:
        summary = coerce_text(item.get("summary"))
        if summary:
            parts.append(f"[{item.get('citation_id')}] {summary}")
    return " ".join(parts)


def build_prior(session: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    risks = []
    risks.extend(collect_evidence_lists(session, "risks_or_unknowns"))
    risks.extend(collect_evidence_lists(session, "warnings"))
    risks.extend(coerce_list(session.get("warnings")))
    prior = {
        "schema_version": PRIOR_SCHEMA_VERSION,
        "prior_id": session["prior_id"],
        "title": session["query"],
        "tags": sorted(set(coerce_list(session.get("tags")) + coerce_list(args.tag))),
        "source_links": evidence_links(session),
        "summary": build_summary(session, coerce_text(args.summary)),
        "stage1_seed_guidance": collect_evidence_lists(session, "stage1_seed_guidance"),
        "stage1_prompt_guidance": collect_evidence_lists(session, "stage1_prompt_guidance"),
        "risks_or_unknowns": dedupe(risks),
        "promotion_candidate": bool(args.promotion_candidate),
        "stage0_metadata": {
            "research_mode": "agent_websearch_webfetch",
            "framework": "dataevolver_stage0_websearch",
            "inspired_by": "local-deep-research source-based framework",
            "framework_references": session.get("framework_references", []),
            "questions_by_iteration": session.get("questions_by_iteration", []),
            "evidence_count": len(session.get("evidence", [])),
            "finalized_at": now_text(),
        },
        "evidence": session.get("evidence", []),
    }
    return prior


def render_report(session: dict[str, Any], prior: dict[str, Any]) -> str:
    question_sections = []
    for block in session.get("questions_by_iteration", []):
        lines = "\n".join(f"- {question}" for question in block.get("questions", []))
        question_sections.append(f"### Iteration {block.get('iteration')}\n\n{lines}")

    evidence_rows = [
        "| # | Source | Type | Key Evidence |",
        "|---|---|---|---|",
    ]
    for item in session.get("evidence", []):
        notes = dedupe(
            [coerce_text(item.get("summary"))]
            + coerce_list(item.get("methodology_notes"))
            + coerce_list(item.get("dataset_workflow"))
            + coerce_list(item.get("quality_gates"))
        )
        note = " ".join(notes)[:500]
        title = escape_md_cell(item.get("title") or item.get("url"))
        url = coerce_text(item.get("url"))
        source = f"[{title}]({url})" if url else title
        evidence_rows.append(
            f"| {item.get('citation_id')} | {source} | {escape_md_cell(item.get('source_type'))} | {escape_md_cell(note)} |"
        )

    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) or "- none"

    return f"""# Research Prior: {session["query"]}

## Query

{session["query"]}

Prior id: `{session["prior_id"]}`

## Search Questions

{chr(10).join(question_sections) if question_sections else "No questions recorded."}

## Source Evidence

{chr(10).join(evidence_rows)}

## Synthesis

{prior.get("summary") or "No synthesis provided."}

## Stage1 Seed Guidance

{bullets(coerce_list(prior.get("stage1_seed_guidance")))}

## Stage1 Prompt Guidance

{bullets(coerce_list(prior.get("stage1_prompt_guidance")))}

## Risks Or Unknowns

{bullets(coerce_list(prior.get("risks_or_unknowns")))}

## Framework

This report was produced by DataEvolver's agent-run Stage0 websearch framework,
which reuses selected local-deep-research source-based research concepts
without calling its LLM/search runtime.
"""


def render_links_summary(
    session: dict[str, Any],
    prior: dict[str, Any],
    *,
    max_papers: int,
) -> str:
    papers = arxiv_papers(session, max_papers=max_papers)
    paper_rows = [
        "| # | Paper | arXiv | Why it matters |",
        "|---|---|---|---|",
    ]
    for index, paper in enumerate(papers, start=1):
        notes = dedupe(
            [paper.get("summary", "")]
            + coerce_list(paper.get("methodology_notes"))
            + coerce_list(paper.get("stage1_seed_guidance"))
            + coerce_list(paper.get("stage1_prompt_guidance"))
        )
        why = compact_sentence_text(notes)
        title = escape_md_cell(paper.get("title"))
        url = coerce_text(paper.get("arxiv_url"))
        paper_rows.append(f"| {index} | {title} | {url} | {escape_md_cell(why)} |")
    if not papers:
        paper_rows.append("| - | No arXiv paper links recorded. | - | - |")

    links = "\n".join(f"- {paper['arxiv_url']}" for paper in papers) or "- none"
    return f"""# ArXiv Links And Summary

## Task

{session["query"]}

Prior id: `{session["prior_id"]}`

## Selected arXiv Papers

{chr(10).join(paper_rows)}

## Links For Next Step

{links}

## Summary For Next Step

{prior.get("summary") or "No summary provided."}
"""


def links_summary_payload(
    session: dict[str, Any],
    prior: dict[str, Any],
    *,
    max_papers: int,
) -> dict[str, Any]:
    return {
        "schema_version": "dataevolver_stage0_arxiv_summary_v1",
        "prior_id": session["prior_id"],
        "query": session["query"],
        "max_papers": max_papers,
        "arxiv_links": [paper["arxiv_url"] for paper in arxiv_papers(session, max_papers=max_papers)],
        "papers": arxiv_papers(session, max_papers=max_papers),
        "summary": coerce_text(prior.get("summary")),
    }


def render_next_ai_repro_prompt(
    session: dict[str, Any],
    prior: dict[str, Any],
    *,
    max_papers: int,
) -> str:
    papers = arxiv_papers(session, max_papers=max_papers)
    links = "\n".join(f"- {paper['arxiv_url']}  # {paper['title']}" for paper in papers)
    if not links:
        links = "- <no arXiv links recorded>"
    return f"""# Next AI Reproduction Prompt

请严格按照 `runtime/AI_REPRODUCTION_GUIDE.md` 的流程，从下面 Stage0
websearch 产出的候选论文中选择最适合复现目标的一篇或多篇，继续完成论文复现。

## Stage0 Research Task

{session["query"]}

## Stage0 Summary

{prior.get("summary") or "No summary provided."}

## Candidate arXiv Links

{links}

## Required Downstream Behavior

- 先确认论文身份，优先使用 arXiv URL，PDF 次之。
- 先拆解论文方法线，再写代码。
- 必须创建 `docs/METHOD_CONSTRAINT_CARD.md`。
- 必须明确区分 `self_generated`、`official_downloaded`、
  `official_result_recomputed`、`mock_or_proxy`、`synthetic_proxy`、
  `not_completed`。
- 官方最终产物只能用于验证，不能直接算作方法复现。
- 不要默认下载全量数据，先用小样本跑通整体架构。
- 如果需要大下载、闭源模型、付费 API、额外账号或超过预算的任务，
  必须暂停并请求用户确认。
- 每完成一个阶段，更新 `REPRO_STATUS.json`。
- 最终报告必须给出复现等级和升级到更高等级所需资源、成本、风险。

## Fill Before Running

```text
工作目录:
<代码目录>

大文件目录:
<超过 1G 的数据、生成图片、候选结果、缓存目录>

服务器:
<ssh 别名或“无”>

算力信息:
<例如 GPU 数量，未知则写“请自行检查”>

限制:
- 不要下载新模型，除非用户明确同意。
- 不要修改已有项目或已有 conda 环境。
- 如需环境，请创建平行环境。
```
"""


def cmd_finalize(args: argparse.Namespace) -> int:
    session_dir = Path(args.session_dir).expanduser().resolve()
    session = load_session(session_dir)
    prior = build_prior(session, args)
    prior_path = Path(args.output_prior).expanduser().resolve() if args.output_prior else session_dir / "research_prior.json"
    report_path = Path(args.output_report).expanduser().resolve() if args.output_report else session_dir / "RESEARCH_PRIOR.md"
    links_summary_path = session_dir / "ARXIV_LINKS_AND_SUMMARY.md"
    links_summary_json_path = session_dir / "arxiv_links_and_summary.json"
    next_prompt_path = session_dir / "NEXT_AI_REPRO_PROMPT.md"
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    prior_path.write_text(json_dump(prior) + "\n", encoding="utf-8")
    report_path.write_text(render_report(session, prior), encoding="utf-8")
    links_summary_path.write_text(
        render_links_summary(session, prior, max_papers=args.max_papers),
        encoding="utf-8",
    )
    links_summary_json_path.write_text(
        json_dump(links_summary_payload(session, prior, max_papers=args.max_papers)) + "\n",
        encoding="utf-8",
    )
    next_prompt_path.write_text(
        render_next_ai_repro_prompt(session, prior, max_papers=args.max_papers),
        encoding="utf-8",
    )
    print(f"Wrote research prior: {prior_path}")
    print(f"Wrote research report: {report_path}")
    print(f"Wrote arXiv links summary: {links_summary_path}")
    print(f"Wrote arXiv links JSON: {links_summary_json_path}")
    print(f"Wrote next AI repro prompt: {next_prompt_path}")
    return validate_prior_file(prior_path, normalize_path=None)


def validate_prior_file(prior_path: Path, normalize_path: Path | None) -> int:
    try:
        prior = read_json(prior_path)
    except Exception as exc:
        print(f"[research_prior] ERROR: failed to read {prior_path}: {exc}", file=sys.stderr)
        return 2

    errors: list[str] = []
    warnings: list[str] = []
    required = prior_schema()["required"]
    for key in required:
        if key not in prior:
            errors.append(f"missing required field: {key}")

    if prior.get("schema_version") != PRIOR_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PRIOR_SCHEMA_VERSION!r}")

    for key in [
        "tags",
        "source_links",
        "stage1_seed_guidance",
        "stage1_prompt_guidance",
        "risks_or_unknowns",
    ]:
        if key in prior and not isinstance(prior[key], list):
            warnings.append(f"coercing {key} to a list")
            prior[key] = coerce_list(prior[key])

    for key in ["prior_id", "title", "summary"]:
        if key in prior and not isinstance(prior[key], str):
            warnings.append(f"coercing {key} to a string")
            prior[key] = str(prior[key])

    if not prior.get("stage1_seed_guidance"):
        warnings.append("stage1_seed_guidance is empty")
    if not prior.get("stage1_prompt_guidance"):
        warnings.append("stage1_prompt_guidance is empty")
    if not prior.get("source_links"):
        warnings.append("source_links is empty")

    for message in warnings:
        print(f"[research_prior] WARN: {message}")

    if errors:
        for message in errors:
            print(f"[research_prior] ERROR: {message}", file=sys.stderr)
        return 2

    if normalize_path:
        normalize_path.parent.mkdir(parents=True, exist_ok=True)
        normalize_path.write_text(json_dump(prior) + "\n", encoding="utf-8")
        print(f"[research_prior] Wrote normalized prior: {normalize_path}")

    print(f"[research_prior] OK: {prior_path}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    normalize_path = Path(args.output_path).expanduser().resolve() if args.output_path else None
    return validate_prior_file(Path(args.prior_path).expanduser().resolve(), normalize_path)


def cmd_status(args: argparse.Namespace) -> int:
    session_dir = Path(args.session_dir).expanduser().resolve()
    session = load_session(session_dir)
    print(f"query: {session['query']}")
    print(f"prior_id: {session['prior_id']}")
    print(f"evidence_count: {len(session.get('evidence', []))}")
    print("questions:")
    for block in session.get("questions_by_iteration", []):
        print(f"  iteration {block.get('iteration')}:")
        for question in block.get("questions", []):
            print(f"    - {question}")
    return 0


def add_common_repeated(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tag", action="append", default=[])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DataEvolver Stage0 websearch framework")
    sub = parser.add_subparsers(dest="command", required=True)

    init_cmd = sub.add_parser("init", help="create a Stage0 research session")
    init_cmd.add_argument("--query", required=True)
    init_cmd.add_argument("--output-dir", required=True)
    init_cmd.add_argument("--prior-id", default=None)
    init_cmd.add_argument("--iterations", type=int, default=2)
    init_cmd.add_argument("--questions-per-iteration", type=int, default=3)
    init_cmd.add_argument("--source-hint", action="append", default=[])
    add_common_repeated(init_cmd)
    init_cmd.set_defaults(func=cmd_init)

    evidence = sub.add_parser("add-evidence", help="record one fetched source")
    evidence.add_argument("--session-dir", required=True)
    evidence.add_argument("--url", required=True)
    evidence.add_argument("--title", default="")
    evidence.add_argument("--source-type", default="web")
    evidence.add_argument("--iteration", type=int, default=1)
    evidence.add_argument("--question", default="")
    evidence.add_argument("--summary", default="")
    evidence.add_argument("--methodology-note", action="append", default=[])
    evidence.add_argument("--dataset-workflow", action="append", default=[])
    evidence.add_argument("--quality-gate", action="append", default=[])
    evidence.add_argument("--failure-mode", action="append", default=[])
    evidence.add_argument("--stage1-seed-guidance", action="append", default=[])
    evidence.add_argument("--stage1-prompt-guidance", action="append", default=[])
    evidence.add_argument("--risk", action="append", default=[])
    evidence.add_argument("--warning", action="append", default=[])
    add_common_repeated(evidence)
    evidence.set_defaults(func=cmd_add_evidence)

    next_iter = sub.add_parser("next-iteration", help="add follow-up search questions")
    next_iter.add_argument("--session-dir", required=True)
    next_iter.add_argument("--iteration", type=int, default=None)
    next_iter.add_argument("--questions-per-iteration", type=int, default=None)
    next_iter.add_argument("--question", action="append", default=[])
    next_iter.add_argument("--allow-extra", action="store_true")
    next_iter.set_defaults(func=cmd_next_iteration)

    finalize = sub.add_parser("finalize", help="write research_prior.json and report")
    finalize.add_argument("--session-dir", required=True)
    finalize.add_argument("--summary", default="")
    finalize.add_argument("--output-prior", default=None)
    finalize.add_argument("--output-report", default=None)
    finalize.add_argument("--max-papers", type=int, default=5)
    finalize.add_argument("--promotion-candidate", action="store_true")
    add_common_repeated(finalize)
    finalize.set_defaults(func=cmd_finalize)

    validate = sub.add_parser("validate", help="validate a research_prior.json")
    validate.add_argument("--prior-path", required=True)
    validate.add_argument("--output-path", default=None)
    validate.set_defaults(func=cmd_validate)

    status = sub.add_parser("status", help="show session status")
    status.add_argument("--session-dir", required=True)
    status.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
