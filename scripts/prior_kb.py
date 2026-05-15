#!/usr/bin/env python3
"""Local research-prior knowledge base for DataEvolver.

Stores structured prior JSON plus readable markdown cards under an ignored
runtime directory. The SQLite index is deliberately small: tags, links, status,
and paths back to the card files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "runtime" / "research_kb"


SCHEMA = """
CREATE TABLE IF NOT EXISTS priors (
    prior_id TEXT PRIMARY KEY,
    title TEXT,
    tags_json TEXT NOT NULL,
    source_links_json TEXT NOT NULL,
    summary TEXT,
    card_json_path TEXT NOT NULL,
    card_md_path TEXT,
    promotion_candidate INTEGER NOT NULL DEFAULT 0,
    result_status TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_priors_updated_at ON priors(updated_at);
CREATE INDEX IF NOT EXISTS idx_priors_result_status ON priors(result_status);
"""


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def kb_root(args: argparse.Namespace) -> Path:
    raw = args.kb_root or os.environ.get("DATAEVOLVER_PRIOR_KB_ROOT")
    return Path(raw).expanduser().resolve() if raw else DEFAULT_ROOT.resolve()


def connect(root: Path) -> sqlite3.Connection:
    root.mkdir(parents=True, exist_ok=True)
    (root / "cards").mkdir(parents=True, exist_ok=True)
    (root / "skill_drafts").mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "prior_kb.sqlite3")
    db.executescript(SCHEMA)
    return db


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Prior JSON must be an object: {path}")
    return payload


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
        out = []
        for item in value:
            if isinstance(item, dict):
                text = coerce_text(
                    item.get("url")
                    or item.get("link")
                    or item.get("title")
                    or item.get("text")
                    or item.get("summary")
                )
            else:
                text = coerce_text(item)
            if text:
                out.append(text)
        return out
    if isinstance(value, dict):
        return [f"{key}: {coerce_text(val)}" for key, val in value.items() if coerce_text(val)]
    text = coerce_text(value)
    return [text] if text else []


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9_-]+", "-", value)
    return value.strip("-")[:80] or "research-prior"


def prior_id_for(prior: dict[str, Any], raw_bytes: bytes) -> str:
    explicit = coerce_text(prior.get("prior_id") or prior.get("id"))
    if explicit:
        return slugify(explicit)
    digest = hashlib.sha256(raw_bytes).hexdigest()[:12]
    title = coerce_text(prior.get("title") or prior.get("summary")).split(".")[0]
    return f"{slugify(title)[:48]}-{digest}" if title else f"prior-{digest}"


def source_links(prior: dict[str, Any]) -> list[str]:
    links: list[str] = []
    for item in coerce_list(prior.get("source_links") or prior.get("sources")):
        if item not in links:
            links.append(item)
    return links


def markdown_card(prior: dict[str, Any], prior_id: str) -> str:
    title = coerce_text(prior.get("title")) or prior_id
    tags = ", ".join(coerce_list(prior.get("tags"))) or "none"
    links = source_links(prior)
    sections = [
        f"# {title}",
        "",
        f"- prior_id: `{prior_id}`",
        f"- tags: {tags}",
        "",
        "## Summary",
        "",
        coerce_text(prior.get("summary")) or "No summary provided.",
        "",
        "## Stage1 Seed Guidance",
        "",
        "\n".join(f"- {item}" for item in coerce_list(prior.get("stage1_seed_guidance"))) or "- none",
        "",
        "## Stage1 Prompt Guidance",
        "",
        "\n".join(f"- {item}" for item in coerce_list(prior.get("stage1_prompt_guidance"))) or "- none",
        "",
        "## Source Links",
        "",
        "\n".join(f"- {item}" for item in links) or "- none",
        "",
    ]
    return "\n".join(sections)


def cmd_add(args: argparse.Namespace) -> None:
    root = kb_root(args)
    prior_path = Path(args.prior).expanduser().resolve()
    raw_bytes = prior_path.read_bytes()
    prior = load_json(prior_path)
    prior_id = prior_id_for(prior, raw_bytes)
    tags = sorted(set(coerce_list(prior.get("tags")) + coerce_list(args.tag)))
    links = source_links(prior)
    title = coerce_text(prior.get("title")) or prior_id
    summary = coerce_text(prior.get("summary"))

    with connect(root) as db:
        cards_dir = root / "cards" / prior_id
        cards_dir.mkdir(parents=True, exist_ok=True)
        json_path = cards_dir / "research_prior.json"
        md_path = cards_dir / "RESEARCH_PRIOR.md"
        json_path.write_text(json.dumps(prior, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.report:
            shutil.copy2(Path(args.report).expanduser().resolve(), md_path)
        else:
            md_path.write_text(markdown_card(prior, prior_id), encoding="utf-8")

        existing = db.execute("SELECT created_at FROM priors WHERE prior_id = ?", (prior_id,)).fetchone()
        created_at = existing[0] if existing else now_text()
        db.execute(
            """
            INSERT OR REPLACE INTO priors (
                prior_id, title, tags_json, source_links_json, summary,
                card_json_path, card_md_path, promotion_candidate, result_status,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prior_id,
                title,
                json.dumps(tags, ensure_ascii=False),
                json.dumps(links, ensure_ascii=False),
                summary,
                str(json_path),
                str(md_path),
                1 if args.promotion_candidate or bool(prior.get("promotion_candidate")) else 0,
                args.result_status,
                created_at,
                now_text(),
            ),
        )
    print(f"Added prior {prior_id}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")


def rows_for_query(db: sqlite3.Connection, args: argparse.Namespace) -> list[sqlite3.Row]:
    db.row_factory = sqlite3.Row
    if getattr(args, "tag", None):
        rows = db.execute("SELECT * FROM priors ORDER BY updated_at DESC").fetchall()
        tag = args.tag.lower()
        return [row for row in rows if tag in [item.lower() for item in json.loads(row["tags_json"])]]
    if getattr(args, "query", None):
        rows = db.execute("SELECT * FROM priors ORDER BY updated_at DESC").fetchall()
        query = args.query.lower()
        out = []
        for row in rows:
            haystack = " ".join(
                [
                    row["prior_id"],
                    row["title"] or "",
                    row["summary"] or "",
                    " ".join(json.loads(row["tags_json"])),
                    " ".join(json.loads(row["source_links_json"])),
                ]
            ).lower()
            if query in haystack:
                out.append(row)
        return out
    limit = int(getattr(args, "limit", 20))
    return db.execute("SELECT * FROM priors ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()


def cmd_list(args: argparse.Namespace) -> None:
    root = kb_root(args)
    with connect(root) as db:
        rows = rows_for_query(db, args)
    for row in rows:
        tags = ", ".join(json.loads(row["tags_json"]))
        print(f"{row['prior_id']}\t{row['result_status'] or '-'}\t{tags}\t{row['title']}")


def cmd_search(args: argparse.Namespace) -> None:
    cmd_list(args)


def get_row(db: sqlite3.Connection, prior_id: str) -> sqlite3.Row:
    db.row_factory = sqlite3.Row
    row = db.execute("SELECT * FROM priors WHERE prior_id = ?", (prior_id,)).fetchone()
    if row is None:
        raise SystemExit(f"Unknown prior_id: {prior_id}")
    return row


def cmd_show(args: argparse.Namespace) -> None:
    root = kb_root(args)
    with connect(root) as db:
        row = get_row(db, args.prior_id)
    print(json.dumps(dict(row), indent=2, ensure_ascii=False))


def cmd_promote_draft(args: argparse.Namespace) -> None:
    root = kb_root(args)
    with connect(root) as db:
        row = get_row(db, args.prior_id)
    prior = load_json(Path(row["card_json_path"]))
    skill_name = slugify(args.skill_name or f"dataevolver-prior-{row['prior_id']}")
    draft_dir = root / "skill_drafts" / skill_name
    draft_dir.mkdir(parents=True, exist_ok=True)
    skill_md = draft_dir / "SKILL.md"
    tags = ", ".join(json.loads(row["tags_json"])) or "dataevolver"
    content = f"""---
name: {skill_name}
description: DataEvolver dataset-construction prior. Tags: {tags}
---

# {row["title"] or row["prior_id"]}

Use this skill when a DataEvolver dataset task matches these tags: {tags}.

## Summary

{coerce_text(prior.get("summary")) or "No summary provided."}

## Stage1 Seed Guidance

{chr(10).join(f"- {item}" for item in coerce_list(prior.get("stage1_seed_guidance"))) or "- none"}

## Stage1 Prompt Guidance

{chr(10).join(f"- {item}" for item in coerce_list(prior.get("stage1_prompt_guidance"))) or "- none"}

## Sources

{chr(10).join(f"- {item}" for item in source_links(prior)) or "- none"}
"""
    skill_md.write_text(content, encoding="utf-8")
    print(f"Wrote skill draft: {skill_md}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage local DataEvolver research priors")
    parser.add_argument("--kb-root", default=None, help="Override KB root; default runtime/research_kb")
    sub = parser.add_subparsers(dest="cmd", required=True)

    add = sub.add_parser("add", help="Add or update a prior JSON")
    add.add_argument("--prior", required=True)
    add.add_argument("--report", default=None)
    add.add_argument("--tag", action="append", default=[])
    add.add_argument("--result-status", default=None, help="Example: untested, smoke_passed, accepted_dataset")
    add.add_argument("--promotion-candidate", action="store_true")
    add.set_defaults(func=cmd_add)

    list_cmd = sub.add_parser("list", help="List priors")
    list_cmd.add_argument("--tag", default=None)
    list_cmd.add_argument("--limit", type=int, default=20)
    list_cmd.set_defaults(func=cmd_list)

    search = sub.add_parser("search", help="Substring search priors")
    search.add_argument("query")
    search.set_defaults(func=cmd_search)

    show = sub.add_parser("show", help="Show index metadata for a prior")
    show.add_argument("prior_id")
    show.set_defaults(func=cmd_show)

    promote = sub.add_parser("promote-draft", help="Generate a local skill draft from a prior")
    promote.add_argument("prior_id")
    promote.add_argument("--skill-name", default=None)
    promote.set_defaults(func=cmd_promote_draft)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
