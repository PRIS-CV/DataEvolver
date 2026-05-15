"""Helpers for soft research-prior guidance in Stage 1.

The prior is intentionally advisory. Bad or missing files should never stop
dataset generation; callers should warn and continue without guidance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
                    item.get("text")
                    or item.get("title")
                    or item.get("summary")
                    or item.get("url")
                    or item.get("link")
                )
            else:
                text = coerce_text(item)
            if text:
                out.append(text)
        return out
    if isinstance(value, dict):
        return [
            f"{key}: {coerce_text(val)}"
            for key, val in value.items()
            if coerce_text(val)
        ]
    text = coerce_text(value)
    return [text] if text else []


def load_research_prior(path: str | None, warn_prefix: str = "[research-prior]") -> dict[str, Any]:
    """Load a research prior JSON file, warning and returning {} on failure."""

    if not path:
        return {}

    prior_path = Path(path)
    if not prior_path.exists():
        print(f"{warn_prefix} WARN: research prior not found: {prior_path}")
        return {}

    try:
        payload = json.loads(prior_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"{warn_prefix} WARN: could not parse research prior {prior_path}: {exc}")
        return {}

    if not isinstance(payload, dict):
        print(f"{warn_prefix} WARN: research prior must be a JSON object: {prior_path}")
        return {}

    return payload


def _nested_dict(prior: dict[str, Any], key: str) -> dict[str, Any]:
    value = prior.get(key)
    return value if isinstance(value, dict) else {}


def source_links(prior: dict[str, Any]) -> list[str]:
    links: list[str] = []
    raw_sources = prior.get("source_links") or prior.get("sources") or []
    if isinstance(raw_sources, dict):
        raw_sources = list(raw_sources.values())
    if not isinstance(raw_sources, list):
        raw_sources = [raw_sources]

    for item in raw_sources:
        if isinstance(item, dict):
            text = coerce_text(item.get("url") or item.get("link") or item.get("pdf_url") or item.get("id"))
        else:
            text = coerce_text(item)
        if text and text not in links:
            links.append(text)
    return links


def prior_metadata(prior: dict[str, Any]) -> dict[str, Any]:
    if not prior:
        return {}
    meta: dict[str, Any] = {}
    prior_id = coerce_text(prior.get("prior_id") or prior.get("id"))
    tags = coerce_list(prior.get("tags"))
    links = source_links(prior)
    if prior_id:
        meta["research_prior_id"] = prior_id
    if tags:
        meta["research_prior_tags"] = tags
    if links:
        meta["research_prior_source_links"] = links[:20]
    return meta


def attach_prior_metadata(items: list[dict[str, Any]], prior: dict[str, Any]) -> list[dict[str, Any]]:
    meta = prior_metadata(prior)
    if not meta:
        return items
    for item in items:
        if isinstance(item, dict):
            item.update(meta)
    return items


def _guidance_candidates(prior: dict[str, Any], audience: str) -> list[str]:
    dataevolver_prior = _nested_dict(prior, "dataevolver_prior")
    candidates: list[str] = []

    if audience == "seed":
        fields = [
            prior.get("stage1_seed_guidance"),
            dataevolver_prior.get("stage1_seed_guidance"),
            dataevolver_prior.get("stage1_seed_rules"),
            dataevolver_prior.get("recommended_object_types"),
            dataevolver_prior.get("avoid_object_types"),
        ]
    else:
        fields = [
            prior.get("stage1_prompt_guidance"),
            dataevolver_prior.get("stage1_prompt_guidance"),
            dataevolver_prior.get("stage1_prompt_rules"),
            dataevolver_prior.get("prompt_rules"),
            dataevolver_prior.get("vlm_gate_guidance"),
        ]

    fields.extend(
        [
            prior.get("summary"),
            prior.get("risks_or_unknowns"),
            dataevolver_prior.get("risks_or_unknowns"),
        ]
    )
    for field in fields:
        candidates.extend(coerce_list(field))
    return candidates


def build_guidance_block(
    prior: dict[str, Any],
    *,
    audience: str,
    max_items: int = 8,
    max_chars: int = 1800,
) -> str:
    """Create a compact advisory block for seed or prompt generation."""

    if not prior:
        return ""

    tags = ", ".join(coerce_list(prior.get("tags"))[:8])
    prior_id = coerce_text(prior.get("prior_id") or prior.get("id")) or "unknown"
    candidates = []
    seen: set[str] = set()
    for item in _guidance_candidates(prior, audience):
        text = " ".join(item.split())
        if text and text not in seen:
            candidates.append(text)
            seen.add(text)
        if len(candidates) >= max_items:
            break

    if not candidates and not tags:
        return ""

    lines = [
        "",
        "Research prior (soft guidance only; do not violate the hard requirements above):",
        f"- prior_id: {prior_id}",
    ]
    if tags:
        lines.append(f"- tags: {tags}")
    for item in candidates:
        lines.append(f"- {item}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text
