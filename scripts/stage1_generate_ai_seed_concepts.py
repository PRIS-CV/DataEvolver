#!/usr/bin/env python3
"""Generate new Stage1 seed concepts for dataset expansion.

The output is a JSON list compatible with:

  pipeline/stage1_text_expansion.py --seed-concepts-file ...
  scripts/build_scene_assets_from_stage1.py --objects-file ...

With --template-only this script emits a deterministic curated list. Without
--template-only it calls Anthropic or an OpenAI-compatible relay and asks for
new object categories that are good candidates for 2D->3D->Blender rotation
data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from research_prior_utils import (  # noqa: E402
    attach_prior_metadata,
    build_guidance_block,
    load_research_prior,
)


CURATED_NAMES = [
    ("obj_051", "ceramic_teapot", "daily_item"),
    ("obj_052", "metal_watering_can", "daily_item"),
    ("obj_053", "wooden_bookshelf", "daily_item"),
    ("obj_054", "rolling_office_chair", "daily_item"),
    ("obj_055", "tool_cart", "daily_item"),
    ("obj_056", "road_safety_cone", "street_object"),
    ("obj_057", "folding_table", "daily_item"),
    ("obj_058", "garden_lantern", "daily_item"),
    ("obj_059", "camera_tripod", "daily_item"),
    ("obj_060", "electric_drill", "daily_item"),
]


SYSTEM_PROMPT = """You create object seed concepts for a synthetic rotation-edit dataset.
Return only JSON, no markdown.
Each object must be a single foreground object that can be generated as an isolated image,
converted to a 3D mesh, inserted into a Blender scene, and rotated in yaw.
Avoid humans, transparent glass-heavy objects, liquids, furry animals, and objects with very thin chaotic structures."""


USER_TEMPLATE = """Generate {count} new object seed concepts.

Requirements:
- IDs must start at obj_{start_index:03d} and increase by 1.
- Names must be lowercase snake_case.
- Categories should be one of: daily_item, street_object, vehicle, sports_item, tool, furniture.
- Favor objects with clear silhouette and useful rotation asymmetry.
- Avoid these existing names: {avoid_names}
{research_guidance}

Return exactly a JSON list:
[
  {{"id": "obj_051", "name": "example_object", "category": "daily_item"}}
]"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage1 seed concepts")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=51)
    parser.add_argument("--avoid-file", action="append", default=[])
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument(
        "--api-provider",
        choices=["anthropic", "openai"],
        default=os.environ.get("STAGE1_API_PROVIDER", "anthropic"),
        help="Use openai for OpenAI-compatible relay endpoints.",
    )
    parser.add_argument(
        "--api-base-url",
        default=(
            os.environ.get("STAGE1_API_BASE_URL")
            or os.environ.get("ANTHROPIC_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
        ),
        help="Relay base URL, for example https://example.com/v1",
    )
    parser.add_argument("--api-group", default=os.environ.get("STAGE1_API_GROUP"))
    parser.add_argument("--api-timeout", type=float, default=float(os.environ.get("STAGE1_API_TIMEOUT", 300)))
    parser.add_argument("--template-only", action="store_true")
    parser.add_argument(
        "--research-prior-path",
        default=None,
        help="Optional local research_prior.json. Soft guidance only; missing or invalid files warn and continue.",
    )
    return parser.parse_args()


def load_avoid_names(paths: list[str]) -> list[str]:
    names: set[str] = set()
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for item in payload:
            if isinstance(item, dict) and item.get("name"):
                names.add(str(item["name"]))
            elif isinstance(item, (list, tuple)) and len(item) > 1:
                names.add(str(item[1]))
    return sorted(names)


def curated(start_index: int, count: int) -> list[dict]:
    out = []
    for idx in range(count):
        if idx < len(CURATED_NAMES):
            _, name, category = CURATED_NAMES[idx]
        else:
            name, category = f"rotation_asset_{start_index + idx:03d}", "daily_item"
        out.append({"id": f"obj_{start_index + idx:03d}", "name": name, "category": category})
    return out


def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError("No JSON list found in model response")
    return json.loads(match.group(0))


def _stage1_api_key(provider: str) -> str | None:
    if provider == "openai":
        return (
            os.environ.get("STAGE1_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("LLM_API_KEY")
        )
    return os.environ.get("STAGE1_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")


def call_anthropic(
    model: str,
    count: int,
    start_index: int,
    avoid_names: list[str],
    research_guidance: str = "",
    base_url: str | None = None,
    timeout: float = 300,
) -> list[dict]:
    api_key = _stage1_api_key("anthropic")
    if not api_key:
        raise SystemExit("STAGE1_API_KEY or ANTHROPIC_API_KEY is required unless --template-only is used")

    endpoint = (base_url or "https://api.anthropic.com").rstrip("/")
    if endpoint.endswith("/v1/messages"):
        pass
    elif endpoint.endswith("/v1"):
        endpoint = endpoint + "/messages"
    else:
        endpoint = endpoint + "/v1/messages"

    body = {
        "model": model,
        "max_tokens": 1600,
        "temperature": 0.7,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    count=count,
                    start_index=start_index,
                    avoid_names=", ".join(avoid_names[:200]) or "none",
                    research_guidance=research_guidance,
                ),
            }
        ],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "User-Agent": "curl/8.5.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Anthropic-compatible relay request failed: HTTP {exc.code}\n{detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Anthropic-compatible relay request failed: {exc}") from exc

    response_payload = json.loads(raw)
    content = response_payload.get("content") or []
    text = ""
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
            text += str(item["text"])
    if not text:
        text = response_payload.get("text") or ""
    payload = extract_json(text)
    if not isinstance(payload, list):
        raise ValueError("Model response is not a JSON list")
    return payload


def call_openai_compatible(
    model: str,
    count: int,
    start_index: int,
    avoid_names: list[str],
    research_guidance: str = "",
    base_url: str | None = None,
    timeout: float = 300,
    group: str | None = None,
) -> list[dict]:
    api_key = _stage1_api_key("openai")
    if not api_key:
        raise SystemExit("STAGE1_API_KEY, OPENAI_API_KEY, or LLM_API_KEY is required for --api-provider openai")
    if not base_url:
        raise SystemExit("STAGE1_API_BASE_URL, OPENAI_BASE_URL, or LLM_BASE_URL is required for --api-provider openai")

    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = endpoint + "/chat/completions"

    body = {
        "model": model,
        "temperature": 0.7,
        "max_tokens": 1600,
        "stream": False,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    count=count,
                    start_index=start_index,
                    avoid_names=", ".join(avoid_names[:200]) or "none",
                    research_guidance=research_guidance,
                ),
            },
        ],
    }
    if group:
        body["group"] = group
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "curl/8.5.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"OpenAI-compatible relay request failed: HTTP {exc.code}\n{detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"OpenAI-compatible relay request failed: {exc}") from exc

    response_payload = json.loads(raw)
    choices = response_payload.get("choices") or []
    if not choices:
        raise ValueError(f"OpenAI-compatible response has no choices: {response_payload}")
    message = choices[0].get("message") or {}
    text = message.get("content") or choices[0].get("text") or ""
    payload = extract_json(text)
    if not isinstance(payload, list):
        raise ValueError("Model response is not a JSON list")
    return payload


def normalize(payload: list[dict], start_index: int, count: int) -> list[dict]:
    out = []
    seen = set()
    for idx, item in enumerate(payload[:count]):
        if not isinstance(item, dict):
            continue
        obj_id = str(item.get("id") or f"obj_{start_index + idx:03d}").strip()
        name = str(item.get("name") or "").strip().lower().replace("-", "_").replace(" ", "_")
        category = str(item.get("category") or "daily_item").strip() or "daily_item"
        if not name or name in seen:
            continue
        seen.add(name)
        out.append({"id": obj_id, "name": name, "category": category})
    next_idx = start_index + len(out)
    while len(out) < count:
        out.append({"id": f"obj_{next_idx:03d}", "name": f"rotation_asset_{next_idx:03d}", "category": "daily_item"})
        next_idx += 1
    for offset, item in enumerate(out):
        item["id"] = f"obj_{start_index + offset:03d}"
    return out


def main() -> None:
    args = parse_args()
    avoid_names = load_avoid_names(args.avoid_file)
    research_prior = load_research_prior(args.research_prior_path, "[Stage 1 seed]")
    research_guidance = build_guidance_block(research_prior, audience="seed")
    if research_guidance:
        print(
            "[Stage 1 seed] Loaded research prior: "
            f"{research_prior.get('prior_id') or research_prior.get('id') or args.research_prior_path}"
        )
    elif args.research_prior_path:
        print("[Stage 1 seed] WARN: research prior provided but no seed guidance was found")
    if args.template_only:
        payload = curated(args.start_index, args.count)
    elif args.api_provider == "openai":
        payload = call_openai_compatible(
            args.model,
            args.count,
            args.start_index,
            avoid_names,
            research_guidance,
            args.api_base_url,
            args.api_timeout,
            args.api_group,
        )
    else:
        payload = call_anthropic(
            args.model,
            args.count,
            args.start_index,
            avoid_names,
            research_guidance,
            args.api_base_url,
            args.api_timeout,
        )
    payload = normalize(payload, args.start_index, args.count)
    payload = attach_prior_metadata(payload, research_prior)
    output = Path(args.output_file).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(payload)} seed concepts -> {output}")


if __name__ == "__main__":
    main()
