from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .schemas import DatasetRequest


def _clean_request(text: str) -> str:
    return " ".join((text or "").strip().split())


def _row_text(row: dict, *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return _clean_request(str(value))
    return ""


def _row_metadata(row: dict) -> Dict[str, object]:
    meta: Dict[str, object] = {}
    for key in (
        "original_prompt",
        "rewritten_prompt",
        "original_edit_instruction",
        "tier",
        "source_prompt_id",
        "rewrite_reason",
        "negative_prompt",
    ):
        value = row.get(key)
        if value is not None and str(value).strip():
            meta[key] = value
    constraints = row.get("constraints")
    if isinstance(constraints, list):
        meta["constraints"] = constraints
    return meta


def _load_prompt_file(path: str) -> List[dict]:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"prompt file not found: {src}")
    suffix = src.suffix.lower()
    if suffix == ".json":
        payload = json.loads(src.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("samples") or payload.get("prompts") or [payload]
        if not isinstance(payload, list):
            raise ValueError("JSON prompt file must contain a list or a dict with samples/prompts")
        return [row if isinstance(row, dict) else {"prompt": str(row)} for row in payload]
    if suffix == ".jsonl":
        rows = []
        for line in src.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append(row if isinstance(row, dict) else {"prompt": str(row)})
        return rows
    return [{"prompt": line.strip()} for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]


def _edit_model_instruction(instruction: str) -> str:
    clean = _clean_request(instruction)
    preservation = (
        "Apply only this requested edit. Preserve the source image's camera view, layout, background, lighting, "
        "non-target objects, object identity, and overall composition. Do not repaint the whole image. "
        "Avoid text, logos, watermarks, blur, or new unrelated objects."
    )
    if any(term in clean.lower() for term in ["preserve", "保持", "保留", "unchanged"]):
        return clean
    return f"{clean}. {preservation}"


def build_prompts(request: DatasetRequest, count: int) -> List[Dict[str, str]]:
    base = _clean_request(request.user_request)
    if not base:
        base = "Create a high-quality multimodal dataset sample"

    file_rows = _load_prompt_file(request.prompt_file) if request.prompt_file else []
    prompts: List[Dict[str, str]] = []
    for idx in range(count):
        file_row = file_rows[idx] if idx < len(file_rows) else {}
        sample_id = _row_text(file_row, "sample_id", "id") or f"{request.sample_prefix}_{idx + 1:04d}"
        if request.route == "t2i":
            prompt = _row_text(file_row, "prompt", "text", "user_request")
            if not prompt:
                prompt = (
                    f"{base}. Produce one clean, inspectable image for dataset sample {idx + 1}. "
                    "Use stable composition, clear subject identity, and no text watermark."
                )
            meta = _row_metadata(file_row)
            meta.setdefault("original_prompt", _row_text(file_row, "original_prompt") or prompt)
            meta.setdefault("rewritten_prompt", prompt)
            prompts.append({"sample_id": sample_id, "prompt": prompt, **meta})
        elif request.route == "edit":
            instruction = _row_text(file_row, "edit_instruction", "instruction", "prompt", "text")
            if not instruction:
                instruction = (
                    f"{base}. Edit only the requested visual attributes for sample {idx + 1}; "
                    "preserve unrelated content, layout, and image identity."
                )
            meta = _row_metadata(file_row)
            meta.setdefault("original_edit_instruction", instruction)
            model_instruction = _edit_model_instruction(instruction)
            prompts.append(
                {
                    "sample_id": sample_id,
                    "prompt": model_instruction,
                    "edit_instruction": model_instruction,
                    **meta,
                }
            )
        elif request.route == "blender":
            prompt = (
                f"{base}. Route this request to the existing Blender dataset pipeline. "
                "Keep scene assets, render metadata, and loop trace outputs reproducible."
            )
            prompts.append({"sample_id": sample_id, "prompt": prompt})
        else:
            prompt = _row_text(file_row, "prompt", "text", "user_request")
            if not prompt:
                prompt = (
                    f"{base}. Generate a short, coherent text-to-video dataset sample. "
                    "Use physically plausible motion, stable camera behavior, consistent subject identity, "
                    "natural lighting, and a clear beginning-middle-end action. Avoid text, logos, watermarks, "
                    "flicker, warped anatomy, impossible object motion, and unrelated scene changes."
                )
            meta = _row_metadata(file_row)
            meta.setdefault("original_prompt", _row_text(file_row, "original_prompt") or prompt)
            meta.setdefault("rewritten_prompt", prompt)
            if "negative_prompt" not in meta:
                meta["negative_prompt"] = (
                    request.negative_prompt
                    or "text, subtitles, logos, watermark, flicker, jitter, distorted anatomy, "
                    "warped geometry, impossible physics, abrupt scene cuts, low quality"
                )
            prompts.append({"sample_id": sample_id, "prompt": prompt, **meta})
    return prompts
