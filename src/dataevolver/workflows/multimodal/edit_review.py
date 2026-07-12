from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Iterable, Optional


VALID_EDIT_STATUS = {"pass", "fail", "uncertain"}


def _constraint(constraint_id: str, kind: str, description: str) -> dict:
    return {
        "constraint_id": constraint_id,
        "type": kind,
        "description": description,
    }


def default_edit_constraints(edit_instruction: str) -> list[dict]:
    return [
        _constraint(
            "c_instruction_following",
            "instruction_following",
            f"The edited image visibly follows the requested edit: {edit_instruction}",
        ),
        _constraint(
            "c_source_preservation",
            "source_preservation",
            "Unmentioned source-image objects, scene layout, camera view, and overall identity remain recognizable.",
        ),
        _constraint(
            "c_edit_locality",
            "edit_locality",
            "The visual change is localized to the requested target or region instead of rewriting the whole image.",
        ),
        _constraint(
            "c_identity_consistency",
            "identity_consistency",
            "The edited target remains the same object category/shape unless the instruction explicitly asks otherwise.",
        ),
        _constraint(
            "c_background_preservation",
            "background_preservation",
            "The background, lighting, and non-target composition are preserved unless explicitly requested.",
        ),
        _constraint(
            "c_artifact_quality",
            "artifact_quality",
            "The edited result is coherent, not blank, not overpainted, and has no obvious artifacts, text, logos, or watermarks.",
        ),
    ]


def normalize_edit_constraints(constraints: Optional[Iterable[dict]], edit_instruction: str) -> list[dict]:
    normalized = []
    seen = set()
    for idx, item in enumerate(constraints or []):
        if not isinstance(item, dict):
            continue
        cid = str(item.get("constraint_id") or item.get("id") or f"c_edit_{idx + 1:02d}").strip()
        desc = str(item.get("description") or item.get("constraint") or "").strip()
        if not cid or not desc or cid in seen:
            continue
        seen.add(cid)
        normalized.append(
            _constraint(
                cid,
                str(item.get("type") or item.get("kind") or "instruction_following").strip(),
                desc,
            )
        )
    return normalized if normalized else default_edit_constraints(edit_instruction)


def normalize_edit_review(review: Optional[dict], constraints: list[dict]) -> dict:
    review = review if isinstance(review, dict) else {}
    by_id = {item["constraint_id"]: item for item in constraints}
    checklist = review.get("edit_checklist")
    if not isinstance(checklist, list):
        checklist = review.get("constraint_checklist")
    if not isinstance(checklist, list):
        checklist = []

    rows = []
    seen = set()
    for item in checklist:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("constraint_id") or item.get("id") or "").strip()
        if not cid:
            continue
        base = by_id.get(cid, _constraint(cid, str(item.get("type") or "instruction_following"), str(item.get("description") or "")))
        status = str(item.get("status") or "").strip().lower()
        if status not in VALID_EDIT_STATUS:
            status = "uncertain"
        confidence = str(item.get("confidence") or "medium").strip().lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"
        rows.append(
            {
                "constraint_id": cid,
                "type": str(item.get("type") or base["type"]),
                "description": str(item.get("description") or base["description"]),
                "status": status,
                "evidence": str(item.get("evidence") or "").strip(),
                "confidence": confidence,
            }
        )
        seen.add(cid)

    for constraint in constraints:
        if constraint["constraint_id"] not in seen:
            rows.append(
                {
                    **constraint,
                    "status": "uncertain",
                    "evidence": "Constraint was not explicitly judged by the reviewer.",
                    "confidence": "low",
                }
            )

    passed = [row for row in rows if row["status"] == "pass"]
    failed = [row for row in rows if row["status"] == "fail"]
    uncertain = [row for row in rows if row["status"] == "uncertain"]
    route = str(review.get("vlm_route") or "").strip().lower()
    if route not in {"pass", "needs_fix", "reject"}:
        route = "pass" if not failed and not uncertain else "needs_fix"
    if (failed or uncertain) and route == "pass":
        route = "needs_fix"
    if any(row["constraint_id"] == "c_artifact_quality" and row["status"] == "fail" for row in failed):
        route = "reject" if len(failed) >= 3 else route

    pass_rate = round(len(passed) / max(1, len(rows)), 4)
    return {
        **review,
        "schema_version": "edit_constraint_review_v1",
        "vlm_route": route,
        "edit_checklist": rows,
        "constraint_checklist": rows,
        "passed_edit_constraints": [row["constraint_id"] for row in passed],
        "failed_edit_constraints": [row["constraint_id"] for row in failed],
        "uncertain_edit_constraints": [row["constraint_id"] for row in uncertain],
        "edit_status_counts": {
            "pass": len(passed),
            "fail": len(failed),
            "uncertain": len(uncertain),
            "total": len(rows),
        },
        "edit_pass_rate": pass_rate,
        "issue_tags": _issue_tags(failed, uncertain),
        "scores": _scores(pass_rate, failed, uncertain),
    }


def _issue_tags(failed: list[dict], uncertain: list[dict]) -> list[str]:
    if not failed and not uncertain:
        return ["none"]
    kinds = {str(row.get("type")) for row in failed + uncertain}
    tags = []
    if "instruction_following" in kinds:
        tags.append("instruction_not_followed")
    if "source_preservation" in kinds or "background_preservation" in kinds:
        tags.append("source_not_preserved")
    if "edit_locality" in kinds:
        tags.append("edit_not_local")
    if "artifact_quality" in kinds:
        tags.append("edit_artifact")
    if "identity_consistency" in kinds:
        tags.append("identity_drift")
    return tags[:4] or ["edit_constraint_miss"]


def _scores(pass_rate: float, failed: list[dict], uncertain: list[dict]) -> dict:
    overall = max(1, min(5, round(pass_rate * 5)))
    if failed:
        overall = min(overall, 3)
    elif uncertain:
        overall = min(overall, 4)
    by_type = {row["type"]: row["status"] for row in failed + uncertain}
    return {
        "instruction_following": 2 if by_type.get("instruction_following") == "fail" else overall,
        "source_preservation": 2 if by_type.get("source_preservation") == "fail" else overall,
        "edit_locality": 2 if by_type.get("edit_locality") == "fail" else overall,
        "identity_consistency": 2 if by_type.get("identity_consistency") == "fail" else overall,
        "artifact_quality": 2 if by_type.get("artifact_quality") == "fail" else overall,
        "overall": overall,
    }


def _build_edit_review_prompt(
    *,
    sample_id: str,
    edit_instruction: str,
    constraints: list[dict],
) -> tuple[str, str]:
    schema = {
        "schema_version": "edit_constraint_review_v1",
        "sample_id": sample_id,
        "vlm_route": "<pass|needs_fix|reject>",
        "edit_checklist": [
            {
                "constraint_id": "<id from provided checklist>",
                "type": "<constraint type>",
                "description": "<constraint being checked>",
                "status": "<pass|fail|uncertain>",
                "evidence": "<brief visual evidence, 20 words or fewer>",
                "confidence": "<low|medium|high>",
            }
        ],
        "failed_edit_constraints": ["<constraint_id>"],
        "uncertain_edit_constraints": ["<constraint_id>"],
        "passed_edit_constraints": ["<constraint_id>"],
        "rewrite_suggestions": ["<specific edit-instruction rewrite suggestion>"],
        "summary": "<one sentence>",
    }
    system_msg = (
        "You are a strict image-editing dataset reviewer. "
        "You compare a SOURCE image and an EDITED image against a natural-language edit instruction. "
        "Return ONLY valid JSON, no markdown fences or extra text. "
        "Do not include chain-of-thought, self-correction, or long reasoning. Keep evidence concise."
    )
    user_msg = (
        "Image 1 is the SOURCE image before editing. Image 2 is the EDITED result.\n\n"
        f"sample_id={sample_id}\n"
        f"Edit instruction:\n{edit_instruction}\n\n"
        "Provided edit checklist. Judge every item independently:\n"
        f"{json.dumps(constraints, indent=2, ensure_ascii=False)}\n\n"
        "Review procedure:\n"
        "1. Compare Image 2 (edited) against Image 1 (source).\n"
        "2. Mark instruction_following pass only if the requested edit is visibly completed.\n"
        "3. Mark source_preservation/background_preservation pass only if unmentioned objects and layout remain recognizable.\n"
        "4. Mark edit_locality fail if the whole image is repainted, the background changes unnecessarily, or non-target objects change substantially.\n"
        "5. Mark artifact_quality fail for blank outputs, solid-color images, severe blur, malformed objects, text/logos/watermarks, or obvious model artifacts.\n"
        "6. Set vlm_route=pass only when all checklist items pass; otherwise use needs_fix or reject.\n\n"
        "JSON schema to return:\n"
        f"{json.dumps(schema, indent=2, ensure_ascii=False)}"
    )
    return system_msg, user_msg


def run_edit_constraint_review(
    *,
    edited_path: str,
    source_path: str,
    sample_id: str,
    edit_instruction: str,
    constraints: Optional[Iterable[dict]] = None,
    device: str = "cuda:0",
    max_retries: int = 2,
    vlm_model_path: Optional[str] = None,
    enable_thinking: bool = False,
) -> dict:
    normalized_constraints = normalize_edit_constraints(constraints, edit_instruction)
    if vlm_model_path:
        os.environ["VLM_MODEL_PATH"] = vlm_model_path

    try:
        import torch
        from qwen_vl_utils import process_vision_info
        from dataevolver.annotation.vlm_review_stage import _extract_json, load_vlm
    except Exception as exc:
        fallback = normalize_edit_review({}, normalized_constraints)
        fallback.update(
            {
                "sample_id": sample_id,
                "edit_instruction": edit_instruction,
                "source_path": source_path,
                "edited_path": edited_path,
                "review_backend": "edit_constraint_import_fallback",
                "parse_mode": "import_failed",
                "error": str(exc),
            }
        )
        return fallback

    model, processor = load_vlm(device)
    system_msg, user_msg = _build_edit_review_prompt(
        sample_id=sample_id,
        edit_instruction=edit_instruction,
        constraints=normalized_constraints,
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_msg}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source_path},
                {"type": "image", "image": edited_path},
                {"type": "text", "text": user_msg},
            ],
        },
    ]
    dialogue = {
        "sample_id": sample_id,
        "system_prompt": system_msg,
        "user_prompt": user_msg,
        "attempts": [],
    }
    for attempt in range(max(1, max_retries)):
        try:
            text_input = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": bool(enable_thinking)},
            )
            text_input = text_input + "{"
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=3072,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            generated = output_ids[0][inputs.input_ids.shape[1] :]
            text_out = "{" + processor.decode(generated, skip_special_tokens=True)
            parsed = _extract_json(text_out)
            dialogue["attempts"].append(
                {
                    "attempt": attempt + 1,
                    "assistant_text": text_out,
                    "parse_mode": "direct_json" if parsed is not None else "failed",
                }
            )
            if parsed is not None:
                review = normalize_edit_review(parsed, normalized_constraints)
                review.update(
                    {
                        "sample_id": sample_id,
                        "edit_instruction": edit_instruction,
                        "source_path": source_path,
                        "edited_path": edited_path,
                        "review_backend": "qwen_edit_constraint_json",
                        "parse_mode": "direct_json",
                        "used_thinking": bool(enable_thinking),
                        "vlm_dialogue": dialogue,
                    }
                )
                return review
        except Exception as exc:
            dialogue["attempts"].append(
                {
                    "attempt": attempt + 1,
                    "parse_mode": "exception",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    fallback = normalize_edit_review({}, normalized_constraints)
    fallback.update(
        {
            "sample_id": sample_id,
            "edit_instruction": edit_instruction,
            "source_path": source_path,
            "edited_path": edited_path,
            "review_backend": "edit_constraint_neutral_fallback",
            "parse_mode": "neutral_fallback",
            "used_thinking": bool(enable_thinking),
            "vlm_dialogue": dialogue,
        }
    )
    return fallback
