from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Iterable, Optional

from .t2i_constraints import constraints_for_prompt, normalize_constraint_review


def _build_constraint_review_prompt(
    *,
    sample_id: str,
    round_idx: int,
    original_prompt: str,
    rewritten_prompt: str,
    constraints: list[dict],
) -> tuple[str, str]:
    schema = {
        "schema_version": "t2i_constraint_review_v1",
        "sample_id": sample_id,
        "round_idx": round_idx,
        "vlm_route": "<pass|needs_fix|reject>",
        "constraint_checklist": [
            {
                "constraint_id": "<id from provided checklist>",
                "type": "<constraint type>",
                "description": "<constraint being checked>",
                "status": "<pass|fail|uncertain>",
                "evidence": "<short visual evidence, not reasoning>",
                "confidence": "<low|medium|high>",
            }
        ],
        "failed_constraints": ["<constraint_id>"],
        "uncertain_constraints": ["<constraint_id>"],
        "passed_constraints": ["<constraint_id>"],
        "rewrite_suggestions": ["<specific prompt rewrite suggestion>"],
        "summary": "<one sentence>",
    }
    system_msg = (
        "You are a strict text-to-image dataset constraint reviewer. "
        "You must judge the image against a prompt-derived checklist before giving any overall route. "
        "Return ONLY a valid JSON object with no markdown fences or extra text. "
        "The first character must be '{' and the last character must be '}'."
    )
    user_msg = (
        f"sample_id={sample_id}, round_idx={round_idx}\n\n"
        f"Original prompt:\n{original_prompt}\n\n"
        f"Prompt used for this generated image:\n{rewritten_prompt}\n\n"
        "Provided constraint checklist. Use every item; do not drop constraints. "
        "You may add an extra checklist item only if the prompt has an obvious missing constraint, but prefer the provided ids.\n"
        f"{json.dumps(constraints, indent=2, ensure_ascii=False)}\n\n"
        "Review procedure:\n"
        "1. First inspect each checklist item independently.\n"
        "2. Mark status=pass only when the image visibly satisfies the constraint.\n"
        "3. Mark status=fail when the image visibly violates the constraint or the required element is missing.\n"
        "4. Mark status=uncertain when the image is ambiguous, too small, occluded beyond inspection, or not enough evidence exists.\n"
        "5. Set vlm_route=pass only if all checklist items pass. Use needs_fix if any item fails or is uncertain. "
        "Use reject only for blank images, unusable images, or a fundamentally wrong scene.\n"
        "6. rewrite_suggestions must name the exact failed/uncertain constraints to repair.\n\n"
        "JSON schema to return:\n"
        f"{json.dumps(schema, indent=2, ensure_ascii=False)}"
    )
    return system_msg, user_msg


def run_t2i_constraint_review(
    *,
    rgb_path: str,
    sample_id: str,
    round_idx: int,
    original_prompt: str,
    rewritten_prompt: str,
    constraints: Optional[Iterable[dict]] = None,
    device: str = "cuda:0",
    max_retries: int = 2,
    vlm_model_path: Optional[str] = None,
    enable_thinking: bool = False,
) -> dict:
    normalized_constraints = constraints_for_prompt(original_prompt, constraints)
    if vlm_model_path:
        os.environ["VLM_MODEL_PATH"] = vlm_model_path

    repo_root = Path(__file__).resolve().parents[2]
    pipeline_dir = repo_root / "pipeline"
    import sys

    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))

    try:
        import torch
        from qwen_vl_utils import process_vision_info
        from pipeline.stage5_5_vlm_review import _extract_json, load_vlm
    except Exception as exc:
        fallback = normalize_constraint_review({}, normalized_constraints)
        fallback.update(
            {
                "sample_id": sample_id,
                "round_idx": round_idx,
                "original_prompt": original_prompt,
                "rewritten_prompt": rewritten_prompt,
                "review_backend": "t2i_constraint_import_fallback",
                "parse_mode": "import_failed",
                "error": str(exc),
            }
        )
        return fallback

    model, processor = load_vlm(device)
    system_msg, user_msg = _build_constraint_review_prompt(
        sample_id=sample_id,
        round_idx=round_idx,
        original_prompt=original_prompt,
        rewritten_prompt=rewritten_prompt,
        constraints=normalized_constraints,
    )
    content = [{"type": "image", "image": rgb_path}, {"type": "text", "text": user_msg}]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_msg}]},
        {"role": "user", "content": content},
    ]
    dialogue = {
        "sample_id": sample_id,
        "round_idx": round_idx,
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
                    max_new_tokens=2048,
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
                review = normalize_constraint_review(parsed, normalized_constraints)
                review.update(
                    {
                        "sample_id": sample_id,
                        "round_idx": round_idx,
                        "original_prompt": original_prompt,
                        "rewritten_prompt": rewritten_prompt,
                        "review_backend": "qwen_t2i_constraint_json",
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

    fallback = normalize_constraint_review({}, normalized_constraints)
    fallback.update(
        {
            "sample_id": sample_id,
            "round_idx": round_idx,
            "original_prompt": original_prompt,
            "rewritten_prompt": rewritten_prompt,
            "review_backend": "t2i_constraint_neutral_fallback",
            "parse_mode": "neutral_fallback",
            "used_thinking": bool(enable_thinking),
            "vlm_dialogue": dialogue,
        }
    )
    return fallback
