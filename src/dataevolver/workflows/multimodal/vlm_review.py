from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

from .schemas import DatasetRequest, DatasetSample


T2I_ISSUE_TAGS = [
    "none",
    "underexposed",
    "overexposed",
    "flat_lighting",
    "weak_subject_separation",
    "background_too_bright",
    "background_too_dark",
    "object_too_small",
    "object_too_large",
    "off_center_left",
    "off_center_right",
    "off_center_up",
    "off_center_down",
    "object_cutoff_left",
    "object_cutoff_right",
    "object_cutoff_top",
    "object_cutoff_bottom",
    "geometry_distortion",
    "background_distracting",
    "color_shift",
    "physical_implausibility",
    "text_artifact",
    "logo_or_watermark",
    "prompt_constraint_miss",
]


def _prompt_appendix(request: DatasetRequest, sample: DatasetSample) -> str:
    base = (
        f"Dataset route: {request.route}. User request: {request.user_request}. "
        f"Prompt/instruction: {sample.edit_instruction or sample.prompt}. "
        "Judge image quality and prompt/instruction alignment using the existing DataEvolver review schema."
    )
    if request.route == "t2i":
        return (
            base
            + "\n\nThis is direct text-to-image generation, not 3D scene insertion. "
            "The main acceptance criterion is whether a human can clearly see the requested subject, "
            "with plausible geometry, correct dominant color, usable framing, and no blank/near-white canvas. "
            "For a simple product/object prompt, accept clean synthetic or studio-rendered imagery if it is visually logical. "
            "For a complex scene prompt, check that all major named elements, spatial relationships, material cues, "
            "lighting constraints, and explicit negative constraints are followed. "
            "Do not pass an image only because it is aesthetically strong; it must satisfy the prompt. "
            "If the prompt forbids readable text, signs, logos, or watermarks and the image contains text-like or logo-like marks, "
            "set vlm_route to needs_fix or reject and include text_artifact or logo_or_watermark. "
            "If important requested objects or constraints are missing, include prompt_constraint_miss. "
            "Use reject for blank images, severe overexposure, absent subject, wrong object type, or wrong dominant color. "
            "Use needs_fix for visible but improvable images."
        )
    if request.route == "edit":
        return (
            base
            + "\n\nThis is image editing. Judge instruction following, preservation of unrelated content, "
            "edit locality, and whether the edited image remains coherent."
        )
    return base


def run_vlm_reviews(request: DatasetRequest, output_root: Path, samples: List[DatasetSample]) -> List[DatasetSample]:
    if not request.vlm_eval:
        return samples

    limit = request.vlm_max_samples if request.vlm_max_samples is not None else len(samples)
    review_dir = output_root / "vlm_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)

    if request.vlm_model_path:
        os.environ["VLM_MODEL_PATH"] = request.vlm_model_path

    reviewed = 0
    for sample in samples:
        if reviewed >= limit:
            sample.vlm_review = {
                "status": "skipped",
                "enabled": True,
                "reason": "beyond --vlm-max-samples",
            }
            continue
        if not sample.output_path:
            sample.vlm_review = {"status": "skipped", "enabled": True, "reason": "missing output_path"}
            continue
        rgb_path = output_root / sample.output_path
        if not rgb_path.exists():
            sample.vlm_review = {"status": "skipped", "enabled": True, "reason": "output file missing"}
            continue

        if request.route == "t2i":
            from .t2i_constraint_review import run_t2i_constraint_review

            original_prompt = str(sample.metadata.get("original_prompt") or sample.prompt)
            rewritten_prompt = str(sample.metadata.get("rewritten_prompt") or sample.prompt)
            constraints = sample.metadata.get("constraints") or []
            review = run_t2i_constraint_review(
                rgb_path=str(rgb_path),
                sample_id=sample.sample_id,
                round_idx=0,
                original_prompt=original_prompt,
                rewritten_prompt=rewritten_prompt,
                constraints=constraints,
                device=request.vlm_device,
                max_retries=request.vlm_max_retries,
                vlm_model_path=request.vlm_model_path,
                enable_thinking=request.vlm_enable_thinking,
            )
            path = review_dir / f"{sample.sample_id}_constraint_review.json"
            path.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            sample.vlm_review = {
                "status": "completed",
                "enabled": True,
                "review_path": str(path.relative_to(output_root)),
                "vlm_route": review.get("vlm_route"),
                "scores": review.get("scores", {}),
                "issue_tags": review.get("issue_tags", []),
                "constraint_checklist": review.get("constraint_checklist", []),
                "constraint_status_counts": review.get("constraint_status_counts", {}),
                "constraint_pass_rate": review.get("constraint_pass_rate"),
                "failed_constraints": review.get("failed_constraints", []),
                "uncertain_constraints": review.get("uncertain_constraints", []),
                "passed_constraints": review.get("passed_constraints", []),
                "rewrite_suggestions": review.get("rewrite_suggestions", []),
                "backend": "dataevolver.workflows.multimodal.t2i_constraint_review.run_t2i_constraint_review",
            }
            reviewed += 1
            continue

        if request.route == "edit":
            from .edit_review import run_edit_constraint_review

            if not sample.input_path:
                sample.vlm_review = {"status": "skipped", "enabled": True, "reason": "missing input_path"}
                continue
            source_path = Path(sample.input_path)
            if not source_path.exists():
                sample.vlm_review = {"status": "skipped", "enabled": True, "reason": "source image missing"}
                continue
            edit_instruction = str(
                sample.metadata.get("original_edit_instruction")
                or sample.edit_instruction
                or sample.prompt
            )
            constraints = sample.metadata.get("constraints") or []
            review = run_edit_constraint_review(
                edited_path=str(rgb_path),
                source_path=str(source_path),
                sample_id=sample.sample_id,
                edit_instruction=edit_instruction,
                constraints=constraints,
                device=request.vlm_device,
                max_retries=request.vlm_max_retries,
                vlm_model_path=request.vlm_model_path,
                enable_thinking=request.vlm_enable_thinking,
            )
            path = review_dir / f"{sample.sample_id}_edit_review.json"
            path.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            sample.vlm_review = {
                "status": "completed",
                "enabled": True,
                "review_path": str(path.relative_to(output_root)),
                "vlm_route": review.get("vlm_route"),
                "scores": review.get("scores", {}),
                "issue_tags": review.get("issue_tags", []),
                "edit_checklist": review.get("edit_checklist", []),
                "constraint_checklist": review.get("constraint_checklist", []),
                "edit_status_counts": review.get("edit_status_counts", {}),
                "edit_pass_rate": review.get("edit_pass_rate"),
                "failed_edit_constraints": review.get("failed_edit_constraints", []),
                "uncertain_edit_constraints": review.get("uncertain_edit_constraints", []),
                "passed_edit_constraints": review.get("passed_edit_constraints", []),
                "rewrite_suggestions": review.get("rewrite_suggestions", []),
                "backend": "dataevolver.workflows.multimodal.edit_review.run_edit_constraint_review",
            }
            reviewed += 1
            continue

        if request.route == "t2v":
            from .t2v_review import run_t2v_keyframe_review

            original_prompt = str(sample.metadata.get("original_prompt") or sample.prompt)
            rewritten_prompt = str(sample.metadata.get("rewritten_prompt") or sample.prompt)
            constraints = sample.metadata.get("constraints") or []
            review = run_t2v_keyframe_review(
                video_path=str(rgb_path),
                sample_id=sample.sample_id,
                round_idx=0,
                original_prompt=original_prompt,
                rewritten_prompt=rewritten_prompt,
                constraints=constraints,
                output_dir=str(review_dir),
                device=request.vlm_device,
                max_retries=request.vlm_max_retries,
                vlm_model_path=request.vlm_model_path,
                enable_thinking=request.vlm_enable_thinking,
            )
            path = review_dir / f"{sample.sample_id}_t2v_review.json"
            path.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            def _relative_or_none(value: object) -> object:
                if not value:
                    return value
                try:
                    return str(Path(str(value)).resolve().relative_to(output_root.resolve()))
                except ValueError:
                    return value

            sample.vlm_review = {
                "status": "completed",
                "enabled": True,
                "review_path": str(path.relative_to(output_root)),
                "vlm_route": review.get("vlm_route"),
                "scores": review.get("scores", {}),
                "issue_tags": review.get("issue_tags", []),
                "video_checklist": review.get("video_checklist", []),
                "constraint_checklist": review.get("constraint_checklist", []),
                "video_status_counts": review.get("video_status_counts", {}),
                "video_pass_rate": review.get("video_pass_rate"),
                "failed_video_constraints": review.get("failed_video_constraints", []),
                "uncertain_video_constraints": review.get("uncertain_video_constraints", []),
                "passed_video_constraints": review.get("passed_video_constraints", []),
                "rewrite_suggestions": review.get("rewrite_suggestions", []),
                "contact_sheet_path": _relative_or_none(review.get("contact_sheet_path")),
                "keyframe_paths": [
                    _relative_or_none(path_value)
                    for path_value in (review.get("keyframe_paths") or [])
                ],
                "backend": "dataevolver.workflows.multimodal.t2v_review.run_t2v_keyframe_review",
            }
            reviewed += 1
            continue

        from dataevolver.annotation.vlm_review_stage import run_vlm_review

        appendix = _prompt_appendix(request, sample)
        issue_tags_whitelist = T2I_ISSUE_TAGS if request.route == "t2i" else None
        review = run_vlm_review(
            str(rgb_path),
            sample.sample_id,
            0,
            "image",
            0,
            0,
            prev_rgb_path=sample.input_path,
            device=request.vlm_device,
            max_retries=request.vlm_max_retries,
            prompt_appendix=appendix,
            issue_tags_whitelist=issue_tags_whitelist,
            review_mode=request.vlm_review_mode,
            use_freeform_first=request.vlm_use_freeform_first,
            enable_thinking=request.vlm_enable_thinking,
        )
        path = review_dir / f"{sample.sample_id}_vlm_review.json"
        path.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        sample.vlm_review = {
            "status": "completed",
            "enabled": True,
            "review_path": str(path.relative_to(output_root)),
            "vlm_route": review.get("vlm_route"),
            "scores": review.get("scores", {}),
            "issue_tags": review.get("issue_tags", []),
            "backend": "dataevolver.annotation.vlm_review_stage.run_vlm_review",
        }
        reviewed += 1
    return samples
