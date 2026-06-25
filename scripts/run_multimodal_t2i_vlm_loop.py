#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.multimodal.t2i_constraints import (
    constraint_improvement,
    constraints_for_prompt,
    failed_constraint_rows,
    has_failure_then_improvement,
    select_prompt_specs,
    uncertain_constraint_rows,
    write_prompt_bank,
)


DEFAULT_PYTHON = "/home/jiazhuangzhuang/miniconda3/envs/aris/bin/python"
DEFAULT_MODEL = "/data/wuwenzhuo/Qwen-Image-2512"
DEFAULT_VLM_MODEL = "/huggingface/model_hub/Qwen3.6-27B"
COMPLEX_SCENE_TERMS = [
    "scene", "interior", "exterior", "environment", "street", "market", "night market",
    "greenhouse", "underwater", "submarine", "coral", "forest", "city", "kitchen",
    "counter", "workbench", "desk", "table", "bridge", "train", "space station",
    "astronaut", "terrarium", "landscape", "crowd", "shoppers", "vendors", "shelves",
    "background elements", "detailed background", "busy background", "foreground and background",
    "surrounded by", "beside", "next to", "behind",
    "in front of", "include", "including", "several", "multiple", "many",
    "场景", "室内", "室外", "街道", "市场", "夜市", "温室", "水下", "珊瑚", "森林",
    "城市", "厨房", "台面", "工作台", "桌面", "桥", "火车", "空间站", "宇航员",
    "玻璃缸", "背景元素", "详细背景", "复杂背景", "前景和背景", "周围", "旁边", "后面",
    "包括", "多个", "许多",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a T2I -> old VLM review prompt refinement loop")
    p.add_argument("--output-root", required=True)
    p.add_argument("--initial-request", default=None)
    p.add_argument("--prompt-id", default=None, help="Use one built-in constraint prompt by id")
    p.add_argument(
        "--prompt-tier",
        default=None,
        choices=["simple_object", "compositional_relation", "hard_constraints"],
        help="When --prompt-id is omitted, pick the first prompt from this built-in tier",
    )
    p.add_argument("--write-prompt-bank", default=None, help="Write the built-in 10/20/20 prompt bank JSON and continue")
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--python", default=DEFAULT_PYTHON)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry-run", action="store_true", help="Run the multimodal dry-run generator for local loop structure tests")
    p.add_argument("--generator", default="qwen-image")
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--vlm-model-path", default=DEFAULT_VLM_MODEL)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--true-cfg-scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=9001)
    p.add_argument("--sample-prefix", default="loop")
    p.add_argument("--vlm-max-retries", type=int, default=2)
    p.add_argument("--vlm-review-mode", default="studio")
    p.add_argument("--vlm-use-freeform-first", dest="vlm_use_freeform_first", action="store_true", default=False)
    p.add_argument("--vlm-json-only", dest="vlm_use_freeform_first", action="store_false")
    p.add_argument("--vlm-enable-thinking", action="store_true")
    p.add_argument("--stop-on-pass", action="store_true", default=True)
    p.add_argument("--no-stop-on-pass", dest="stop_on_pass", action="store_false")
    p.add_argument("--require-fail-then-improve", dest="require_fail_then_improve", action="store_true", default=True)
    p.add_argument("--allow-one-round-pass", dest="require_fail_then_improve", action="store_false")
    return p.parse_args()


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compact_review(row: dict) -> dict:
    review = row.get("vlm_review") or {}
    validation = row.get("validation") or {}
    return {
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
        "validation_status": validation.get("status"),
        "validation_issues": validation.get("issues", []),
        "mean_rgb": validation.get("mean_rgb"),
        "luma_mean": validation.get("luma_mean"),
        "foreground_fraction": validation.get("foreground_fraction"),
        "foreground_bbox_xyxy": validation.get("foreground_bbox_xyxy"),
        "output_path": row.get("output_path"),
        "review_path": review.get("review_path"),
    }


def _infer_target(request: str) -> Tuple[str, str | None, str | None]:
    text = request.strip()
    lower = text.lower()
    color = None
    if "红" in text or "red" in lower:
        color = "red"
    elif "蓝" in text or "blue" in lower:
        color = "blue"
    elif "绿" in text or "green" in lower:
        color = "green"
    elif "黄" in text or "yellow" in lower:
        color = "yellow"
    elif "黑" in text or "black" in lower:
        color = "black"
    elif "白" in text or "white" in lower:
        color = "white"

    obj = None
    if any(token in text for token in ["立方体", "方块", "正方体"]) or "cube" in lower:
        obj = "cube"
    elif "球" in text or "sphere" in lower or "ball" in lower:
        obj = "sphere"
    elif "杯" in text or "cup" in lower or "mug" in lower:
        obj = "mug"

    if color and obj:
        return f"a single matte {color} {obj}", color, obj
    if obj:
        return f"a single clear {obj}", color, obj
    return f"the requested subject: {request}", color, obj


def _looks_complex_scene(request: str) -> bool:
    text = request.strip()
    lower = text.lower()
    if any(term in lower for term in COMPLEX_SCENE_TERMS):
        return True
    comma_like_count = sum(text.count(sep) for sep in [",", "，", ";", "；"])
    relation_count = sum(lower.count(term) for term in [" with ", " and ", " on ", " under ", " beside "])
    return comma_like_count >= 3 or relation_count >= 4


def _requests_text_or_logo(request: str) -> bool:
    lower = request.lower()
    negated_terms = [
        "no text", "without text", "no readable", "without readable",
        "no logo", "no logos", "without logo", "without logos",
        "不要文字", "无文字", "没有文字", "不要logo", "无logo", "没有logo",
    ]
    if any(term in lower for term in negated_terms):
        return False
    return any(
        term in lower
        for term in [
            "readable text", "legible text", "text that says", "sign that says",
            "label reading", "readable label", "with the word", "spells",
            "company logo", "brand logo", "with logo", "typography",
            "文字内容", "写着", "标语写着", "标签写着", "徽标为",
        ]
    )


def _refine_complex_scene_prompt(base_request: str, review: Dict[str, object], round_idx: int) -> str:
    tags = set(review.get("issue_tags") or [])
    issues = set(review.get("validation_issues") or [])
    problems = tags | issues
    scores = review.get("scores") or {}
    prompt_parts = [
        "Create a detailed, visually coherent dataset image that fully matches this scene request:",
        base_request.strip(),
        "Preserve all named subjects, objects, colors, materials, lighting cues, and spatial relationships from the request.",
        "Use coherent perspective, natural depth, balanced exposure, visible shadows, and clear separation between foreground, midground, and background.",
        "Keep the main focal elements easy to inspect while retaining the requested environment and supporting objects.",
    ]
    if not _requests_text_or_logo(base_request):
        prompt_parts.append("Avoid readable text, fake lettering, logos, watermarks, and sign-like artifacts.")
    if {"near_white", "washed_out_low_contrast", "foreground_missing", "weak_subject_separation"} & problems:
        prompt_parts.append(
            "Avoid overexposure and washed-out contrast; make the scene clearly visible with distinct silhouettes and tonal separation."
        )
    if {"object_too_small", "foreground_too_small"} & problems:
        prompt_parts.append("Make the requested focal subjects larger and easier to inspect without removing the scene context.")
    if "color_shift" in problems:
        prompt_parts.append("Preserve the requested object colors and material colors accurately.")
    if "geometry_distortion" in problems or scores.get("object_integrity", 3) <= 2:
        prompt_parts.append("Prioritize stable geometry, consistent anatomy, and physically plausible object structure.")
    if {"text_artifact", "logo_or_watermark", "prompt_constraint_miss"} & problems:
        prompt_parts.append("Strictly satisfy negative constraints from the request, especially no-text, no-logo, and no-watermark constraints.")
    if round_idx > 0:
        prompt_parts.append(f"This is revision {round_idx}; correct the previous issues while preserving the original complex scene request.")
    return " ".join(prompt_parts)


def refine_prompt(base_request: str, review: Dict[str, object] | None, round_idx: int) -> str:
    review = review or {}
    tags = set(review.get("issue_tags") or [])
    issues = set(review.get("validation_issues") or [])
    problems = tags | issues
    scores = review.get("scores") or {}
    if _looks_complex_scene(base_request):
        return _refine_complex_scene_prompt(base_request, review, round_idx)
    target, color, obj = _infer_target(base_request)

    background = "an off-white or very light gray studio background, not a pure white blank canvas"
    if "白" in base_request or "white" in base_request.lower():
        background = "an off-white studio background that reads as white but still has visible tonal separation"

    prompt_parts = [
        f"Create one visually logical dataset image of {target}.",
        "The subject is centered and large, occupying about 45-60 percent of the frame.",
        f"Use {background}.",
        "Use balanced exposure, moderate contrast, and a visible soft shadow under the subject.",
        "No text, watermark, extra objects, abstract blur, blank canvas, or low-contrast washed-out result.",
    ]
    if obj == "cube":
        prompt_parts.append(
            "The cube has crisp straight edges, three visible square faces, coherent perspective, and a solid 3D form."
        )
    if color == "red":
        prompt_parts.append("The dominant object color is saturated matte red, not green, gray, pink, or yellow.")
    elif color:
        prompt_parts.append(f"The dominant object color is clearly {color}.")

    if {"near_white", "washed_out_low_contrast", "foreground_missing", "weak_subject_separation"} & problems:
        prompt_parts.append(
            "Avoid overexposure: keep the background below pure white, add edge separation, and make the object clearly visible."
        )
    if {"object_too_small", "foreground_too_small"} & problems:
        prompt_parts.append("Make the subject much larger and easier to inspect.")
    if "color_shift" in problems and color:
        prompt_parts.append(f"Correct the color shift: the subject must be {color}.")
    if "geometry_distortion" in problems or scores.get("object_integrity", 3) <= 2:
        prompt_parts.append("Prioritize simple stable geometry over decorative texture.")
    if "background_too_bright" in problems or "overexposed" in problems:
        prompt_parts.append("Use a light gray floor/background gradient and avoid clipping highlights.")
    if round_idx > 0:
        prompt_parts.append(f"This is revision {round_idx}; ignore previous failed artifacts and regenerate cleanly from the request.")
    return " ".join(prompt_parts)


def build_rewrite(base_request: str, review: Dict[str, object] | None, round_idx: int) -> Tuple[str, str]:
    review = review or {}
    prompt = refine_prompt(base_request, review, round_idx)
    repair_rows = failed_constraint_rows(review) + uncertain_constraint_rows(review)
    if not repair_rows:
        if round_idx == 0:
            return prompt, "initial_prompt_rewrite: preserve the original request while making it dataset-oriented"
        return prompt, "no_failed_constraints_available: regenerated from the original request"

    repair_lines = []
    for row in repair_rows:
        repair_lines.append(
            f"{row.get('constraint_id')}: {row.get('description')} (previous status={row.get('status')})"
        )
        kind = str(row.get("type") or "")
        if kind == "text_ban":
            repair_lines.append(
                "text_ban repair detail: make every prop blank and unmarked; use a plain silver coin or disc "
                "with no numerals, letters, logos, embossed symbols, labels, or readable ruler markings"
            )
        elif kind == "occlusion":
            repair_lines.append(
                "occlusion repair detail: put the occluding object physically in front of the named target, "
                "covering only the requested target and not the other primary subjects"
            )
        elif kind == "part_detail":
            repair_lines.append(
                "part_detail repair detail: simplify local geometry so the requested parts are countable and unambiguous"
            )
    passed_ids = review.get("passed_constraints") or []
    pass_rows = [
        row
        for row in review.get("constraint_checklist", [])
        if isinstance(row, dict) and row.get("status") == "pass"
    ]
    preserve_lines = [
        f"{row.get('constraint_id')}: {row.get('description')}"
        for row in pass_rows[:12]
        if row.get("constraint_id") and row.get("description")
    ]
    prompt = (
        f"{prompt} Constraint repair checklist for this revision: "
        + " ; ".join(repair_lines)
        + ". Anti-regression checklist: preserve every previously passing constraint exactly"
    )
    if preserve_lines:
        prompt += ": " + " ; ".join(preserve_lines)
    elif passed_ids:
        prompt += f" ({', '.join(str(x) for x in passed_ids[:8])})"
    prompt += "."
    return (
        prompt,
        "repair_failed_constraints: "
        + "; ".join(str(row.get("constraint_id")) for row in repair_rows),
    )


def review_rank(review: Dict[str, object]) -> Tuple[int, int, float, float]:
    route_score = {"pass": 3, "needs_fix": 2, "reject": 1}.get(str(review.get("vlm_route")), 0)
    validation_score = 1 if review.get("validation_status") == "pass" else 0
    constraint_rate = review.get("constraint_pass_rate")
    if isinstance(constraint_rate, (int, float)):
        return route_score, validation_score, float(constraint_rate) * 100.0, -float(
            (review.get("constraint_status_counts") or {}).get("fail", 0)
        )
    scores = review.get("scores") or {}
    score_sum = float(sum(float(v) for v in scores.values() if isinstance(v, (int, float))))
    foreground = float(review.get("foreground_fraction") or 0.0)
    return route_score, validation_score, score_sum, foreground


def resolve_initial_prompt(args: argparse.Namespace) -> Tuple[str, List[dict], Dict[str, object]]:
    if args.write_prompt_bank:
        write_prompt_bank(Path(args.write_prompt_bank))
    selected = []
    if args.prompt_id or args.prompt_tier:
        selected = select_prompt_specs(tier=args.prompt_tier, prompt_id=args.prompt_id)
        if not selected:
            raise SystemExit(f"No built-in prompt matched prompt_id={args.prompt_id!r}, prompt_tier={args.prompt_tier!r}")
    if selected:
        spec = selected[0]
        prompt = str(spec["prompt"])
        constraints = constraints_for_prompt(prompt, spec.get("constraints"))
        return (
            prompt,
            constraints,
            {
                "tier": spec.get("tier"),
                "source_prompt_id": spec.get("id"),
                "prompt_bank_selected": True,
            },
        )
    if not args.initial_request:
        raise SystemExit("--initial-request is required unless --prompt-id or --prompt-tier is provided")
    prompt = args.initial_request
    return prompt, constraints_for_prompt(prompt), {"prompt_bank_selected": False}


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT
    entrypoint = repo_root / "pipeline" / "multimodal" / "run_multimodal_dataset.py"
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    original_prompt, constraints, prompt_meta = resolve_initial_prompt(args)
    prompt, rewrite_reason = build_rewrite(original_prompt, None, 0)
    history = []
    best_event = None
    for round_idx in range(args.max_rounds):
        round_dir = output_root / f"round_{round_idx:02d}"
        sample_id = f"{args.sample_prefix}{round_idx:02d}_0001"
        prompt_file = round_dir / "model_inputs" / "loop_prompt.json"
        write_json(
            prompt_file,
            [
                {
                    "id": sample_id,
                    "prompt": prompt,
                    "original_prompt": original_prompt,
                    "rewritten_prompt": prompt,
                    "constraints": constraints,
                    "rewrite_reason": rewrite_reason,
                    "tier": prompt_meta.get("tier"),
                    "source_prompt_id": prompt_meta.get("source_prompt_id"),
                }
            ],
        )
        generator = "dryrun" if args.dry_run and args.generator == "qwen-image" else args.generator
        cmd = [
            args.python,
            str(entrypoint),
            "--route",
            "t2i",
            "--user-request",
            original_prompt,
            "--output-root",
            str(round_dir),
            "--generator",
            generator,
            "--prompt-file",
            str(prompt_file),
            "--device",
            args.device,
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--steps",
            str(args.steps),
            "--true-cfg-scale",
            str(args.true_cfg_scale),
            "--num-samples",
            "1",
            "--seed",
            str(args.seed + round_idx),
            "--sample-prefix",
            f"{args.sample_prefix}{round_idx:02d}",
            "--vlm-eval",
            "--vlm-device",
            args.device,
            "--vlm-model-path",
            args.vlm_model_path,
            "--vlm-max-samples",
            "1",
            "--vlm-max-retries",
            str(args.vlm_max_retries),
            "--vlm-review-mode",
            args.vlm_review_mode,
            "--no-skip-existing",
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        else:
            cmd.extend(["--allow-model-inference", "--model-path", args.model_path])
        if args.vlm_use_freeform_first:
            cmd.append("--vlm-use-freeform-first")
        if args.vlm_enable_thinking:
            cmd.append("--vlm-enable-thinking")
        started = datetime.now(timezone.utc).isoformat()
        result = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
        logs_dir = round_dir / "loop_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "stdout.log").write_text(result.stdout or "", encoding="utf-8")
        (logs_dir / "stderr.log").write_text(result.stderr or "", encoding="utf-8")

        row = {}
        if (round_dir / "manifest.jsonl").exists():
            rows = read_jsonl(round_dir / "manifest.jsonl")
            row = rows[0] if rows else {}
        review = compact_review(row)
        prev_review = history[-1]["review"] if history else None
        improvement = constraint_improvement(prev_review, review) if prev_review else {}
        event = {
            "round_idx": round_idx,
            "started_at": started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "original_prompt": original_prompt,
            "rewritten_prompt": prompt,
            "prompt": prompt,
            "rewrite_reason": rewrite_reason,
            "failed_constraints": review.get("failed_constraints", []),
            "uncertain_constraints": review.get("uncertain_constraints", []),
            "passed_constraints": review.get("passed_constraints", []),
            "image_path": row.get("output_path"),
            "review_json": review.get("review_path"),
            "constraint_improvement_from_prev": improvement,
            "round_dir": str(round_dir),
            "returncode": result.returncode,
            "review": review,
        }
        history.append(event)
        if best_event is None or review_rank(review) > review_rank(best_event["review"]):
            best_event = event
        success, success_evidence = has_failure_then_improvement(history)
        state = {
            "status": "running",
            "success_criterion": "failure_then_constraint_improvement"
            if args.require_fail_then_improve
            else "one_round_pass_allowed",
            "success_evidence": success_evidence,
            "prompt_meta": prompt_meta,
            "history": history,
            "best": best_event,
        }
        write_json(output_root / "loop_state.json", state)

        if result.returncode != 0:
            write_json(output_root / "loop_state.json", {"status": "failed", "history": history, "best": best_event})
            raise SystemExit(result.returncode)
        visual_pass = review.get("validation_status") == "pass"
        if args.require_fail_then_improve and success:
            write_json(
                output_root / "loop_state.json",
                {
                    **state,
                    "status": "improved_after_failure",
                    "success_evidence": success_evidence,
                    "history": history,
                    "best": best_event,
                },
            )
            return
        if (
            not args.require_fail_then_improve
            and args.stop_on_pass
            and visual_pass
            and review.get("vlm_route") == "pass"
        ):
            write_json(
                output_root / "loop_state.json",
                {
                    **state,
                    "status": "passed",
                    "history": history,
                    "best": best_event,
                },
            )
            return
        prompt, rewrite_reason = build_rewrite(original_prompt, review, round_idx + 1)

    ever_failed = any((event.get("failed_constraints") or event.get("uncertain_constraints")) for event in history)
    status = "max_rounds_reached" if ever_failed else "no_failure_observed"
    success, success_evidence = has_failure_then_improvement(history)
    if success:
        status = "improved_after_failure"
    write_json(
        output_root / "loop_state.json",
        {
            "status": status,
            "success_criterion": "failure_then_constraint_improvement"
            if args.require_fail_then_improve
            else "one_round_pass_allowed",
            "success_evidence": success_evidence,
            "prompt_meta": prompt_meta,
            "history": history,
            "best": best_event,
        },
    )


if __name__ == "__main__":
    main()
