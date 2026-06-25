#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.multimodal.t2i_constraints import constraints_for_prompt


DEFAULT_PYTHON = "/home/jiazhuangzhuang/miniconda3/envs/aris/bin/python"
DEFAULT_MODEL = "/data/jiazhuangzhuang/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_VLM_MODEL = "/huggingface/model_hub/Qwen3.6-27B"
DEFAULT_NEGATIVE_PROMPT = (
    "text, subtitles, logos, watermark, flicker, jitter, distorted anatomy, "
    "warped geometry, impossible physics, abrupt scene cuts, low quality"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a T2V -> keyframe VLM review prompt refinement loop")
    p.add_argument("--output-root", required=True)
    p.add_argument("--initial-request", required=True)
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--python", default=DEFAULT_PYTHON)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry-run", action="store_true", help="Use the local GIF dry-run generator")
    p.add_argument("--generator", default="wan-t2v")
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--vlm-model-path", default=DEFAULT_VLM_MODEL)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=33)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--sample-prefix", default="t2vloop")
    p.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    p.add_argument("--vlm-eval", dest="vlm_eval", action="store_true", default=True)
    p.add_argument("--no-vlm-eval", dest="vlm_eval", action="store_false")
    p.add_argument("--vlm-max-retries", type=int, default=2)
    p.add_argument("--vlm-enable-thinking", action="store_true")
    p.add_argument("--stop-on-pass", action="store_true", default=True)
    p.add_argument("--no-stop-on-pass", dest="stop_on_pass", action="store_false")
    return p.parse_args()


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _rel(value: object, base: Path) -> str:
    if not value:
        return ""
    path = Path(str(value))
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def build_initial_prompt(user_request: str) -> str:
    return (
        "Create a short, coherent text-to-video dataset sample for this request: "
        f"{user_request.strip()} "
        "Use one continuous scene with a clear beginning, middle, and end. Keep subject identity, camera viewpoint, "
        "lighting, and scene layout consistent across frames. Motion should be physically plausible and easy to inspect. "
        "Avoid text, subtitles, logos, watermarks, heavy blur, flicker, warped geometry, and abrupt scene cuts."
    )


def _checklist_rows(review: Dict[str, object], statuses: set[str]) -> List[dict]:
    rows = review.get("video_checklist") or review.get("constraint_checklist") or []
    return [
        row
        for row in rows
        if isinstance(row, dict) and str(row.get("status")) in statuses
    ]


def refine_prompt(original_request: str, review: Dict[str, object] | None, round_idx: int) -> Tuple[str, str]:
    if not review:
        return build_initial_prompt(original_request), "initial_video_prompt"

    issue_tags = [str(item) for item in (review.get("issue_tags") or []) if item]
    validation_issues = [str(item) for item in (review.get("validation_issues") or []) if item]
    suggestions = [str(item) for item in (review.get("rewrite_suggestions") or []) if item]
    repair_rows = _checklist_rows(review, {"fail", "uncertain"})
    pass_rows = _checklist_rows(review, {"pass"})

    prompt_parts = [
        "Regenerate a short text-to-video dataset sample for this original request:",
        original_request.strip(),
        f"This is revision {round_idx}; repair the previous round while preserving the intended scene and action.",
        "Use a single continuous scene, stable camera behavior, coherent lighting, consistent subject identity, and plausible motion.",
        "Make the main action readable in the keyframes and avoid abrupt scene changes.",
        "Avoid text, subtitles, logos, watermarks, flicker, jitter, warped geometry, impossible physics, and unusable blur.",
    ]
    if repair_rows:
        prompt_parts.append("Repair these failed or uncertain checks:")
        prompt_parts.extend(
            f"{row.get('constraint_id')}: {row.get('description')} (previous status={row.get('status')})"
            for row in repair_rows[:12]
        )
    if suggestions:
        prompt_parts.append("Use these reviewer suggestions:")
        prompt_parts.extend(suggestions[:6])
    if issue_tags or validation_issues:
        prompt_parts.append(
            "Previous issue tags: "
            + ", ".join((issue_tags + validation_issues)[:12])
            + "."
        )
    if pass_rows:
        prompt_parts.append("Preserve these checks that already passed:")
        prompt_parts.extend(
            f"{row.get('constraint_id')}: {row.get('description')}"
            for row in pass_rows[:8]
            if row.get("constraint_id") and row.get("description")
        )
    reason_ids = [
        str(row.get("constraint_id"))
        for row in repair_rows
        if row.get("constraint_id")
    ]
    reason = "repair_video_constraints: " + ", ".join(reason_ids[:12]) if reason_ids else "generic_video_quality_rewrite"
    return " ".join(prompt_parts), reason


def compact_review(row: dict) -> dict:
    review = row.get("vlm_review") or {}
    validation = row.get("validation") or {}
    return {
        "vlm_status": review.get("status"),
        "vlm_route": review.get("vlm_route"),
        "scores": review.get("scores", {}),
        "issue_tags": review.get("issue_tags", []),
        "video_checklist": review.get("video_checklist", []),
        "video_status_counts": review.get("video_status_counts", {}),
        "video_pass_rate": review.get("video_pass_rate"),
        "failed_video_constraints": review.get("failed_video_constraints", []),
        "uncertain_video_constraints": review.get("uncertain_video_constraints", []),
        "passed_video_constraints": review.get("passed_video_constraints", []),
        "rewrite_suggestions": review.get("rewrite_suggestions", []),
        "validation_status": validation.get("status"),
        "validation_issues": validation.get("issues", []),
        "width": validation.get("width"),
        "height": validation.get("height"),
        "num_frames": validation.get("num_frames"),
        "container": validation.get("container"),
        "file_size": validation.get("file_size"),
        "output_path": row.get("output_path"),
        "review_path": review.get("review_path"),
        "contact_sheet_path": review.get("contact_sheet_path"),
        "keyframe_paths": review.get("keyframe_paths", []),
    }


def review_rank(review: Dict[str, object]) -> Tuple[int, int, float, int]:
    route_score = {"pass": 3, "needs_fix": 2, "reject": 1}.get(str(review.get("vlm_route")), 0)
    validation_score = 1 if review.get("validation_status") == "pass" else 0
    pass_rate = review.get("video_pass_rate")
    if isinstance(pass_rate, (int, float)):
        fail_count = int((review.get("video_status_counts") or {}).get("fail", 0))
        return route_score, validation_score, float(pass_rate) * 100.0, -fail_count
    frame_count = int(review.get("num_frames") or 0)
    return route_score, validation_score, 0.0, frame_count


def _is_pass(review: Dict[str, object]) -> bool:
    if review.get("validation_status") != "pass":
        return False
    if review.get("vlm_status") in {None, "pending", "skipped"}:
        return False
    return review.get("vlm_route") == "pass"


def write_reports(output_root: Path, state: dict) -> None:
    history = state.get("history") or []
    md_lines = [
        "# DataEvolver T2V VLM Loop Report",
        "",
        f"- Status: `{state.get('status')}`",
        f"- Initial request: {state.get('initial_request')}",
        f"- Max rounds: {state.get('max_rounds')}",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Rounds",
        "",
    ]
    for event in history:
        review = event.get("review") or {}
        md_lines.extend(
            [
                f"### Round {event.get('round_idx')}",
                "",
                f"- Return code: `{event.get('returncode')}`",
                f"- Rewrite reason: `{event.get('rewrite_reason')}`",
                f"- Video: `{event.get('video_path')}`",
                f"- Contact sheet: `{review.get('contact_sheet_path') or ''}`",
                f"- VLM route: `{review.get('vlm_route')}`",
                f"- Video pass rate: `{review.get('video_pass_rate')}`",
                f"- Failed constraints: `{review.get('failed_video_constraints')}`",
                f"- Uncertain constraints: `{review.get('uncertain_video_constraints')}`",
                "",
                "Prompt:",
                "",
                "```text",
                str(event.get("prompt") or ""),
                "```",
                "",
            ]
        )
    (output_root / "T2V_LOOP_REPORT.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    cards = []
    for event in history:
        review = event.get("review") or {}
        round_dir = Path(str(event.get("round_dir")))
        video_src = _rel((round_dir / str(event.get("video_path"))).resolve(), output_root.resolve()) if event.get("video_path") else ""
        sheet_src = _rel((round_dir / str(review.get("contact_sheet_path"))).resolve(), output_root.resolve()) if review.get("contact_sheet_path") else ""
        cards.append(
            f"""
            <section class="round">
              <div class="round-head">
                <h2>Round {html.escape(str(event.get('round_idx')))}</h2>
                <span>{html.escape(str(review.get('vlm_route')))}</span>
              </div>
              <video controls src="{html.escape(video_src)}"></video>
              {'<img src="' + html.escape(sheet_src) + '" alt="keyframe contact sheet">' if sheet_src else ''}
              <dl>
                <dt>Rewrite</dt><dd>{html.escape(str(event.get('rewrite_reason')))}</dd>
                <dt>Validation</dt><dd>{html.escape(str(review.get('validation_status')))} {html.escape(str(review.get('validation_issues') or []))}</dd>
                <dt>Pass rate</dt><dd>{html.escape(str(review.get('video_pass_rate')))}</dd>
                <dt>Failed</dt><dd>{html.escape(str(review.get('failed_video_constraints') or []))}</dd>
                <dt>Uncertain</dt><dd>{html.escape(str(review.get('uncertain_video_constraints') or []))}</dd>
              </dl>
              <details><summary>Prompt</summary><pre>{html.escape(str(event.get('prompt') or ''))}</pre></details>
            </section>
            """
        )
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DataEvolver T2V Loop Report</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1e2428; background: #f6f7f8; }}
    header {{ padding: 28px 36px 20px; background: #111820; color: white; }}
    header h1 {{ margin: 0 0 8px; font-size: 26px; letter-spacing: 0; }}
    header p {{ margin: 4px 0; max-width: 980px; color: #d7dde2; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    .round {{ background: white; border: 1px solid #d7dde2; border-radius: 8px; padding: 18px; margin-bottom: 20px; }}
    .round-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: center; }}
    .round-head h2 {{ margin: 0 0 12px; font-size: 20px; }}
    .round-head span {{ padding: 4px 8px; border: 1px solid #b9c2ca; border-radius: 999px; font-size: 13px; }}
    video, img {{ display: block; width: 100%; max-height: 520px; object-fit: contain; background: #101418; border-radius: 6px; margin: 10px 0 14px; }}
    dl {{ display: grid; grid-template-columns: 140px 1fr; gap: 8px 14px; margin: 0; }}
    dt {{ font-weight: 700; color: #4a545d; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    details {{ margin-top: 14px; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f0f2f4; padding: 12px; border-radius: 6px; }}
  </style>
</head>
<body>
  <header>
    <h1>DataEvolver T2V Loop Report</h1>
    <p>Status: {html.escape(str(state.get('status')))}</p>
    <p>Initial request: {html.escape(str(state.get('initial_request')))}</p>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    (output_root / "T2V_LOOP_REPORT.html").write_text(html_doc, encoding="utf-8")


def main() -> None:
    args = parse_args()
    entrypoint = REPO_ROOT / "pipeline" / "multimodal" / "run_multimodal_dataset.py"
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    original_prompt = args.initial_request
    constraints = constraints_for_prompt(original_prompt)
    prompt, rewrite_reason = refine_prompt(original_prompt, None, 0)
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
                    "negative_prompt": args.negative_prompt,
                }
            ],
        )

        generator = "dryrun" if args.dry_run else args.generator
        cmd = [
            args.python,
            str(entrypoint),
            "--route",
            "t2v",
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
            "--num-frames",
            str(args.num_frames),
            "--fps",
            str(args.fps),
            "--steps",
            str(args.steps),
            "--guidance-scale",
            str(args.guidance_scale),
            "--negative-prompt",
            args.negative_prompt,
            "--num-samples",
            "1",
            "--seed",
            str(args.seed + round_idx),
            "--sample-prefix",
            f"{args.sample_prefix}{round_idx:02d}",
            "--no-skip-existing",
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        else:
            cmd.extend(["--allow-model-inference", "--model-path", args.model_path])
        if args.vlm_eval:
            cmd.extend(
                [
                    "--vlm-eval",
                    "--vlm-device",
                    args.device,
                    "--vlm-model-path",
                    args.vlm_model_path,
                    "--vlm-max-samples",
                    "1",
                    "--vlm-max-retries",
                    str(args.vlm_max_retries),
                ]
            )
            if args.vlm_enable_thinking:
                cmd.append("--vlm-enable-thinking")

        started = datetime.now(timezone.utc).isoformat()
        result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
        logs_dir = round_dir / "loop_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "stdout.log").write_text(result.stdout or "", encoding="utf-8")
        (logs_dir / "stderr.log").write_text(result.stderr or "", encoding="utf-8")
        (logs_dir / "command.json").write_text(json.dumps(cmd, indent=2) + "\n", encoding="utf-8")

        row = {}
        if (round_dir / "manifest.jsonl").exists():
            rows = read_jsonl(round_dir / "manifest.jsonl")
            row = rows[0] if rows else {}
        review = compact_review(row)
        event = {
            "round_idx": round_idx,
            "started_at": started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "original_prompt": original_prompt,
            "prompt": prompt,
            "rewrite_reason": rewrite_reason,
            "video_path": row.get("output_path"),
            "review_json": review.get("review_path"),
            "round_dir": str(round_dir),
            "returncode": result.returncode,
            "review": review,
        }
        history.append(event)
        if best_event is None or review_rank(review) > review_rank(best_event["review"]):
            best_event = event

        state = {
            "status": "running",
            "initial_request": original_prompt,
            "max_rounds": args.max_rounds,
            "history": history,
            "best": best_event,
        }
        write_json(output_root / "loop_state.json", state)
        write_reports(output_root, state)

        if result.returncode != 0:
            state["status"] = "failed"
            write_json(output_root / "loop_state.json", state)
            write_reports(output_root, state)
            raise SystemExit(result.returncode)
        if args.stop_on_pass and _is_pass(review):
            state["status"] = "passed"
            write_json(output_root / "loop_state.json", state)
            write_reports(output_root, state)
            return
        prompt, rewrite_reason = refine_prompt(original_prompt, review, round_idx + 1)

    state = {
        "status": "max_rounds_reached",
        "initial_request": original_prompt,
        "max_rounds": args.max_rounds,
        "history": history,
        "best": best_event,
    }
    write_json(output_root / "loop_state.json", state)
    write_reports(output_root, state)


if __name__ == "__main__":
    main()
