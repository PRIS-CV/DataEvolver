#!/usr/bin/env python3
"""Create or update FEEDBACK_STATE.json for feedback-loop runs."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


DEFAULT_WORKDIR = "/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build"
DEFAULT_BASELINE_DATASET = (
    "dataset_scene_v7_full50_rotation8_trainready_front2others_"
    "splitobj_seed42_bboxmask_bright_objinfo_20260418"
)
DEFAULT_BASELINE_CHECKPOINT = "DiffSynth-Studio/output/rotation8_bright_objinfo_rank32/epoch_0029/lora.safetensors"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def command_file(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def get_round(state: dict, round_number: int, run_root: Path) -> dict:
    rounds = state.setdefault("rounds", [])
    for item in rounds:
        if item.get("round") == round_number:
            return item
    item = {
        "round": round_number,
        "run_root": str(run_root),
        "created_at": now_iso(),
        "paths": {
            "dataset": str(run_root / "dataset"),
            "checkpoint": str(run_root / "output" / "epoch_0029" / "lora.safetensors"),
            "eval_results": str(run_root / "eval_results"),
            "intervention_plan": str(run_root / "INTERVENTION_PLAN.md"),
            "compare_report": str(run_root / "compare_report.json"),
        },
        "commands": {},
        "steps": {},
    }
    rounds.append(item)
    return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update feedback-loop state")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--step", required=True)
    parser.add_argument("--status", default="completed")
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--validation-report", default=None)
    parser.add_argument("--compare-report", default=None)
    parser.add_argument("--dataset-feedback-plan", default=None)
    parser.add_argument("--augmented-manifest", default=None)
    parser.add_argument("--pinned-split-path", default=None)
    parser.add_argument("--golden-config-path", default=None)
    parser.add_argument("--wwz-run-root", default=None)
    parser.add_argument("--server68-run-root", default=None)
    parser.add_argument("--message", default=None)
    parser.add_argument("--workdir", default=os.environ.get("WORKDIR", DEFAULT_WORKDIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    state_path = Path(args.state_path).resolve() if args.state_path else run_root.parent / "FEEDBACK_STATE.json"
    workdir = Path(args.workdir)

    state = read_json(state_path) or {
        "schema_version": "feedback_state_v2",
        "experiment_id": args.experiment_id,
        "status": "in_progress",
        "baseline_dataset": str(workdir / DEFAULT_BASELINE_DATASET),
        "baseline_checkpoint": str(workdir / DEFAULT_BASELINE_CHECKPOINT),
        "current_dataset": None,
        "baseline": {
            "experiment": "exp5_objinfo",
            "checkpoint_rule": "epoch_29_fixed",
            "dataset_root": str(workdir / DEFAULT_BASELINE_DATASET),
            "checkpoint": str(workdir / DEFAULT_BASELINE_CHECKPOINT),
            "eval_dir": os.environ.get("BASELINE_EVAL_DIR", str(run_root.parent / "round_0" / "eval_results")),
        },
        "rounds": [],
    }
    state["experiment_id"] = args.experiment_id
    state["schema_version"] = "feedback_state_v2"
    state["current_round"] = args.round
    state.setdefault("created_at", now_iso())
    state["updated_at"] = now_iso()
    if args.status == "failed":
        state["status"] = "needs_attention"

    round_item = get_round(state, args.round, run_root)
    round_item["updated_at"] = now_iso()
    round_item["steps"][args.step] = {"status": args.status, "updated_at": now_iso()}
    if args.message:
        round_item["steps"][args.step]["message"] = args.message

    commands = round_item.setdefault("commands", {})
    for name in ("train", "eval", "compare"):
        path = command_file(run_root / f"{name}_command.sh")
        if path:
            commands[name] = path

    validation_path = Path(args.validation_report) if args.validation_report else run_root / "validation_report.json"
    validation = read_json(validation_path)
    if validation:
        round_item["steps"]["validation"] = {
            "status": "completed" if validation.get("status") == "passed" else "failed",
            "result": validation.get("status"),
            "pair_counts": validation.get("pair_counts"),
            "warnings": validation.get("warnings", []),
            "errors": validation.get("errors", []),
            "report": str(validation_path),
        }

    feedback_plan_path = Path(args.dataset_feedback_plan) if args.dataset_feedback_plan else run_root / "dataset_feedback_plan.json"
    feedback_plan = read_json(feedback_plan_path)
    augmented_manifest_path = (
        Path(args.augmented_manifest)
        if args.augmented_manifest
        else run_root / "dataset" / "augmented_manifest.json"
    )
    augmented_manifest = read_json(augmented_manifest_path)
    if feedback_plan or augmented_manifest or args.pinned_split_path or args.golden_config_path:
        construction = round_item.setdefault("v2_dataset_construction", {})
        construction.update({
            "wwz_run_root": args.wwz_run_root,
            "server68_run_root": args.server68_run_root or str(run_root),
            "pinned_split_path": args.pinned_split_path,
            "golden_config_path": args.golden_config_path,
        })
        if augmented_manifest:
            construction.update({
                "augmented_manifest_path": str(augmented_manifest_path),
                "new_object_ids": augmented_manifest.get("new_object_ids"),
                "object_count_delta": augmented_manifest.get("object_count_delta"),
                "weak_target_rotations": augmented_manifest.get("weak_target_rotations"),
                "pair_count_delta": augmented_manifest.get("pair_count_delta"),
            })
            state["current_dataset"] = str(run_root / "dataset")
        if feedback_plan:
            construction["dataset_feedback_plan_path"] = str(feedback_plan_path)
            construction.setdefault("weak_target_rotations", feedback_plan.get("weak_target_rotations"))
            round_item["cost"] = {
                **(round_item.get("cost") or {}),
                **(feedback_plan.get("cost") or {}),
            }

    compare_path = Path(args.compare_report) if args.compare_report else run_root / "compare_report.json"
    compare = read_json(compare_path)
    if compare:
        round_item["steps"]["compare"] = {
            "status": "completed",
            "verdict": compare.get("verdict"),
            "report": str(compare_path),
        }
        round_item["verdict"] = compare.get("verdict")
        round_item["delta_summary"] = {
            source: {
                "overall_delta": payload.get("overall_delta"),
                "weak_angles": payload.get("weak_angles"),
                "weak_delta": payload.get("weak_delta"),
            }
            for source, payload in (compare.get("sources") or {}).items()
        }
        primary = (compare.get("sources") or {}).get("spatialedit")
        if primary is None and compare.get("sources"):
            primary = next(iter(compare["sources"].values()))
        if primary:
            round_item["diagnostics"] = {
                "weak_angles": primary.get("weak_angles"),
                "strong_angles": primary.get("strong_angles"),
                "all_angle_bad_objects": primary.get("all_angle_bad_objects"),
                "object_angle_outliers": primary.get("object_angle_outliers"),
                "per_angle_delta": primary.get("per_angle_delta"),
                "per_object_delta": primary.get("per_object_delta"),
                "verdict": compare.get("verdict"),
                "strong_angle_regressions": primary.get("strong_angle_regressions"),
            }

    write_json(state_path, state)
    print(f"Wrote {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
