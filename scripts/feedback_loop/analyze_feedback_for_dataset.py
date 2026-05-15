#!/usr/bin/env python3
"""Turn compare diagnostics into a dataset-construction plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_STAGE_COST_HOURS = {
    "stage1_text_expansion": 2 / 60,
    "stage2_t2i_20_objects": 10 / 60,
    "stage2_5_sam2": 5 / 60,
    "stage3_hunyuan3d_20_objects": 50 / 60,
    "stage4_vlm_loop_per_object_min": 15 / 60,
    "stage4_vlm_loop_per_object_max": 30 / 60,
    "rotation_export": 20 / 60,
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def pick_primary_source(compare_report: dict) -> dict:
    sources = compare_report.get("sources") or {}
    if "spatialedit" in sources:
        return sources["spatialedit"]
    if "testset" in sources:
        return sources["testset"]
    if sources:
        return next(iter(sources.values()))
    raise ValueError("compare_report has no sources")


def estimate_cost(new_object_count: int) -> dict:
    scale = max(float(new_object_count) / 20.0, 0.0)
    fixed = (
        DEFAULT_STAGE_COST_HOURS["stage1_text_expansion"]
        + DEFAULT_STAGE_COST_HOURS["stage2_t2i_20_objects"] * scale
        + DEFAULT_STAGE_COST_HOURS["stage2_5_sam2"] * scale
        + DEFAULT_STAGE_COST_HOURS["stage3_hunyuan3d_20_objects"] * scale
        + DEFAULT_STAGE_COST_HOURS["rotation_export"]
    )
    stage4_min = DEFAULT_STAGE_COST_HOURS["stage4_vlm_loop_per_object_min"] * new_object_count
    stage4_max = DEFAULT_STAGE_COST_HOURS["stage4_vlm_loop_per_object_max"] * new_object_count
    return {
        "estimated_hours_min": round(fixed + stage4_min, 2),
        "estimated_hours_max": round(fixed + stage4_max, 2),
        "assumption": "Stage4 VLM loop uses 15-30 min/object after render-prior warm start.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dataset_feedback_plan.json from compare_report.json")
    parser.add_argument("--compare-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--new-object-budget", type=int, default=20)
    parser.add_argument("--max-weak-angles", type=int, default=4)
    parser.add_argument("--cost-cap-hours", type=float, default=12.0)
    args = parser.parse_args()

    compare_report = read_json(Path(args.compare_report))
    source = pick_primary_source(compare_report)
    weak_angles = [int(angle) for angle in source.get("weak_angles", []) if int(angle) != 360]
    weak_angles = weak_angles[: max(1, args.max_weak_angles)]
    if not weak_angles:
        raise ValueError(
            "No weak target rotations were found in compare_report; refusing to write an empty dataset feedback plan."
        )
    strong_angles = [int(angle) for angle in source.get("strong_angles", []) if int(angle) != 360]
    object_budget = args.new_object_budget
    cost = estimate_cost(object_budget)
    if cost["estimated_hours_max"] > args.cost_cap_hours:
        object_budget = max(1, min(args.new_object_budget, int(args.new_object_budget * args.cost_cap_hours / cost["estimated_hours_max"])))
        cost = estimate_cost(object_budget)

    plan = {
        "schema_version": "dataset_feedback_plan_v2",
        "source_compare_report": str(Path(args.compare_report).resolve()),
        "primary_source": source.get("source"),
        "weak_target_rotations": weak_angles,
        "strong_target_rotations": strong_angles,
        "new_object_budget": object_budget,
        "max_weak_angles": args.max_weak_angles,
        "category_quota": {},
        "diagnostics": {
            "verdict": compare_report.get("verdict"),
            "all_angle_bad_objects": source.get("all_angle_bad_objects", []),
            "object_angle_outliers": source.get("object_angle_outliers", {}),
            "systematic_weak_angles": source.get("systematic_weak_angles", []),
            "object_specific_weak_angles": source.get("object_specific_weak_angles", []),
            "strong_angle_regressions": source.get("strong_angle_regressions", []),
        },
        "cost": cost,
    }
    write_json(Path(args.output), plan)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
