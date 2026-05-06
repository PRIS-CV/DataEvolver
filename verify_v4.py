#!/usr/bin/env python3
"""verify_v4.py -- relative assertion verification: v4 vs v2 baseline
Fix 5: filter by exit_reason, report @0.80 and @0.78 accept counts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
V2_SUMMARY_PATH = ROOT / "pipeline" / "data" / "evolution_v2" / "evolution_summary.json"
V4_SUMMARY_PATH = ROOT / "pipeline" / "data" / "evolution_v4" / "evolution_summary.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def safe_int(value: Any) -> int:
    if value is None:
        return 0
    return int(value)


def mean_final_score(summary: dict[str, Any], exclude_hard_fail: bool = False) -> float:
    """Compute mean final score. If exclude_hard_fail, filter out objects with
    exit_reason=render_hard_fail."""
    final_scores = summary.get("final_scores", {})
    if not isinstance(final_scores, dict) or not final_scores:
        return 0.0

    if exclude_hard_fail:
        results = summary.get("results", {})
        scores = []
        for obj_id, score in final_scores.items():
            obj_result = results.get(obj_id, {})
            exit_reason = obj_result.get("exit_reason")
            if exit_reason == "render_hard_fail":
                continue
            scores.append(float(score))
    else:
        scores = [float(score) for score in final_scores.values()]

    return sum(scores) / len(scores) if scores else 0.0


def count_accepted(summary: dict[str, Any], threshold: float = 0.80) -> int:
    """Count objects with final_hybrid >= threshold, excluding hard_fail."""
    final_scores = summary.get("final_scores", {})
    results = summary.get("results", {})
    count = 0
    for obj_id, score in final_scores.items():
        obj_result = results.get(obj_id, {})
        exit_reason = obj_result.get("exit_reason")
        if exit_reason == "render_hard_fail":
            continue
        if float(score) >= threshold:
            count += 1
    return count


def normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def flat_lighting_ratio(summary_path: Path) -> float:
    evolution_dir = summary_path.parent
    flat_count = 0
    non_none_count = 0

    for result_path in sorted(evolution_dir.glob("obj_*/evolution_result.json")):
        result = load_json(result_path)
        state_log = result.get("state_log", [])
        if not isinstance(state_log, list):
            continue
        for entry in state_log:
            if not isinstance(entry, dict):
                continue
            for tag in normalize_tags(entry.get("issue_tags")):
                tag_lower = tag.lower()
                if tag_lower == "none":
                    continue
                non_none_count += 1
                if tag_lower == "flat_lighting":
                    flat_count += 1

    if non_none_count == 0:
        return 0.0
    return flat_count / non_none_count


def main() -> None:
    errors: list[str] = []

    try:
        v2 = load_json(V2_SUMMARY_PATH)
        v4 = load_json(V4_SUMMARY_PATH)
    except Exception as exc:
        print(f"FAIL: {[str(exc)]}")
        sys.exit(1)

    v2_total = safe_int(v2.get("total"))
    v4_total = safe_int(v4.get("total"))
    v2_accepted = safe_int(v2.get("accepted"))

    # Fix 5: Use exit_reason filtering for avg score
    v2_avg = mean_final_score(v2, exclude_hard_fail=True)
    v4_avg = mean_final_score(v4, exclude_hard_fail=True)

    # Dual threshold reporting
    v4_accepted_80 = count_accepted(v4, 0.80)
    v4_accepted_78 = count_accepted(v4, 0.78)

    v4_flat_ratio = flat_lighting_ratio(V4_SUMMARY_PATH)

    if v4_total != 10:
        errors.append(f"v4 total must be 10, got {v4_total}")

    if v4_accepted_80 < v2_accepted:
        errors.append(
            f"v4 accepted@0.80 regressed: v4={v4_accepted_80} < v2={v2_accepted}"
        )

    score_improved = v4_avg >= v2_avg
    flat_ok = v4_flat_ratio < 0.80
    if not (score_improved or flat_ok):
        errors.append(
            "improvement check failed: "
            f"v4 avg_score={v4_avg:.4f}, v2 avg_score={v2_avg:.4f}, "
            f"v4 flat_lighting_ratio={v4_flat_ratio:.2%}"
        )

    print(f"v2: accepted={v2_accepted}/{v2_total} avg={v2_avg:.4f} (excl hard_fail)")
    print(f"v4: accepted@0.80={v4_accepted_80}/{v4_total} "
          f"accepted@0.78={v4_accepted_78}/{v4_total} "
          f"avg={v4_avg:.4f} (excl hard_fail) flat={v4_flat_ratio:.0%}")

    if errors:
        print(f"FAIL: {errors}")
        sys.exit(1)

    print("PASS")


if __name__ == "__main__":
    main()
