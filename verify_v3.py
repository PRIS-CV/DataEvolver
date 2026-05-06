#!/usr/bin/env python3
"""verify_v3.py — 相对断言验证：v3 vs v2 baseline"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
V2_SUMMARY_PATH = ROOT / "pipeline" / "data" / "evolution_v2" / "evolution_summary.json"
V3_SUMMARY_PATH = ROOT / "pipeline" / "data" / "evolution_v3" / "evolution_summary.json"


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


def mean_final_score(summary: dict[str, Any]) -> float:
    final_scores = summary.get("final_scores", {})
    if not isinstance(final_scores, dict) or not final_scores:
        return 0.0
    scores = [float(score) for score in final_scores.values()]
    return sum(scores) / len(scores)


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
        v3 = load_json(V3_SUMMARY_PATH)
    except Exception as exc:
        print(f"FAIL: {[str(exc)]}")
        sys.exit(1)

    v2_total = safe_int(v2.get("total"))
    v3_total = safe_int(v3.get("total"))
    v2_accepted = safe_int(v2.get("accepted"))
    v3_accepted = safe_int(v3.get("accepted"))
    v2_avg = mean_final_score(v2)
    v3_avg = mean_final_score(v3)
    v3_flat_ratio = flat_lighting_ratio(V3_SUMMARY_PATH)

    if v3_total != 10:
        errors.append(f"v3 total must be 10, got {v3_total}")

    if v3_accepted < v2_accepted:
        errors.append(
            f"v3 accepted regressed: v3={v3_accepted} < v2={v2_accepted}"
        )

    score_improved = v3_avg >= v2_avg + 0.01
    flat_ok = v3_flat_ratio < 0.80
    if not (score_improved or flat_ok):
        errors.append(
            "improvement check failed: "
            f"v3 avg_score={v3_avg:.4f}, v2 avg_score={v2_avg:.4f}, "
            f"v3 flat_lighting_ratio={v3_flat_ratio:.2%}"
        )

    print(f"v2: accepted={v2_accepted}/{v2_total} avg={v2_avg:.4f}")
    print(f"v3: accepted={v3_accepted}/{v3_total} avg={v3_avg:.4f} flat={v3_flat_ratio:.0%}")

    if errors:
        print(f"FAIL: {errors}")
        sys.exit(1)

    print("PASS")


if __name__ == "__main__":
    main()
