#!/usr/bin/env python3
"""
VLM-based evaluation of a trainready rotation dataset.

Runs VLM review on each target render, aggregates scores per-object and
per-angle, and outputs a compare_report-compatible JSON that downstream
scripts (analyze_feedback_for_dataset.py, retire_and_replace.py) can consume.

Usage:
    python scripts/feedback_loop/eval_dataset_vlm.py \
        --dataset-root /path/to/trainready_split \
        --output /path/to/vlm_eval_report.json \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

VLM_WEIGHTS = {
    "lighting": 0.25,
    "object_integrity": 0.30,
    "composition": 0.20,
    "render_quality_semantic": 0.10,
    "overall": 0.15,
}


def compute_vlm_score(scores: dict) -> float:
    return sum(VLM_WEIGHTS[k] * (scores.get(k, 3) - 1) / 4.0 for k in VLM_WEIGHTS)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_args():
    p = argparse.ArgumentParser(description="VLM evaluation of trainready dataset")
    p.add_argument("--dataset-root", required=True, help="Trainready split root")
    p.add_argument("--split", default="train",
                   help="Which split to evaluate (default: train)")
    p.add_argument("--output", required=True, help="Output compare_report JSON path")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--review-mode", default="scene_insert",
                   choices=["scene_insert", "studio"])
    p.add_argument("--enable-thinking", action="store_true", default=True)
    p.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    p.add_argument("--bad-percentile", type=float, default=0.20,
                   help="Bottom percentile for all_angle_bad (default: 0.20)")
    p.add_argument("--weak-angle-count", type=int, default=3,
                   help="Number of weakest angles to report (default: 3)")
    p.add_argument("--profile", default=None,
                   help="Gate profile JSON for review settings")
    return p.parse_args()


def find_pairs_jsonl(dataset_root: Path, split: str) -> Path | None:
    for candidate in [
        dataset_root / "pairs" / f"{split}_pairs.jsonl",
        dataset_root / f"{split}_pairs.jsonl",
    ]:
        if candidate.exists():
            return candidate
    return None


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_path = Path(args.output)

    profile = {}
    if args.profile and Path(args.profile).exists():
        with open(args.profile) as f:
            profile = json.load(f)

    pairs_path = find_pairs_jsonl(dataset_root, args.split)
    if not pairs_path:
        pairs_path = find_pairs_jsonl(dataset_root, "all")
    if not pairs_path:
        print(f"[vlm_eval] ERROR: No pairs JSONL found at {dataset_root}")
        sys.exit(1)

    rows = read_jsonl(pairs_path)
    print(f"[vlm_eval] Evaluating {len(rows)} pairs from {pairs_path}")
    print(f"[vlm_eval] Device: {args.device}, Review mode: {args.review_mode}")

    from pipeline.stage5_5_vlm_review import run_vlm_review

    per_pair_results = []
    t0 = time.time()

    for idx, row in enumerate(rows, 1):
        obj_id = str(row.get("obj_id", "unknown"))
        target_rot = int(row.get("target_rotation_deg", 0))
        target_path = dataset_root / row["target_image"]
        source_path = dataset_root / row["source_image"]
        pair_id = row.get("pair_id", f"{obj_id}_{target_rot}")

        if not target_path.exists():
            print(f"  [{idx}/{len(rows)}] {pair_id}: SKIP (target not found)")
            continue

        print(f"  [{idx}/{len(rows)}] {pair_id} ...", end=" ", flush=True)

        try:
            review = run_vlm_review(
                rgb_path=str(target_path),
                sample_id=f"{obj_id}_rot{target_rot:03d}_vlmeval",
                round_idx=0,
                active_group="object",
                az=target_rot,
                el=0,
                device=args.device,
                review_mode=args.review_mode,
                enable_thinking=args.enable_thinking,
                prompt_appendix=profile.get("prompt_appendix", ""),
                reference_image_path=str(source_path)
                if source_path.exists() else None,
            )

            scores = review.get("scores", {})
            vlm_score = compute_vlm_score(scores)
            hybrid_score = review.get("hybrid_score", vlm_score)

            result = {
                "pair_id": pair_id,
                "obj_id": obj_id,
                "target_rotation_deg": target_rot,
                "vlm_score": round(vlm_score, 4),
                "hybrid_score": round(hybrid_score, 4),
                "scores": scores,
                "vlm_route": review.get("vlm_route", "unknown"),
                "hybrid_route": review.get("hybrid_route", "unknown"),
                "issue_tags": review.get("issue_tags", []),
            }
            per_pair_results.append(result)
            print(f"vlm={vlm_score:.3f} hybrid={hybrid_score:.3f} "
                  f"route={review.get('hybrid_route', '?')}")

        except Exception as e:
            print(f"ERROR: {e}")
            per_pair_results.append({
                "pair_id": pair_id,
                "obj_id": obj_id,
                "target_rotation_deg": target_rot,
                "vlm_score": 0.0,
                "hybrid_score": 0.0,
                "error": str(e),
            })

    elapsed = time.time() - t0
    print(f"\n[vlm_eval] {len(per_pair_results)} reviews done in {elapsed:.0f}s")

    # ── Aggregate per-object ────────────────────────────────────────────────
    obj_scores = defaultdict(list)
    for r in per_pair_results:
        obj_scores[r["obj_id"]].append(r["hybrid_score"])

    per_object = {}
    for obj_id, scores_list in obj_scores.items():
        avg = round(mean(scores_list), 4)
        per_object[obj_id] = {
            "psnr": avg,
            "vlm_score": avg,
            "n_angles": len(scores_list),
        }

    # ── Aggregate per-angle ─────────────────────────────────────────────────
    angle_scores = defaultdict(list)
    for r in per_pair_results:
        angle_scores[str(r["target_rotation_deg"])].append(r["hybrid_score"])

    per_angle = {}
    for angle, scores_list in angle_scores.items():
        avg = round(mean(scores_list), 4)
        per_angle[angle] = {
            "psnr": avg,
            "vlm_score": avg,
            "n_objects": len(scores_list),
        }

    # ── Per-object-angle ────────────────────────────────────────────────────
    per_object_angle: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in per_pair_results:
        per_object_angle[r["obj_id"]][str(r["target_rotation_deg"])] = {
            "psnr": r["hybrid_score"],
            "vlm_score": r["hybrid_score"],
        }

    # ── Identify weak angles (lowest N non-0° angles) ──────────────────────
    non_zero_angles = [
        (a, v["vlm_score"])
        for a, v in per_angle.items()
        if a != "0"
    ]
    non_zero_angles.sort(key=lambda x: x[1])
    weak_angles = [a for a, _ in non_zero_angles[:args.weak_angle_count]]
    strong_angles = [a for a, _ in non_zero_angles[args.weak_angle_count:]]

    # ── Identify all_angle_bad objects (bottom percentile) ──────────────────
    all_angle_bad_objects = []
    if per_object:
        sorted_objs = sorted(per_object.items(), key=lambda x: x[1]["vlm_score"])
        cutoff_idx = max(1, int(len(sorted_objs) * args.bad_percentile))
        cutoff_score = sorted_objs[cutoff_idx - 1][1]["vlm_score"]
        for obj_id, metrics in sorted_objs:
            angle_count = len(per_object_angle.get(obj_id, {}))
            if metrics["vlm_score"] <= cutoff_score and angle_count >= 3:
                all_angle_bad_objects.append(obj_id)

    # ── Object-angle outliers (below mean - 1σ per angle) ──────────────────
    object_angle_outliers = {}
    for angle, scores_list in angle_scores.items():
        if len(scores_list) < 3:
            continue
        angle_mean = mean(scores_list)
        angle_std = pstdev(scores_list) if len(scores_list) > 1 else 0.0
        threshold = angle_mean - angle_std
        outliers = []
        for r in per_pair_results:
            if str(r["target_rotation_deg"]) == angle and r["hybrid_score"] < threshold:
                outliers.append({
                    "object_id": r["obj_id"],
                    "baseline_psnr": round(r["hybrid_score"], 4),
                })
        if outliers:
            object_angle_outliers[angle] = outliers

    # ── Systematic weak angles ──────────────────────────────────────────────
    systematic_weak_angles = []
    if per_angle:
        overall_mean = mean(v["vlm_score"] for v in per_angle.values())
        for angle in weak_angles:
            if per_angle[angle]["vlm_score"] < overall_mean * 0.90:
                systematic_weak_angles.append(angle)

    # ── Build compare_report-compatible output ──────────────────────────────
    overall_score = round(
        mean(r["hybrid_score"] for r in per_pair_results), 4
    ) if per_pair_results else 0

    report = {
        "schema": "vlm_dataset_eval_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "total_pairs": len(per_pair_results),
        "elapsed_seconds": round(elapsed, 1),
        "verdict": "continue",
        "sources": {
            "vlm_dataset_review": {
                "source": "vlm_dataset_review",
                # Fields at source level (for analyze_feedback_for_dataset.py)
                "weak_angles": weak_angles,
                "strong_angles": strong_angles,
                "all_angle_bad_objects": all_angle_bad_objects,
                "object_angle_outliers": object_angle_outliers,
                "systematic_weak_angles": systematic_weak_angles,
                "object_specific_weak_angles": [],
                "strong_angle_regressions": [],
                "overall_delta": {},
                "per_angle_delta": {},
                "per_object_delta": {},
                # Data fields (for retire_and_replace.py)
                "per_object": per_object,
                "per_angle": per_angle,
                "per_object_angle": dict(per_object_angle),
            }
        },
        "summary": {
            "overall_vlm_score": overall_score,
            "per_angle_vlm_scores": {
                a: v["vlm_score"] for a, v in sorted(per_angle.items(),
                                                      key=lambda x: int(x[0]))
            },
            "bad_object_count": len(all_angle_bad_objects),
            "bad_objects": all_angle_bad_objects,
            "weak_angle_count": len(weak_angles),
            "weak_angles": weak_angles,
        },
        "per_pair_results": per_pair_results,
    }

    save_json(output_path, report)

    print(f"\n[vlm_eval] Report saved to {output_path}")
    print(f"  Overall VLM score: {overall_score}")
    print(f"  Bad objects ({len(all_angle_bad_objects)}): {all_angle_bad_objects}")
    print(f"  Weak angles ({len(weak_angles)}): {weak_angles}")
    for angle in sorted(per_angle.keys(), key=lambda x: int(x)):
        s = per_angle[angle]
        print(f"    {angle:>3}°: {s['vlm_score']:.3f} (n={s['n_objects']})")


if __name__ == "__main__":
    main()
