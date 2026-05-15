#!/usr/bin/env python3
"""Compare feedback-loop eval outputs against a baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Optional


LOWER_IS_BETTER = {"lpips", "fid"}
VERDICT_METRICS = ("psnr", "ssim", "lpips", "clip_i", "dino")
DEGRADE_THRESHOLDS = {
    "psnr": -0.5,
    "ssim": -0.01,
    "lpips": -0.02,
    "clip_i": -0.01,
    "dino": -0.01,
}
IMPROVE_THRESHOLDS = {
    "psnr": 0.1,
    "ssim": 0.003,
    "lpips": 0.005,
    "clip_i": 0.003,
    "dino": 0.003,
}
STRONG_ANGLE_REGRESSION_THRESHOLDS = {
    "psnr": -0.3,
    "ssim": -0.008,
    "lpips": -0.015,
    "clip_i": -0.008,
    "dino": -0.008,
}
MODE_PRIORITY = ("ours_feedback", "ours_objinfo", "ours", "feedback", "current", "base", "fal")
ANGLE_IDX_TO_DEG = {0: 45, 1: 90, 2: 135, 3: 180, 4: 225, 5: 270, 6: 315, 7: 360}
METRIC_ALIASES = {
    "psnr": ("psnr", "PSNR"),
    "ssim": ("ssim", "SSIM"),
    "lpips": ("lpips", "LPIPS"),
    "clip_i": ("clip_i", "clip_similarity", "clip-i", "CLIP-I"),
    "dino": ("dino", "dino_similarity", "DINO"),
    "fid": ("fid", "FID"),
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def pick_metric(row: dict, metric: str) -> Optional[float]:
    for key in METRIC_ALIASES[metric]:
        if key in row:
            return to_float(row[key])
    return None


def normalize_metrics(row: dict) -> dict:
    out = {}
    for metric in METRIC_ALIASES:
        value = pick_metric(row, metric)
        if value is not None:
            out[metric] = value
    return out


def pick_mode(summary: dict, requested: Optional[str]) -> Optional[str]:
    if requested and requested in summary:
        return requested
    for mode in MODE_PRIORITY:
        if mode in summary:
            return mode
    return sorted(summary.keys())[0] if summary else None


def find_testset_summary(root: Path, requested_mode: Optional[str]) -> Optional[dict]:
    candidates = list(root.rglob("combined_summary.json")) + list(root.rglob("eval1_summary.json"))
    for path in sorted(candidates):
        payload = read_json(path)
        if "eval1_testset" in payload:
            payload = payload["eval1_testset"]
        if not isinstance(payload, dict):
            continue
        mode = pick_mode(payload, requested_mode)
        if not mode:
            continue
        record = payload[mode]
        if "overall" not in record:
            continue
        per_angle = {
            str(int(float(angle))): normalize_metrics(metrics)
            for angle, metrics in (record.get("per_angle") or {}).items()
        }
        return {
            "source": "testset",
            "mode": mode,
            "path": str(path),
            "overall": normalize_metrics(record.get("overall") or {}),
            "per_angle": per_angle,
            "n_pairs": record.get("n_pairs"),
        }
    return None


def parse_angle(row: dict) -> Optional[int]:
    for key in ("angle", "target_rotation_deg", "angle_deg"):
        value = row.get(key)
        if value not in (None, ""):
            try:
                return int(float(value))
            except ValueError:
                pass
    value = row.get("angle_idx")
    if value not in (None, ""):
        try:
            return ANGLE_IDX_TO_DEG.get(int(float(value)), int(float(value)))
        except ValueError:
            pass

    name = str(row.get("image_name") or row.get("pair_id") or row.get("filename") or "")
    match = re.search(r"angle(\d+)", name)
    if match:
        idx = int(match.group(1))
        return ANGLE_IDX_TO_DEG.get(idx, idx)
    match = re.search(r"yaw(\d{3})", name)
    if match:
        return int(match.group(1))
    return None


def parse_object_id(row: dict) -> Optional[str]:
    for key in ("obj_id", "object_id", "obj_name", "object_name", "object", "name"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)

    for key in ("image_name", "pair_id", "filename", "path", "source_image", "target_image"):
        value = str(row.get(key) or "")
        match = re.search(r"(obj_\d+)", value)
        if match:
            return match.group(1)
    return None


def csv_mode_score(path: Path, requested_mode: Optional[str]) -> int:
    name = path.name.lower()
    if requested_mode and requested_mode.lower() in name:
        return 100
    for idx, mode in enumerate(MODE_PRIORITY):
        if mode.lower() in name:
            return 90 - idx
    return 0


def find_spatialedit_csv(root: Path, requested_mode: Optional[str]) -> Optional[Path]:
    csvs = [
        path for path in root.rglob("*metrics.csv")
        if "per_pair" not in path.name
    ]
    if not csvs:
        return None

    def parseable_score(path: Path) -> int:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if normalize_metrics(row) and parse_angle(row) is not None:
                        return 10 + (1 if parse_object_id(row) else 0)
        except OSError:
            return 0
        return 0

    ranked = sorted(
        csvs,
        key=lambda p: (parseable_score(p), csv_mode_score(p, requested_mode), -len(str(p))),
        reverse=True,
    )
    return ranked[0] if parseable_score(ranked[0]) > 0 else None


def load_spatialedit_metrics(root: Path, requested_mode: Optional[str]) -> Optional[dict]:
    path = find_spatialedit_csv(root, requested_mode)
    if path is None:
        return None

    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics = normalize_metrics(row)
            angle = parse_angle(row)
            if metrics and angle is not None:
                rows.append({
                    "angle": angle,
                    "object_id": parse_object_id(row),
                    "metrics": metrics,
                })
    if not rows:
        return None

    overall = {}
    for metric in VERDICT_METRICS:
        values = [row["metrics"][metric] for row in rows if metric in row["metrics"]]
        if values:
            overall[metric] = float(mean(values))

    per_angle = {}
    for angle in sorted({row["angle"] for row in rows}):
        angle_rows = [row["metrics"] for row in rows if row["angle"] == angle]
        per_angle[str(angle)] = {}
        for metric in VERDICT_METRICS:
            values = [metrics[metric] for metrics in angle_rows if metric in metrics]
            if values:
                per_angle[str(angle)][metric] = float(mean(values))

    object_ids = sorted({row["object_id"] for row in rows if row["object_id"]})
    per_object = {}
    for object_id in object_ids:
        object_rows = [row["metrics"] for row in rows if row["object_id"] == object_id]
        per_object[object_id] = {}
        for metric in VERDICT_METRICS:
            values = [metrics[metric] for metrics in object_rows if metric in metrics]
            if values:
                per_object[object_id][metric] = float(mean(values))

    per_object_angle = {}
    for object_id in object_ids:
        per_object_angle[object_id] = {}
        for angle in sorted({row["angle"] for row in rows if row["object_id"] == object_id}):
            object_angle_rows = [
                row["metrics"] for row in rows
                if row["object_id"] == object_id and row["angle"] == angle
            ]
            per_object_angle[object_id][str(angle)] = {}
            for metric in VERDICT_METRICS:
                values = [metrics[metric] for metrics in object_angle_rows if metric in metrics]
                if values:
                    per_object_angle[object_id][str(angle)][metric] = float(mean(values))

    return {
        "source": "spatialedit",
        "mode": requested_mode or path.name.replace("_metrics.csv", ""),
        "path": str(path),
        "overall": overall,
        "per_angle": per_angle,
        "per_object": per_object,
        "per_object_angle": per_object_angle,
        "n_pairs": len(rows),
    }


def load_eval_sources(root: Path, requested_mode: Optional[str] = None) -> dict:
    root = root.resolve()
    sources = {}
    testset = find_testset_summary(root, requested_mode)
    if testset:
        sources["testset"] = testset
    spatialedit = load_spatialedit_metrics(root, requested_mode)
    if spatialedit:
        sources["spatialedit"] = spatialedit
    return sources


def metric_delta(metric: str, current: float, baseline: float) -> float:
    if metric in LOWER_IS_BETTER:
        return baseline - current
    return current - baseline


def compare_metric_maps(current_map: dict, baseline_map: dict) -> dict:
    delta = {}
    for key in sorted(set(current_map) & set(baseline_map)):
        delta[key] = {}
        for metric in sorted(set(current_map[key]) & set(baseline_map[key])):
            delta[key][metric] = metric_delta(metric, current_map[key][metric], baseline_map[key][metric])
    return delta


def compare_nested_metric_maps(current_map: dict, baseline_map: dict) -> dict:
    delta = {}
    for outer_key in sorted(set(current_map) & set(baseline_map)):
        inner = compare_metric_maps(current_map[outer_key], baseline_map[outer_key])
        if inner:
            delta[outer_key] = inner
    return delta


def percentile(values: list[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    index = min(max(int(round((len(values) - 1) * fraction)), 0), len(values) - 1)
    return values[index]


def analyze_object_failures(baseline: dict, weak_angles: list[str]) -> dict:
    per_object = baseline.get("per_object") or {}
    per_object_angle = baseline.get("per_object_angle") or {}
    object_scores = {
        obj_id: metrics["psnr"]
        for obj_id, metrics in per_object.items()
        if "psnr" in metrics
    }
    low_cutoff = percentile(list(object_scores.values()), 0.2)
    all_angle_bad_objects = []
    if low_cutoff is not None:
        for obj_id, score in sorted(object_scores.items(), key=lambda item: item[1]):
            angle_count = len([
                angle for angle, metrics in (per_object_angle.get(obj_id) or {}).items()
                if "psnr" in metrics and int(angle) != 360
            ])
            if score <= low_cutoff and angle_count >= 3:
                all_angle_bad_objects.append({"object_id": obj_id, "baseline_psnr": score, "angle_count": angle_count})

    object_angle_outliers = {}
    for angle in sorted({angle for angles in per_object_angle.values() for angle in angles}, key=lambda x: int(x)):
        angle_values = []
        for obj_id, angles in per_object_angle.items():
            metrics = angles.get(angle) or {}
            if "psnr" in metrics:
                angle_values.append((obj_id, metrics["psnr"]))
        if len(angle_values) < 3:
            continue
        values = [value for _, value in angle_values]
        threshold = mean(values) - (pstdev(values) if len(values) > 1 else 0.0)
        outliers = [
            {"object_id": obj_id, "baseline_psnr": value}
            for obj_id, value in sorted(angle_values, key=lambda item: item[1])
            if value <= threshold
        ]
        if outliers:
            object_angle_outliers[angle] = outliers

    systematic_weak_angles = []
    object_specific_weak_angles = []
    for angle in weak_angles:
        object_count = len([
            obj_id for obj_id, angles in per_object_angle.items()
            if "psnr" in (angles.get(angle) or {})
        ])
        outlier_count = len(object_angle_outliers.get(angle, []))
        if object_count == 0 or outlier_count / max(object_count, 1) >= 0.4:
            systematic_weak_angles.append(angle)
        else:
            object_specific_weak_angles.append(angle)

    return {
        "all_angle_bad_objects": all_angle_bad_objects,
        "object_angle_outliers": object_angle_outliers,
        "systematic_weak_angles": systematic_weak_angles,
        "object_specific_weak_angles": object_specific_weak_angles,
    }


def strong_angle_regressions(
    per_angle_delta: dict,
    strong_angles: list[str],
    missing_current_angles: list[str],
) -> list[dict]:
    regressions = []
    for angle in missing_current_angles:
        regressions.append({
            "angle": angle,
            "metric": "__missing_angle__",
            "delta": None,
            "threshold": None,
        })
    for angle in strong_angles:
        for metric, threshold in STRONG_ANGLE_REGRESSION_THRESHOLDS.items():
            value = (per_angle_delta.get(angle) or {}).get(metric)
            if value is not None and value < threshold:
                regressions.append({
                    "angle": angle,
                    "metric": metric,
                    "delta": value,
                    "threshold": threshold,
                })
    return regressions


def compare_source(current: dict, baseline: dict) -> dict:
    overall_delta = {}
    for metric in sorted(set(current["overall"]) & set(baseline["overall"])):
        overall_delta[metric] = metric_delta(metric, current["overall"][metric], baseline["overall"][metric])

    per_angle_delta = {}
    current_angles = set(current["per_angle"])
    baseline_angles = set(baseline["per_angle"])
    common_angles = sorted(current_angles & baseline_angles, key=lambda x: int(x))
    for angle in common_angles:
        per_angle_delta[angle] = {}
        for metric in sorted(set(current["per_angle"][angle]) & set(baseline["per_angle"][angle])):
            per_angle_delta[angle][metric] = metric_delta(
                metric,
                current["per_angle"][angle][metric],
                baseline["per_angle"][angle][metric],
            )

    weak_candidates = [
        (angle, metrics["psnr"])
        for angle, metrics in baseline["per_angle"].items()
        if "psnr" in metrics and int(angle) != 360
    ]
    weak_angles = [angle for angle, _ in sorted(weak_candidates, key=lambda item: item[1])[:3]]
    weak_values = [
        per_angle_delta[angle]["psnr"]
        for angle in weak_angles
        if angle in per_angle_delta and "psnr" in per_angle_delta[angle]
    ]
    weak_delta = float(mean(weak_values)) if weak_values else None
    strong_angles = [
        angle for angle in sorted(baseline_angles, key=lambda x: int(x))
        if angle not in weak_angles and int(angle) != 360
    ]
    missing_current_strong_angles = [angle for angle in strong_angles if angle not in current_angles]
    regressions = strong_angle_regressions(per_angle_delta, strong_angles, missing_current_strong_angles)
    object_analysis = analyze_object_failures(baseline, weak_angles)
    per_object_delta = compare_metric_maps(
        current.get("per_object") or {},
        baseline.get("per_object") or {},
    )
    per_object_angle_delta = compare_nested_metric_maps(
        current.get("per_object_angle") or {},
        baseline.get("per_object_angle") or {},
    )

    has_degradation = any(
        overall_delta.get(metric, 0.0) < DEGRADE_THRESHOLDS[metric]
        for metric in VERDICT_METRICS
        if metric in overall_delta
    ) or bool(regressions)
    has_improvement = any(
        overall_delta.get(metric, 0.0) > IMPROVE_THRESHOLDS[metric]
        for metric in VERDICT_METRICS
        if metric in overall_delta
    ) or (weak_delta is not None and weak_delta > 0.1)

    if has_degradation and has_improvement:
        verdict = "inspect"
    elif has_degradation:
        verdict = "stop_or_revert"
    elif has_improvement:
        verdict = "continue"
    else:
        verdict = "no_signal"

    return {
        "source": current["source"],
        "current_path": current["path"],
        "baseline_path": baseline["path"],
        "current_mode": current["mode"],
        "baseline_mode": baseline["mode"],
        "verdict": verdict,
        "overall_delta": overall_delta,
        "per_angle_delta": per_angle_delta,
        "per_object_delta": per_object_delta,
        "per_object_angle_delta": per_object_angle_delta,
        "weak_angles": weak_angles,
        "strong_angles": strong_angles,
        "missing_current_strong_angles": missing_current_strong_angles,
        "weak_delta": weak_delta,
        "strong_angle_regressions": regressions,
        **object_analysis,
    }


def combine_verdict(source_reports: dict) -> str:
    verdicts = [report["verdict"] for report in source_reports.values()]
    if "stop_or_revert" in verdicts:
        return "stop_or_revert"
    if "inspect" in verdicts:
        return "inspect"
    if "continue" in verdicts:
        return "continue"
    return "no_signal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare feedback-loop eval results")
    parser.add_argument("--current", required=True, help="Current round eval directory")
    parser.add_argument("--baseline", required=True, help="Baseline eval directory")
    parser.add_argument("--output", required=True, help="Output compare_report.json")
    parser.add_argument("--current-mode", default=None)
    parser.add_argument("--baseline-mode", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    current_sources = load_eval_sources(Path(args.current), args.current_mode)
    baseline_sources = load_eval_sources(Path(args.baseline), args.baseline_mode)
    common = sorted(set(current_sources) & set(baseline_sources))
    if not common:
        print("No comparable Test Set or SpatialEdit metrics found.", file=sys.stderr)
        return 2

    source_reports = {
        source: compare_source(current_sources[source], baseline_sources[source])
        for source in common
    }
    report = {
        "verdict": combine_verdict(source_reports),
        "sources": source_reports,
        "current_dir": str(Path(args.current).resolve()),
        "baseline_dir": str(Path(args.baseline).resolve()),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
