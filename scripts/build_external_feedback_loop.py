#!/usr/bin/env python3
"""
Build an external feedback plan from training/benchmark metrics.

This script is the bridge for the outer loop:
    train/bench results -> dataset construction requirements -> more weak-case data

It reads SpatialEdit-style metrics CSVs from a directory or zip archive, compares
the current run against an optional baseline, finds weak samples/angles, and
writes:
    - feedback_summary.json
    - augmentation_requirements.json
    - weak_samples.csv
    - DATASET_FEEDBACK_PLAN.md

The output requirements JSON is intentionally simple so it can be consumed by
dataset synthesis/orchestration scripts or by a human operator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional


METRICS_HIGHER_IS_BETTER = {
    "psnr": True,
    "ssim": True,
    "lpips": False,
    "dino_similarity": True,
    "clip_similarity": True,
}


WEAKNESS_RULES = {
    "semantic_instruction_coverage": {
        "stage_route": "stage1_stage2",
        "signals": ["low_clip_similarity"],
        "missing_data": "more semantic/instruction-aligned rotation edits",
    },
    "geometry_view_consistency": {
        "stage_route": "stage3_stage4",
        "signals": ["low_dino_similarity"],
        "missing_data": "more geometry-consistent hard-angle views",
    },
    "appearance_fidelity": {
        "stage_route": "stage2_stage4",
        "signals": ["high_lpips", "low_ssim"],
        "missing_data": "more appearance-preserving source-target pairs",
    },
}


@dataclass
class MetricRow:
    image_name: str
    psnr: Optional[float]
    ssim: Optional[float]
    lpips: Optional[float]
    dino_similarity: Optional[float]
    clip_similarity: Optional[float]
    angle: str
    sample_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build external feedback requirements from eval metrics")
    parser.add_argument("--eval-zip", default=None, help="Zip archive containing metrics CSV files")
    parser.add_argument("--eval-dir", default=None, help="Directory containing metrics CSV files")
    parser.add_argument("--current", default="ours_objinfo_metrics.csv", help="Current metrics CSV filename")
    parser.add_argument("--baseline", default="base_metrics.csv", help="Baseline metrics CSV filename")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=40, help="Number of weak samples to report")
    parser.add_argument("--angle-bottom-k", type=int, default=3, help="Number of weak angles to prioritize")
    parser.add_argument("--clip-threshold", type=float, default=0.82)
    parser.add_argument("--dino-threshold", type=float, default=0.82)
    parser.add_argument("--lpips-threshold", type=float, default=0.40)
    parser.add_argument("--ssim-threshold", type=float, default=0.65)
    parser.add_argument("--underperform-margin", type=float, default=0.01)
    parser.add_argument("--keep-extracted", action="store_true", help="Keep extracted zip contents in output dir")
    return parser.parse_args()


def extract_zip(eval_zip: Path, output_dir: Path, keep: bool) -> Path:
    if keep:
        extract_root = output_dir / "_extracted_eval"
    else:
        extract_root = output_dir / "_extract_work"
    if extract_root.exists():
        for child in extract_root.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(eval_zip, "r") as zf:
        zf.extractall(extract_root)
    return extract_root


def find_file(root: Path, name: str) -> Path:
    direct = root / name
    if direct.exists():
        return direct
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    return matches[0]


def clean_key(key: Optional[str]) -> str:
    return (key or "").replace("\ufeff", "").strip()


def parse_angle(image_name: str) -> str:
    match = re.search(r"_angle(\d+)", image_name)
    return match.group(1) if match else "NA"


def parse_sample_key(image_name: str) -> str:
    return re.sub(r"_angle\d+(?=\.[^.]+$)", "", image_name)


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        value_f = float(text)
    except ValueError:
        return None
    if math.isnan(value_f):
        return None
    return value_f


def read_metrics_csv(path: Path) -> list[MetricRow]:
    rows: list[MetricRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            r = {clean_key(k): v for k, v in raw.items()}
            image_name = str(r.get("image_name") or "").strip()
            if not image_name:
                continue
            rows.append(
                MetricRow(
                    image_name=image_name,
                    psnr=to_float(r.get("psnr")),
                    ssim=to_float(r.get("ssim")),
                    lpips=to_float(r.get("lpips")),
                    dino_similarity=to_float(r.get("dino_similarity")),
                    clip_similarity=to_float(r.get("clip_similarity")),
                    angle=parse_angle(image_name),
                    sample_key=parse_sample_key(image_name),
                )
            )
    if not rows:
        raise ValueError(f"No metric rows found in {path}")
    return rows


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def rounded(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def row_score(row: MetricRow, thresholds: dict[str, float]) -> tuple[float, list[str], list[str]]:
    penalties: list[float] = []
    signals: list[str] = []
    weak_types: list[str] = []

    if row.clip_similarity is not None and row.clip_similarity < thresholds["clip_similarity"]:
        penalties.append(thresholds["clip_similarity"] - row.clip_similarity)
        signals.append("low_clip_similarity")
        weak_types.append("semantic_instruction_coverage")
    if row.dino_similarity is not None and row.dino_similarity < thresholds["dino_similarity"]:
        penalties.append(thresholds["dino_similarity"] - row.dino_similarity)
        signals.append("low_dino_similarity")
        weak_types.append("geometry_view_consistency")
    if row.lpips is not None and row.lpips > thresholds["lpips"]:
        penalties.append(row.lpips - thresholds["lpips"])
        signals.append("high_lpips")
        weak_types.append("appearance_fidelity")
    if row.ssim is not None and row.ssim < thresholds["ssim"]:
        penalties.append(thresholds["ssim"] - row.ssim)
        signals.append("low_ssim")
        weak_types.append("appearance_fidelity")

    return round(sum(penalties), 6), sorted(set(signals)), sorted(set(weak_types))


def metric_delta(current: Optional[float], baseline: Optional[float], metric: str) -> Optional[float]:
    if current is None or baseline is None:
        return None
    delta = current - baseline
    return delta if METRICS_HIGHER_IS_BETTER[metric] else -delta


def summarize_by_angle(rows: list[MetricRow]) -> dict[str, dict]:
    grouped: dict[str, list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault(row.angle, []).append(row)
    out = {}
    for angle, group in sorted(grouped.items()):
        out[angle] = {
            "count": len(group),
            "psnr": rounded(mean(r.psnr for r in group)),
            "ssim": rounded(mean(r.ssim for r in group)),
            "lpips": rounded(mean(r.lpips for r in group)),
            "dino_similarity": rounded(mean(r.dino_similarity for r in group)),
            "clip_similarity": rounded(mean(r.clip_similarity for r in group)),
        }
    return out


def angle_weakness_value(stats: dict) -> float:
    clip = stats.get("clip_similarity")
    dino = stats.get("dino_similarity")
    lpips = stats.get("lpips")
    ssim = stats.get("ssim")
    value = 0.0
    if clip is not None:
        value += max(0.0, 0.90 - clip)
    if dino is not None:
        value += max(0.0, 0.89 - dino)
    if lpips is not None:
        value += max(0.0, lpips - 0.25)
    if ssim is not None:
        value += max(0.0, 0.73 - ssim)
    return round(value, 6)


def build_feedback(
    current_rows: list[MetricRow],
    baseline_rows: Optional[list[MetricRow]],
    args: argparse.Namespace,
) -> dict:
    thresholds = {
        "clip_similarity": args.clip_threshold,
        "dino_similarity": args.dino_threshold,
        "lpips": args.lpips_threshold,
        "ssim": args.ssim_threshold,
    }
    baseline_by_name = {r.image_name: r for r in baseline_rows or []}

    weak_records = []
    weakness_counts: dict[str, int] = {}
    signal_counts: dict[str, int] = {}
    angle_counts: dict[str, int] = {}
    underperform_records = []

    for row in current_rows:
        score, signals, weak_types = row_score(row, thresholds)
        baseline = baseline_by_name.get(row.image_name)
        underperforming_metrics = []
        if baseline:
            for metric in METRICS_HIGHER_IS_BETTER:
                delta = metric_delta(getattr(row, metric), getattr(baseline, metric), metric)
                if delta is not None and delta < -args.underperform_margin:
                    underperforming_metrics.append(metric)

        if score > 0 or underperforming_metrics:
            for weak_type in weak_types:
                weakness_counts[weak_type] = weakness_counts.get(weak_type, 0) + 1
            for signal in signals:
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
            angle_counts[row.angle] = angle_counts.get(row.angle, 0) + 1
            record = {
                "image_name": row.image_name,
                "sample_key": row.sample_key,
                "angle": row.angle,
                "weakness_score": rounded(score, 6),
                "signals": signals,
                "weakness_types": weak_types,
                "underperforming_metrics": underperforming_metrics,
                "psnr": rounded(row.psnr),
                "ssim": rounded(row.ssim),
                "lpips": rounded(row.lpips),
                "dino_similarity": rounded(row.dino_similarity),
                "clip_similarity": rounded(row.clip_similarity),
            }
            weak_records.append(record)
            if underperforming_metrics:
                underperform_records.append(record)

    weak_records.sort(
        key=lambda r: (
            float(r["weakness_score"] or 0.0),
            len(r["underperforming_metrics"]),
        ),
        reverse=True,
    )

    angle_summary = summarize_by_angle(current_rows)
    weak_angles = sorted(
        [
            {
                "angle": angle,
                "weakness_value": angle_weakness_value(stats),
                **stats,
                "weak_record_count": angle_counts.get(angle, 0),
            }
            for angle, stats in angle_summary.items()
        ],
        key=lambda r: (r["weakness_value"], r["weak_record_count"]),
        reverse=True,
    )

    prioritized_angles = [r["angle"] for r in weak_angles[: args.angle_bottom_k] if r["angle"] != "NA"]
    top_weak = weak_records[: args.top_k]
    prioritized_samples = sorted({r["sample_key"] for r in top_weak})[: args.top_k]

    missing_data = []
    for weak_type, count in sorted(weakness_counts.items(), key=lambda item: item[1], reverse=True):
        rule = WEAKNESS_RULES.get(weak_type)
        if rule:
            missing_data.append(rule["missing_data"])
    if prioritized_angles:
        missing_data.append("more rotation data for weak angles: " + ", ".join(f"angle{a}" for a in prioritized_angles))
    missing_data = list(dict.fromkeys(missing_data))

    stage_routes = {}
    for weak_type, count in weakness_counts.items():
        rule = WEAKNESS_RULES.get(weak_type)
        if not rule:
            continue
        route = rule["stage_route"]
        stage_routes[route] = stage_routes.get(route, 0) + count

    return {
        "source": {
            "current_rows": len(current_rows),
            "baseline_rows": len(baseline_rows or []),
            "current_file": args.current,
            "baseline_file": args.baseline if baseline_rows else None,
        },
        "thresholds": thresholds,
        "metric_means": {
            metric: rounded(mean(getattr(r, metric) for r in current_rows))
            for metric in METRICS_HIGHER_IS_BETTER
        },
        "angle_summary": angle_summary,
        "weak_angles": weak_angles,
        "weakness_counts": weakness_counts,
        "signal_counts": signal_counts,
        "stage_route_counts": stage_routes,
        "weak_sample_count": len(weak_records),
        "underperform_sample_count": len(underperform_records),
        "top_weak_samples": top_weak,
        "augmentation_requirements": {
            "missing_data": missing_data,
            "priority_angles": prioritized_angles,
            "priority_samples": prioritized_samples,
            "weakness_counts": weakness_counts,
            "stage_route_counts": stage_routes,
            "recommended_loop": "bench_metrics -> dataset_construction -> train/bench",
            "notes": [
                "Use priority_angles to oversample rotation/yaw cases in the next dataset build.",
                "Use priority_samples as hard examples for targeted reconstruction or prompt repair.",
                "Low CLIP suggests Stage1/Stage2 prompt/image coverage; low DINO suggests Stage3/Stage4 geometry/view coverage; high LPIPS or low SSIM suggests appearance-preserving pairs.",
            ],
        },
    }


def write_weak_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "image_name",
        "sample_key",
        "angle",
        "weakness_score",
        "signals",
        "weakness_types",
        "underperforming_metrics",
        "psnr",
        "ssim",
        "lpips",
        "dino_similarity",
        "clip_similarity",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["signals"] = ";".join(out.get("signals") or [])
            out["weakness_types"] = ";".join(out.get("weakness_types") or [])
            out["underperforming_metrics"] = ";".join(out.get("underperforming_metrics") or [])
            writer.writerow({k: out.get(k) for k in fieldnames})


def build_markdown(feedback: dict) -> str:
    req = feedback["augmentation_requirements"]
    lines = [
        "# External Feedback Plan",
        "",
        "## Summary",
        f"- Current rows: `{feedback['source']['current_rows']}`",
        f"- Weak samples: `{feedback['weak_sample_count']}`",
        f"- Underperforming samples vs baseline: `{feedback['underperform_sample_count']}`",
        f"- Recommended loop: `{req['recommended_loop']}`",
        "",
        "## Metric Means",
    ]
    for metric, value in feedback["metric_means"].items():
        lines.append(f"- `{metric}`: `{value}`")

    lines.extend(["", "## Priority Angles"])
    for item in feedback["weak_angles"][:8]:
        lines.append(
            "- angle `{angle}`: weakness `{weakness_value}`, clip `{clip_similarity}`, "
            "dino `{dino_similarity}`, lpips `{lpips}`, weak records `{weak_record_count}`".format(**item)
        )

    lines.extend(["", "## Weakness Counts"])
    for key, value in sorted(feedback["weakness_counts"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Stage Routes"])
    for key, value in sorted(feedback["stage_route_counts"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Dataset Requirements"])
    for item in req["missing_data"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Top Weak Samples"])
    for row in feedback["top_weak_samples"][:20]:
        lines.append(
            "- `{image}` angle `{angle}` score `{score}` signals `{signals}`".format(
                image=row["image_name"],
                angle=row["angle"],
                score=row["weakness_score"],
                signals=",".join(row.get("signals") or row.get("underperforming_metrics") or []),
            )
        )

    lines.extend(["", "## Notes"])
    for note in req["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_zip:
        root = extract_zip(Path(args.eval_zip), output_dir, args.keep_extracted)
    elif args.eval_dir:
        root = Path(args.eval_dir)
    else:
        raise SystemExit("Provide --eval-zip or --eval-dir")

    current_path = find_file(root, args.current)
    baseline_path = find_file(root, args.baseline) if args.baseline else None
    current_rows = read_metrics_csv(current_path)
    baseline_rows = read_metrics_csv(baseline_path) if baseline_path and baseline_path.exists() else None

    feedback = build_feedback(current_rows, baseline_rows, args)

    summary_path = output_dir / "feedback_summary.json"
    requirements_path = output_dir / "augmentation_requirements.json"
    weak_csv_path = output_dir / "weak_samples.csv"
    report_path = output_dir / "DATASET_FEEDBACK_PLAN.md"

    summary_path.write_text(json.dumps(feedback, indent=2, ensure_ascii=False), encoding="utf-8")
    requirements_path.write_text(
        json.dumps(feedback["augmentation_requirements"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_weak_csv(weak_csv_path, feedback["top_weak_samples"])
    report_path.write_text(build_markdown(feedback), encoding="utf-8")

    print(f"[external-feedback] current: {current_path}")
    print(f"[external-feedback] baseline: {baseline_path}")
    print(f"[external-feedback] summary: {summary_path}")
    print(f"[external-feedback] requirements: {requirements_path}")
    print(f"[external-feedback] weak samples: {weak_csv_path}")
    print(f"[external-feedback] report: {report_path}")
    print(f"[external-feedback] weak_sample_count={feedback['weak_sample_count']}")


if __name__ == "__main__":
    main()
