#!/usr/bin/env python3
"""
Apply external benchmark feedback to a train-ready rotation dataset.

This is the second half of the outer loop:
    augmentation_requirements.json -> next training dataset root

The script does not regenerate images by itself. Instead, it creates a derived
training dataset root that oversamples pairs matching weak benchmark angles and
records the hard-sample plan. This makes the output immediately usable by
training code that reads pairs/all_pairs.csv or pairs/*.jsonl.

Typical input requirements are produced by scripts/build_external_feedback_loop.py.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_ANGLE_ORDER = [0, 45, 90, 135, 180, 225, 270, 315]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an external-feedback augmented dataset root")
    parser.add_argument("--source-dataset", required=True, help="Existing train-ready dataset root")
    parser.add_argument("--requirements", required=True, help="augmentation_requirements.json")
    parser.add_argument("--weak-samples", default=None, help="weak_samples.csv from external feedback analysis")
    parser.add_argument("--output-dir", required=True, help="New dataset root to create")
    parser.add_argument("--oversample-factor", type=int, default=2,
                        help="Total copies for weak-angle pairs in train split. 1 means no duplication.")
    parser.add_argument("--include-val-test", action="store_true",
                        help="Also duplicate weak-angle val/test rows. Default only duplicates train rows.")
    parser.add_argument("--angle-order", default="0,45,90,135,180,225,270,315",
                        help="Mapping from benchmark angle index 00..07 to rotation degrees")
    parser.add_argument("--selection-mode", choices=["target", "source_or_target"], default="target",
                        help="Whether priority angles match target rotation only or either source/target")
    parser.add_argument("--asset-mode", choices=["symlink", "copy", "none"], default="symlink",
                        help="How to materialize views/objects links in output root")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_angle_order(raw: str) -> list[int]:
    values = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not values:
        values = list(DEFAULT_ANGLE_ORDER)
    return values


def priority_angle_degrees(requirements: dict, angle_order: list[int]) -> list[int]:
    out = []
    for raw in requirements.get("priority_angles") or []:
        text = str(raw).strip().lower().replace("angle", "")
        if not text:
            continue
        try:
            idx = int(text)
        except ValueError:
            continue
        if 0 <= idx < len(angle_order):
            deg = angle_order[idx]
        else:
            deg = idx
        if deg not in out:
            out.append(deg)
    return out


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fieldnames:
        raise ValueError(f"CSV has no header: {path}")
    return fieldnames, rows


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_int(value: object) -> Optional[int]:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def is_priority_pair(row: dict, degrees: set[int], mode: str) -> bool:
    target = as_int(row.get("target_rotation_deg"))
    source = as_int(row.get("source_rotation_deg"))
    if mode == "target":
        return target in degrees
    return target in degrees or source in degrees


def stable_augmented_id(row: dict, copy_idx: int) -> str:
    base = str(row.get("pair_id") or row.get("id") or "pair").strip()
    return f"{base}__externalfb_dup{copy_idx:02d}"


def augment_rows(
    rows: list[dict],
    *,
    priority_degrees: set[int],
    oversample_factor: int,
    include_val_test: bool,
    selection_mode: str,
) -> tuple[list[dict], list[dict]]:
    if oversample_factor < 1:
        raise ValueError("--oversample-factor must be >= 1")

    augmented = [dict(row) for row in rows]
    feedback_rows = []
    extra_copies = max(0, oversample_factor - 1)
    for row in rows:
        split = str(row.get("split") or "").strip().lower()
        if split != "train" and not include_val_test:
            continue
        if not is_priority_pair(row, priority_degrees, selection_mode):
            continue
        for copy_idx in range(1, extra_copies + 1):
            dup = dict(row)
            if "pair_id" in dup:
                dup["pair_id"] = stable_augmented_id(row, copy_idx)
            dup["external_feedback_augmented"] = "1"
            dup["external_feedback_reason"] = "priority_angle_oversample"
            dup["external_feedback_copy_index"] = str(copy_idx)
            augmented.append(dup)
            feedback_rows.append(dup)
    return augmented, feedback_rows


def ensure_feedback_fields(fieldnames: list[str]) -> list[str]:
    out = list(fieldnames)
    for name in [
        "external_feedback_augmented",
        "external_feedback_reason",
        "external_feedback_copy_index",
    ]:
        if name not in out:
            out.append(name)
    return out


def split_rows(rows: list[dict]) -> dict[str, list[dict]]:
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        split = str(row.get("split") or "").strip().lower()
        if split in splits:
            splits[split].append(row)
    return splits


def materialize_asset(output_root: Path, source_root: Path, name: str, mode: str) -> Optional[str]:
    src = source_root / name
    dst = output_root / name
    if not src.exists() and not src.is_symlink():
        return None
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if mode == "none":
        return None
    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve(), target_is_directory=True)
            return "symlink"
        except OSError:
            mode = "copy"
    if mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)
        return "copy"
    return None


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def load_weak_samples(path: Optional[str]) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    _, rows = read_csv(p)
    return rows


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_dataset).resolve()
    output_root = Path(args.output_dir).resolve()
    requirements_path = Path(args.requirements).resolve()
    weak_samples_path = Path(args.weak_samples).resolve() if args.weak_samples else None

    if not source_root.exists():
        raise FileNotFoundError(f"source dataset not found: {source_root}")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output dir already exists: {output_root}; pass --overwrite to replace")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    requirements = load_json(requirements_path)
    angle_order = parse_angle_order(args.angle_order)
    priority_degrees = priority_angle_degrees(requirements, angle_order)
    if not priority_degrees:
        raise ValueError("No priority angle degrees resolved from requirements")

    pairs_dir = source_root / "pairs"
    all_pairs_csv = pairs_dir / "all_pairs.csv"
    fieldnames, rows = read_csv(all_pairs_csv)
    out_fieldnames = ensure_feedback_fields(fieldnames)
    augmented_rows, feedback_rows = augment_rows(
        rows,
        priority_degrees=set(priority_degrees),
        oversample_factor=args.oversample_factor,
        include_val_test=args.include_val_test,
        selection_mode=args.selection_mode,
    )

    out_pairs_dir = output_root / "pairs"
    write_csv(out_pairs_dir / "all_pairs.csv", out_fieldnames, augmented_rows)
    write_jsonl(out_pairs_dir / "all_pairs.jsonl", augmented_rows)
    write_csv(out_pairs_dir / "external_feedback_pairs.csv", out_fieldnames, feedback_rows)
    write_jsonl(out_pairs_dir / "external_feedback_pairs.jsonl", feedback_rows)

    for split, split_group in split_rows(augmented_rows).items():
        write_csv(out_pairs_dir / f"{split}_pairs.csv", out_fieldnames, split_group)
        write_jsonl(out_pairs_dir / f"{split}_pairs.jsonl", split_group)

    asset_modes = {}
    for asset_name in ["views", "objects"]:
        mode = materialize_asset(output_root, source_root, asset_name, args.asset_mode)
        if mode:
            asset_modes[asset_name] = mode

    for name in ["object_splits.json", "object_splits.csv", "manifest.json"]:
        copy_if_exists(source_root / name, output_root / name)

    weak_samples = load_weak_samples(str(weak_samples_path) if weak_samples_path else None)
    original_split_counts = {k: len(v) for k, v in split_rows(rows).items()}
    augmented_split_counts = {k: len(v) for k, v in split_rows(augmented_rows).items()}
    selected_original_rows = [
        row for row in rows
        if (args.include_val_test or str(row.get("split", "")).lower() == "train")
        and is_priority_pair(row, set(priority_degrees), args.selection_mode)
    ]

    summary = {
        "success": True,
        "created_at": now(),
        "dataset_type": "rotation_edit_trainready_external_feedback_augmented",
        "source_dataset": str(source_root),
        "output_root": str(output_root),
        "requirements_path": str(requirements_path),
        "weak_samples_path": str(weak_samples_path) if weak_samples_path else None,
        "priority_angle_indices": requirements.get("priority_angles") or [],
        "angle_order": angle_order,
        "priority_rotation_degrees": priority_degrees,
        "selection_mode": args.selection_mode,
        "oversample_factor": args.oversample_factor,
        "include_val_test": bool(args.include_val_test),
        "asset_modes": asset_modes,
        "original_pair_counts": {
            "total": len(rows),
            **original_split_counts,
        },
        "selected_priority_pair_count": len(selected_original_rows),
        "added_feedback_pair_count": len(feedback_rows),
        "augmented_pair_counts": {
            "total": len(augmented_rows),
            **augmented_split_counts,
        },
        "requirements": requirements,
        "weak_sample_count": len(weak_samples),
        "weak_sample_preview": weak_samples[:20],
    }

    save_json(output_root / "external_feedback_manifest.json", summary)
    save_json(output_root / "summary.json", summary)
    save_json(output_root / "augmentation_requirements.json", requirements)
    if weak_samples_path and weak_samples_path.exists():
        copy_if_exists(weak_samples_path, output_root / "weak_samples.csv")

    print(f"[external-feedback-apply] source: {source_root}")
    print(f"[external-feedback-apply] output: {output_root}")
    print(f"[external-feedback-apply] priority_rotation_degrees: {priority_degrees}")
    print(f"[external-feedback-apply] selected_priority_pair_count: {len(selected_original_rows)}")
    print(f"[external-feedback-apply] added_feedback_pair_count: {len(feedback_rows)}")
    print(f"[external-feedback-apply] augmented_total_pairs: {len(augmented_rows)}")
    print(f"[external-feedback-apply] manifest: {output_root / 'external_feedback_manifest.json'}")


if __name__ == "__main__":
    main()
