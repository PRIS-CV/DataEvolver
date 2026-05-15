#!/usr/bin/env python3
"""Append new weak-angle train pairs while preserving a pinned baseline split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


SPLITS = ("train", "val", "test")


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fieldnames:
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_pairs_jsonl(dataset_root: Path, split: str) -> Optional[Path]:
    candidates = [
        dataset_root / "pairs" / f"{split}_pairs.jsonl",
        dataset_root / f"{split}_pairs.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_split_rows(dataset_root: Path) -> Dict[str, List[dict]]:
    split_rows = {}
    for split in SPLITS:
        path = find_pairs_jsonl(dataset_root, split)
        split_rows[split] = read_jsonl(path) if path else []
    return split_rows


def load_new_train_rows(dataset_root: Path) -> List[dict]:
    train_path = find_pairs_jsonl(dataset_root, "train")
    if train_path:
        return read_jsonl(train_path)
    all_path = dataset_root / "pairs" / "all_pairs.jsonl"
    if all_path.exists():
        return [row for row in read_jsonl(all_path) if row.get("split", "train") == "train"]
    manifest = read_json(dataset_root / "manifest.json", default={}) or {}
    return [dict(row) for row in manifest.get("pairs", [])]


def parse_rotations(raw: Optional[str]) -> Optional[set[int]]:
    if raw is None or raw == "":
        return None
    return {int(item.strip()) for item in str(raw).split(",") if item.strip()}


def parse_required_int(row: dict, field: str) -> int:
    value = row.get(field)
    if value in (None, ""):
        raise ValueError(f"{row.get('pair_id', '<unknown_pair>')}: missing required field {field!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{row.get('pair_id', '<unknown_pair>')}: invalid integer field {field!r}: {value!r}"
        ) from exc


def pinned_object_map(path: Path) -> Dict[str, str]:
    payload = read_json(path, default={}) or {}
    mapping = payload.get("object_to_split") or {}
    if mapping:
        return {str(obj_id): str(split) for obj_id, split in mapping.items()}
    out = {}
    for split in SPLITS:
        for obj_id in payload.get(f"{split}_objects", []) or []:
            out[str(obj_id)] = split
    return out


def materialize_object_dir(src_root: Path, dst_root: Path, obj_id: str, mode: str) -> None:
    for dirname in ("views", "objects"):
        src = src_root / dirname / obj_id
        dst = dst_root / dirname / obj_id
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            continue
        if mode == "symlink":
            dst.symlink_to(os.path.relpath(src, dst.parent), target_is_directory=True)
        else:
            shutil.copytree(src, dst)


def extract_objects(rows: Iterable[dict]) -> List[str]:
    return sorted({str(row.get("obj_id")) for row in rows if row.get("obj_id")})


def build_augmented_dataset(
    *,
    baseline_split_root: Path,
    new_trainready_root: Optional[Path],
    output_root: Path,
    pinned_split_path: Path,
    allowed_target_rotations: Optional[set[int]],
    asset_mode: str,
    retire_object_ids: Optional[set[str]] = None,
) -> dict:
    baseline_split_root = baseline_split_root.resolve()
    new_trainready_root = new_trainready_root.resolve()
    output_root = output_root.resolve()
    pinned = pinned_object_map(pinned_split_path)
    if not pinned:
        raise FileNotFoundError(f"No pinned object split found in {pinned_split_path}")

    baseline_rows = load_split_rows(baseline_split_root)
    for split, rows in baseline_rows.items():
        for row in rows:
            obj_id = str(row.get("obj_id"))
            expected = pinned.get(obj_id)
            if expected and expected != split:
                raise ValueError(f"Pinned object {obj_id} is in {split}, expected {expected}")

    retired_pairs_count = 0
    if retire_object_ids:
        before = len(baseline_rows["train"])
        baseline_rows["train"] = [
            row for row in baseline_rows["train"]
            if str(row.get("obj_id")) not in retire_object_ids
        ]
        retired_pairs_count = before - len(baseline_rows["train"])
        if retired_pairs_count:
            print(f"  [retire] Removed {retired_pairs_count} train pairs from "
                  f"{len(retire_object_ids)} retired objects: {sorted(retire_object_ids)}")
        val_retired = [row for row in baseline_rows["val"]
                       if str(row.get("obj_id")) in retire_object_ids]
        if val_retired:
            print(f"  [WARN] {len(val_retired)} val pairs belong to retired objects "
                  f"(kept to preserve eval baseline)")

    new_rows = load_new_train_rows(new_trainready_root) if new_trainready_root else []
    filtered_new_rows = []
    skipped_rows = []
    for row in new_rows:
        obj_id = str(row.get("obj_id"))
        if obj_id in pinned:
            raise ValueError(f"New train-ready root contains pinned baseline object {obj_id}")
        target_rot = parse_required_int(row, "target_rotation_deg")
        if allowed_target_rotations is not None and target_rot not in allowed_target_rotations:
            skipped_rows.append(row)
            continue
        updated = dict(row)
        updated["split"] = "train"
        filtered_new_rows.append(updated)

    if not filtered_new_rows and not retire_object_ids:
        raise ValueError("No new train rows remain after target-rotation filtering")

    output_root.mkdir(parents=True, exist_ok=True)
    for rows in baseline_rows.values():
        for obj_id in extract_objects(rows):
            materialize_object_dir(baseline_split_root, output_root, obj_id, asset_mode)
    for obj_id in extract_objects(filtered_new_rows):
        materialize_object_dir(new_trainready_root, output_root, obj_id, asset_mode)

    split_rows = {
        "train": [dict(row) for row in baseline_rows["train"]] + filtered_new_rows,
        "val": [dict(row) for row in baseline_rows["val"]],
        "test": [dict(row) for row in baseline_rows["test"]],
    }
    all_rows = split_rows["train"] + split_rows["val"] + split_rows["test"]
    seen_pair_ids = set()
    duplicate_pair_ids = set()
    for row in all_rows:
        pair_id = row.get("pair_id")
        if not pair_id:
            raise ValueError(f"{row.get('obj_id', '<unknown_obj>')}: missing required field 'pair_id'")
        if pair_id in seen_pair_ids:
            duplicate_pair_ids.add(str(pair_id))
        seen_pair_ids.add(str(pair_id))
    if duplicate_pair_ids:
        raise ValueError(f"Duplicate pair_id values in augmented dataset: {sorted(duplicate_pair_ids)}")

    for split in SPLITS:
        write_jsonl(output_root / "pairs" / f"{split}_pairs.jsonl", split_rows[split])
        write_csv(output_root / "pairs" / f"{split}_pairs.csv", split_rows[split])
    write_jsonl(output_root / "pairs" / "all_pairs.jsonl", all_rows)
    write_csv(output_root / "pairs" / "all_pairs.csv", all_rows)

    new_objects = extract_objects(filtered_new_rows)
    object_to_split = dict(pinned)
    for obj_id in new_objects:
        object_to_split[obj_id] = "train"

    object_split_rows = [{"obj_id": obj_id, "split": object_to_split[obj_id]} for obj_id in sorted(object_to_split)]
    write_csv(output_root / "object_splits.csv", object_split_rows)
    write_json(
        output_root / "object_splits.json",
        {
            "schema_version": "augmented_object_splits_v1",
            "pinned_split_path": str(pinned_split_path.resolve()),
            "train_objects": sorted(obj for obj, split in object_to_split.items() if split == "train"),
            "val_objects": sorted(obj for obj, split in object_to_split.items() if split == "val"),
            "test_objects": sorted(obj for obj, split in object_to_split.items() if split == "test"),
            "object_to_split": object_to_split,
        },
    )

    pair_counts = {split: len(rows) for split, rows in split_rows.items()}
    object_counts = {
        split: len({row["obj_id"] for row in rows if row.get("obj_id")})
        for split, rows in split_rows.items()
    }
    manifest = {
        "schema_version": "augmented_rotation_dataset_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_split_root": str(baseline_split_root),
        "new_trainready_root": str(new_trainready_root) if new_trainready_root else None,
        "output_root": str(output_root),
        "pinned_split_path": str(pinned_split_path.resolve()),
        "asset_mode": asset_mode,
        "new_object_ids": new_objects,
        "retired_object_ids": sorted(retire_object_ids) if retire_object_ids else [],
        "retired_pairs_removed": retired_pairs_count,
        "weak_target_rotations": sorted(allowed_target_rotations) if allowed_target_rotations else sorted(
            {int(row["target_rotation_deg"]) for row in filtered_new_rows}
        ),
        "expected_pair_counts": pair_counts,
        "pair_count_delta": {
            "train": len(filtered_new_rows),
            "val": 0,
            "test": 0,
            "total": len(filtered_new_rows),
        },
        "object_count_delta": len(new_objects),
        "skipped_new_pair_count": len(skipped_rows),
        "pair_counts": pair_counts,
        "object_counts": object_counts,
    }
    write_json(output_root / "augmented_manifest.json", manifest)
    write_json(output_root / "summary.json", {**manifest, "dataset_type": "rotation_edit_trainready_augmented"})
    write_json(output_root / "manifest.json", {"summary": manifest, "pairs": all_rows, "object_splits": object_to_split})
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build augmented rotation train-ready dataset")
    parser.add_argument("--baseline-split-root", required=True)
    parser.add_argument("--new-trainready-root", default=None,
                        help="New train-ready root to merge (optional if only retiring)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pinned-split-path", required=True)
    parser.add_argument("--target-rotations", default=None, help="Optional comma-separated weak target rotations")
    parser.add_argument("--asset-mode", choices=["symlink", "copy"], default="copy")
    parser.add_argument("--retire-objects", default=None,
                        help="JSON file with list of obj_ids to remove from train split")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    retire_ids = None
    if args.retire_objects:
        retire_path = Path(args.retire_objects)
        if retire_path.exists():
            retire_data = read_json(retire_path, default=[])
            if isinstance(retire_data, list):
                retire_ids = set(str(x) for x in retire_data)
            elif isinstance(retire_data, dict):
                retire_ids = set(str(x) for x in retire_data.get("retired_objects", []))

    new_root = Path(args.new_trainready_root) if args.new_trainready_root else None
    if not new_root and not retire_ids:
        raise ValueError("At least one of --new-trainready-root or --retire-objects is required")

    manifest = build_augmented_dataset(
        baseline_split_root=Path(args.baseline_split_root),
        new_trainready_root=new_root,
        output_root=Path(args.output_dir),
        pinned_split_path=Path(args.pinned_split_path),
        allowed_target_rotations=parse_rotations(args.target_rotations),
        asset_mode=args.asset_mode,
        retire_object_ids=retire_ids,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
