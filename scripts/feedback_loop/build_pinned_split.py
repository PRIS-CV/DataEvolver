#!/usr/bin/env python3
"""Freeze object-disjoint split assignments for augmented rotation datasets."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional


SPLITS = ("train", "val", "test")


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
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


def extract_obj_id(row: dict) -> str:
    if row.get("obj_id"):
        return str(row["obj_id"])
    match = re.search(r"(obj_\d+)", str(row.get("pair_id", "")))
    return match.group(1) if match else ""


def load_from_object_splits_json(path: Path) -> Dict[str, str]:
    payload = read_json(path)
    if not payload:
        return {}

    assignments: Dict[str, str] = {}
    for split in SPLITS:
        for obj_id in payload.get(f"{split}_objects", []) or []:
            assignments[str(obj_id)] = split
    if assignments:
        return assignments

    for obj_id, item in (payload.get("object_splits") or {}).items():
        split = item.get("split") if isinstance(item, dict) else item
        if split in SPLITS:
            assignments[str(obj_id)] = str(split)
    return assignments


def load_from_object_splits_csv(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    assignments: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj_id = row.get("obj_id") or row.get("object_id")
            split = row.get("split")
            if obj_id and split in SPLITS:
                assignments[str(obj_id)] = str(split)
    return assignments


def find_pairs_jsonl(dataset_root: Path, split: str) -> Optional[Path]:
    candidates = [
        dataset_root / "pairs" / f"{split}_pairs.jsonl",
        dataset_root / f"{split}_pairs.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_from_pair_files(dataset_root: Path) -> Dict[str, str]:
    assignments: Dict[str, str] = {}
    conflicts: Dict[str, List[str]] = {}
    for split in SPLITS:
        path = find_pairs_jsonl(dataset_root, split)
        if not path:
            continue
        for row in read_jsonl(path):
            obj_id = extract_obj_id(row)
            if not obj_id:
                continue
            previous = assignments.get(obj_id)
            if previous and previous != split:
                conflicts.setdefault(obj_id, [previous]).append(split)
            assignments[obj_id] = split
    if conflicts:
        raise ValueError(f"Object split conflicts in pair files: {conflicts}")
    return assignments


def freeze_split(dataset_root: Path, output_path: Path, source_note: str = "") -> dict:
    dataset_root = dataset_root.resolve()
    assignments = (
        load_from_object_splits_json(dataset_root / "object_splits.json")
        or load_from_object_splits_csv(dataset_root / "object_splits.csv")
        or load_from_pair_files(dataset_root)
    )
    if not assignments:
        raise FileNotFoundError(f"No split assignments found in {dataset_root}")

    by_split = {
        split: sorted(obj for obj, obj_split in assignments.items() if obj_split == split)
        for split in SPLITS
    }
    payload = {
        "schema_version": "pinned_split_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_dataset_root": str(dataset_root),
        "source_note": source_note,
        "object_to_split": {obj_id: assignments[obj_id] for obj_id in sorted(assignments)},
        "train_objects": by_split["train"],
        "val_objects": by_split["val"],
        "test_objects": by_split["test"],
        "object_counts": {split: len(by_split[split]) for split in SPLITS},
        "pinned_object_count": len(assignments),
    }
    write_json(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create pinned_split.json from an existing split dataset")
    parser.add_argument("--dataset-root", required=True, help="Existing split dataset root")
    parser.add_argument("--output", required=True, help="Output pinned_split.json")
    parser.add_argument("--source-note", default="", help="Optional human note for lineage")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = freeze_split(Path(args.dataset_root), Path(args.output), args.source_note)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
