#!/usr/bin/env python3
"""Validate a train-ready rotation dataset before feedback-loop training."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REQUIRED_SPLITS = ("train", "val", "test")
REQUIRED_FIELDS = ("instruction", "target_rotation_deg", "source_image", "target_image")
DEFAULT_EXPECTED_COUNTS = {"train": 245, "val": 49, "test": 56}
MAX_REPORTED_ERRORS = 200


class DatasetValidationError(Exception):
    """Raised when a dataset has blocking validation errors."""


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def find_pairs_jsonl(dataset_root: Path, split: str) -> Optional[Path]:
    candidates = [
        dataset_root / "pairs" / f"{split}_pairs.jsonl",
        dataset_root / f"{split}_pairs.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_dataset_path(dataset_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return dataset_root / path


def uses_bbox_view(value: str) -> bool:
    normalized = value.replace("\\", "/")
    return "bbox_views/" in normalized or normalized == "bbox_views"


def extract_obj_id(row: dict) -> str:
    if row.get("obj_id"):
        return str(row["obj_id"])
    pair_id = str(row.get("pair_id", ""))
    match = re.search(r"(obj_\d+)", pair_id)
    if match:
        return match.group(1)
    return ""


def load_expected_counts(dataset_root: Path, expected_manifest: Optional[Path], cli_expected: Dict[str, Optional[int]]) -> Dict[str, int]:
    explicit = {split: value for split, value in cli_expected.items() if value is not None}
    if explicit:
        return {**DEFAULT_EXPECTED_COUNTS, **explicit}

    manifest_candidates = []
    if expected_manifest:
        manifest_candidates.append(expected_manifest)
    manifest_candidates.extend([
        dataset_root / "augmented_manifest.json",
        dataset_root / "summary.json",
    ])
    for path in manifest_candidates:
        payload = read_json(path)
        if not payload:
            continue
        counts = payload.get("expected_pair_counts") or payload.get("pair_counts")
        if isinstance(counts, dict) and all(split in counts for split in REQUIRED_SPLITS):
            return {split: int(counts[split]) for split in REQUIRED_SPLITS}
    return dict(DEFAULT_EXPECTED_COUNTS)


def read_json(path: Path) -> Optional[dict]:
    if not path or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pinned_split(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    payload = read_json(path) or {}
    mapping = payload.get("object_to_split") or {}
    if mapping:
        return {str(obj_id): str(split) for obj_id, split in mapping.items()}
    out = {}
    for split in REQUIRED_SPLITS:
        for obj_id in payload.get(f"{split}_objects", []) or []:
            out[str(obj_id)] = split
    return out


def add_limited_error(errors: List[str], message: str, omitted: Dict[str, int]) -> None:
    if len(errors) < MAX_REPORTED_ERRORS:
        errors.append(message)
    else:
        omitted["errors"] += 1


def validate_dataset(
    dataset_root: Path,
    expected_counts: Dict[str, int],
    pinned_split_path: Optional[Path] = None,
    splits: Iterable[str] = REQUIRED_SPLITS,
) -> dict:
    dataset_root = dataset_root.resolve()
    errors: List[str] = []
    warnings: List[str] = []
    omitted = {"errors": 0}
    split_rows: Dict[str, List[dict]] = {}
    split_paths: Dict[str, str] = {}

    if not dataset_root.exists():
        errors.append(f"Dataset root does not exist: {dataset_root}")
    elif not dataset_root.is_dir():
        errors.append(f"Dataset root is not a directory: {dataset_root}")

    for split in splits:
        jsonl = find_pairs_jsonl(dataset_root, split)
        if jsonl is None:
            add_limited_error(errors, f"Missing {split} pairs JSONL", omitted)
            split_rows[split] = []
            continue
        split_paths[split] = str(jsonl)
        try:
            split_rows[split] = read_jsonl(jsonl)
        except DatasetValidationError as exc:
            add_limited_error(errors, str(exc), omitted)
            split_rows[split] = []

    for split, rows in split_rows.items():
        for idx, row in enumerate(rows, start=1):
            pair_id = row.get("pair_id", f"{split}:{idx}")
            row_split = row.get("split")
            if row_split and row_split != split:
                warnings.append(f"{pair_id}: row split is {row_split!r}, file split is {split!r}")

            for field in REQUIRED_FIELDS:
                if field not in row or row[field] in (None, ""):
                    add_limited_error(errors, f"{pair_id}: missing field {field!r}", omitted)

            for field in ("source_image", "target_image"):
                value = str(row.get(field, ""))
                if not value:
                    continue
                if uses_bbox_view(value):
                    add_limited_error(errors, f"{pair_id}: {field} points to bbox_views/: {value}", omitted)
                path = resolve_dataset_path(dataset_root, value)
                if not path.exists():
                    add_limited_error(errors, f"{pair_id}: missing {field}: {path}", omitted)

    split_objects: Dict[str, set] = {
        split: {obj_id for obj_id in (extract_obj_id(row) for row in rows) if obj_id}
        for split, rows in split_rows.items()
    }
    pinned_split = load_pinned_split(pinned_split_path)
    pinned_checked = 0
    pinned_present = set()
    for split, rows in split_rows.items():
        for row in rows:
            obj_id = extract_obj_id(row)
            if not obj_id or obj_id not in pinned_split:
                continue
            pinned_present.add(obj_id)
            pinned_checked += 1
            expected_split = pinned_split[obj_id]
            if expected_split != split:
                add_limited_error(
                    errors,
                    f"Pinned split violation for {obj_id}: found in {split}, expected {expected_split}",
                    omitted,
                )
    for obj_id, expected_split in sorted(pinned_split.items()):
        if obj_id not in pinned_present:
            add_limited_error(
                errors,
                f"Pinned object missing from dataset: {obj_id} expected in {expected_split}",
                omitted,
            )

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        leak = sorted(split_objects.get(left, set()) & split_objects.get(right, set()))
        if leak:
            add_limited_error(errors, f"Object leak {left} intersection {right}: {leak}", omitted)

    counts = {split: len(rows) for split, rows in split_rows.items()}
    for split, expected in expected_counts.items():
        actual = counts.get(split)
        if actual is not None and actual != expected:
            warnings.append(
                f"{split}: {actual} pairs (expected {expected}); "
                "if intentional, document the reason in INTERVENTION_PLAN.md"
            )

    return {
        "status": "failed" if errors else "passed",
        "dataset_root": str(dataset_root),
        "split_paths": split_paths,
        "pair_counts": counts,
        "expected_pair_counts": expected_counts,
        "object_counts": {split: len(objs) for split, objs in split_objects.items()},
        "pinned_split_path": str(pinned_split_path) if pinned_split_path else None,
        "pinned_objects": len(pinned_split),
        "pinned_rows_checked": pinned_checked,
        "warnings": warnings,
        "errors": errors,
        "omitted": omitted,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a rotation feedback-loop dataset")
    parser.add_argument("dataset_root", help="Dataset root to validate")
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    parser.add_argument("--expected-manifest", default=None, help="augmented_manifest.json with expected_pair_counts")
    parser.add_argument("--pinned-split-path", default=None, help="pinned_split.json to verify old objects were not moved")
    parser.add_argument("--expected-train", type=int, default=None)
    parser.add_argument("--expected-val", type=int, default=None)
    parser.add_argument("--expected-test", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    expected = load_expected_counts(
        dataset_root,
        Path(args.expected_manifest) if args.expected_manifest else None,
        {"train": args.expected_train, "val": args.expected_val, "test": args.expected_test},
    )
    report = validate_dataset(
        dataset_root,
        expected_counts=expected,
        pinned_split_path=Path(args.pinned_split_path) if args.pinned_split_path else None,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
