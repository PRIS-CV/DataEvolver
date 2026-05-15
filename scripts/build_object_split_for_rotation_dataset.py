"""
Build object-disjoint train/val/test splits for a train-ready rotation dataset.

The source dataset is expected to contain:
- summary.json
- manifest.json
- views/
- objects/
- pairs/train_pairs.csv
- pairs/train_pairs.jsonl

The output dataset:
- reuses the existing views directory via symlink by default
- writes object split assignment files
- writes split-specific JSONL/CSV pair files
- writes a new manifest/summary with split metadata
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Build object-disjoint splits for rotation dataset")
    parser.add_argument("--source-root", required=True, help="Train-ready dataset root")
    parser.add_argument("--output-dir", required=True, help="Split dataset output directory")
    parser.add_argument("--train-objects", type=int, default=14)
    parser.add_argument("--val-objects", type=int, default=3)
    parser.add_argument("--test-objects", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--asset-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to materialize shared asset directories (views/objects)",
    )
    return parser.parse_args()


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def ensure_dir_link_or_copy(src: Path, dst: Path, mode: str):
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    if mode == "symlink":
        dst.symlink_to(src, target_is_directory=True)
    else:
        shutil.copytree(src, dst)


def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_splits(
    source_root: Path,
    output_root: Path,
    train_objects: int,
    val_objects: int,
    test_objects: int,
    seed: int,
    asset_mode: str,
) -> dict:
    manifest = load_json(source_root / "manifest.json")
    if not manifest:
        raise FileNotFoundError(f"manifest.json not found in {source_root}")

    summary = manifest.get("summary") or {}
    objects = manifest.get("objects") or {}
    pairs = manifest.get("pairs") or []
    obj_ids = sorted(objects.keys())

    total_requested = train_objects + val_objects + test_objects
    if total_requested != len(obj_ids):
        raise ValueError(
            f"Requested split sizes {train_objects}+{val_objects}+{test_objects}={total_requested}, "
            f"but source dataset has {len(obj_ids)} objects"
        )

    rng = random.Random(seed)
    shuffled = obj_ids[:]
    rng.shuffle(shuffled)

    train_ids = sorted(shuffled[:train_objects])
    val_ids = sorted(shuffled[train_objects : train_objects + val_objects])
    test_ids = sorted(shuffled[train_objects + val_objects :])

    obj_to_split = {obj_id: "train" for obj_id in train_ids}
    obj_to_split.update({obj_id: "val" for obj_id in val_ids})
    obj_to_split.update({obj_id: "test" for obj_id in test_ids})

    output_root.mkdir(parents=True, exist_ok=True)
    ensure_dir_link_or_copy(source_root / "views", output_root / "views", asset_mode)
    ensure_dir_link_or_copy(source_root / "objects", output_root / "objects", asset_mode)

    split_rows: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    updated_pairs: List[dict] = []
    for pair in pairs:
        row = dict(pair)
        split = obj_to_split[row["obj_id"]]
        row["split"] = split
        split_rows[split].append(row)
        updated_pairs.append(row)

    write_jsonl(output_root / "pairs" / "train_pairs.jsonl", split_rows["train"])
    write_jsonl(output_root / "pairs" / "val_pairs.jsonl", split_rows["val"])
    write_jsonl(output_root / "pairs" / "test_pairs.jsonl", split_rows["test"])

    write_csv(output_root / "pairs" / "train_pairs.csv", split_rows["train"])
    write_csv(output_root / "pairs" / "val_pairs.csv", split_rows["val"])
    write_csv(output_root / "pairs" / "test_pairs.csv", split_rows["test"])
    write_jsonl(output_root / "pairs" / "all_pairs.jsonl", updated_pairs)
    write_csv(output_root / "pairs" / "all_pairs.csv", updated_pairs)

    object_split_rows = []
    object_manifest = {}
    for obj_id in obj_ids:
        split = obj_to_split[obj_id]
        row = {"obj_id": obj_id, "split": split}
        object_split_rows.append(row)
        object_manifest[obj_id] = {
            "split": split,
            "source_object_manifest": f"objects/{obj_id}/object_manifest.json",
        }

    write_csv(output_root / "object_splits.csv", object_split_rows)
    save_json(
        output_root / "object_splits.json",
        {
            "seed": seed,
            "train_objects": train_ids,
            "val_objects": val_ids,
            "test_objects": test_ids,
        },
    )

    split_summary = {
        "success": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_type": "rotation_edit_trainready_split",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "asset_mode": asset_mode,
        "seed": seed,
        "source_dataset_summary": summary,
        "object_counts": {
            "total": len(obj_ids),
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "pair_counts": {
            "total": len(updated_pairs),
            "train": len(split_rows["train"]),
            "val": len(split_rows["val"]),
            "test": len(split_rows["test"]),
        },
    }
    save_json(output_root / "summary.json", split_summary)

    output_manifest = {
        "summary": split_summary,
        "object_splits": object_manifest,
        "pairs": updated_pairs,
    }
    save_json(output_root / "manifest.json", output_manifest)
    return split_summary


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_dir).resolve()
    summary = build_splits(
        source_root=source_root,
        output_root=output_root,
        train_objects=args.train_objects,
        val_objects=args.val_objects,
        test_objects=args.test_objects,
        seed=args.seed,
        asset_mode=args.asset_mode,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
