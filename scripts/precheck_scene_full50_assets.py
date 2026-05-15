"""
Precheck the obj_021..obj_050 expansion assets and seed concept metadata.

This script supports two validation stages:
- metadata precheck: validates IDs and seed concept metadata only
- mesh precheck: additionally requires mesh assets to exist

It never modifies any existing dataset roots.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_MESHES_DIR = REPO_ROOT / "pipeline" / "data" / "meshes"
DEFAULT_SUBSET_OBJECTS = REPO_ROOT / "configs" / "seed_concepts" / "scene_obj021_050_objects.json"
DEFAULT_FULL_OBJECTS = REPO_ROOT / "configs" / "seed_concepts" / "scene_full50_objects.json"
PLACEHOLDER_PREFIX = "TODO__"


def parse_args():
    parser = argparse.ArgumentParser(description="Precheck full50 scene expansion assets")
    parser.add_argument("--meshes-dir", default=str(DEFAULT_MESHES_DIR))
    parser.add_argument("--objects-file", default=str(DEFAULT_SUBSET_OBJECTS))
    parser.add_argument("--full-objects-file", default=str(DEFAULT_FULL_OBJECTS))
    parser.add_argument("--start-idx", type=int, default=21)
    parser.add_argument("--end-idx", type=int, default=50)
    parser.add_argument("--require-meshes", action="store_true")
    parser.add_argument("--report-path", default=None)
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


def expected_ids(start_idx: int, end_idx: int) -> List[str]:
    return [f"obj_{idx:03d}" for idx in range(start_idx, end_idx + 1)]


def find_asset(meshes_dir: Path, obj_id: str) -> Optional[Path]:
    for suffix in (".glb", ".blend"):
        candidate = meshes_dir / f"{obj_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def metadata_is_placeholder(value: Optional[str]) -> bool:
    if value is None:
        return True
    normalized = str(value).strip()
    return (not normalized) or normalized.startswith(PLACEHOLDER_PREFIX)


def summarize_object(item: dict, meshes_dir: Path) -> dict:
    obj_id = str(item.get("id", "")).strip()
    name = item.get("name")
    category = item.get("category")
    asset_path = find_asset(meshes_dir, obj_id) if obj_id else None
    return {
        "id": obj_id,
        "name": name,
        "category": category,
        "asset_path": str(asset_path) if asset_path else None,
        "asset_exists": bool(asset_path),
        "metadata_complete": not metadata_is_placeholder(name) and not metadata_is_placeholder(category),
    }


def validate_object_range(items: List[dict], start_idx: int, end_idx: int) -> dict:
    expected = expected_ids(start_idx, end_idx)
    provided_ids = [str(item.get("id", "")).strip() for item in items]
    duplicates = sorted({obj_id for obj_id in provided_ids if provided_ids.count(obj_id) > 1})
    missing = sorted(set(expected) - set(provided_ids))
    unexpected = sorted(set(provided_ids) - set(expected))
    ordered_match = provided_ids == expected
    return {
        "expected_ids": expected,
        "provided_ids": provided_ids,
        "duplicates": duplicates,
        "missing_ids": missing,
        "unexpected_ids": unexpected,
        "ordered_match": ordered_match,
        "count_ok": len(items) == len(expected),
    }


def validate_full_objects(full_items: List[dict], start_idx: int, end_idx: int) -> dict:
    expected_total = end_idx
    expected_prefix = expected_ids(1, start_idx - 1)
    expected_suffix = expected_ids(start_idx, end_idx)
    provided_ids = [str(item.get("id", "")).strip() for item in full_items]
    return {
        "count_ok": len(full_items) == expected_total,
        "prefix_ok": provided_ids[: len(expected_prefix)] == expected_prefix,
        "suffix_ok": provided_ids[len(expected_prefix) : len(expected_prefix) + len(expected_suffix)] == expected_suffix,
        "provided_ids_head": provided_ids[:10],
        "provided_ids_tail": provided_ids[-10:],
    }


def main():
    args = parse_args()
    meshes_dir = Path(args.meshes_dir).resolve()
    objects_file = Path(args.objects_file).resolve()
    full_objects_file = Path(args.full_objects_file).resolve()

    subset_items = load_json(objects_file, default=None)
    full_items = load_json(full_objects_file, default=None)
    if not isinstance(subset_items, list):
        raise SystemExit(f"Invalid objects file: {objects_file}")
    if not isinstance(full_items, list):
        raise SystemExit(f"Invalid full objects file: {full_objects_file}")

    subset_validation = validate_object_range(subset_items, args.start_idx, args.end_idx)
    full_validation = validate_full_objects(full_items, args.start_idx, args.end_idx)
    object_rows = [summarize_object(item, meshes_dir) for item in subset_items]

    missing_assets = [row["id"] for row in object_rows if not row["asset_exists"]]
    incomplete_metadata = [row["id"] for row in object_rows if not row["metadata_complete"]]

    success = all(
        [
            subset_validation["count_ok"],
            subset_validation["ordered_match"],
            not subset_validation["duplicates"],
            not subset_validation["missing_ids"],
            not subset_validation["unexpected_ids"],
            full_validation["count_ok"],
            full_validation["prefix_ok"],
            full_validation["suffix_ok"],
            not incomplete_metadata,
            (not args.require_meshes) or (meshes_dir.exists() and not missing_assets),
        ]
    )

    report = {
        "success": success,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meshes_dir": str(meshes_dir),
        "objects_file": str(objects_file),
        "full_objects_file": str(full_objects_file),
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "require_meshes": bool(args.require_meshes),
        "subset_validation": subset_validation,
        "full_validation": full_validation,
        "missing_assets": missing_assets,
        "incomplete_metadata": incomplete_metadata,
        "objects": object_rows,
    }

    if args.report_path:
        save_json(Path(args.report_path).resolve(), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
