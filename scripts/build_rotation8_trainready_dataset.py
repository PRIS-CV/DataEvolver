"""
Build a train-ready rotation editing dataset from a consistent rotation8 render set.

Expected source dataset structure:
- summary.json
- manifest.json
- objects/obj_xxx/yaw000.png ... yaw315.png
- objects/obj_xxx/yaw000_mask.png ...
- objects/obj_xxx/yaw000_render_metadata.json ...
- objects/obj_xxx/yaw000_control.json ...

This script keeps the rendered view set and additionally creates front-view-to-target
editing pairs suitable for downstream image-edit training.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional


VIEW_NAMES = {
    0: "front view",
    45: "front-right view",
    90: "right side view",
    135: "back-right view",
    180: "back view",
    225: "back-left view",
    270: "left side view",
    315: "front-left view",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build train-ready rotation8 dataset")
    parser.add_argument("--source-root", required=True, help="Consistent rotation8 dataset root")
    parser.add_argument("--output-dir", required=True, help="Train-ready dataset output directory")
    parser.add_argument(
        "--source-rotation-deg",
        type=int,
        default=0,
        help="Source view rotation for editing pairs (default: 0/front view)",
    )
    parser.add_argument(
        "--target-rotations",
        default="45,90,135,180,225,270,315",
        help="Comma-separated target rotations for editing pairs",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="How to materialize files into output dataset",
    )
    parser.add_argument(
        "--prompts-json",
        default=None,
        help="Path to stage1 prompts.json containing T2I prompts per object",
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


def _copy_or_link(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)
    return str(dst)


def _rel(path: Optional[Path], base: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _rot_slug(rotation_deg: int) -> str:
    return f"yaw{rotation_deg:03d}"


def _view_name(rotation_deg: int) -> str:
    return VIEW_NAMES.get(rotation_deg, f"{rotation_deg} degree view")


def build_dataset(
    source_root: Path,
    output_root: Path,
    source_rotation_deg: int,
    target_rotations: List[int],
    copy_mode: str,
    prompts_json: Optional[Path] = None,
) -> dict:
    source_manifest = load_json(source_root / "manifest.json")
    if not source_manifest:
        raise FileNotFoundError(f"manifest.json not found in {source_root}")

    objects_payload = source_manifest.get("objects") or {}
    if not objects_payload:
        raise ValueError("No objects found in source manifest")

    prompt_lookup: Dict[str, dict] = {}
    if prompts_json and prompts_json.exists():
        prompts_data = load_json(prompts_json, default=[])
        for entry in prompts_data:
            prompt_lookup[entry["id"]] = entry

    output_root.mkdir(parents=True, exist_ok=True)
    views_root = output_root / "views"
    pairs_root = output_root / "pairs"
    objects_root = output_root / "objects"
    views_root.mkdir(parents=True, exist_ok=True)
    pairs_root.mkdir(parents=True, exist_ok=True)
    objects_root.mkdir(parents=True, exist_ok=True)

    source_rot_slug = _rot_slug(source_rotation_deg)
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    pair_rows: List[dict] = []
    object_manifests: Dict[str, dict] = {}
    rendered_view_count = 0
    rendered_mask_count = 0

    for obj_id in sorted(objects_payload.keys()):
        source_obj_manifest = load_json(source_root / "objects" / obj_id / "object_manifest.json", default={}) or {}
        rotations_list = source_obj_manifest.get("rotations") or []
        rotation_map = {int(item["rotation_deg"]): item for item in rotations_list}

        required_rotations = [source_rotation_deg] + target_rotations
        missing = [rot for rot in required_rotations if rot not in rotation_map]
        if missing:
            raise ValueError(f"{obj_id} missing required rotations: {missing}")

        obj_out_dir = views_root / obj_id
        obj_out_dir.mkdir(parents=True, exist_ok=True)

        out_rotations = {}
        for rotation_deg in sorted(rotation_map.keys()):
            record = rotation_map[rotation_deg]
            rot_slug = _rot_slug(rotation_deg)

            rgb_src = source_root / record["rgb_path"]
            mask_src = source_root / record["mask_path"]
            render_meta_src = source_root / record["render_metadata_path"]
            control_src = source_root / record["control_state_path"]

            rgb_out = Path(_copy_or_link(rgb_src, obj_out_dir / f"{rot_slug}.png", copy_mode))
            mask_out = Path(_copy_or_link(mask_src, obj_out_dir / f"{rot_slug}_mask.png", copy_mode))
            render_meta_out = Path(_copy_or_link(render_meta_src, obj_out_dir / f"{rot_slug}_render_metadata.json", copy_mode))
            control_out = Path(_copy_or_link(control_src, obj_out_dir / f"{rot_slug}_control.json", copy_mode))

            rendered_view_count += 1
            rendered_mask_count += 1

            out_rotations[rot_slug] = {
                "rotation_deg": rotation_deg,
                "view_name": _view_name(rotation_deg),
                "rgb_path": _rel(rgb_out, output_root),
                "mask_path": _rel(mask_out, output_root),
                "render_metadata_path": _rel(render_meta_out, output_root),
                "control_state_path": _rel(control_out, output_root),
                "base_pair_name": record.get("pair_name"),
                "base_best_round_idx": record.get("base_best_round_idx"),
                "base_hybrid_score": record.get("base_hybrid_score"),
            }

        source_record = out_rotations[source_rot_slug]
        prompt_entry = prompt_lookup.get(obj_id, {})
        obj_name = prompt_entry.get("name", "object").replace("_", " ")
        t2i_prompt = prompt_entry.get("prompt", "")

        pair_entries = []
        for target_rotation_deg in target_rotations:
            target_slug = _rot_slug(target_rotation_deg)
            target_record = out_rotations[target_slug]
            pair_id = f"{obj_id}_{source_rot_slug}_to_{target_slug}"
            instruction = f"Rotate this {obj_name} clockwise from front view to {_view_name(target_rotation_deg)}."
            pair_payload = {
                "pair_id": pair_id,
                "obj_id": obj_id,
                "task_type": "rotation_edit",
                "split": "train",
                "prompt_version": "v3",
                "source_rotation_deg": source_rotation_deg,
                "target_rotation_deg": target_rotation_deg,
                "source_view_name": _view_name(source_rotation_deg),
                "target_view_name": _view_name(target_rotation_deg),
                "instruction": instruction,
                "object_description": obj_name,
                "t2i_prompt": t2i_prompt,
                "source_image": source_record["rgb_path"],
                "target_image": target_record["rgb_path"],
                "source_mask": source_record["mask_path"],
                "target_mask": target_record["mask_path"],
                "source_render_metadata": source_record["render_metadata_path"],
                "target_render_metadata": target_record["render_metadata_path"],
                "source_control_state": source_record["control_state_path"],
                "target_control_state": target_record["control_state_path"],
            }
            pair_entries.append(pair_payload)
            pair_rows.append(pair_payload)

        object_manifest = {
            "obj_id": obj_id,
            "source_rotation_deg": source_rotation_deg,
            "source_view_name": _view_name(source_rotation_deg),
            "target_rotations": target_rotations,
            "views": out_rotations,
            "training_pairs": pair_entries,
        }
        save_json(objects_root / obj_id / "object_manifest.json", object_manifest)
        object_manifests[obj_id] = object_manifest

    pair_rows = sorted(pair_rows, key=lambda item: item["pair_id"])
    jsonl_path = pairs_root / "train_pairs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in pair_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = pairs_root / "train_pairs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_id",
                "obj_id",
                "task_type",
                "split",
                "prompt_version",
                "source_rotation_deg",
                "target_rotation_deg",
                "source_view_name",
                "target_view_name",
                "instruction",
                "object_description",
                "t2i_prompt",
                "source_image",
                "target_image",
                "source_mask",
                "target_mask",
                "source_render_metadata",
                "target_render_metadata",
                "source_control_state",
                "target_control_state",
            ],
        )
        writer.writeheader()
        writer.writerows(pair_rows)

    summary = {
        "success": True,
        "created_at": created_at,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "dataset_type": "rotation_edit_trainready",
        "source_rotation_deg": source_rotation_deg,
        "source_view_name": _view_name(source_rotation_deg),
        "target_rotations": target_rotations,
        "target_view_names": [_view_name(rot) for rot in target_rotations],
        "total_objects": len(object_manifests),
        "rendered_views": rendered_view_count,
        "rendered_masks": rendered_mask_count,
        "training_pairs": len(pair_rows),
        "pairs_per_object": len(target_rotations),
        "copy_mode": copy_mode,
    }
    save_json(output_root / "summary.json", summary)

    manifest = {
        "summary": summary,
        "objects": object_manifests,
        "pairs": pair_rows,
    }
    save_json(output_root / "manifest.json", manifest)

    return summary


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_dir).resolve()
    target_rotations = [int(item) for item in args.target_rotations.split(",") if item.strip()]

    summary = build_dataset(
        source_root=source_root,
        output_root=output_root,
        source_rotation_deg=args.source_rotation_deg,
        target_rotations=target_rotations,
        copy_mode=args.copy_mode,
        prompts_json=Path(args.prompts_json) if args.prompts_json else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
