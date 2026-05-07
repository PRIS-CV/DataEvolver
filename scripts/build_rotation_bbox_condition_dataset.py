"""
Build a bbox-conditioned rotation dataset from an existing train-ready/split root.

This script is read-only with respect to the source dataset root.
It derives per-view 2D bounding boxes from mask images, renders bbox overlays,
and writes a new parallel dataset root suitable for bbox-conditioned training.

Supported source roots:
- rotation_edit_trainready
- rotation_edit_trainready_split
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description="Build bbox-conditioned rotation dataset from masks")
    parser.add_argument("--source-root", required=True, help="Existing train-ready or split dataset root")
    parser.add_argument("--output-dir", required=True, help="New bbox-conditioned dataset root")
    parser.add_argument(
        "--asset-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to materialize shared views/objects assets into the new root",
    )
    parser.add_argument("--mask-threshold", type=int, default=127, help="Mask threshold in [0,255]")
    parser.add_argument("--bbox-line-width", type=int, default=4)
    parser.add_argument(
        "--bbox-color",
        default="255,0,0",
        help="BBox RGB color, e.g. 255,0,0",
    )
    parser.add_argument(
        "--overwrite-pair-image-fields",
        action="store_true",
        help="Replace source_image/target_image with bbox overlay image paths in pair rows",
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


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]):
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


def ensure_dir_link_or_copy(src: Path, dst: Path, mode: str):
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src, target_is_directory=True)
    else:
        shutil.copytree(src, dst)


def relpath(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def parse_color(raw: str) -> Tuple[int, int, int]:
    parts = [int(item.strip()) for item in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid bbox color: {raw}")
    return parts[0], parts[1], parts[2]


def compute_bbox_from_mask(mask_path: Path, threshold: int) -> dict:
    mask = np.array(Image.open(mask_path).convert("L"))
    fg = mask > threshold
    if not fg.any():
        raise ValueError(f"Foreground missing in mask: {mask_path}")

    ys, xs = np.where(fg)
    left = int(xs.min())
    right = int(xs.max())
    top = int(ys.min())
    bottom = int(ys.max())
    height, width = mask.shape[:2]
    bbox_w = right - left + 1
    bbox_h = bottom - top + 1

    return {
        "image_width": int(width),
        "image_height": int(height),
        "bbox_xyxy": [left, top, right, bottom],
        "bbox_xywh": [left, top, bbox_w, bbox_h],
        "bbox_xyxy_norm": [
            round(left / width, 6),
            round(top / height, 6),
            round((right + 1) / width, 6),
            round((bottom + 1) / height, 6),
        ],
        "bbox_xywh_norm": [
            round(left / width, 6),
            round(top / height, 6),
            round(bbox_w / width, 6),
            round(bbox_h / height, 6),
        ],
        "area_pixels": int(fg.sum()),
        "area_ratio": round(float(fg.mean()), 6),
        "mask_threshold": int(threshold),
    }


def render_bbox_overlay(image_path: Path, bbox_xyxy: List[int], out_path: Path, color: Tuple[int, int, int], line_width: int):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = bbox_xyxy
    for offset in range(max(1, line_width)):
        draw.rectangle(
            [left - offset, top - offset, right + offset, bottom + offset],
            outline=color,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def collect_pair_files(pairs_root: Path) -> List[Path]:
    preferred = [
        pairs_root / "train_pairs.jsonl",
        pairs_root / "val_pairs.jsonl",
        pairs_root / "test_pairs.jsonl",
        pairs_root / "all_pairs.jsonl",
    ]
    return [path for path in preferred if path.exists()]


def build_dataset(
    source_root: Path,
    output_root: Path,
    asset_mode: str,
    mask_threshold: int,
    bbox_line_width: int,
    bbox_color: Tuple[int, int, int],
    overwrite_pair_image_fields: bool,
) -> dict:
    summary = load_json(source_root / "summary.json", default={}) or {}
    manifest = load_json(source_root / "manifest.json", default={}) or {}
    if not summary or not manifest:
        raise FileNotFoundError(f"summary/manifest missing under {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    views_src = source_root / "views"
    objects_src = source_root / "objects"
    if not views_src.exists():
        raise FileNotFoundError(f"views dir missing in {source_root}")
    if objects_src.exists():
        ensure_dir_link_or_copy(objects_src, output_root / "objects", asset_mode)
    ensure_dir_link_or_copy(views_src, output_root / "views", asset_mode)

    pairs_src = source_root / "pairs"
    pairs_dst = output_root / "pairs"
    pairs_dst.mkdir(parents=True, exist_ok=True)

    if (source_root / "object_splits.json").exists():
        shutil.copy2(source_root / "object_splits.json", output_root / "object_splits.json")
    if (source_root / "object_splits.csv").exists():
        shutil.copy2(source_root / "object_splits.csv", output_root / "object_splits.csv")

    bbox_root = output_root / "bbox_annotations"
    bbox_image_root = output_root / "bbox_views"
    bbox_root.mkdir(parents=True, exist_ok=True)
    bbox_image_root.mkdir(parents=True, exist_ok=True)

    pair_files = collect_pair_files(pairs_src)
    if not pair_files:
        raise FileNotFoundError(f"No pair jsonl files found in {pairs_src}")

    cached_views: Dict[Tuple[str, str], dict] = {}
    all_updated_pairs: List[dict] = []

    def ensure_view_assets(image_rel: str, mask_rel: str) -> dict:
        cache_key = (image_rel, mask_rel)
        if cache_key in cached_views:
            return cached_views[cache_key]

        image_abs = source_root / image_rel
        mask_abs = source_root / mask_rel
        bbox_meta = compute_bbox_from_mask(mask_abs, mask_threshold)

        image_rel_path = Path(image_rel)
        obj_dir = image_rel_path.parent.name
        stem = image_rel_path.stem
        bbox_json_path = bbox_root / obj_dir / f"{stem}_bbox.json"
        bbox_overlay_path = bbox_image_root / obj_dir / f"{stem}.png"

        bbox_payload = {
            "source_image": image_rel,
            "source_mask": mask_rel,
            **bbox_meta,
        }
        save_json(bbox_json_path, bbox_payload)
        render_bbox_overlay(
            image_abs,
            bbox_meta["bbox_xyxy"],
            bbox_overlay_path,
            color=bbox_color,
            line_width=bbox_line_width,
        )

        record = {
            "bbox_json": relpath(bbox_json_path, output_root),
            "bbox_overlay_image": relpath(bbox_overlay_path, output_root),
            **bbox_meta,
        }
        cached_views[cache_key] = record
        return record

    for pair_file in pair_files:
        rows = load_jsonl(pair_file)
        updated_rows: List[dict] = []
        for row in rows:
            updated = dict(row)

            source_image_raw = str(row["source_image"])
            target_image_raw = str(row["target_image"])
            source_mask_rel = str(row["source_mask"])
            target_mask_rel = str(row["target_mask"])

            src_bbox = ensure_view_assets(source_image_raw, source_mask_rel)
            tgt_bbox = ensure_view_assets(target_image_raw, target_mask_rel)

            updated["source_image_raw"] = source_image_raw
            updated["target_image_raw"] = target_image_raw
            updated["source_image_bbox"] = src_bbox["bbox_overlay_image"]
            updated["target_image_bbox"] = tgt_bbox["bbox_overlay_image"]
            updated["source_bbox_json"] = src_bbox["bbox_json"]
            updated["target_bbox_json"] = tgt_bbox["bbox_json"]
            updated["source_bbox_xyxy"] = src_bbox["bbox_xyxy"]
            updated["target_bbox_xyxy"] = tgt_bbox["bbox_xyxy"]
            updated["source_bbox_xywh"] = src_bbox["bbox_xywh"]
            updated["target_bbox_xywh"] = tgt_bbox["bbox_xywh"]
            updated["source_bbox_xyxy_norm"] = src_bbox["bbox_xyxy_norm"]
            updated["target_bbox_xyxy_norm"] = tgt_bbox["bbox_xyxy_norm"]
            updated["source_bbox_xywh_norm"] = src_bbox["bbox_xywh_norm"]
            updated["target_bbox_xywh_norm"] = tgt_bbox["bbox_xywh_norm"]

            if overwrite_pair_image_fields:
                updated["source_image"] = src_bbox["bbox_overlay_image"]
                updated["target_image"] = tgt_bbox["bbox_overlay_image"]

            updated_rows.append(updated)
            all_updated_pairs.append(updated)

        dst_jsonl = pairs_dst / pair_file.name
        write_jsonl(dst_jsonl, updated_rows)

        dst_csv = pairs_dst / pair_file.name.replace(".jsonl", ".csv")
        write_csv(dst_csv, updated_rows)

    unique_pairs = {row["pair_id"]: row for row in all_updated_pairs}

    new_summary = {
        "success": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_type": f"{summary.get('dataset_type', 'rotation_edit')}_bbox_condition",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "asset_mode": asset_mode,
        "mask_threshold": mask_threshold,
        "bbox_line_width": bbox_line_width,
        "bbox_color_rgb": list(bbox_color),
        "overwrite_pair_image_fields": overwrite_pair_image_fields,
        "source_summary": summary,
        "bbox_view_count": len(cached_views),
        "pair_count": len(unique_pairs),
    }
    save_json(output_root / "summary.json", new_summary)

    serialized_bbox_views = {
        f"{image_rel}|{mask_rel}": payload for (image_rel, mask_rel), payload in cached_views.items()
    }

    new_manifest = {
        "summary": new_summary,
        "source_manifest": manifest,
        "pairs": list(unique_pairs.values()),
        "bbox_views": serialized_bbox_views,
    }
    save_json(output_root / "manifest.json", new_manifest)
    return new_summary


def main():
    args = parse_args()
    summary = build_dataset(
        source_root=Path(args.source_root).resolve(),
        output_root=Path(args.output_dir).resolve(),
        asset_mode=args.asset_mode,
        mask_threshold=args.mask_threshold,
        bbox_line_width=args.bbox_line_width,
        bbox_color=parse_color(args.bbox_color),
        overwrite_pair_image_fields=args.overwrite_pair_image_fields,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
