"""
Build a brighter parallel rotation dataset by brightening only the masked object region.

This script is read-only with respect to the source dataset root. It creates a new
parallel dataset root, keeps the original dataset untouched, and writes brightened
images under `bright_views/`.

Supported source roots:
- rotation_edit_trainready
- rotation_edit_trainready_split
- bbox-conditioned variants built on top of the above
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def parse_args():
    parser = argparse.ArgumentParser(description="Build an object-brightened parallel rotation dataset")
    parser.add_argument("--source-root", required=True, help="Existing train-ready or split dataset root")
    parser.add_argument("--output-dir", required=True, help="New brightened dataset root")
    parser.add_argument(
        "--asset-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to materialize shared assets into the new root",
    )
    parser.add_argument("--mask-threshold", type=int, default=127, help="Mask threshold in [0,255]")
    parser.add_argument(
        "--mask-feather-radius",
        type=float,
        default=2.0,
        help="Gaussian blur radius applied to the binary mask before compositing",
    )
    parser.add_argument(
        "--target-object-value-mean",
        type=float,
        default=None,
        help="Target masked HSV-V mean in [0,1]. Required unless --reference-image/--reference-mask is provided.",
    )
    parser.add_argument("--reference-image", default=None, help="Reference image used to derive target object brightness")
    parser.add_argument("--reference-mask", default=None, help="Reference mask used to derive target object brightness")
    parser.add_argument("--min-gain", type=float, default=1.0, help="Minimum brightness multiplier")
    parser.add_argument("--max-gain", type=float, default=1.35, help="Maximum brightness multiplier")
    parser.add_argument(
        "--gain-scale",
        type=float,
        default=1.0,
        help="Extra multiplier applied on top of the raw target/current brightness ratio",
    )
    parser.add_argument(
        "--overwrite-pair-image-fields",
        action="store_true",
        help="Replace source_image/target_image with brightened image paths in pair rows",
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


def ensure_link_or_copy(src: Path, dst: Path, mode: str):
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src, target_is_directory=src.is_dir())
    elif src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def relpath(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def collect_pair_files(pairs_root: Path) -> List[Path]:
    preferred = [
        pairs_root / "train_pairs.jsonl",
        pairs_root / "val_pairs.jsonl",
        pairs_root / "test_pairs.jsonl",
        pairs_root / "all_pairs.jsonl",
    ]
    return [path for path in preferred if path.exists()]


def load_binary_mask(mask_path: Path, threshold: int) -> np.ndarray:
    mask = np.array(Image.open(mask_path).convert("L"))
    fg = mask > threshold
    if not fg.any():
        raise ValueError(f"Foreground missing in mask: {mask_path}")
    return fg


def compute_masked_value_mean(image: Image.Image, fg_mask: np.ndarray) -> float:
    hsv = np.array(image.convert("HSV"), dtype=np.float32) / 255.0
    value = hsv[..., 2]
    return float(value[fg_mask].mean())


def build_soft_alpha(mask_path: Path, threshold: int, feather_radius: float) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    binary = mask.point(lambda p: 255 if p > threshold else 0)
    if feather_radius > 0:
        binary = binary.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    alpha = np.array(binary, dtype=np.float32) / 255.0
    return alpha[..., None]


def normalize_bbox_xyxy(value) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        value = json.loads(value)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    return [int(round(float(v))) for v in value]


def render_bbox_overlay(
    image_path: Path,
    bbox_xyxy: List[int],
    out_path: Path,
    color: Tuple[int, int, int] = (255, 0, 0),
    line_width: int = 4,
):
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


def brighten_object_only(
    image_path: Path,
    mask_path: Path,
    out_path: Path,
    threshold: int,
    feather_radius: float,
    target_mean_value: float,
    min_gain: float,
    max_gain: float,
    gain_scale: float,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    fg_mask = load_binary_mask(mask_path, threshold)
    before_mean = compute_masked_value_mean(image, fg_mask)
    raw_gain = (target_mean_value / max(before_mean, 1e-6)) * gain_scale
    gain_applied = max(min_gain, min(max_gain, raw_gain))

    brightened = ImageEnhance.Brightness(image).enhance(gain_applied)

    base_arr = np.array(image, dtype=np.float32)
    bright_arr = np.array(brightened, dtype=np.float32)
    alpha = build_soft_alpha(mask_path, threshold, feather_radius)
    out_arr = np.clip(base_arr * (1.0 - alpha) + bright_arr * alpha, 0, 255).astype(np.uint8)
    out_img = Image.fromarray(out_arr, mode="RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)

    after_mean = compute_masked_value_mean(out_img, fg_mask)

    return {
        "image_width": int(base_arr.shape[1]),
        "image_height": int(base_arr.shape[0]),
        "object_value_mean_before": round(before_mean, 6),
        "object_value_mean_after": round(after_mean, 6),
        "target_object_value_mean": round(target_mean_value, 6),
        "raw_gain": round(raw_gain, 6),
        "gain_applied": round(gain_applied, 6),
        "mask_threshold": int(threshold),
        "mask_feather_radius": float(feather_radius),
    }


def resolve_target_mean_value(
    target_object_value_mean: Optional[float],
    reference_image: Optional[Path],
    reference_mask: Optional[Path],
    threshold: int,
) -> Tuple[float, dict]:
    if reference_image and reference_mask:
        image = Image.open(reference_image).convert("RGB")
        fg_mask = load_binary_mask(reference_mask, threshold)
        target = compute_masked_value_mean(image, fg_mask)
        return target, {
            "mode": "reference_image",
            "reference_image": str(reference_image),
            "reference_mask": str(reference_mask),
            "target_object_value_mean": round(target, 6),
        }

    if target_object_value_mean is None:
        raise ValueError("Either --target-object-value-mean or both --reference-image/--reference-mask must be provided")

    target = float(target_object_value_mean)
    if not 0.0 < target <= 1.0:
        raise ValueError("--target-object-value-mean must be in (0, 1]")
    return target, {
        "mode": "explicit_value",
        "target_object_value_mean": round(target, 6),
    }


def build_dataset(
    source_root: Path,
    output_root: Path,
    asset_mode: str,
    mask_threshold: int,
    mask_feather_radius: float,
    target_object_value_mean: Optional[float],
    reference_image: Optional[Path],
    reference_mask: Optional[Path],
    min_gain: float,
    max_gain: float,
    gain_scale: float,
    overwrite_pair_image_fields: bool,
) -> dict:
    summary = load_json(source_root / "summary.json", default={}) or {}
    manifest = load_json(source_root / "manifest.json", default={}) or {}
    if not summary or not manifest:
        raise FileNotFoundError(f"summary/manifest missing under {source_root}")

    target_mean_value, target_meta = resolve_target_mean_value(
        target_object_value_mean=target_object_value_mean,
        reference_image=reference_image,
        reference_mask=reference_mask,
        threshold=mask_threshold,
    )

    output_root.mkdir(parents=True, exist_ok=True)

    for asset_name in ["views", "objects", "bbox_views", "bbox_annotations"]:
        src_asset = source_root / asset_name
        if src_asset.exists():
            ensure_link_or_copy(src_asset, output_root / asset_name, asset_mode)

    if (source_root / "object_splits.json").exists():
        shutil.copy2(source_root / "object_splits.json", output_root / "object_splits.json")
    if (source_root / "object_splits.csv").exists():
        shutil.copy2(source_root / "object_splits.csv", output_root / "object_splits.csv")

    pairs_src = source_root / "pairs"
    pairs_dst = output_root / "pairs"
    pairs_dst.mkdir(parents=True, exist_ok=True)
    pair_files = collect_pair_files(pairs_src)
    if not pair_files:
        raise FileNotFoundError(f"No pair jsonl files found in {pairs_src}")

    bright_root = output_root / "bright_views"
    bright_bbox_root = output_root / "bright_bbox_views"
    annotations_root = output_root / "brightness_annotations"
    bright_root.mkdir(parents=True, exist_ok=True)
    bright_bbox_root.mkdir(parents=True, exist_ok=True)
    annotations_root.mkdir(parents=True, exist_ok=True)

    cached_views: Dict[Tuple[str, str], dict] = {}
    all_updated_pairs: List[dict] = []

    def ensure_view_assets(image_rel: str, mask_rel: str, raw_image_rel: Optional[str] = None, bbox_xyxy: Optional[List[int]] = None) -> dict:
        base_image_rel = raw_image_rel or image_rel
        cache_key = (image_rel, mask_rel, base_image_rel, json.dumps(bbox_xyxy, ensure_ascii=False) if bbox_xyxy else "")
        if cache_key in cached_views:
            return cached_views[cache_key]

        image_abs = source_root / base_image_rel
        mask_abs = source_root / mask_rel
        image_rel_path = Path(base_image_rel)
        obj_dir = image_rel_path.parent.name
        stem = image_rel_path.stem

        bright_path = bright_root / obj_dir / f"{stem}.png"
        ann_path = annotations_root / obj_dir / f"{stem}_brightness.json"
        overlay_path: Optional[Path] = None

        payload = brighten_object_only(
            image_path=image_abs,
            mask_path=mask_abs,
            out_path=bright_path,
            threshold=mask_threshold,
            feather_radius=mask_feather_radius,
            target_mean_value=target_mean_value,
            min_gain=min_gain,
            max_gain=max_gain,
            gain_scale=gain_scale,
        )
        if bbox_xyxy:
            overlay_path = bright_bbox_root / obj_dir / f"{stem}.png"
            render_bbox_overlay(bright_path, bbox_xyxy, overlay_path)
        payload.update(
            {
                "source_image": image_rel,
                "source_image_raw": base_image_rel,
                "source_mask": mask_rel,
                "bright_image_raw": relpath(bright_path, output_root),
                "bright_image": relpath(overlay_path, output_root) if overlay_path else relpath(bright_path, output_root),
            }
        )
        save_json(ann_path, payload)

        record = {
            "bright_image": relpath(bright_path, output_root),
            "brightness_json": relpath(ann_path, output_root),
            **payload,
        }
        cached_views[cache_key] = record
        return record

    for pair_file in pair_files:
        rows = load_jsonl(pair_file)
        updated_rows: List[dict] = []
        for row in rows:
            updated = dict(row)

            source_image_before = str(row["source_image"])
            target_image_before = str(row["target_image"])
            source_image_raw = str(row.get("source_image_raw", source_image_before))
            target_image_raw = str(row.get("target_image_raw", target_image_before))
            source_mask_rel = str(row["source_mask"])
            target_mask_rel = str(row["target_mask"])
            source_bbox_xyxy = normalize_bbox_xyxy(row.get("source_bbox_xyxy"))
            target_bbox_xyxy = normalize_bbox_xyxy(row.get("target_bbox_xyxy"))

            src_bright = ensure_view_assets(
                source_image_before,
                source_mask_rel,
                raw_image_rel=source_image_raw,
                bbox_xyxy=source_bbox_xyxy,
            )
            tgt_bright = ensure_view_assets(
                target_image_before,
                target_mask_rel,
                raw_image_rel=target_image_raw,
                bbox_xyxy=target_bbox_xyxy,
            )

            updated["source_image_before_brighten"] = source_image_before
            updated["target_image_before_brighten"] = target_image_before
            updated["source_image_bright"] = src_bright["bright_image"]
            updated["target_image_bright"] = tgt_bright["bright_image"]
            updated["source_image_bright_raw"] = src_bright["bright_image_raw"]
            updated["target_image_bright_raw"] = tgt_bright["bright_image_raw"]
            updated["source_brightness_json"] = src_bright["brightness_json"]
            updated["target_brightness_json"] = tgt_bright["brightness_json"]
            updated["source_object_value_mean_before"] = src_bright["object_value_mean_before"]
            updated["target_object_value_mean_before"] = tgt_bright["object_value_mean_before"]
            updated["source_object_value_mean_after"] = src_bright["object_value_mean_after"]
            updated["target_object_value_mean_after"] = tgt_bright["object_value_mean_after"]
            updated["source_brightness_gain_applied"] = src_bright["gain_applied"]
            updated["target_brightness_gain_applied"] = tgt_bright["gain_applied"]

            if overwrite_pair_image_fields:
                updated["source_image"] = src_bright["bright_image"]
                updated["target_image"] = tgt_bright["bright_image"]

            updated_rows.append(updated)
            all_updated_pairs.append(updated)

        dst_jsonl = pairs_dst / pair_file.name
        write_jsonl(dst_jsonl, updated_rows)
        dst_csv = pairs_dst / pair_file.name.replace(".jsonl", ".csv")
        write_csv(dst_csv, updated_rows)

    unique_pairs = {row["pair_id"]: row for row in all_updated_pairs}
    serialized_views = {"|".join(map(str, cache_key)): payload for cache_key, payload in cached_views.items()}

    new_summary = {
        "success": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_type": f"{summary.get('dataset_type', 'rotation_edit')}_object_brightened",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "asset_mode": asset_mode,
        "mask_threshold": int(mask_threshold),
        "mask_feather_radius": float(mask_feather_radius),
        "min_gain": float(min_gain),
        "max_gain": float(max_gain),
        "gain_scale": float(gain_scale),
        "overwrite_pair_image_fields": bool(overwrite_pair_image_fields),
        "target_brightness": target_meta,
        "source_summary": summary,
        "bright_view_count": len(cached_views),
        "pair_count": len(unique_pairs),
    }
    save_json(output_root / "summary.json", new_summary)

    new_manifest = {
        "summary": new_summary,
        "source_manifest": manifest,
        "pairs": list(unique_pairs.values()),
        "bright_views": serialized_views,
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
        mask_feather_radius=args.mask_feather_radius,
        target_object_value_mean=args.target_object_value_mean,
        reference_image=Path(args.reference_image).resolve() if args.reference_image else None,
        reference_mask=Path(args.reference_mask).resolve() if args.reference_mask else None,
        min_gain=args.min_gain,
        max_gain=args.max_gain,
        gain_scale=args.gain_scale,
        overwrite_pair_image_fields=args.overwrite_pair_image_fields,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
