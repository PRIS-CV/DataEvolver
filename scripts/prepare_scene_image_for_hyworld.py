#!/usr/bin/env python3
"""Prepare Stage2 scene images before HYWorld conditioning.

This script fixes a narrow class of T2I failures that are safe to repair
automatically: blank white/black borders or letterbox padding around an
otherwise usable scene photograph. It also rejects obvious source-image
artifacts before they are propagated into HYWorld reconstruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop and quality-gate scene images for HYWorld")
    parser.add_argument("images", nargs="+", help="Input scene image paths")
    parser.add_argument("--output-dir", required=True, help="Directory for cleaned scene images")
    parser.add_argument("--report-path", default=None, help="Optional JSON report path")
    parser.add_argument("--crop-padding", type=int, default=0)
    parser.add_argument("--content-row-frac", type=float, default=0.04)
    parser.add_argument("--content-col-frac", type=float, default=0.04)
    parser.add_argument("--white-threshold", type=int, default=245)
    parser.add_argument("--black-threshold", type=int, default=8)
    parser.add_argument("--max-border-blank", type=float, default=0.18)
    parser.add_argument("--max-bottom-blank", type=float, default=0.15)
    parser.add_argument("--max-border-dark", type=float, default=0.18)
    parser.add_argument("--max-saturated-color", type=float, default=0.08)
    parser.add_argument("--min-crop-area-ratio", type=float, default=0.35)
    parser.add_argument("--require-usable", action="store_true", help="Exit non-zero if any image fails")
    return parser.parse_args()


def blank_masks(rgb: np.ndarray, white_threshold: int, black_threshold: int) -> tuple[np.ndarray, np.ndarray]:
    white = (rgb >= white_threshold).all(axis=2)
    black = (rgb <= black_threshold).all(axis=2)
    return white, black


def crop_blank_border(
    rgb: np.ndarray,
    white_threshold: int,
    black_threshold: int,
    row_frac: float,
    col_frac: float,
    padding: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    white, black = blank_masks(rgb, white_threshold, black_threshold)
    content = ~(white | black)
    row_has = content.mean(axis=1) > row_frac
    col_has = content.mean(axis=0) > col_frac
    if not row_has.any() or not col_has.any():
        h, w = rgb.shape[:2]
        return rgb, (0, 0, w, h)

    ys = np.where(row_has)[0]
    xs = np.where(col_has)[0]
    h, w = rgb.shape[:2]
    y0 = max(int(ys[0]) - padding, 0)
    y1 = min(int(ys[-1]) + 1 + padding, h)
    x0 = max(int(xs[0]) - padding, 0)
    x1 = min(int(xs[-1]) + 1 + padding, w)
    return rgb[y0:y1, x0:x1], (x0, y0, x1, y1)


def edge_pixels(rgb: np.ndarray, width: int = 20) -> np.ndarray:
    h, w = rgb.shape[:2]
    bw = max(1, min(width, h // 4, w // 4))
    return np.concatenate(
        [
            rgb[:bw].reshape(-1, 3),
            rgb[-bw:].reshape(-1, 3),
            rgb[:, :bw].reshape(-1, 3),
            rgb[:, -bw:].reshape(-1, 3),
        ],
        axis=0,
    )


def image_metrics(rgb: np.ndarray, white_threshold: int, black_threshold: int) -> dict:
    white, black = blank_masks(rgb, white_threshold, black_threshold)
    border = edge_pixels(rgb)
    border_white = (border >= white_threshold).all(axis=1)
    border_black = (border <= black_threshold).all(axis=1)
    border_dark = border.mean(axis=1) < 35
    bottom = rgb[int(rgb.shape[0] * 0.75) :]
    bottom_white, bottom_black = blank_masks(bottom, white_threshold, black_threshold)
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    saturated_yellow = (red > 220) & (green > 220) & (blue < 120)
    saturated_blue = (blue > 170) & (red < 120) & (green < 165)
    return {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "border_white_ratio": float(border_white.mean()),
        "border_black_ratio": float(border_black.mean()),
        "border_dark_ratio": float(border_dark.mean()),
        "bottom_white_ratio": float(bottom_white.mean()),
        "bottom_black_ratio": float(bottom_black.mean()),
        "saturated_yellow_ratio": float(saturated_yellow.mean()),
        "saturated_blue_ratio": float(saturated_blue.mean()),
    }


def assess(metrics: dict, args: argparse.Namespace, crop_area_ratio: float) -> list[str]:
    reasons: list[str] = []
    border_blank = metrics["border_white_ratio"] + metrics["border_black_ratio"]
    bottom_blank = metrics["bottom_white_ratio"] + metrics["bottom_black_ratio"]
    max_saturated = max(metrics["saturated_yellow_ratio"], metrics["saturated_blue_ratio"])
    if crop_area_ratio < args.min_crop_area_ratio:
        reasons.append(f"crop_area_ratio_too_small:{crop_area_ratio:.3f}")
    if border_blank > args.max_border_blank:
        reasons.append(f"border_blank:{border_blank:.3f}")
    if bottom_blank > args.max_bottom_blank:
        reasons.append(f"bottom_blank:{bottom_blank:.3f}")
    if metrics["border_dark_ratio"] > args.max_border_dark:
        reasons.append(f"border_dark:{metrics['border_dark_ratio']:.3f}")
    if max_saturated > args.max_saturated_color:
        reasons.append(f"saturated_color:{max_saturated:.3f}")
    return reasons


def process_one(path: Path, args: argparse.Namespace) -> dict:
    image = Image.open(path).convert("RGB")
    original = np.asarray(image)
    cropped, bbox = crop_blank_border(
        original,
        args.white_threshold,
        args.black_threshold,
        args.content_row_frac,
        args.content_col_frac,
        args.crop_padding,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / path.name
    Image.fromarray(cropped).save(out_path)

    crop_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    crop_area_ratio = crop_area / float(original.shape[0] * original.shape[1])
    metrics = image_metrics(cropped, args.white_threshold, args.black_threshold)
    reasons = assess(metrics, args, crop_area_ratio)
    return {
        "input_path": str(path),
        "output_path": str(out_path),
        "usable": not reasons,
        "reasons": reasons,
        "bbox_xyxy": [int(v) for v in bbox],
        "crop_area_ratio": float(crop_area_ratio),
        "metrics": metrics,
    }


def main() -> int:
    args = parse_args()
    records = [process_one(Path(raw), args) for raw in args.images]
    report = {"records": records, "all_usable": all(item["usable"] for item in records)}
    report_path = Path(args.report_path) if args.report_path else Path(args.output_dir) / "scene_image_quality.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for item in records:
        status = "PASS" if item["usable"] else "FAIL"
        reasons = ",".join(item["reasons"]) or "ok"
        print(f"[{status}] {item['input_path']} -> {item['output_path']} ({reasons})")
    if args.require_usable and not report["all_usable"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
