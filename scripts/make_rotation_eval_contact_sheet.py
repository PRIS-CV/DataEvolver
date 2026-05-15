#!/usr/bin/env python3
"""Create contact sheets for rotation eval predictions.

Each row shows source, prediction, and target for one pair. This is meant for
quick visual inspection after LoRA inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rotation eval contact sheet")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--meta", default=None, help="Defaults to <pred-dir>/eval_meta.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=21)
    parser.add_argument("--thumb-width", type=int, default=256)
    parser.add_argument("--select", choices=["first", "even_by_angle"], default="even_by_angle")
    return parser.parse_args()


def load_rows(meta_path: Path) -> list[dict]:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_rows(rows: list[dict], limit: int, mode: str) -> list[dict]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    if mode == "first":
        return rows[:limit]
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(str(row.get("target_rotation_deg") or "NA"), []).append(row)
    out = []
    while len(out) < limit and buckets:
        for angle in sorted(list(buckets), key=lambda x: int(float(x)) if x != "NA" else 9999):
            bucket = buckets.get(angle) or []
            if bucket and len(out) < limit:
                out.append(bucket.pop(0))
            if not bucket:
                buckets.pop(angle, None)
    return out


def resize(image: Image.Image, width: int) -> Image.Image:
    h = max(1, round(image.height * width / image.width))
    return image.resize((width, h), Image.BICUBIC)


def paste_labeled(canvas: Image.Image, xy: tuple[int, int], image: Image.Image, label: str, font) -> None:
    draw = ImageDraw.Draw(canvas)
    x, y = xy
    draw.text((x, y), label, fill=(0, 0, 0), font=font)
    canvas.paste(image, (x, y + 20))


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    pred_dir = Path(args.pred_dir).resolve()
    meta_path = Path(args.meta).resolve() if args.meta else pred_dir / "eval_meta.json"
    output = Path(args.output).resolve()
    rows = select_rows(load_rows(meta_path), args.limit, args.select)
    font = ImageFont.load_default()

    triplets = []
    for row in rows:
        source = resize(Image.open(dataset_root / row["source_image"]).convert("RGB"), args.thumb_width)
        pred = resize(Image.open(pred_dir / row["pred_image"]).convert("RGB"), args.thumb_width)
        target = resize(Image.open(dataset_root / row["target_image"]).convert("RGB"), args.thumb_width)
        triplets.append((row, source, pred, target))

    pad = 12
    label_h = 20
    row_h = max(img.height for _, *imgs in triplets for img in imgs) + label_h + pad
    col_w = args.thumb_width
    width = pad * 4 + col_w * 3
    height = pad + row_h * len(triplets)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    y = pad
    for row, source, pred, target in triplets:
        title = f"{row.get('pair_id')} | target={row.get('target_rotation_deg')} | {row.get('instruction', '')}"
        draw.text((pad, y), title[:180], fill=(0, 0, 0), font=font)
        image_y = y + 18
        paste_labeled(canvas, (pad, image_y), source, "source", font)
        paste_labeled(canvas, (pad * 2 + col_w, image_y), pred, "prediction", font)
        paste_labeled(canvas, (pad * 3 + col_w * 2, image_y), target, "target", font)
        y += row_h

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"Saved contact sheet: {output}")


if __name__ == "__main__":
    main()
