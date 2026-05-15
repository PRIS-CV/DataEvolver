#!/usr/bin/env python3
"""Calculate image metrics for rotation-edit inference outputs.

Input is the eval_meta.json written by scripts/eval_rotation_lora_inference.py.
The script compares each predicted image against its target image and writes a
SpatialEdit-style metrics CSV that can be consumed by build_external_feedback_loop.py.

Metrics:
  - PSNR
  - SSIM
  - LPIPS if the lpips package is installed, otherwise blank

CLIP/DINO columns are emitted but left blank in this lightweight evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


ANGLE_ORDER = [0, 45, 90, 135, 180, 225, 270, 315]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate rotation edit metrics")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--meta", default=None, help="Defaults to <pred-dir>/eval_meta.json")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-lpips", action="store_true")
    return parser.parse_args()


def load_rgb(path: Path, size: Optional[tuple[int, int]] = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BICUBIC)
    return image


def image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image).astype(np.float32) / 255.0


def calc_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def calc_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(target, pred, channel_axis=2, data_range=1.0))
    except Exception:
        # Global SSIM fallback. Less precise than skimage's windowed SSIM, but
        # deterministic and dependency-free.
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu_x = float(np.mean(target))
        mu_y = float(np.mean(pred))
        sigma_x = float(np.var(target))
        sigma_y = float(np.var(pred))
        sigma_xy = float(np.mean((target - mu_x) * (pred - mu_y)))
        return float(((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)))


def make_lpips(device: str, disabled: bool):
    if disabled:
        return None
    try:
        import lpips
        import torch

        model = lpips.LPIPS(net="alex").to(device)
        model.eval()
        return model
    except Exception as exc:
        print(f"[Warn] LPIPS unavailable, leaving lpips column blank: {exc}")
        return None


def calc_lpips(model, pred_img: Image.Image, target_img: Image.Image, device: str) -> Optional[float]:
    if model is None:
        return None
    import torch

    def to_tensor(image: Image.Image):
        arr = np.asarray(image).astype(np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return ten.to(device) * 2.0 - 1.0

    with torch.no_grad():
        value = model(to_tensor(pred_img), to_tensor(target_img))
    return float(value.item())


def angle_name(deg: object) -> str:
    try:
        deg_i = int(float(str(deg)))
    except Exception:
        return "angleNA"
    if deg_i in ANGLE_ORDER:
        return f"angle{ANGLE_ORDER.index(deg_i):02d}"
    return f"angle{deg_i:03d}"


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    pred_dir = Path(args.pred_dir).resolve()
    meta_path = Path(args.meta).resolve() if args.meta else pred_dir / "eval_meta.json"
    out_csv = Path(args.output_csv).resolve()

    with meta_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    lpips_model = make_lpips(args.device, args.no_lpips)
    out_rows = []
    for idx, row in enumerate(rows, start=1):
        pair_id = str(row["pair_id"])
        pred_path = pred_dir / row["pred_image"]
        target_path = dataset_root / row["target_image"]
        pred_img = load_rgb(pred_path)
        target_img = load_rgb(target_path, size=pred_img.size)
        pred_arr = image_array(pred_img)
        target_arr = image_array(target_img)

        psnr = calc_psnr(pred_arr, target_arr)
        ssim = calc_ssim(pred_arr, target_arr)
        lpips_value = calc_lpips(lpips_model, pred_img, target_img, args.device)
        angle = angle_name(row.get("target_rotation_deg"))
        image_name = f"{pair_id}_{angle}.png"
        out_rows.append(
            {
                "image_name": image_name,
                "pair_id": pair_id,
                "obj_id": row.get("obj_id") or "",
                "target_rotation_deg": row.get("target_rotation_deg") or "",
                "psnr": round(psnr, 6),
                "ssim": round(ssim, 6),
                "lpips": "" if lpips_value is None else round(lpips_value, 6),
                "dino_similarity": "",
                "clip_similarity": "",
                "source_image": row.get("source_image") or "",
                "target_image": row.get("target_image") or "",
                "pred_image": row.get("pred_image") or "",
                "instruction": row.get("instruction") or "",
            }
        )
        print(f"[{idx}/{len(rows)}] {pair_id}: psnr={psnr:.4f} ssim={ssim:.4f} lpips={lpips_value}")

    fieldnames = [
        "image_name",
        "pair_id",
        "obj_id",
        "target_rotation_deg",
        "psnr",
        "ssim",
        "lpips",
        "dino_similarity",
        "clip_similarity",
        "source_image",
        "target_image",
        "pred_image",
        "instruction",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Done: {len(out_rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
