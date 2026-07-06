"""
Stage 2.5: RGBA quality gate — validates rembg output before 3D mesh generation.

Checks (ordered, short-circuit on first failure):
  1. File readable / PIL can open it
  2. Exactly 4 channels (RGBA)
  3. alpha_min < 255  (rembg actually removed some background)
  4. alpha_max > 0    (not completely transparent)
  5. foreground_ratio ∈ [FG_RATIO_MIN, FG_RATIO_MAX]

Exposed API:
  check_rgba(image_path)         -> GateResult  (file-level)
  check_rgba_from_pil(pil_image) -> GateResult  (inline use in stage3)

CLI usage:
  python pipeline/stage2_5_rgba_gate.py \
    --image-dir pipeline/data/images_rgba/ \
    --output gate_manifest.json
"""

import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

FG_RATIO_MIN = 0.05
FG_RATIO_MAX = 0.95


@dataclass
class GateResult:
    passed: bool
    reason: str               # "ok" or failure reason string
    image_path: str
    rgba_path: Optional[str]
    num_channels: Optional[int]
    alpha_min: Optional[int]
    alpha_max: Optional[int]
    foreground_ratio: Optional[float]


def check_rgba(image_path: str) -> GateResult:
    """File-level RGBA quality check."""
    from PIL import Image

    if not os.path.exists(image_path):
        return GateResult(
            passed=False, reason="unreadable", image_path=image_path,
            rgba_path=None, num_channels=None, alpha_min=None, alpha_max=None,
            foreground_ratio=None,
        )

    try:
        img = Image.open(image_path)
    except Exception as e:
        return GateResult(
            passed=False, reason=f"unreadable: {e}", image_path=image_path,
            rgba_path=None, num_channels=None, alpha_min=None, alpha_max=None,
            foreground_ratio=None,
        )

    return check_rgba_from_pil(img, image_path=image_path)


def check_rgba_from_pil(image_pil, image_path: str = "") -> GateResult:
    """PIL Image-level RGBA quality check (for inline use in stage3)."""
    num_channels = len(image_pil.getbands())

    if num_channels != 4:
        return GateResult(
            passed=False, reason="not_rgba", image_path=image_path,
            rgba_path=image_path if image_path else None,
            num_channels=num_channels, alpha_min=None, alpha_max=None,
            foreground_ratio=None,
        )

    arr = np.array(image_pil)
    alpha = arr[:, :, 3]
    alpha_min = int(alpha.min())
    alpha_max = int(alpha.max())

    if alpha_min >= 255:
        return GateResult(
            passed=False, reason="alpha_all_white", image_path=image_path,
            rgba_path=image_path if image_path else None,
            num_channels=4, alpha_min=alpha_min, alpha_max=alpha_max,
            foreground_ratio=None,
        )

    if alpha_max <= 0:
        return GateResult(
            passed=False, reason="alpha_all_black", image_path=image_path,
            rgba_path=image_path if image_path else None,
            num_channels=4, alpha_min=alpha_min, alpha_max=alpha_max,
            foreground_ratio=None,
        )

    total_pixels = alpha.size
    fg_pixels = int((alpha > 127).sum())
    fg_ratio = fg_pixels / max(1, total_pixels)

    if fg_ratio < FG_RATIO_MIN or fg_ratio > FG_RATIO_MAX:
        return GateResult(
            passed=False, reason="fg_ratio_out_of_range", image_path=image_path,
            rgba_path=image_path if image_path else None,
            num_channels=4, alpha_min=alpha_min, alpha_max=alpha_max,
            foreground_ratio=round(fg_ratio, 4),
        )

    return GateResult(
        passed=True, reason="ok", image_path=image_path,
        rgba_path=image_path if image_path else None,
        num_channels=4, alpha_min=alpha_min, alpha_max=alpha_max,
        foreground_ratio=round(fg_ratio, 4),
    )


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2.5: RGBA quality gate for rembg outputs")
    p.add_argument("--image-dir", required=True, help="Directory with RGBA PNG images")
    p.add_argument("--output", default="gate_manifest.json",
                   help="Output gate manifest JSON (default: gate_manifest.json)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = []
    passed = 0
    failed = 0

    for fname in sorted(os.listdir(args.image_dir)):
        if not fname.lower().endswith(".png"):
            continue
        img_path = os.path.join(args.image_dir, fname)
        gate = check_rgba(img_path)
        results.append(asdict(gate))
        if gate.passed:
            passed += 1
            print(f"  OK  [{gate.fg_ratio:.1%} fg]: {fname}")
        else:
            print(f"  FAIL [{gate.reason}]: {fname}")
            failed += 1

    manifest = {
        "total":   len(results),
        "passed":  passed,
        "failed":  failed,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Stage 2.5] RGBA gate: {passed}/{len(results)} passed → {args.output}")
