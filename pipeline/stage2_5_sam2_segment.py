"""
Stage 2.5: SAM3 Foreground Segmentation — between T2I and Image-to-3D.

Reads pipeline/data/images/{id}.png (RGB, white/gray background),
outputs pipeline/data/images_rgba/{id}.png (RGBA, transparent background).

Uses SAM3 with text prompt "the main object":
  1. processor.set_image(pil_img)  → image state
  2. processor.set_text_prompt("the main object", state)  → masks + scores
  3. Pick the highest-score mask
  4. Morphological close to fill small holes
  Fallback to edge_flood_fill_reference() + color_fallback() when SAM3
  returns no masks.

Stage 3 (Image-to-3D) auto-detects images_rgba/ and skips rembg when present.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

SAM3_CKPT = "/huggingface/model_hub/sam3/sam3.pt"
SAM3_DIR = "<data-build-root>"
TEXT_PROMPT = "the main object"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
IMAGES_RGBA_DIR = os.path.join(DATA_DIR, "images_rgba")
PROMPTS_PATH = os.path.join(DATA_DIR, "prompts.json")

FG_RATIO_MIN = 0.05
FG_RATIO_MAX = 0.85


def prepare_runtime_device(device: str) -> tuple[str, str | None]:
    """Remap a physical cuda:N target into a single visible local cuda:0.

    SAM3 internals appear to assume cuda:0 in some indexing paths. When we want
    to run one worker per GPU in parallel, the safest approach is to isolate each
    process with CUDA_VISIBLE_DEVICES=<physical-id> and let the worker use the
    local cuda:0 inside that narrowed device set.
    """
    raw = str(device).strip()
    if not raw.startswith("cuda:"):
        return raw, None
    physical = raw.split(":", 1)[1]
    if not physical.isdigit():
        return raw, None
    if os.environ.get("ARIS_DISABLE_CUDA_VISIBLE_REMAP") == "1":
        return raw, os.environ.get("CUDA_VISIBLE_DEVICES")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "cuda:0", os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = physical
    return "cuda:0", physical


def build_model(device):
    """Load SAM3 image model and return a Sam3Processor."""
    if SAM3_DIR not in sys.path:
        sys.path.insert(0, SAM3_DIR)

    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as e:
        print(f"[Stage 2.5] ERROR: sam3 import failed: {e}")
        print(f"  Expected sam3 package at: {SAM3_DIR}")
        sys.exit(1)

    if not os.path.exists(SAM3_CKPT):
        print(f"[Stage 2.5] ERROR: SAM3 checkpoint not found: {SAM3_CKPT}")
        sys.exit(1)

    print(f"[Stage 2.5] Loading SAM3 from {SAM3_CKPT} on {device}...")
    model = build_sam3_image_model(
        checkpoint_path=SAM3_CKPT,
        load_from_HF=False,
        device=device,
    )

    # Keep model in native bfloat16 — SAM3 uses bf16 internally throughout.
    # Do NOT call .float(); inference is wrapped with autocast(bfloat16).
    model = model.to(device)

    processor = Sam3Processor(model, device=device, confidence_threshold=0.0)
    print("[Stage 2.5] SAM3 model loaded.")
    return processor


def color_fallback(image_np):
    """Color-distance segmentation fallback when SAM3 returns no masks.

    Estimates background color from four corner patches, thresholds per-pixel
    Euclidean distance to background. Morphological cleanup; returns largest
    connected foreground component.

    Args:
        image_np: H x W x 3 uint8 RGB numpy array

    Returns:
        mask: H x W bool numpy array (True = foreground)
    """
    import cv2

    corners = [
        image_np[:10, :10],
        image_np[:10, -10:],
        image_np[-10:, :10],
        image_np[-10:, -10:],
    ]
    bg = np.mean([c.mean(axis=(0, 1)) for c in corners], axis=0)
    dist = np.sqrt(((image_np.astype(float) - bg) ** 2).sum(axis=2))
    thresh = max(30.0, float(np.percentile(dist, 50)))
    fg = (dist > thresh).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n_labels <= 1:
        return fg > 127
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def edge_flood_fill_reference(image_np):
    """CPU-only reference foreground mask via edge-guided flood fill.

    Used as fallback when SAM3 returns no masks. Not called in the main path.

    Args:
        image_np: H x W x 3 uint8 RGB numpy array

    Returns:
        mask: H x W bool numpy array (True = foreground reference)
    """
    import cv2

    H, W = image_np.shape[:2]

    corners = [
        image_np[:10, :10], image_np[:10, -10:],
        image_np[-10:, :10], image_np[-10:, -10:],
    ]
    bg_color = np.mean([c.mean(axis=(0, 1)) for c in corners], axis=0)
    dist = np.sqrt(((image_np.astype(float) - bg_color) ** 2).sum(axis=2))

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 20, 80)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    bg_like = ((dist < 30) & (edges == 0)).astype(np.uint8) * 255
    bg_like = cv2.morphologyEx(bg_like, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    flood_mask = np.zeros((H + 2, W + 2), np.uint8)
    for x in range(0, W, 2):
        if bg_like[0, x] == 255:
            cv2.floodFill(bg_like, flood_mask, (x, 0), 128)
        if bg_like[H - 1, x] == 255:
            cv2.floodFill(bg_like, flood_mask, (x, H - 1), 128)
    for y in range(0, H, 2):
        if bg_like[y, 0] == 255:
            cv2.floodFill(bg_like, flood_mask, (0, y), 128)
        if bg_like[y, W - 1] == 255:
            cv2.floodFill(bg_like, flood_mask, (W - 1, y), 128)

    return bg_like != 128


def segment_image_sam3(processor, image_np):
    """Segment foreground using SAM3 text prompt 'the main object'.

    Steps:
      1. set_image(pil_img) → image state
      2. set_text_prompt(TEXT_PROMPT, state) → {masks, scores, boxes}
      3. Pick highest-score mask (shape H x W bool)
      4. Morphological close to fill small holes
      Fallback: color_fallback() when SAM3 returns no masks.

    Args:
        processor: Sam3Processor instance
        image_np: H x W x 3 uint8 RGB numpy array

    Returns:
        mask: H x W bool numpy array (True = foreground)
    """
    import cv2

    pil_img = Image.fromarray(image_np)

    try:
        import torch
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = processor.set_image(pil_img)
            output = processor.set_text_prompt(prompt=TEXT_PROMPT, state=state)
    except TypeError:
        import torch
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            processor.set_image(pil_img)
            output = processor.set_text_prompt(prompt=TEXT_PROMPT)

    masks = output.get("masks", None)
    scores = output.get("scores", None)

    if masks is None or (hasattr(masks, '__len__') and len(masks) == 0):
        print("(SAM3 no masks, fallback)", end=" ")
        return color_fallback(image_np)

    # Convert to numpy if torch tensor; cast to float32 first (bfloat16 unsupported by numpy)
    if hasattr(masks, "cpu"):
        masks = masks.float().cpu().numpy()
    if hasattr(scores, "cpu"):
        scores = scores.float().cpu().numpy()

    best_idx = int(np.argmax(scores)) if scores is not None and len(scores) > 0 else 0
    mask = masks[best_idx]
    # Squeeze to 2D (SAM3 may return (1, H, W) or (H, W))
    while mask.ndim > 2:
        mask = mask.squeeze(0)
    mask = mask.astype(bool)

    # Morphological close: fill small holes
    mask_u8 = mask.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return mask_u8 > 127


def segment_all(obj_ids, device, skip_existing, images_dir=IMAGES_DIR, rgba_dir=IMAGES_RGBA_DIR):
    """Run SAM3 text-prompted segmentation for all given object IDs."""
    os.makedirs(rgba_dir, exist_ok=True)

    to_process = []
    for oid in obj_ids:
        out_path = os.path.join(rgba_dir, f"{oid}.png")
        if skip_existing and os.path.exists(out_path):
            print(f"[Stage 2.5] Skip (exists): {oid}")
            continue
        src_path = os.path.join(images_dir, f"{oid}.png")
        if not os.path.exists(src_path):
            print(f"[WARN] Source image not found: {src_path}, skipping")
            continue
        to_process.append(oid)

    if not to_process:
        print("[Stage 2.5] All RGBA images already exist, nothing to do.")
        return {"ok": 0, "warn": 0, "error": 0, "processed": 0}

    processor = build_model(device)

    ok = 0
    warn = 0
    error = 0
    for i, oid in enumerate(to_process):
        src_path = os.path.join(images_dir, f"{oid}.png")
        out_path = os.path.join(rgba_dir, f"{oid}.png")
        print(f"[Stage 2.5] ({i+1}/{len(to_process)}) {oid} ...", end=" ")

        try:
            rgb_img = Image.open(src_path).convert("RGB")
            image_np = np.array(rgb_img)
            mask = segment_image_sam3(processor, image_np)

            # QC: check foreground ratio
            fg_ratio = mask.sum() / mask.size
            if fg_ratio < FG_RATIO_MIN or fg_ratio > FG_RATIO_MAX:
                print(f"WARN fg={fg_ratio:.1%} (expected {FG_RATIO_MIN:.0%}–{FG_RATIO_MAX:.0%})")
                warn += 1
            else:
                print(f"fg={fg_ratio:.1%} OK")
                ok += 1

            # Build RGBA: copy RGB channels, set alpha from mask
            rgba_arr = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
            rgba_arr[:, :, :3] = image_np
            rgba_arr[:, :, 3] = (mask * 255).astype(np.uint8)
            Image.fromarray(rgba_arr, "RGBA").save(out_path)

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            error += 1

    print(f"[Stage 2.5] Done: {ok} OK, {warn} warnings, {error} errors, out of {len(to_process)} processed.")
    return {"ok": ok, "warn": warn, "error": error, "processed": len(to_process)}


def main():
    parser = argparse.ArgumentParser(description="Stage 2.5: SAM3 foreground segmentation")
    parser.add_argument("--device", default="cuda:0",
                        help="CUDA device (e.g. cuda:0)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Force re-segmentation even if RGBA output already exists")
    parser.add_argument("--ids", default=None,
                        help="Comma-separated object IDs to process (default: all from prompts.json)")
    parser.add_argument("--images-dir", default=IMAGES_DIR,
                        help="Input RGB images directory")
    parser.add_argument("--rgba-dir", default=IMAGES_RGBA_DIR,
                        help="Output RGBA images directory")
    parser.add_argument("--input-dir", default=None,
                        help="Alias of --images-dir for replacement runner compatibility")
    parser.add_argument("--output-dir", default=None,
                        help="Alias of --rgba-dir for replacement runner compatibility")
    parser.add_argument("--prompts-path", default=PROMPTS_PATH,
                        help="Prompt file used to enumerate valid object IDs")
    args = parser.parse_args()

    if args.input_dir:
        args.images_dir = args.input_dir
    if args.output_dir:
        args.rgba_dir = args.output_dir
    runtime_device, visible_devices = prepare_runtime_device(args.device)

    with open(args.prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    all_ids = [p["id"] for p in prompts]

    if args.ids:
        obj_ids = [x.strip() for x in args.ids.split(",") if x.strip()]
        invalid = [oid for oid in obj_ids if oid not in all_ids]
        if invalid:
            print(f"[WARN] Unknown object IDs: {invalid}")
        obj_ids = [oid for oid in obj_ids if oid in all_ids]
    else:
        obj_ids = all_ids

    if visible_devices:
        print(f"[Stage 2.5] CUDA_VISIBLE_DEVICES={visible_devices}; logical device={runtime_device}")
    print(f"[Stage 2.5] Processing {len(obj_ids)} objects on {args.device}")
    result = segment_all(
        obj_ids,
        device=runtime_device,
        skip_existing=not args.no_skip,
        images_dir=args.images_dir,
        rgba_dir=args.rgba_dir,
    )
    if result["error"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
