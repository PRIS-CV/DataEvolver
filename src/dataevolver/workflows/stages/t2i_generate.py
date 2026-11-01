"""
Stage 2: Text-to-Image — Generate images using Qwen-Image-2.1.
Reads .dataevolver/runtime/prompts.json, outputs .dataevolver/runtime/images/{id}.png
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from dataevolver.paths import runtime_path

MODEL_PATH = os.environ.get("QWEN_IMAGE_MODEL_PATH", "")
DATA_DIR = os.fspath(runtime_path())
PROMPTS_PATH = os.path.join(DATA_DIR, "prompts.json")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, multiple objects, background clutter"


def load_prompts(path=PROMPTS_PATH, ids=None):
    with open(path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if ids:
        wanted = {x.strip() for x in ids if str(x).strip()}
        prompts = [p for p in prompts if p["id"] in wanted]
    return prompts


def _make_obj_seed(obj_id: str, run_salt: int) -> int:
    digest = hashlib.sha256(f"{obj_id}:{run_salt}".encode("utf-8")).hexdigest()
    return int(digest[:10], 16) % (2**31)


def generate_images(prompts, device="cuda:0", height=1024, width=1024,
                    num_steps=30, skip_existing=True, output_dir=IMAGES_DIR,
                    seed_base=None, negative_prompt=DEFAULT_NEGATIVE_PROMPT, true_cfg_scale=4.0):
    """Load Qwen-Image-2.1 and generate images for all prompts."""
    if seed_base is None:
        seed_base = int(time.time() * 1000) % (2**31)
        print(f"[Stage 2] Auto seed_base={seed_base}  (pass --seed-base to reproduce)")
    else:
        print(f"[Stage 2] Fixed seed_base={seed_base}")
    os.makedirs(output_dir, exist_ok=True)

    # Check which objects already have images
    if skip_existing:
        remaining = [p for p in prompts
                     if not os.path.exists(os.path.join(output_dir, f"{p['id']}.png"))]
        if len(remaining) < len(prompts):
            print(f"[Stage 2] Skipping {len(prompts) - len(remaining)} existing images")
        prompts = remaining

    if not prompts:
        print("[Stage 2] All images already exist, skipping model load")
        return

    if not MODEL_PATH:
        raise ValueError("Set QWEN_IMAGE_MODEL_PATH explicitly before using this optional adapter")
    import torch
    from diffusers import QwenImage21Pipeline

    print(f"[Stage 2] Loading Qwen-Image-2.1 from {MODEL_PATH} ...")
    pipe = QwenImage21Pipeline.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True,
    ).to(device)

    print(f"[Stage 2] Model loaded. Generating {len(prompts)} images ...")

    for i, obj in enumerate(prompts):
        out_path = os.path.join(output_dir, f"{obj['id']}.png")
        print(f"[Stage 2] ({i+1}/{len(prompts)}) Generating: {obj['name']} ...")

        try:
            obj_seed = _make_obj_seed(obj["id"], seed_base)
            with torch.no_grad():
                result = pipe(
                    prompt=obj["prompt"],
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_steps,
                    true_cfg_scale=true_cfg_scale,
                    generator=torch.Generator(device=device).manual_seed(obj_seed),
                )
            result.images[0].save(out_path)
            print(f"    Saved → {out_path}")
        except Exception as e:
            print(f"[WARN] Failed to generate {obj['id']}: {e}")

    # Free GPU memory after all generation
    del pipe
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Stage 2] GPU memory released.")


def verify_outputs(prompts, output_dir=IMAGES_DIR):
    """Check all expected images exist and can be decoded."""
    from PIL import Image

    invalid = []
    for obj in prompts:
        path = os.path.join(output_dir, f"{obj['id']}.png")
        if not os.path.isfile(path):
            invalid.append(obj['id'])
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            invalid.append(obj['id'])

    if invalid:
        print(f"[Stage 2] WARN: Missing/invalid images: {invalid}")
    else:
        print(f"[Stage 2] ✓ All {len(prompts)} images generated successfully")
    return len(invalid) == 0


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Text-to-Image generation")
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=None,
                        help="Base seed for reproducibility (default: auto-randomize)")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT,
                        help="Negative prompt used for every selected object")
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--no-skip", action="store_true", help="Re-generate even if file exists")
    parser.add_argument("--prompts-path", default=PROMPTS_PATH, help="Path to prompts.json")
    parser.add_argument("--output-dir", default=IMAGES_DIR, help="Where to write generated RGB images")
    parser.add_argument("--ids", default=None, help="Comma-separated object IDs to generate")
    args = parser.parse_args()

    ids = [x.strip() for x in args.ids.split(",")] if args.ids else None
    prompts = load_prompts(path=args.prompts_path, ids=ids)
    print(f"[Stage 2] Loaded {len(prompts)} prompts from {args.prompts_path}")

    generate_images(
        prompts,
        device=args.device,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        skip_existing=not args.no_skip,
        output_dir=args.output_dir,
        seed_base=args.seed_base,
        negative_prompt=args.negative_prompt,
        true_cfg_scale=args.true_cfg_scale,
    )
    if not verify_outputs(prompts, output_dir=args.output_dir):
        sys.exit(1)


if __name__ == "__main__":
    main()
