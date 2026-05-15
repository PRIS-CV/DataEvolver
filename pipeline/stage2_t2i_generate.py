"""
Stage 2: Text-to-Image — Generate images using Qwen-Image-2512.
Reads pipeline/data/prompts.json, outputs pipeline/data/images/{id}.png
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time
import torch


MODEL_PATH = "/data/wuwenzhuo/Qwen-Image-2512"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PROMPTS_PATH = os.path.join(DATA_DIR, "prompts.json")
IMAGES_DIR = os.path.join(DATA_DIR, "images")


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
                    seed_base=None):
    """Load Qwen-Image-2512 and generate images for all prompts."""
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

    print(f"[Stage 2] Loading Qwen-Image-2512 from {MODEL_PATH} ...")
    print(f"[Stage 2] This model is ~56GB; ensure {device} has enough VRAM (80GB A800)")

    try:
        from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
    except ImportError:
        print("[ERROR] QwenImagePipeline not available from diffsynth.")
        sys.exit(1)

    transformer_files = sorted(
        glob.glob(os.path.join(MODEL_PATH, "transformer", "diffusion_pytorch_model*.safetensors"))
    )
    text_encoder_files = sorted(
        glob.glob(os.path.join(MODEL_PATH, "text_encoder", "model*.safetensors"))
    )
    vae_file = os.path.join(MODEL_PATH, "vae", "diffusion_pytorch_model.safetensors")
    if not transformer_files or not text_encoder_files or not os.path.exists(vae_file):
        print("[Stage 2] ERROR: Qwen-Image-2512 model files incomplete.")
        sys.exit(1)

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=transformer_files),
            ModelConfig(path=text_encoder_files),
            ModelConfig(path=vae_file),
        ],
        tokenizer_config=ModelConfig(path=os.path.join(MODEL_PATH, "tokenizer")),
        vram_limit=torch.cuda.mem_get_info(device)[1] / (1024 ** 3) - 0.5,
    )

    print(f"[Stage 2] Model loaded. Generating {len(prompts)} images ...")

    for i, obj in enumerate(prompts):
        out_path = os.path.join(output_dir, f"{obj['id']}.png")
        print(f"[Stage 2] ({i+1}/{len(prompts)}) Generating: {obj['name']} ...")

        try:
            obj_seed = _make_obj_seed(obj["id"], seed_base)
            with torch.no_grad():
                result = pipe(
                    prompt=obj["prompt"],
                    negative_prompt="blurry, low quality, distorted, multiple objects, background clutter",
                    height=height,
                    width=width,
                    num_inference_steps=num_steps,
                    cfg_scale=7.5,
                    seed=obj_seed,
                )
            if isinstance(result, list):
                result = result[0]
            result.save(out_path)
            print(f"    Saved → {out_path}")
        except Exception as e:
            print(f"[WARN] Failed to generate {obj['id']}: {e}")

    # Free GPU memory after all generation
    del pipe
    torch.cuda.empty_cache()
    print("[Stage 2] GPU memory released.")


def verify_outputs(prompts, output_dir=IMAGES_DIR):
    """Check all expected images exist and are non-empty."""
    missing = []
    for obj in prompts:
        path = os.path.join(output_dir, f"{obj['id']}.png")
        if not os.path.exists(path) or os.path.getsize(path) < 1024:
            missing.append(obj['id'])

    if missing:
        print(f"[Stage 2] WARN: Missing/empty images: {missing}")
    else:
        print(f"[Stage 2] ✓ All {len(prompts)} images generated successfully")
    return len(missing) == 0


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Text-to-Image generation")
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=None,
                        help="Base seed for reproducibility (default: auto-randomize)")
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
    )
    if not verify_outputs(prompts, output_dir=args.output_dir):
        sys.exit(1)


if __name__ == "__main__":
    main()
