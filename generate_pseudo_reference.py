"""
Generate pseudo-reference insertion candidates with Qwen-Image-Edit-2511.

This script is designed for remote execution on wwz. It uses a single process
with multi-GPU sharding (device_map=balanced) and generates candidates
sequentially after one model load.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

from PIL import Image


DEFAULT_PROMPT = (
    "Use the first image as the target background scene and the second image as the "
    "isolated foreground object source. Insert the object from the "
    "second image naturally into the first image. Keep the object's identity, structure, "
    "texture and main colors consistent. Choose a physically plausible position, scale "
    "and orientation. Ensure the object is grounded in the scene with natural contact and "
    "shadow cues. Do not significantly alter the background scene."
)

DEFAULT_NEGATIVE_PROMPT = (
    "different object, duplicated object, extra objects, severe color shift, "
    "destroyed background, unrealistic deformation, blurry, low quality"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pseudo-reference images with Qwen Image Edit 2511")
    parser.add_argument("--model-path", default="/data/wuwenzhuo/Qwen-Image-Edit-2511")
    parser.add_argument("--rgba-path", required=True)
    parser.add_argument("--background-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", default="0,1,2", help="Comma-separated physical GPU ids")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-memory-gib", type=int, default=75)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--matte-color",
        default="255,255,255",
        help="RGB matte color used to flatten RGBA input for the edit model",
    )
    parser.add_argument("--select-index", type=int, default=None, help="1-based candidate index to copy as pseudo_reference.png")
    return parser.parse_args()


def build_prompt(args) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    return DEFAULT_PROMPT


def ensure_inputs(args):
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not os.path.exists(args.rgba_path):
        raise FileNotFoundError(f"RGBA image not found: {args.rgba_path}")
    if not os.path.exists(args.background_path):
        raise FileNotFoundError(f"Background image not found: {args.background_path}")


def configure_cuda(gpus: str) -> List[str]:
    gpu_list = [g.strip() for g in gpus.split(",") if g.strip()]
    if not gpu_list:
        raise ValueError("No GPUs specified")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return gpu_list


def serialize_device_map(device_map) -> Dict[str, str]:
    if isinstance(device_map, dict):
        return {str(k): str(v) for k, v in device_map.items()}
    return {"mode": str(device_map)}


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_rgb_triplet(raw: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid matte color '{raw}', expected R,G,B")
    values = tuple(max(0, min(255, int(p))) for p in parts)
    return values


def prepare_foreground_rgb(rgba_path: str, output_dir: Path, matte_color: tuple[int, int, int]) -> Image.Image:
    source = Image.open(rgba_path)
    if source.mode == "RGBA":
        matte = Image.new("RGBA", source.size, matte_color + (255,))
        flattened = Image.alpha_composite(matte, source).convert("RGB")
    else:
        flattened = source.convert("RGB")
    flattened.save(output_dir / "input_foreground_rgb.png")
    return flattened


def main():
    args = parse_args()
    ensure_inputs(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(args)
    visible_gpu_list = configure_cuda(args.gpus)
    matte_color = parse_rgb_triplet(args.matte_color)

    request = {
        "model_path": args.model_path,
        "rgba_path": os.path.abspath(args.rgba_path),
        "background_path": os.path.abspath(args.background_path),
        "output_dir": str(output_dir),
        "visible_gpus": visible_gpu_list,
        "num_candidates": args.num_candidates,
        "seed_base": args.seed_base,
        "num_inference_steps": args.num_inference_steps,
        "true_cfg_scale": args.true_cfg_scale,
        "guidance_scale": args.guidance_scale,
        "matte_color": list(matte_color),
        "negative_prompt": args.negative_prompt,
        "select_index": args.select_index,
    }
    save_json(output_dir / "request.json", request)
    (output_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    shutil.copy2(args.rgba_path, output_dir / "input_rgba.png")
    shutil.copy2(args.background_path, output_dir / "input_background.png")

    start = time.time()
    success = False
    generated_files: List[str] = []
    error = None
    load_config = {}

    try:
        import diffusers
        import torch
        from diffusers import QwenImageEditPlusPipeline

        if torch.cuda.device_count() < len(visible_gpu_list):
            raise RuntimeError(
                f"CUDA visible device count={torch.cuda.device_count()} < requested GPUs={len(visible_gpu_list)}"
            )

        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        max_memory = {idx: f"{args.max_memory_gib}GiB" for idx in range(len(visible_gpu_list))}

        load_config = {
            "visible_gpus": visible_gpu_list,
            "device_map_mode": "balanced",
            "dtype": args.dtype,
            "max_memory": max_memory,
            "diffusers_version": diffusers.__version__,
        }
        save_json(output_dir / "load_config.json", load_config)

        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            device_map="balanced",
            max_memory=max_memory,
            local_files_only=True,
        )
        pipeline.set_progress_bar_config(disable=None)

        hf_device_map = getattr(pipeline, "hf_device_map", None)
        if hf_device_map:
            load_config["hf_device_map"] = serialize_device_map(hf_device_map)
            save_json(output_dir / "load_config.json", load_config)

        background = Image.open(args.background_path).convert("RGB")
        foreground_rgb = prepare_foreground_rgb(args.rgba_path, output_dir, matte_color)

        with torch.inference_mode():
            for idx in range(args.num_candidates):
                seed = args.seed_base + idx
                generator = torch.Generator(device="cpu").manual_seed(seed)
                output = pipeline(
                    image=[background, foreground_rgb],
                    prompt=prompt,
                    generator=generator,
                    true_cfg_scale=args.true_cfg_scale,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=1,
                )
                out_path = output_dir / f"candidate_{idx + 1:02d}.png"
                output.images[0].save(out_path)
                generated_files.append(str(out_path))

        if args.select_index is not None:
            if args.select_index < 1 or args.select_index > args.num_candidates:
                raise ValueError(f"select-index {args.select_index} out of range 1..{args.num_candidates}")
            selected_name = f"candidate_{args.select_index:02d}.png"
            shutil.copy2(output_dir / selected_name, output_dir / "pseudo_reference.png")
            save_json(
                output_dir / "selection.json",
                {
                    "mode": "explicit_index",
                    "selected_index": args.select_index,
                    "selected_file": selected_name,
                },
            )

        success = True
    except Exception as exc:
        error = repr(exc)
        print(f"[pseudo-ref] ERROR: {exc}", file=sys.stderr)
    finally:
        elapsed = round(time.time() - start, 3)
        runtime = {
            "visible_gpus": visible_gpu_list,
            "device_map_mode": "balanced",
            "dtype": args.dtype,
            "seed_base": args.seed_base,
            "num_candidates": args.num_candidates,
            "elapsed_seconds": elapsed,
            "success": success,
            "generated_files": generated_files,
            "error": error,
        }
        save_json(output_dir / "runtime.json", runtime)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
