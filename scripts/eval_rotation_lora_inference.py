#!/usr/bin/env python3
"""Run Qwen-Image-Edit rotation inference for a dataset split.

This is a parameterized replacement for the older eval_inference.py that had
hard-coded dataset and LoRA paths. It supports:
  - base model
  - ARIS-trained LoRA
  - fal/community LoRA

Outputs:
  - one prediction PNG per pair
  - eval_meta.json mapping source/target/pred paths
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file


DEFAULT_MODEL_ROOT = "/data/wuwenzhuo/Qwen-Image-Edit-2511"
DEFAULT_DIFFSYNTH_ROOT = "<diffsynth-root>"
DEFAULT_FAL_LORA = (
    "/data/wuwenzhuo/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/"
    "qwen-image-edit-2511-multiple-angles-lora.safetensors"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rotation LoRA inference")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pairs-file", default=None, help="Defaults to <dataset-root>/pairs/test_pairs.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["base", "ours", "fal"], default="ours")
    parser.add_argument("--lora-path", default=None, help="Required for --mode ours unless using base/fal")
    parser.add_argument("--fal-lora-path", default=DEFAULT_FAL_LORA)
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--diffsynth-root", default=DEFAULT_DIFFSYNTH_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Optional max pairs for smoke tests")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def convert_our_lora_keys(state_dict: dict) -> dict:
    out = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("pipe.dit."):
            new_key = new_key[len("pipe.dit.") :]
        new_key = new_key.replace(".lora_A.default.", ".lora_A.")
        new_key = new_key.replace(".lora_B.default.", ".lora_B.")
        out[new_key] = value
    return out


def convert_fal_lora_keys(state_dict: dict) -> dict:
    out = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("transformer."):
            new_key = new_key[len("transformer.") :]
        out[new_key] = value
    return out


def resolve_model_paths(model_root: Path) -> str:
    transformer = sorted(glob.glob(str(model_root / "transformer" / "diffusion_pytorch_model-*.safetensors")))
    text_encoder = sorted(glob.glob(str(model_root / "text_encoder" / "model-*.safetensors")))
    vae = model_root / "vae" / "diffusion_pytorch_model.safetensors"
    missing = []
    if not transformer:
        missing.append(str(model_root / "transformer" / "diffusion_pytorch_model-*.safetensors"))
    if not text_encoder:
        missing.append(str(model_root / "text_encoder" / "model-*.safetensors"))
    if not vae.exists():
        missing.append(str(vae))
    if missing:
        raise FileNotFoundError("Missing model files: " + ", ".join(missing))
    return json.dumps([transformer, text_encoder, str(vae)])


def load_pipeline(args: argparse.Namespace):
    sys.path.insert(0, args.diffsynth_root)
    from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

    model_root = Path(args.model_root)
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(path=paths)
            for paths in json.loads(resolve_model_paths(model_root))
        ],
        tokenizer_config=ModelConfig(str(model_root / "tokenizer")),
        processor_config=ModelConfig(str(model_root / "processor")),
    )
    return pipe


def load_lora(pipe, path: Path, mode: str) -> None:
    state = load_file(str(path))
    if mode == "ours":
        state = convert_our_lora_keys(state)
    elif mode == "fal":
        state = convert_fal_lora_keys(state)
    pipe.load_lora(pipe.dit, state_dict=state)


def first_image(output):
    if isinstance(output, Image.Image):
        return output
    if isinstance(output, (list, tuple)):
        return output[0]
    if isinstance(output, dict):
        image = output.get("image") or output.get("images")
        if isinstance(image, (list, tuple)):
            return image[0]
        if isinstance(image, Image.Image):
            return image
    if hasattr(output, "images"):
        return output.images[0]
    raise TypeError(f"Unsupported pipeline output type: {type(output)}")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    pairs_file = Path(args.pairs_file).resolve() if args.pairs_file else dataset_root / "pairs" / "test_pairs.jsonl"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "<repo-root>/runtime/model_cache/diffsynth_models")
    os.environ.setdefault("MODELSCOPE_CACHE", "<repo-root>/runtime/model_cache/modelscope")
    os.environ.setdefault("HF_HOME", "<repo-root>/runtime/model_cache/huggingface")

    with pairs_file.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"Mode: {args.mode}")
    print(f"Dataset: {dataset_root}")
    print(f"Pairs: {pairs_file} ({len(rows)} rows)")
    print(f"Output: {output_dir}")
    print("Loading pipeline...")
    pipe = load_pipeline(args)

    if args.mode == "ours":
        if not args.lora_path:
            raise ValueError("--lora-path is required for --mode ours")
        print(f"Loading LoRA: {args.lora_path}")
        load_lora(pipe, Path(args.lora_path), mode="ours")
    elif args.mode == "fal":
        print(f"Loading fal LoRA: {args.fal_lora_path}")
        load_lora(pipe, Path(args.fal_lora_path), mode="fal")
    else:
        print("Using base model without LoRA.")

    meta = []
    started = time.time()
    for idx, row in enumerate(rows, start=1):
        pair_id = str(row["pair_id"])
        out_name = f"{pair_id}.png"
        out_path = output_dir / out_name
        if out_path.exists() and not args.overwrite:
            print(f"[{idx}/{len(rows)}] skip existing {out_name}")
        else:
            src = Image.open(dataset_root / row["source_image"]).convert("RGB")
            output = pipe(
                prompt=row["instruction"],
                edit_image=[src],
                seed=args.seed,
                num_inference_steps=args.num_steps,
                height=src.size[1],
                width=src.size[0],
                edit_image_auto_resize=True,
                zero_cond_t=True,
            )
            image = first_image(output)
            image.save(out_path)
            elapsed = time.time() - started
            avg = elapsed / idx
            eta = avg * (len(rows) - idx)
            print(f"[{idx}/{len(rows)}] {pair_id} -> {out_name} ({avg:.1f}s/img, ETA {eta/60:.1f}m)")

        meta.append(
            {
                "pair_id": pair_id,
                "obj_id": row.get("obj_id"),
                "source_image": row.get("source_image"),
                "target_image": row.get("target_image"),
                "pred_image": out_name,
                "instruction": row.get("instruction"),
                "source_rotation_deg": row.get("source_rotation_deg"),
                "target_rotation_deg": row.get("target_rotation_deg"),
            }
        )

    with (output_dir / "eval_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Done: {len(meta)} predictions -> {output_dir}")


if __name__ == "__main__":
    main()
