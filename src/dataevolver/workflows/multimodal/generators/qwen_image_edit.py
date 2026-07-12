from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List

from dataevolver.paths import DATAEVOLVER_ROOT

from ..prompt_writer import build_prompts
from ..schemas import DatasetRequest, DatasetSample, relpath
from .base import GeneratorAdapter
from .dryrun import iter_images


LOCAL_ROOT = Path(os.environ.get("DATAEVOLVER_LOCAL_ROOT", os.fspath(DATAEVOLVER_ROOT / "local")))
LOCAL_MODEL_ROOT = Path(os.environ.get("DATAEVOLVER_MODEL_ROOT", os.fspath(LOCAL_ROOT / "model_hub")))
DEFAULT_QWEN_IMAGE_EDIT_PATH = os.fspath(LOCAL_MODEL_ROOT / "Qwen-Image-Edit-2511")


class QwenImageEditGenerator(GeneratorAdapter):
    name = "qwen-image-edit"

    def generate(self, request: DatasetRequest, output_root: Path) -> List[DatasetSample]:
        if not request.allow_model_inference or request.dry_run:
            raise RuntimeError("Qwen image edit inference requires --allow-model-inference and non-dry-run mode")
        if request.route != "edit":
            raise ValueError("QwenImageEditGenerator only supports route=edit")

        input_dir = Path(request.input_image_dir or "")
        image_paths = list(iter_images(input_dir))[: request.num_samples]
        if not image_paths:
            raise FileNotFoundError(f"no images found in input image directory: {input_dir}")

        samples: List[DatasetSample] = []
        prompt_rows = build_prompts(request, len(image_paths))
        planned = []
        for idx, (prompt_row, src) in enumerate(zip(prompt_rows, image_paths)):
            sample_id = prompt_row["sample_id"]
            out_path = output_root / "samples" / "edited" / f"{sample_id}.png"
            planned.append((idx, prompt_row, src, out_path, out_path.exists()))

        if request.skip_existing and all(item[4] for item in planned):
            for idx, prompt_row, src, out_path, existed in planned:
                samples.append(
                    DatasetSample(
                        sample_id=prompt_row["sample_id"],
                        route=request.route,
                        prompt=prompt_row["prompt"],
                        input_path=str(src),
                        output_path=relpath(out_path, output_root),
                        edit_instruction=prompt_row["edit_instruction"],
                        generator=self.name,
                        status="skipped_existing",
                        metadata={
                            "dry_run": False,
                            "backend": "diffusers.QwenImageEditPlusPipeline",
                            "model_name": request.model_name,
                            "model_path": request.model_path
                            or os.environ.get("QWEN_IMAGE_EDIT_MODEL_PATH", DEFAULT_QWEN_IMAGE_EDIT_PATH),
                            "device": request.device,
                            "height": request.height,
                            "width": request.width,
                            "steps": request.steps,
                            "elapsed_seconds": 0.0,
                            "resume_skip": True,
                            "original_edit_instruction": prompt_row.get(
                                "original_edit_instruction",
                                prompt_row["edit_instruction"],
                            ),
                            "model_edit_instruction": prompt_row["edit_instruction"],
                            "constraints": prompt_row.get("constraints", []),
                            "tier": prompt_row.get("tier"),
                            "source_prompt_id": prompt_row.get("source_prompt_id"),
                            "rewrite_reason": prompt_row.get("rewrite_reason"),
                        },
                        vlm_review={
                            "status": "pending",
                            "enabled": False,
                            "reason": "VLM evaluation is deferred",
                        },
                    )
                )
            return samples

        import torch
        from PIL import Image
        from diffusers import QwenImageEditPlusPipeline

        model_path = request.model_path or os.environ.get("QWEN_IMAGE_EDIT_MODEL_PATH", DEFAULT_QWEN_IMAGE_EDIT_PATH)
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipe.to(request.device)
        pipe.set_progress_bar_config(disable=None)

        for idx, prompt_row, src, out_path, existed in planned:
            sample_id = prompt_row["sample_id"]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            start = time.time()
            if not (request.skip_existing and existed):
                image = Image.open(src).convert("RGB")
                generator = torch.Generator(device=request.device).manual_seed(request.seed + idx)
                with torch.inference_mode():
                    output = pipe(
                        image=image,
                        prompt=prompt_row["edit_instruction"],
                        generator=generator,
                        true_cfg_scale=request.true_cfg_scale,
                        negative_prompt=" ",
                        num_inference_steps=request.steps,
                        guidance_scale=request.guidance_scale,
                        height=request.height,
                        width=request.width,
                        num_images_per_prompt=1,
                    )
                output.images[0].save(out_path)
            samples.append(
                DatasetSample(
                    sample_id=sample_id,
                    route=request.route,
                    prompt=prompt_row["prompt"],
                    input_path=str(src),
                    output_path=relpath(out_path, output_root),
                    edit_instruction=prompt_row["edit_instruction"],
                    generator=self.name,
                    status="skipped_existing" if existed and request.skip_existing else "generated",
                    metadata={
                        "dry_run": False,
                        "backend": "diffusers.QwenImageEditPlusPipeline",
                        "model_name": request.model_name,
                        "model_path": model_path,
                        "device": request.device,
                        "height": request.height,
                        "width": request.width,
                        "steps": request.steps,
                        "elapsed_seconds": round(time.time() - start, 3),
                        "original_edit_instruction": prompt_row.get(
                            "original_edit_instruction",
                            prompt_row["edit_instruction"],
                        ),
                        "model_edit_instruction": prompt_row["edit_instruction"],
                        "constraints": prompt_row.get("constraints", []),
                        "tier": prompt_row.get("tier"),
                        "source_prompt_id": prompt_row.get("source_prompt_id"),
                        "rewrite_reason": prompt_row.get("rewrite_reason"),
                    },
                    vlm_review={
                        "status": "pending",
                        "enabled": False,
                        "reason": "VLM evaluation is deferred",
                    },
                )
            )

        del pipe
        if request.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return samples
