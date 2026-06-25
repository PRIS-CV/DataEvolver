from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

from ..prompt_writer import build_prompts
from ..schemas import DatasetRequest, DatasetSample, relpath
from .base import GeneratorAdapter


class WanT2VGenerator(GeneratorAdapter):
    name = "wan-t2v"

    def generate(self, request: DatasetRequest, output_root: Path) -> List[DatasetSample]:
        if not request.allow_model_inference or request.dry_run:
            raise RuntimeError("Wan T2V inference requires --allow-model-inference and non-dry-run mode")
        if request.route != "t2v":
            raise ValueError("WanT2VGenerator only supports route=t2v")
        if not request.model_path:
            raise ValueError("Wan T2V requires --model-path or WAN_T2V_MODEL_PATH")

        prompt_rows = build_prompts(request, request.num_samples)
        prompt_path = output_root / "model_inputs" / "t2v_prompts.json"
        video_dir = output_root / "samples" / "videos"
        log_dir = output_root / "logs"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        prompt_payload = [
            {
                "id": row["sample_id"],
                "prompt": row["prompt"],
                "negative_prompt": row.get("negative_prompt", request.negative_prompt),
            }
            for row in prompt_rows
        ]
        prompt_path.write_text(json.dumps(prompt_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        existed_before = {
            row["sample_id"]: (video_dir / f"{row['sample_id']}.mp4").exists()
            for row in prompt_rows
        }
        if request.skip_existing and all(existed_before.values()):
            (log_dir / "wan_t2v.log").write_text("[multimodal] skipped existing outputs\n", encoding="utf-8")
            return self._samples(request, output_root, prompt_rows, video_dir, log_dir, existed_before)

        import torch
        from diffusers import WanPipeline
        from diffusers.utils import export_to_video

        model_path = Path(request.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Wan model path not found: {model_path}")

        log_lines = [
            f"model_path={model_path}",
            f"device={request.device}",
            f"height={request.height} width={request.width} frames={request.num_frames}",
            f"steps={request.steps} guidance_scale={request.guidance_scale}",
        ]
        pipe = WanPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        if request.device.startswith("cuda") and os.environ.get("WAN_T2V_CPU_OFFLOAD") == "1":
            pipe.enable_model_cpu_offload()
            log_lines.append("offload=model_cpu_offload")
        else:
            pipe = pipe.to(request.device)
            log_lines.append(f"pipeline_to={request.device}")

        for idx, row in enumerate(prompt_rows):
            sample_id = row["sample_id"]
            out_path = video_dir / f"{sample_id}.mp4"
            if request.skip_existing and out_path.exists():
                log_lines.append(f"skip_existing={out_path}")
                continue
            seed = int(request.seed) + idx
            generator_device = request.device if request.device.startswith("cuda") else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(seed)
            negative_prompt = str(row.get("negative_prompt") or request.negative_prompt or "")
            result = pipe(
                prompt=row["prompt"],
                negative_prompt=negative_prompt or None,
                height=request.height,
                width=request.width,
                num_frames=request.num_frames,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance_scale,
                generator=generator,
                output_type="pil",
            )
            frames = result.frames[0]
            export_to_video(frames, str(out_path), fps=request.fps)
            log_lines.append(f"generated={out_path} seed={seed} frames={len(frames)}")

        (log_dir / "wan_t2v.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return self._samples(request, output_root, prompt_rows, video_dir, log_dir, existed_before)

    def _samples(
        self,
        request: DatasetRequest,
        output_root: Path,
        prompt_rows: List[dict],
        video_dir: Path,
        log_dir: Path,
        existed_before: dict,
    ) -> List[DatasetSample]:
        samples: List[DatasetSample] = []
        for row in prompt_rows:
            out_path = video_dir / f"{row['sample_id']}.mp4"
            if not out_path.exists():
                raise FileNotFoundError(f"expected T2V output missing: {out_path}")
            samples.append(
                DatasetSample(
                    sample_id=row["sample_id"],
                    route=request.route,
                    prompt=row["prompt"],
                    output_path=relpath(out_path, output_root),
                    generator=self.name,
                    status="skipped_existing" if existed_before.get(row["sample_id"]) and request.skip_existing else "generated",
                    metadata={
                        "dry_run": False,
                        "backend": "diffusers.WanPipeline",
                        "model_name": request.model_name,
                        "model_path": request.model_path,
                        "device": request.device,
                        "height": request.height,
                        "width": request.width,
                        "num_frames": request.num_frames,
                        "fps": request.fps,
                        "steps": request.steps,
                        "guidance_scale": request.guidance_scale,
                        "negative_prompt": row.get("negative_prompt", request.negative_prompt),
                        "seed": request.seed,
                        "original_prompt": row.get("original_prompt", row["prompt"]),
                        "rewritten_prompt": row.get("rewritten_prompt", row["prompt"]),
                        "constraints": row.get("constraints", []),
                        "tier": row.get("tier"),
                        "source_prompt_id": row.get("source_prompt_id"),
                        "rewrite_reason": row.get("rewrite_reason"),
                        "wan_log": relpath(log_dir / "wan_t2v.log", output_root),
                    },
                    vlm_review={
                        "status": "pending",
                        "enabled": False,
                        "reason": "VLM evaluation is deferred",
                    },
                )
            )
        return samples
