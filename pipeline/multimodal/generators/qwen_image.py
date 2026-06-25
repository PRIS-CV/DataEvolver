from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List

from ..prompt_writer import build_prompts
from ..schemas import DatasetRequest, DatasetSample, relpath
from .base import GeneratorAdapter


class QwenImageGenerator(GeneratorAdapter):
    name = "qwen-image"

    def generate(self, request: DatasetRequest, output_root: Path) -> List[DatasetSample]:
        if not request.allow_model_inference or request.dry_run:
            raise RuntimeError("Qwen image inference requires --allow-model-inference and non-dry-run mode")
        if request.route != "t2i":
            raise ValueError("QwenImageGenerator only supports route=t2i")

        prompt_rows = build_prompts(request, request.num_samples)
        prompt_path = output_root / "model_inputs" / "t2i_prompts.json"
        image_dir = output_root / "samples" / "images"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        prompt_payload = [
            {
                "id": row["sample_id"],
                "name": row["sample_id"],
                "prompt": row["prompt"],
                "features": {},
            }
            for row in prompt_rows
        ]
        prompt_path.write_text(json.dumps(prompt_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        repo_root = Path(__file__).resolve().parents[3]
        stage2 = repo_root / "pipeline" / "stage2_t2i_generate.py"
        existed_before = {
            row["sample_id"]: (image_dir / f"{row['sample_id']}.png").exists()
            for row in prompt_rows
        }
        log_dir = output_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        if not (request.skip_existing and all(existed_before.values())):
            cmd = [
                sys.executable,
                str(stage2),
                "--prompts-path",
                str(prompt_path),
                "--output-dir",
                str(image_dir),
                "--device",
                request.device,
                "--height",
                str(request.height),
                "--width",
                str(request.width),
                "--steps",
                str(request.steps),
                "--cfg-scale",
                str(request.true_cfg_scale),
                "--seed-base",
                str(request.seed),
            ]
            if not request.skip_existing:
                cmd.append("--no-skip")
            env = None
            if request.model_path:
                import os

                env = dict(os.environ)
                env["QWEN_IMAGE_MODEL_PATH"] = request.model_path
            result = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, env=env, timeout=3600)
            (log_dir / "qwen_image_stage2.stdout.log").write_text(result.stdout or "", encoding="utf-8")
            (log_dir / "qwen_image_stage2.stderr.log").write_text(result.stderr or "", encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(f"stage2_t2i_generate.py failed with code {result.returncode}; see {log_dir}")
        else:
            (log_dir / "qwen_image_stage2.stdout.log").write_text("[multimodal] skipped existing outputs\n", encoding="utf-8")
            (log_dir / "qwen_image_stage2.stderr.log").write_text("", encoding="utf-8")

        samples: List[DatasetSample] = []
        for row in prompt_rows:
            out_path = image_dir / f"{row['sample_id']}.png"
            if not out_path.exists():
                raise FileNotFoundError(f"expected T2I output missing: {out_path}")
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
                        "backend": "pipeline.stage2_t2i_generate",
                        "model_name": request.model_name,
                        "model_path": request.model_path,
                        "device": request.device,
                        "height": request.height,
                        "width": request.width,
                        "steps": request.steps,
                        "cfg_scale": request.true_cfg_scale,
                        "original_prompt": row.get("original_prompt", row["prompt"]),
                        "rewritten_prompt": row.get("rewritten_prompt", row["prompt"]),
                        "constraints": row.get("constraints", []),
                        "tier": row.get("tier"),
                        "source_prompt_id": row.get("source_prompt_id"),
                        "rewrite_reason": row.get("rewrite_reason"),
                        "stage2_stdout_log": relpath(log_dir / "qwen_image_stage2.stdout.log", output_root),
                        "stage2_stderr_log": relpath(log_dir / "qwen_image_stage2.stderr.log", output_root),
                    },
                    vlm_review={
                        "status": "pending",
                        "enabled": False,
                        "reason": "VLM evaluation is deferred",
                    },
                )
            )
        return samples
