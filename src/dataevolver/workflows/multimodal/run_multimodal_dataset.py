#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dataevolver.workflows.multimodal.generators.dryrun import DryRunGenerator
from dataevolver.workflows.multimodal.router import ROUTE_DESCRIPTIONS, build_request
from dataevolver.workflows.multimodal.schemas import DatasetRequest, DatasetSample
from dataevolver.workflows.multimodal.validation import validate_image, validate_video
from dataevolver.workflows.multimodal.vlm_review import run_vlm_reviews


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multimodal DataEvolver dataset route")
    parser.add_argument("--route", required=True, choices=sorted(ROUTE_DESCRIPTIONS))
    parser.add_argument("--user-request", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Use local mock generation; no model inference")
    parser.add_argument("--allow-model-inference", action="store_true", help="Required for any real model adapter")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--input-image-dir", default=None)
    parser.add_argument("--prompt-file", default=None, help="Optional .txt/.json/.jsonl prompt or instruction list")
    parser.add_argument("--generator", default="dryrun", help="Generator adapter name. Only dryrun is local-safe now")
    parser.add_argument("--model-name", default=None, help="Registered or custom model name to record in metadata")
    parser.add_argument("--model-path", default=None, help="Cloud/local model path to record for later real adapters")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--sample-prefix", default="sample")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--validate-outputs", dest="validate_outputs", action="store_true", default=True)
    parser.add_argument("--no-validate-outputs", dest="validate_outputs", action="store_false")
    parser.add_argument("--vlm-eval", action="store_true", help="Run the existing DataEvolver VLM review on generated images")
    parser.add_argument("--vlm-device", default="cuda:0")
    parser.add_argument("--vlm-model-path", default=None)
    parser.add_argument("--vlm-max-samples", type=int, default=None)
    parser.add_argument("--vlm-max-retries", type=int, default=1)
    parser.add_argument("--vlm-review-mode", default="studio")
    parser.add_argument("--vlm-use-freeform-first", action="store_true")
    parser.add_argument("--vlm-enable-thinking", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_generator(request: DatasetRequest):
    if request.dry_run and request.generator == "dryrun":
        return DryRunGenerator()
    if not request.allow_model_inference:
        raise SystemExit("Real model adapters require --allow-model-inference.")
    if request.generator == "qwen-image-edit" and request.route == "edit":
        from dataevolver.workflows.multimodal.generators.qwen_image_edit import QwenImageEditGenerator

        return QwenImageEditGenerator()
    if request.generator == "qwen-image" and request.route == "t2i":
        from dataevolver.workflows.multimodal.generators.qwen_image import QwenImageGenerator

        return QwenImageGenerator()
    if request.generator in {"wan-t2v", "wan2.1-t2v-1.3b"} and request.route == "t2v":
        from dataevolver.workflows.multimodal.generators.wan_t2v import WanT2VGenerator

        return WanT2VGenerator()
    raise SystemExit(f"Unsupported generator={request.generator!r} for route={request.route!r}")


def run(request: DatasetRequest) -> List[DatasetSample]:
    output_root = Path(request.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "request.json", request.to_dict())

    generator = select_generator(request)
    samples = generator.generate(request, output_root)
    if request.validate_outputs:
        for sample in samples:
            output_path = output_root / sample.output_path if sample.output_path else None
            if request.route == "t2v":
                sample.validation = validate_video(
                    output_path,
                    expected_width=request.width,
                    expected_height=request.height,
                    expected_num_frames=request.num_frames,
                )
            else:
                sample.validation = validate_image(
                    output_path,
                    expected_width=request.width if request.route in {"t2i", "edit"} else None,
                    expected_height=request.height if request.route in {"t2i", "edit"} else None,
                )
    samples = run_vlm_reviews(request, output_root, samples)

    sample_rows = [sample.to_dict() for sample in samples]
    write_jsonl(output_root / "manifest.jsonl", sample_rows)
    write_json(
        output_root / "dataset_summary.json",
        {
            "schema": "dataevolver_multimodal_dataset_summary_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "route": request.route,
            "dry_run": request.dry_run,
            "generator": generator.name,
            "sample_count": len(samples),
            "status_counts": {
                status: sum(1 for sample in samples if sample.status == status)
                for status in sorted({sample.status for sample in samples})
            },
            "validation_counts": {
                status: sum(1 for sample in samples if sample.validation.get("status") == status)
                for status in sorted({sample.validation.get("status") for sample in samples if sample.validation})
            },
            "vlm_counts": {
                status: sum(1 for sample in samples if sample.vlm_review.get("status") == status)
                for status in sorted({sample.vlm_review.get("status") for sample in samples if sample.vlm_review})
            },
            "outputs": {
                "request": "request.json",
                "manifest": "manifest.jsonl",
            },
            "notes": request.notes,
        },
    )
    return samples


def main() -> None:
    args = parse_args()
    try:
        request = build_request(
            route=args.route,
            user_request=args.user_request,
            output_root=args.output_root,
            dry_run=args.dry_run,
            allow_model_inference=args.allow_model_inference,
            num_samples=args.num_samples,
            input_image_dir=args.input_image_dir,
            prompt_file=args.prompt_file,
            generator=args.generator,
            model_name=args.model_name,
            model_path=args.model_path,
            device=args.device,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            fps=args.fps,
            steps=args.steps,
            true_cfg_scale=args.true_cfg_scale,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            sample_prefix=args.sample_prefix,
            skip_existing=args.skip_existing,
            validate_outputs=args.validate_outputs,
            vlm_eval=args.vlm_eval,
            vlm_device=args.vlm_device,
            vlm_model_path=args.vlm_model_path,
            vlm_max_samples=args.vlm_max_samples,
            vlm_max_retries=args.vlm_max_retries,
            vlm_review_mode=args.vlm_review_mode,
            vlm_use_freeform_first=args.vlm_use_freeform_first,
            vlm_enable_thinking=args.vlm_enable_thinking,
        )
        samples = run(request)
    except Exception as exc:
        raise SystemExit(f"[multimodal] ERROR: {exc}") from exc

    print(
        f"[multimodal] route={request.route} dry_run={request.dry_run} "
        f"samples={len(samples)} output_root={request.output_root}"
    )


if __name__ == "__main__":
    main()
