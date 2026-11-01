"""Multimodal workload library used by registered Harness operators."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List


from dataevolver.workflows.multimodal.generators.dryrun import DryRunGenerator
from dataevolver.workflows.multimodal.schemas import DatasetRequest, DatasetSample
from dataevolver.workflows.multimodal.validation import validate_image, validate_video
from dataevolver.workflows.multimodal.vlm_review import run_vlm_reviews




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


if __name__ == "__main__":
    raise SystemExit("Use python -m dataevolver; the standalone multimodal CLI is retired.")
