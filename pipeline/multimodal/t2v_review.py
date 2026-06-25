from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ImageDraw, ImageSequence

from .t2i_constraints import constraints_for_prompt
from .t2i_constraint_review import run_t2i_constraint_review


def _pil_from_array(frame) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    return Image.fromarray(frame).convert("RGB")


def _extract_gif_frames(path: Path) -> list[Image.Image]:
    with Image.open(path) as img:
        return [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(img)]


def _extract_imageio_frames(path: Path) -> list[Image.Image]:
    try:
        import imageio.v3 as iio

        return [_pil_from_array(frame) for frame in iio.imiter(path)]
    except Exception:
        import imageio

        reader = imageio.get_reader(str(path))
        try:
            return [_pil_from_array(frame) for frame in reader]
        finally:
            reader.close()


def extract_keyframes(video_path: Path, frame_dir: Path, *, max_frames: int = 4) -> list[Path]:
    video_path = Path(video_path)
    frame_dir.mkdir(parents=True, exist_ok=True)
    if video_path.suffix.lower() == ".gif":
        frames = _extract_gif_frames(video_path)
    else:
        frames = _extract_imageio_frames(video_path)
    if not frames:
        raise ValueError(f"no frames decoded from video: {video_path}")
    count = min(max(1, max_frames), len(frames))
    if count == 1:
        indices = [0]
    else:
        indices = sorted({round(i * (len(frames) - 1) / (count - 1)) for i in range(count)})
    paths = []
    for out_idx, frame_idx in enumerate(indices):
        out_path = frame_dir / f"frame_{out_idx:02d}_src{frame_idx:04d}.png"
        frames[frame_idx].save(out_path)
        paths.append(out_path)
    return paths


def make_contact_sheet(frame_paths: list[Path], out_path: Path) -> Path:
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    if not images:
        raise ValueError("cannot build contact sheet without frames")
    thumb_w = 320
    pad = 12
    thumbs = []
    for img in images:
        img.thumbnail((thumb_w, thumb_w), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb_w, thumb_w), (245, 245, 242))
        tile.paste(img, ((thumb_w - img.width) // 2, (thumb_w - img.height) // 2))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, thumb_w - 1, thumb_w - 1), outline=(55, 60, 64), width=2)
        thumbs.append(tile)
    width = len(thumbs) * thumb_w + (len(thumbs) + 1) * pad
    height = thumb_w + 2 * pad
    sheet = Image.new("RGB", (width, height), (238, 238, 235))
    for idx, tile in enumerate(thumbs):
        sheet.paste(tile, (pad + idx * (thumb_w + pad), pad))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def _video_constraints(original_prompt: str, constraints: Optional[Iterable[dict]]) -> list[dict]:
    base = constraints_for_prompt(original_prompt, constraints)
    existing = {str(row.get("constraint_id")) for row in base if isinstance(row, dict)}
    extra = [
        {
            "constraint_id": "video_prompt_alignment",
            "type": "prompt_alignment",
            "description": "The keyframes collectively show the requested subject, setting, and action from the prompt.",
        },
        {
            "constraint_id": "video_temporal_coherence",
            "type": "temporal_coherence",
            "description": "The subject identity, scene layout, lighting, and camera viewpoint remain coherent across frames.",
        },
        {
            "constraint_id": "video_common_sense_motion",
            "type": "physical_plausibility",
            "description": "The visible motion is physically plausible and follows common sense for the requested scene.",
        },
        {
            "constraint_id": "video_artifact_quality",
            "type": "artifact_quality",
            "description": "The frames avoid severe flicker, warped geometry, unusable blur, text artifacts, logos, or watermarks.",
        },
    ]
    return base + [row for row in extra if row["constraint_id"] not in existing]


def run_t2v_keyframe_review(
    *,
    video_path: str,
    sample_id: str,
    round_idx: int,
    original_prompt: str,
    rewritten_prompt: str,
    constraints: Optional[Iterable[dict]] = None,
    output_dir: str,
    device: str = "cuda:0",
    max_retries: int = 2,
    vlm_model_path: Optional[str] = None,
    enable_thinking: bool = False,
) -> dict:
    output = Path(output_dir)
    frames_dir = output / "frames" / sample_id
    contact_sheet = output / "contact_sheets" / f"{sample_id}_keyframes.png"
    normalized_constraints = _video_constraints(original_prompt, constraints)
    try:
        frame_paths = extract_keyframes(Path(video_path), frames_dir, max_frames=4)
        make_contact_sheet(frame_paths, contact_sheet)
    except Exception as exc:
        return {
            "schema_version": "t2v_keyframe_review_v1",
            "sample_id": sample_id,
            "round_idx": round_idx,
            "vlm_route": "needs_fix",
            "status": "frame_extraction_failed",
            "error": str(exc),
            "video_path": video_path,
            "constraint_checklist": normalized_constraints,
            "failed_video_constraints": ["video_decode"],
            "rewrite_suggestions": ["Regenerate a valid playable video file."],
        }

    review_prompt = (
        "This image is a contact sheet of keyframes extracted from a generated text-to-video sample. "
        "Judge the video by comparing these ordered frames against the original video prompt. "
        "Treat temporal coherence, common-sense motion, and prompt alignment as first-class constraints. "
        f"Original video prompt: {original_prompt}"
    )
    review = run_t2i_constraint_review(
        rgb_path=str(contact_sheet),
        sample_id=sample_id,
        round_idx=round_idx,
        original_prompt=review_prompt,
        rewritten_prompt=rewritten_prompt,
        constraints=normalized_constraints,
        device=device,
        max_retries=max_retries,
        vlm_model_path=vlm_model_path,
        enable_thinking=enable_thinking,
    )
    checklist = review.get("constraint_checklist") or []
    counts = {
        "pass": sum(1 for row in checklist if isinstance(row, dict) and row.get("status") == "pass"),
        "fail": sum(1 for row in checklist if isinstance(row, dict) and row.get("status") == "fail"),
        "uncertain": sum(1 for row in checklist if isinstance(row, dict) and row.get("status") == "uncertain"),
        "total": len(checklist),
    }
    pass_rate = counts["pass"] / counts["total"] if counts["total"] else 0.0
    review.update(
        {
            "schema_version": "t2v_keyframe_review_v1",
            "video_path": video_path,
            "keyframe_paths": [str(path) for path in frame_paths],
            "contact_sheet_path": str(contact_sheet),
            "video_checklist": checklist,
            "video_status_counts": counts,
            "video_pass_rate": pass_rate,
            "failed_video_constraints": review.get("failed_constraints", []),
            "uncertain_video_constraints": review.get("uncertain_constraints", []),
            "passed_video_constraints": review.get("passed_constraints", []),
            "review_backend": "t2v_keyframe_contact_sheet_vlm",
        }
    )
    return review
