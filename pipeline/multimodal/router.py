from __future__ import annotations

from .schemas import DatasetRequest, normalize_route
from .model_registry import resolve_model


ROUTE_DESCRIPTIONS = {
    "t2i": "text prompt to image dataset",
    "edit": "input image plus edit instruction to edited image dataset",
    "blender": "existing Blender / 3D dataset route",
    "t2v": "text prompt to video dataset",
}


def build_request(
    *,
    route: str,
    user_request: str,
    output_root: str,
    dry_run: bool,
    allow_model_inference: bool,
    num_samples: int,
    input_image_dir: str | None,
    prompt_file: str | None,
    generator: str,
    model_name: str | None,
    model_path: str | None,
    device: str,
    height: int,
    width: int,
    steps: int,
    num_frames: int,
    fps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    negative_prompt: str,
    seed: int,
    sample_prefix: str,
    skip_existing: bool,
    validate_outputs: bool,
    vlm_eval: bool,
    vlm_device: str,
    vlm_model_path: str | None,
    vlm_max_samples: int | None,
    vlm_max_retries: int,
    vlm_review_mode: str,
    vlm_use_freeform_first: bool,
    vlm_enable_thinking: bool,
) -> DatasetRequest:
    normalized = normalize_route(route)
    model = resolve_model(normalized, model_name, model_path)
    notes = [f"route={normalized}: {ROUTE_DESCRIPTIONS[normalized]}"]
    notes.append(f"model={model['model_name']} path={model['model_path'] or '<unset>'}")
    if normalized == "edit" and not input_image_dir:
        raise ValueError("--input-image-dir is required for route=edit")
    if prompt_file:
        notes.append(f"prompt_file={prompt_file}")
    if normalized == "t2v":
        notes.append("t2v uses a text-to-video generator plus keyframe VLM review loop")
    if not dry_run and not allow_model_inference:
        notes.append("real model inference requested but blocked until --allow-model-inference is set")
    elif not dry_run:
        notes.append("real model inference is enabled through the selected adapter")
    return DatasetRequest(
        route=normalized,
        user_request=user_request.strip(),
        output_root=output_root,
        dry_run=dry_run,
        allow_model_inference=allow_model_inference,
        num_samples=max(1, int(num_samples)),
        input_image_dir=input_image_dir,
        prompt_file=prompt_file,
        generator=generator,
        model_name=model["model_name"],
        model_path=model["model_path"],
        device=device,
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
        fps=int(fps),
        steps=int(steps),
        true_cfg_scale=float(true_cfg_scale),
        guidance_scale=float(guidance_scale),
        negative_prompt=(negative_prompt or "").strip(),
        seed=int(seed),
        sample_prefix=sample_prefix,
        skip_existing=bool(skip_existing),
        validate_outputs=bool(validate_outputs),
        vlm_eval=bool(vlm_eval),
        vlm_device=vlm_device,
        vlm_model_path=vlm_model_path,
        vlm_max_samples=vlm_max_samples,
        vlm_max_retries=max(1, int(vlm_max_retries)),
        vlm_review_mode=vlm_review_mode,
        vlm_use_freeform_first=bool(vlm_use_freeform_first),
        vlm_enable_thinking=bool(vlm_enable_thinking),
        notes=notes,
    )
