from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


SUPPORTED_ROUTES = {"t2i", "edit", "blender", "t2v"}


@dataclass
class DatasetRequest:
    route: str
    user_request: str
    output_root: str
    dry_run: bool
    allow_model_inference: bool = False
    num_samples: int = 4
    input_image_dir: Optional[str] = None
    prompt_file: Optional[str] = None
    generator: str = "dryrun"
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    device: str = "cuda:0"
    height: int = 512
    width: int = 512
    num_frames: int = 49
    fps: int = 16
    steps: int = 30
    true_cfg_scale: float = 4.0
    guidance_scale: float = 1.0
    negative_prompt: str = ""
    seed: int = 11
    sample_prefix: str = "sample"
    skip_existing: bool = True
    validate_outputs: bool = True
    vlm_eval: bool = False
    vlm_device: str = "cuda:0"
    vlm_model_path: Optional[str] = None
    vlm_max_samples: Optional[int] = None
    vlm_max_retries: int = 1
    vlm_review_mode: str = "studio"
    vlm_use_freeform_first: bool = False
    vlm_enable_thinking: bool = False
    status: str = "planned"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSample:
    sample_id: str
    route: str
    prompt: str
    output_path: Optional[str]
    input_path: Optional[str] = None
    edit_instruction: Optional[str] = None
    generator: str = "dryrun"
    status: str = "generated"
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    vlm_review: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_route(route: str) -> str:
    normalized = (route or "").strip().lower().replace("-", "_")
    aliases = {
        "text_to_image": "t2i",
        "text2image": "t2i",
        "image_edit": "edit",
        "i2i_edit": "edit",
        "t_i_to_i": "edit",
        "3d": "blender",
        "blender_3d": "blender",
        "text_to_video": "t2v",
        "text2video": "t2v",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_ROUTES:
        raise ValueError(f"Unsupported route '{route}'. Expected one of: {sorted(SUPPORTED_ROUTES)}")
    return normalized


def relpath(path: Optional[Path], root: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
