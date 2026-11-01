"""Environment configuration for the optional HYWorld backend."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROFILE_SCHEMA = "dataevolver.hyworld_environment.v1"


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_profile_path(profile_path: Path, value: str | None) -> Path | None:
    if not value:
        return None
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if expanded.is_absolute():
        return expanded
    return (profile_path.parent / expanded).resolve()


def load_profile(path: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    profile_path = Path(path).resolve()
    profile = load_json(profile_path)
    if profile.get("schema") not in (PROFILE_SCHEMA, "dataevolver.production.v1"):
        raise ValueError(f"Unsupported production profile schema: {profile.get('schema')}")
    return profile_path, profile


def build_runtime_environment(profile_path: Path, profile: dict[str, Any]) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        str(key): os.path.expandvars(str(value))
        for key, value in (profile.get("environment") or {}).items()
        if value is not None
    })
    offline = profile.get("offline") or {}
    environment.update({
        "HF_HUB_OFFLINE": "1" if offline.get("hf_hub_offline", True) else "0",
        "TRANSFORMERS_OFFLINE": "1" if offline.get("hf_hub_offline", True) else "0",
        "DIFFUSERS_OFFLINE": "1" if offline.get("hf_hub_offline", True) else "0",
    })
    # HYWorld uses HYWORLD_NO_MODEL_DOWNLOADS=1 to skip MoGe/ZIM/GD loading and
    # require precomputed depth. Production real-3D must load local cached
    # models while HF remains offline; only opt into skipping via the explicit
    # legacy flag.
    skip_hyworld_depth_models = bool(offline.get("skip_hyworld_depth_models", False))
    environment["HYWORLD_NO_MODEL_DOWNLOADS"] = "1" if skip_hyworld_depth_models else "0"

    workspace = resolve_profile_path(profile_path, profile.get("workspace_root"))
    paths = profile.get("paths") or {}
    env_path_map = {
        "hyworld_moge_model": ("HYWORLD_MOGE_MODEL_PATH", "MOGE_MODEL_PATH"),
        "hyworld_zim_model": "HYWORLD_ZIM_MODEL_PATH",
        "hyworld_grounding_dino_model": "HYWORLD_GROUNDING_DINO_MODEL_PATH",
        "hyworld_sam3_model": "HYWORLD_SAM3_MODEL_PATH",
        "hyworld_worldstereo_weights": "HYWORLD_WORLDSTEREO_PATH",
        "hyworld_wan_base_model": ("HYWORLD_WAN_BASE_MODEL", "WORLDSTEREO_BASE_MODEL_PATH"),
    }
    for key, env_names in env_path_map.items():
        candidate = resolve_profile_path(profile_path, paths.get(key))
        if candidate:
            if isinstance(env_names, str):
                env_names = (env_names,)
            for env_name in env_names:
                environment[env_name] = str(candidate)
    pythonpath_entries: list[str] = []
    if workspace:
        pythonpath_entries.append(str(workspace / "src"))
    for key in ("hyworld_source", "sam3_source", "hunyuan3d_source"):
        candidate = resolve_profile_path(profile_path, paths.get(key))
        if candidate and candidate.exists():
            pythonpath_entries.append(str(candidate))
            if key == "hyworld_source":
                pythonpath_entries.extend([
                    str(candidate / "hyworld2" / "panogen"),
                    str(candidate / "hyworld2" / "worldgen"),
                ])
    current_pythonpath = environment.get("PYTHONPATH")
    if current_pythonpath:
        pythonpath_entries.append(current_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath_entries))
    return environment
