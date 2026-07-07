from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from dataevolver.paths import DATAEVOLVER_ROOT


LOCAL_ROOT = Path(os.environ.get("DATAEVOLVER_LOCAL_ROOT", os.fspath(DATAEVOLVER_ROOT / "local")))
LOCAL_MODEL_ROOT = Path(os.environ.get("DATAEVOLVER_MODEL_ROOT", os.fspath(LOCAL_ROOT / "model_hub")))


def _local_model_path(name: str) -> str:
    return os.fspath(LOCAL_MODEL_ROOT / name)


MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "qwen-image-2512": {
        "route": "t2i",
        "env": "QWEN_IMAGE_MODEL_PATH",
        "default_path": _local_model_path("Qwen-Image-2512"),
    },
    "qwen-image-edit-2511": {
        "route": "edit",
        "env": "QWEN_IMAGE_EDIT_MODEL_PATH",
        "default_path": _local_model_path("Qwen-Image-Edit-2511"),
    },
    "dataevolver-blender-existing": {
        "route": "blender",
        "env": "DATAEVOLVER_BLENDER_ENTRYPOINT",
        "default_path": "src/dataevolver/cli/run_all.sh",
    },
    "wan2.1-t2v-1.3b": {
        "route": "t2v",
        "env": "WAN_T2V_MODEL_PATH",
        "default_path": _local_model_path("Wan2.1-T2V-1.3B-Diffusers"),
    },
    "t2v-placeholder": {
        "route": "t2v",
        "env": "DATAEVOLVER_T2V_MODEL_PATH",
        "default_path": _local_model_path("Wan2.1-T2V-1.3B-Diffusers"),
    },
}

DEFAULT_MODEL_BY_ROUTE = {
    "t2i": "qwen-image-2512",
    "edit": "qwen-image-edit-2511",
    "blender": "dataevolver-blender-existing",
    "t2v": "wan2.1-t2v-1.3b",
}


def resolve_model(route: str, model_name: Optional[str], model_path: Optional[str]) -> dict:
    name = (model_name or DEFAULT_MODEL_BY_ROUTE[route]).strip()
    spec = dict(MODEL_REGISTRY.get(name, {}))
    if spec and spec.get("route") != route:
        raise ValueError(f"model '{name}' is registered for route={spec.get('route')}, not route={route}")

    env_name = spec.get("env")
    resolved_path = model_path or (os.environ.get(env_name) if env_name else None) or spec.get("default_path") or ""
    return {
        "model_name": name,
        "model_path": resolved_path,
        "registry_hit": bool(spec),
        "env": env_name,
    }
