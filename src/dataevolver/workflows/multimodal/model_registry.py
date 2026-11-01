"""Resolve explicit workload bindings without choosing a provider or checkpoint."""
from __future__ import annotations

import os


def resolve_model(route: str, model_name: str | None, model_path: str | None) -> dict:
    prefix = "DATAEVOLVER_" + route.upper()
    return {
        "model_name": (model_name or os.environ.get(prefix + "_MODEL", "")).strip(),
        "model_path": model_path or os.environ.get(prefix + "_MODEL_PATH", ""),
        "registry_hit": False,
        "env": prefix + "_MODEL",
    }
