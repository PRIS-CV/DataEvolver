"""Shared filesystem paths for the src-based DataEvolver layout."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent
CONFIGS_ROOT = REPO_ROOT / "configs"
DATAEVOLVER_ROOT = REPO_ROOT / ".dataevolver"
RUNTIME_ROOT = Path(os.environ.get("DATAEVOLVER_RUNTIME_ROOT", DATAEVOLVER_ROOT / "runtime"))


def runtime_path(*parts: str) -> Path:
    """Return a path under the local DataEvolver runtime directory."""

    return RUNTIME_ROOT.joinpath(*parts)


def repo_path(*parts: str) -> Path:
    """Return a path under the repository root."""

    return REPO_ROOT.joinpath(*parts)
