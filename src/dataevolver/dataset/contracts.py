"""Source-neutral contracts extracted from the local DataProcess primitives.

Versioned parsing is deliberately stricter than the legacy ContractMixin:
unknown fields, non-finite numbers and unbound artifact paths fail closed.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Mapping


def canonical_json(value: Any) -> str:
    """Encode strict dataset contracts; reject non-finite numbers."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identifier(value: str, name: str = "identifier") -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"{name} must be a non-empty path-safe identifier")


def positive_int(value: int, name: str, *, zero: bool = False) -> None:
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'non-negative' if zero else 'positive'} integer")


class Contract:
    schema: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json({"schema_version": self.schema, **asdict(self)}))

    def compute_hash(self) -> str:
        return payload_sha256(self.to_dict())

    @classmethod
    def read_fields(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping) or payload.get("schema_version") != cls.schema:
            raise ValueError(f"expected schema {cls.schema}")
        allowed = {item.name for item in fields(cls)}
        unknown = set(payload) - allowed - {"schema_version"}
        if unknown:
            raise ValueError(f"unknown {cls.__name__} fields: {', '.join(sorted(unknown))}")
        canonical_json(payload)
        return {name: value for name, value in payload.items() if name != "schema_version"}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]):
        return cls(**cls.read_fields(payload))


@dataclass(frozen=True)
class ArtifactRef(Contract):
    schema: ClassVar[str] = "dataevolver.artifact_ref.v1"
    store: str
    path: str
    sha256: str
    media_type: str = "application/json"

    def __post_init__(self):
        identifier(self.store, "artifact store")
        path = PurePosixPath(self.path)
        if (not self.path or path.is_absolute() or ".." in path.parts or
                "\\" in self.path or ":" in self.path or str(path) != self.path or self.path == "."):
            raise ValueError("artifact path must be a normalized relative POSIX path")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("artifact sha256 must contain 64 lowercase hex digits")
        if not isinstance(self.media_type, str) or "/" not in self.media_type:
            raise ValueError("artifact media_type is required")


class ArtifactStore:
    """Explicit stores; no implicit runtime discovery or source imports."""

    def __init__(self, roots: Mapping[str, str | Path]):
        self.roots = {key: Path(value).expanduser().resolve() for key, value in roots.items()}
        for key in self.roots:
            identifier(key, "artifact store")

    def resolve(self, ref: ArtifactRef) -> Path:
        if ref.store not in self.roots:
            raise ValueError("artifact store is not registered")
        root = self.roots[ref.store]
        path = (root / ref.path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("artifact escapes its registered store")
        if not path.is_file() or file_sha256(path) != ref.sha256:
            raise ValueError("artifact is missing or its hash changed")
        return path
