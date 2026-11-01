"""Typed, backend-neutral model requests; secrets remain environment references."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar
from urllib.parse import urlsplit

from dataevolver.dataset.contracts import ArtifactRef, Contract, canonical_json, identifier, positive_int


def origin(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("model endpoint must be an HTTP(S) URL without credentials, query or fragment")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    host = f"[{parsed.hostname.lower()}]" if ":" in parsed.hostname else parsed.hostname.lower()
    return f"{parsed.scheme.lower()}://{host}:{port}"


@dataclass(frozen=True)
class ModelProfile(Contract):
    schema: ClassVar[str] = "dataevolver.model_profile.v1"
    profile_id: str
    backend: str
    model: str
    revision: str
    capabilities: tuple[str, ...] = ("json",)
    endpoint_env: str = ""
    api_key_env: str = ""
    model_env: str = ""
    mock_output: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        identifier(self.profile_id, "profile_id")
        if self.backend not in {"mock", "local_endpoint", "local_python", "external_api"}:
            raise ValueError("unregistered model backend")
        if not self.model or not self.revision or not set(self.capabilities).issubset({"json", "vision"}) or "json" not in self.capabilities:
            raise ValueError("model identity, revision and supported capabilities are required")
        if self.backend == "local_endpoint" and not self.endpoint_env:
            raise ValueError("local endpoint requires endpoint_env")
        for value in (self.endpoint_env, self.api_key_env, self.model_env):
            if value:
                identifier(value, "environment variable name")
        if not isinstance(self.mock_output, dict):
            raise ValueError("mock_output must be an object")
        canonical_json(self.mock_output)


@dataclass(frozen=True)
class ModelPolicy(Contract):
    schema: ClassVar[str] = "dataevolver.model_policy.v1"
    bindings: dict[str, str]
    max_calls: int | None
    allowed_local_origins: tuple[str, ...] = ()
    external_api_enabled: bool = False
    usage_mode: str = "bounded"

    def __post_init__(self):
        if not isinstance(self.bindings, dict) or not self.bindings:
            raise ValueError("model role bindings are required")
        for key, value in self.bindings.items():
            identifier(key, "model role")
            identifier(value, "profile binding")
        if self.usage_mode == "bounded":
            positive_int(self.max_calls, "max_calls")
        elif self.usage_mode != "accounting" or self.max_calls is not None:
            raise ValueError("accounting mode requires an explicit null call ceiling")
        if self.external_api_enabled is not False:
            raise ValueError("external_api backend is disabled")
        for value in self.allowed_local_origins:
            if origin(value) != value:
                raise ValueError("allowed origins must be canonical scheme://host:port values")

    def to_dict(self):
        result = super().to_dict()
        # Existing bounded policies retain their exact serialized identity.
        if self.usage_mode == "bounded":
            result.pop("usage_mode")
        return result


@dataclass(frozen=True)
class ModelRequest(Contract):
    schema: ClassVar[str] = "dataevolver.model_request.v1"
    request_id: str
    role: str
    messages: tuple[dict[str, str], ...]
    response_schema: dict[str, Any]
    artifacts: tuple[ArtifactRef, ...] = ()
    max_tokens: int = 256
    timeout_seconds: int = 30
    max_attempts: int = 1
    seed: int = 0

    def __post_init__(self):
        identifier(self.request_id, "request_id")
        identifier(self.role, "role")
        if not self.messages:
            raise ValueError("model messages are required")
        for message in self.messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"} or
                    message["role"] not in {"system", "user", "assistant"} or not isinstance(message["content"], str)):
                raise ValueError("messages require a supported role and text content")
        if not isinstance(self.response_schema, dict) or self.response_schema.get("type") != "object":
            raise ValueError("structured responses require a JSON object schema")
        if any(not isinstance(x, ArtifactRef) for x in self.artifacts):
            raise ValueError("model media must use ArtifactRef, not arbitrary URLs")
        for name in ("max_tokens", "timeout_seconds", "max_attempts"):
            positive_int(getattr(self, name), name)
        positive_int(self.seed, "seed", zero=True)
        canonical_json(self.messages)
        canonical_json(self.response_schema)

    @classmethod
    def from_dict(cls, payload):
        values = cls.read_fields(payload)
        values["artifacts"] = tuple(ArtifactRef.from_dict(x) for x in values.get("artifacts", []))
        return cls(**values)

    def to_dict(self):
        result = super().to_dict()
        result["artifacts"] = [x.to_dict() for x in self.artifacts]
        return result


@dataclass(frozen=True)
class ModelReceipt(Contract):
    schema: ClassVar[str] = "dataevolver.model_receipt.v1"
    request_id: str
    request_hash: str
    profile_hash: str
    model: str
    revision: str
    backend: str
    output: dict[str, Any]
    usage: dict[str, int | float | None]
    finish_reason: str
    provider_request_id: str = ""
    reused: bool = False

    def __post_init__(self):
        identifier(self.request_id, "request_id")
        for name in ("request_hash", "profile_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a SHA-256 digest")
        if not self.model or not self.revision or self.backend not in {"mock", "local_endpoint", "local_python"}:
            raise ValueError("verified receipt requires a bound local model identity")
        if not isinstance(self.output, dict) or not isinstance(self.usage, dict):
            raise ValueError("model output and usage must be objects")
        if set(self.usage) - {"input_tokens", "output_tokens", "images", "cost", "wall_seconds"}:
            raise ValueError("unknown model usage fields")
        for name in ("input_tokens", "output_tokens", "images"):
            value = self.usage.get(name)
            if value is not None:
                positive_int(value, name, zero=True)
        if self.finish_reason != "stop":
            raise ValueError("only completed, non-truncated outputs can be verified")
        for value in self.usage.values():
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0):
                raise ValueError("model usage must be non-negative or unknown")
        canonical_json(self.output)
        canonical_json(self.usage)

    def without_reuse_hash(self):
        payload = self.to_dict()
        payload["reused"] = False
        from dataevolver.dataset.contracts import payload_sha256
        return payload_sha256(payload)
