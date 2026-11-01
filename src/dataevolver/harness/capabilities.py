from __future__ import annotations

import os
from pathlib import Path

from .models import Capability, canonical_json


class CapabilityStore:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.failures_path = self.path.with_name("capability_failures.jsonl")

    @staticmethod
    def _append_line(path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def list_capabilities(self) -> tuple[Capability, ...]:
        if not self.path.exists():
            return ()
        capabilities = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                import json

                capabilities.append(Capability.from_dict(json.loads(line)))
        return tuple(capabilities)

    def append_candidate(self, capability: Capability) -> bool:
        for existing in self.list_capabilities():
            if existing.capability_id == capability.capability_id:
                return False
            if capability.novelty_key and (
                existing.novelty_key == capability.novelty_key
                and existing.compatibility_hash == capability.compatibility_hash
            ):
                return False
        self._append_line(self.path, capability)
        return True

    def _known_failure_keys(self, compatibility_hash: str) -> set[str]:
        if not self.failures_path.exists():
            return set()
        import json

        return {
            str(payload.get("novelty_key") or "")
            for line in self.failures_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for payload in (json.loads(line),)
            if payload.get("compatibility_hash") == compatibility_hash
        }

    def query(
        self,
        context_signature: str,
        failure_code: str,
        compatibility_hash: str,
        *,
        attempted: tuple[str, ...] | set[str] = (),
        limit: int = 5,
    ) -> tuple[Capability, ...]:
        excluded = set(attempted) | self._known_failure_keys(compatibility_hash)
        matches = [
            item
            for item in self.list_capabilities()
            if item.context_signature == context_signature
            and item.failure_signature.code == failure_code
            and item.compatibility_hash == compatibility_hash
            and item.novelty_key not in excluded
        ]
        matches.sort(
            key=lambda item: (
                min(item.protected_metric_delta.values(), default=0.0) >= 0.0,
                sum(item.metric_delta.values()),
                item.capability_id,
            ),
            reverse=True,
        )
        return tuple(matches[: max(0, limit)])

    def mark_known_failure(
        self,
        novelty_key: str,
        *,
        asset_id: str,
        compatibility_hash: str,
        reason: str,
    ) -> bool:
        payload = {
            "schema_version": "dataevolver.harness.capability_failure.v1",
            "novelty_key": novelty_key,
            "asset_id": asset_id,
            "compatibility_hash": compatibility_hash,
            "reason": reason,
        }
        if self.failures_path.exists():
            import json

            for line in self.failures_path.read_text(encoding="utf-8").splitlines():
                if line.strip() and json.loads(line) == payload:
                    return False
        self._append_line(self.failures_path, payload)
        return True
