from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import JsonModel, canonical_json


@dataclass(frozen=True)
class HarnessEvent(JsonModel):
    schema_version: str
    sequence: int
    event_id: str
    campaign_id: str
    event_type: str
    timestamp: str
    payload: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HarnessEvent":
        return cls(
            schema_version=str(payload["schema_version"]),
            sequence=int(payload["sequence"]),
            event_id=str(payload["event_id"]),
            campaign_id=str(payload["campaign_id"]),
            event_type=str(payload["event_type"]),
            timestamp=str(payload["timestamp"]),
            payload=dict(payload.get("payload") or {}),
            idempotency_key=str(payload.get("idempotency_key") or ""),
        )


class EventLog:
    def __init__(self, path: str | os.PathLike[str], *, campaign_id: str):
        self.path = Path(path)
        self.campaign_id = campaign_id

    def read(self) -> list[HarnessEvent]:
        if not self.path.exists():
            return []
        lines = self.path.read_bytes().splitlines()
        events: list[HarnessEvent] = []
        for index, raw in enumerate(lines):
            if not raw.strip():
                continue
            try:
                events.append(HarnessEvent.from_dict(json.loads(raw.decode("utf-8"))))
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                if index == len(lines) - 1:
                    break
                raise ValueError(f"invalid event JSONL at line {index + 1}")
        return events

    def find_idempotency_key(self, key: str) -> HarnessEvent | None:
        if not key:
            return None
        # shortcut: replay scans the event log; add an index when replay cost is measured as a bottleneck.
        return next((event for event in self.read() if event.idempotency_key == key), None)

    def next_sequence(self) -> int:
        events = self.read()
        return events[-1].sequence + 1 if events else 1

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str = "",
    ) -> HarnessEvent:
        existing = self.find_idempotency_key(idempotency_key)
        if existing is not None:
            return existing
        event = HarnessEvent(
            schema_version="dataevolver.harness.event.v1",
            sequence=self.next_sequence(),
            event_id=str(uuid.uuid4()),
            campaign_id=self.campaign_id,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload=payload,
            idempotency_key=idempotency_key,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(event) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event
