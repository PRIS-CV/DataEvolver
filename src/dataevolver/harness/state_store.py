from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .events import EventLog
from .models import CampaignState, canonical_json


class LeaseBusyError(RuntimeError):
    pass


class CampaignLease:
    def __init__(self, path: str | os.PathLike[str], *, ttl_seconds: int):
        self.path = Path(path)
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.token = str(uuid.uuid4())
        self._held = False

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if os.name == "nt":
            import ctypes

            process_query_limited_information = 0x1000
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
            if not handle:
                return False
            kernel32.CloseHandle(handle)
            return True
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError, ValueError):
            return False

    def _existing_is_stale(self) -> bool:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True
        if float(payload.get("expires_at", 0)) <= time.time():
            return True
        return (
            payload.get("hostname") == socket.gethostname()
            and not self._pid_alive(int(payload.get("pid", -1)))
        )

    def acquire(self) -> "CampaignLease":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            payload = {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "token": self.token,
                "expires_at": time.time() + self.ttl_seconds,
            }
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if not self._existing_is_stale():
                    raise LeaseBusyError(f"campaign lease is held: {self.path}")
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(canonical_json(payload) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._held = True
            return self
        raise LeaseBusyError(f"could not acquire campaign lease: {self.path}")

    def renew(self) -> None:
        if not self._held:
            raise RuntimeError("cannot renew a lease that is not held")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("token") != self.token:
            raise LeaseBusyError("campaign lease ownership changed")
        payload["expires_at"] = time.time() + self.ttl_seconds
        atomic_write_json(self.path, payload)

    def release(self) -> None:
        if not self._held:
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("token") == self.token:
                self.path.unlink()
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        self._held = False

    def __enter__(self) -> "CampaignLease":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
            newline="\n",
        ) as handle:
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


@dataclass(frozen=True)
class CampaignLayout:
    campaigns_root: Path
    campaign_id: str

    def __init__(self, campaigns_root: str | os.PathLike[str], campaign_id: str):
        object.__setattr__(self, "campaigns_root", Path(campaigns_root))
        object.__setattr__(self, "campaign_id", campaign_id)

    @property
    def campaign_dir(self) -> Path:
        return self.campaigns_root / self.campaign_id

    @property
    def goal_path(self) -> Path:
        return self.campaign_dir / "goal.json"

    @property
    def frozen_contract_path(self) -> Path:
        return self.campaign_dir / "frozen_contract.json"

    @property
    def events_path(self) -> Path:
        return self.campaign_dir / "events.jsonl"

    @property
    def state_path(self) -> Path:
        return self.campaign_dir / "state.json"

    @property
    def final_report_path(self) -> Path:
        return self.campaign_dir / "final_report.json"

    @property
    def lease_path(self) -> Path:
        return self.campaign_dir / ".controller.lease.json"

    @property
    def pause_request_path(self) -> Path:
        return self.campaign_dir / "pause.request.json"


class StateStore:
    def __init__(self, layout: CampaignLayout):
        self.layout = layout
        self.events = EventLog(layout.events_path, campaign_id=layout.campaign_id)

    def initialize_state(self, state: CampaignState) -> None:
        self.layout.campaign_dir.mkdir(parents=True, exist_ok=True)
        if not self.layout.state_path.exists():
            atomic_write_json(self.layout.state_path, state)

    def commit(
        self,
        state: CampaignState,
        event_type: str,
        payload: dict[str, Any],
        idempotency_key: str = "",
    ) -> CampaignState:
        existing = self.events.find_idempotency_key(idempotency_key)
        if existing is not None:
            return CampaignState.from_dict(existing.payload["state_after"])
        sequence = self.events.next_sequence()
        committed = state.with_updates(last_event_seq=sequence)
        event = self.events.append(
            event_type,
            {**payload, "state_after": committed.to_dict()},
            idempotency_key=idempotency_key,
        )
        if event.sequence != sequence:
            return CampaignState.from_dict(event.payload["state_after"])
        atomic_write_json(self.layout.state_path, committed)
        return committed

    def load_state(self) -> CampaignState:
        events = self.events.read()
        snapshot: CampaignState | None = None
        if self.layout.state_path.exists():
            try:
                snapshot = CampaignState.from_dict(json.loads(self.layout.state_path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                snapshot = None
        if events and (snapshot is None or snapshot.last_event_seq != events[-1].sequence):
            snapshot = CampaignState.from_dict(events[-1].payload["state_after"])
            atomic_write_json(self.layout.state_path, snapshot)
        if snapshot is None:
            raise FileNotFoundError(f"campaign state not found: {self.layout.state_path}")
        return snapshot

    def replay(self) -> CampaignState:
        events = self.events.read()
        if not events:
            return self.load_state()
        return CampaignState.from_dict(events[-1].payload["state_after"])
