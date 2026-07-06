"""Durable snapshots for HYWorld scene-construction intermediates."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode
except ModuleNotFoundError:  # pragma: no cover - local/unit envs may not install OTel.
    class _NoopSpan:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def set_attributes(self, *args, **kwargs):
            return None

        def record_exception(self, *args, **kwargs):
            return None

        def set_status(self, *args, **kwargs):
            return None

    class _NoopTracer:
        def start_as_current_span(self, *args, **kwargs):
            return _NoopSpan()

    class _NoopInstrument:
        def add(self, *args, **kwargs):
            return None

        def record(self, *args, **kwargs):
            return None

    class _NoopMeter:
        def create_counter(self, *args, **kwargs):
            return _NoopInstrument()

        def create_histogram(self, *args, **kwargs):
            return _NoopInstrument()

    class _NoopTrace:
        @staticmethod
        def get_tracer(*args, **kwargs):
            return _NoopTracer()

    class _NoopMetrics:
        @staticmethod
        def get_meter(*args, **kwargs):
            return _NoopMeter()

    class StatusCode:
        ERROR = "ERROR"

    class Status:
        def __init__(self, *args, **kwargs):
            pass

    metrics = _NoopMetrics()
    trace = _NoopTrace()


logger = logging.getLogger("dataevolver.hyworld_intermediates")
tracer = trace.get_tracer("dataevolver.hyworld_intermediates")
meter = metrics.get_meter("dataevolver.hyworld_intermediates")
captures = meter.create_counter("dataevolver.hyworld.intermediate_captures", unit="1")
captured_files = meter.create_counter("dataevolver.hyworld.intermediate_files", unit="1")
captured_bytes = meter.create_counter("dataevolver.hyworld.intermediate_bytes", unit="By")
capture_duration = meter.create_histogram("dataevolver.hyworld.intermediate_capture_duration", unit="s")

SCHEMA_VERSION = "dataevolver.hyworld.intermediates.v1"
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _iter_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    matches: set[Path] = set()
    if not root.exists():
        return []
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                matches.add(path.resolve())
    return sorted(matches)


def _copy_with_sha256(source: Path, destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as input_file, temporary.open("wb") as output_file:
        while chunk := input_file.read(8 * 1024 * 1024):
            output_file.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        output_file.flush()
        os.fsync(output_file.fileno())
    shutil.copystat(source, temporary)
    os.replace(temporary, destination)
    return size, digest.hexdigest()


class HYWorldIntermediateStore:
    """Create an immutable, per-run snapshot of selected HYWorld artifacts."""

    def __init__(self, root: Path, *, scene_id: str, run_id: str | None = None) -> None:
        self.root = root.resolve()
        self.run_id = run_id or default_run_id()
        if not _SAFE_RUN_ID.fullmatch(self.run_id):
            raise ValueError("Intermediate run id may only contain letters, digits, '.', '_', and '-'")
        self.run_dir = self.root / self.run_id
        if self.run_dir.exists():
            raise FileExistsError(f"Intermediate run already exists: {self.run_dir}")
        self.manifest_path = self.run_dir / "manifest.json"
        self.manifest: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "scene_id": scene_id,
            "run_id": self.run_id,
            "status": "running",
            "created_at": _utc_now(),
            "completed_at": None,
            "stages": [],
        }
        self._captured_state: dict[str, tuple[int, int]] = {}
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._flush()

    def _flush(self) -> None:
        _write_json_atomic(self.manifest_path, self.manifest)
        _write_json_atomic(
            self.root / "latest.json",
            {
                "schema": SCHEMA_VERSION,
                "scene_id": self.manifest["scene_id"],
                "run_id": self.run_id,
                "status": self.manifest["status"],
                "manifest_path": str(self.manifest_path),
                "updated_at": _utc_now(),
            },
        )

    def capture(
        self,
        stage_id: str,
        sources: Iterable[tuple[str, Path, Iterable[str]]],
        *,
        required: Iterable[tuple[str, Path]] = (),
    ) -> dict[str, Any]:
        """Copy stage artifacts into the run directory and append the manifest."""

        started = time.perf_counter()
        file_count = 0
        byte_count = 0
        outcome = "failure"
        with tracer.start_as_current_span("hyworld.preserve_intermediates") as span:
            span.set_attributes({
                "hyworld.scene_id": self.manifest["scene_id"],
                "hyworld.intermediate.stage": stage_id,
            })
            try:
                missing = [
                    {"label": label, "path": str(path.resolve())}
                    for label, path in required
                    if not path.is_file()
                ]
                artifacts: list[dict[str, Any]] = []
                unchanged_count = 0
                for source_label, source_root, patterns in sources:
                    source_root = source_root.resolve()
                    for source in _iter_files(source_root, patterns):
                        try:
                            source.relative_to(self.root)
                            continue
                        except ValueError:
                            pass
                        stat = source.stat()
                        source_key = str(source)
                        source_state = (stat.st_size, stat.st_mtime_ns)
                        if self._captured_state.get(source_key) == source_state:
                            unchanged_count += 1
                            continue
                        relative = source.relative_to(source_root)
                        destination = self.run_dir / "stages" / stage_id / source_label / relative
                        size, sha256 = _copy_with_sha256(source, destination)
                        artifacts.append({
                            "source_label": source_label,
                            "source_path": str(source),
                            "snapshot_path": str(destination),
                            "relative_path": str(relative),
                            "size_bytes": size,
                            "sha256": sha256,
                        })
                        self._captured_state[source_key] = source_state
                        file_count += 1
                        byte_count += size
                entry = {
                    "stage_id": stage_id,
                    "captured_at": _utc_now(),
                    "status": "complete" if not missing else "incomplete",
                    "file_count": file_count,
                    "size_bytes": byte_count,
                    "unchanged_file_count": unchanged_count,
                    "missing_required": missing,
                    "artifacts": artifacts,
                }
                self.manifest["stages"].append(entry)
                self._flush()
                outcome = entry["status"]
                if missing:
                    raise FileNotFoundError(
                        "Required HYWorld intermediate artifacts are missing: "
                        + ", ".join(item["path"] for item in missing)
                    )
                logger.info(
                    "preserved HYWorld intermediates stage=%s files=%d bytes=%d",
                    stage_id,
                    file_count,
                    byte_count,
                )
                return entry
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("failed to preserve HYWorld intermediates stage=%s", stage_id)
                raise
            finally:
                elapsed = time.perf_counter() - started
                span.set_attributes({
                    "outcome": outcome,
                    "hyworld.intermediate.files": file_count,
                    "hyworld.intermediate.bytes": byte_count,
                })
                dimensions = {"stage": stage_id, "outcome": outcome}
                captures.add(1, dimensions)
                captured_files.add(file_count, dimensions)
                captured_bytes.add(byte_count, dimensions)
                capture_duration.record(elapsed, dimensions)

    def finish(self, status: str, *, error: str | None = None) -> None:
        if status not in {"success", "failure"}:
            raise ValueError(f"Unsupported intermediate run status: {status}")
        self.manifest["status"] = status
        self.manifest["completed_at"] = _utc_now()
        if error:
            self.manifest["error"] = error
        self._flush()

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.manifest["status"],
            "root": str(self.root),
            "run_dir": str(self.run_dir),
            "manifest_path": str(self.manifest_path),
            "stage_count": len(self.manifest["stages"]),
            "file_count": sum(int(stage["file_count"]) for stage in self.manifest["stages"]),
            "size_bytes": sum(int(stage["size_bytes"]) for stage in self.manifest["stages"]),
        }
