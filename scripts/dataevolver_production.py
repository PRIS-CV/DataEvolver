#!/usr/bin/env python3
"""Unified production doctor, worker start, and resumable run command."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode
except ModuleNotFoundError:  # pragma: no cover - doctor must run in lean onboarding envs.
    class _NoopSpan:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def set_attribute(self, *args, **kwargs):
            return None

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


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

try:
    from observability import init_observability, shutdown_observability
except ModuleNotFoundError:  # pragma: no cover - optional production telemetry.
    def init_observability(*args, **kwargs):
        return None

    def shutdown_observability(*args, **kwargs):
        return None
from production_runtime import doctor_profile, run_resumable_manifest, start_workers


logger = logging.getLogger("dataevolver.production")
tracer = trace.get_tracer("dataevolver.production")
meter = metrics.get_meter("dataevolver.production")
operations = meter.create_counter("dataevolver.production.operations", unit="1")
operation_duration = meter.create_histogram("dataevolver.production.duration", unit="s")


def parse_args():
    parser = argparse.ArgumentParser(description="DataEvolver production runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--profile", required=True)
    doctor.add_argument("--no-imports", action="store_true")
    doctor.add_argument("--output", default=None)
    start = sub.add_parser("start")
    start.add_argument("--profile", required=True)
    start.add_argument("--skip-doctor", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--state", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_observability("dataevolver.production_runtime")
    started = time.perf_counter()
    outcome = "failure"
    exit_code = 1
    payload = None
    with tracer.start_as_current_span("production.execute") as span:
        span.set_attribute("production.command", args.command)
        try:
            if args.command == "doctor":
                payload = doctor_profile(args.profile, run_imports=not args.no_imports)
                exit_code = 0 if payload["success"] else 1
                if args.output:
                    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            elif args.command == "start":
                payload = start_workers(args.profile, skip_doctor=args.skip_doctor)
                exit_code = 0
            else:
                payload = run_resumable_manifest(args.manifest, args.state)
                exit_code = 0
            outcome = "success" if exit_code == 0 else "failure"
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            logger.exception("production operation failed")
            payload = {"success": False, "error": str(exc)}
            exit_code = 1
        finally:
            elapsed = time.perf_counter() - started
            span.set_attributes({"outcome": outcome, "production.duration_seconds": elapsed})
            operations.add(1, {"command": args.command, "outcome": outcome})
            operation_duration.record(elapsed, {"command": args.command, "outcome": outcome})
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    shutdown_observability()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
