#!/usr/bin/env python3
"""Render all objects/yaws for one HYWorld scene in one Blender process."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode
except Exception:  # pragma: no cover - production servers may not install OTel.
    class _NoopSpan:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def set_attributes(self, *_args, **_kwargs):
            return None

        def record_exception(self, *_args, **_kwargs):
            return None

        def set_status(self, *_args, **_kwargs):
            return None

    class _NoopTracer:
        def start_as_current_span(self, *_args, **_kwargs):
            return _NoopSpan()

    class _NoopMeter:
        def create_counter(self, *_args, **_kwargs):
            return self

        def create_histogram(self, *_args, **_kwargs):
            return self

        def add(self, *_args, **_kwargs):
            return None

        def record(self, *_args, **_kwargs):
            return None

    class _NoopTrace:
        @staticmethod
        def get_tracer(*_args, **_kwargs):
            return _NoopTracer()

    class _NoopMetrics:
        @staticmethod
        def get_meter(*_args, **_kwargs):
            return _NoopMeter()

    class _NoopStatus:
        def __init__(self, *_args, **_kwargs):
            pass

    class _NoopStatusCode:
        ERROR = "ERROR"

    metrics = _NoopMetrics()
    trace = _NoopTrace()
    Status = _NoopStatus
    StatusCode = _NoopStatusCode


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

try:
    from observability import init_observability, shutdown_observability
except Exception:  # pragma: no cover - keep render usable without telemetry deps.
    def init_observability(*_args, **_kwargs):
        return None

    def shutdown_observability(*_args, **_kwargs):
        return None


logger = logging.getLogger("dataevolver.blender_batch")
tracer = trace.get_tracer("dataevolver.blender_batch")
meter = metrics.get_meter("dataevolver.blender_batch")
batch_runs = meter.create_counter("dataevolver.blender_batch.runs", unit="1")
batch_duration = meter.create_histogram("dataevolver.blender_batch.duration", unit="s")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_rotations(value: str) -> list[int]:
    rotations = [int(item.strip()) % 360 for item in value.split(",") if item.strip()]
    if not rotations or len(set(rotations)) != len(rotations):
        raise ValueError("Rotations must be a non-empty unique comma-separated list")
    return rotations


def parse_args():
    parser = argparse.ArgumentParser(description="One-process rotation8 render for one HYWorld scene")
    parser.add_argument("--scene-template", required=True)
    parser.add_argument("--objects-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--blender-bin", default=os.environ.get("BLENDER_BIN"))
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--engine", choices=["EEVEE", "CYCLES"], default="EEVEE")
    parser.add_argument("--rotations", default="0,45,90,135,180,225,270,315")
    parser.add_argument("--xvfb", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_jobs(args, objects: list[dict], rotations: list[int], temp_root: Path):
    jobs = []
    for item in objects:
        base_control = item.get("control_state") or {}
        for yaw in rotations:
            control = json.loads(json.dumps(base_control))
            control.setdefault("object", {})["yaw_deg"] = float(yaw)
            job_output = temp_root / item["obj_id"] / f"yaw{yaw:03d}"
            jobs.append({
                "job_id": f"{item['obj_id']}_yaw{yaw:03d}",
                "obj_id": item["obj_id"],
                "asset_path": item["asset_path"],
                "reference_image": item.get("reference_image"),
                "output_dir": str(job_output),
                "control_state": control,
                "resolution": args.resolution,
                "engine": args.engine,
            })
    return jobs


def consolidate(output_root: Path, temp_root: Path, objects: list[dict], rotations: list[int]):
    results = []
    for item in objects:
        obj_id = item["obj_id"]
        destination = output_root / "objects" / obj_id
        destination.mkdir(parents=True, exist_ok=True)
        rotations_meta = []
        for yaw in rotations:
            slug = f"yaw{yaw:03d}"
            source = temp_root / obj_id / slug / obj_id
            rgb = source / "az000_el+00.png"
            mask = source / "az000_el+00_mask.png"
            metadata = source / "metadata.json"
            if not rgb.exists() or not mask.exists() or not metadata.exists():
                raise RuntimeError(f"Missing Blender output for {obj_id} {slug}")
            render_meta = load_json(metadata)
            if not render_meta.get("joint_3d_render"):
                raise RuntimeError(f"Render did not use the HYWorld 3D scene contract: {obj_id} {slug}")
            shutil.copy2(rgb, destination / f"{slug}.png")
            shutil.copy2(mask, destination / f"{slug}_mask.png")
            shutil.copy2(metadata, destination / f"{slug}_render_metadata.json")
            for candidate in source.iterdir():
                if "depth" in candidate.name and candidate.suffix in {".png", ".exr"}:
                    shutil.copy2(candidate, destination / f"{slug}_depth{candidate.suffix}")
                elif "normal" in candidate.name and candidate.suffix == ".png":
                    shutil.copy2(candidate, destination / f"{slug}_normal.png")
            rotations_meta.append({
                "obj_id": obj_id,
                "rotation_deg": yaw,
                "rgb_path": str((destination / f"{slug}.png").relative_to(output_root)),
                "mask_path": str((destination / f"{slug}_mask.png").relative_to(output_root)),
                "render_metadata_path": str((destination / f"{slug}_render_metadata.json").relative_to(output_root)),
                "return_code": 0,
            })
        write_json(destination / "object_manifest.json", {"obj_id": obj_id, "rotations": rotations_meta})
        results.extend(rotations_meta)
    return results


def main() -> int:
    args = parse_args()
    if not args.blender_bin:
        raise SystemExit("--blender-bin or BLENDER_BIN is required")
    rotations = parse_rotations(args.rotations)
    manifest = load_json(Path(args.objects_manifest))
    if manifest.get("schema") != "dataevolver.scene_objects.v1":
        raise SystemExit("Unsupported objects manifest schema")
    objects = manifest.get("objects") or []
    if not objects:
        raise SystemExit("Objects manifest is empty")

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    temp_root = output_root / "_batch_tmp"
    jobs = build_jobs(args, objects, rotations, temp_root)
    jobs_path = output_root / "blender_jobs.json"
    batch_result = output_root / "blender_batch_result.json"
    write_json(jobs_path, {
        "schema": "dataevolver.blender_jobs.v1",
        "scene_id": manifest.get("scene_id"),
        "jobs": jobs,
        "summary_output": str(batch_result),
    })

    command = [
        args.blender_bin,
        "-b",
        "-P",
        str(PIPELINE_DIR / "stage4_scene_render.py"),
        "--",
        "--jobs-manifest",
        str(jobs_path),
        "--scene-template",
        str(Path(args.scene_template).resolve()),
        "--output-dir",
        str(batch_result),
        "--resolution",
        str(args.resolution),
        "--engine",
        args.engine,
    ]
    if args.xvfb == "on" or (args.xvfb == "auto" and not os.environ.get("DISPLAY")):
        command = ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24", *command]
    if args.dry_run:
        print(json.dumps({"success": True, "command": command, "jobs": len(jobs)}, indent=2))
        return 0

    init_observability("dataevolver.blender_scene_batch")
    started = time.perf_counter()
    outcome = "failure"
    exit_code = 1
    with tracer.start_as_current_span("blender.render_scene_batch") as span:
        span.set_attributes({
            "scene.id": str(manifest.get("scene_id", "unknown")),
            "batch.size": len(jobs),
            "objects.count": len(objects),
        })
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            (output_root / "blender.log").write_text(completed.stdout, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"Blender batch failed with rc={completed.returncode}")
            batch = load_json(batch_result)
            if batch.get("scene_load_count") != 1 or not batch.get("success"):
                raise RuntimeError("Blender batch did not preserve one scene load")
            rotations_meta = consolidate(output_root, temp_root, objects, rotations)
            summary = {
                "schema": "dataevolver.rotation8_scene.v1",
                "success": True,
                "scene_id": manifest.get("scene_id"),
                "scene_load_count": 1,
                "objects": len(objects),
                "rotations_per_object": len(rotations),
                "total_renders": len(rotations_meta),
                "joint_3d_render": True,
                "renders": rotations_meta,
            }
            write_json(output_root / "summary.json", summary)
            shutil.rmtree(temp_root, ignore_errors=True)
            outcome = "success"
            exit_code = 0
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            logger.exception("Blender scene batch failed")
        finally:
            elapsed = time.perf_counter() - started
            span.set_attributes({"outcome": outcome, "batch.duration_seconds": elapsed})
            batch_runs.add(1, {"outcome": outcome, "engine": args.engine})
            batch_duration.record(elapsed, {"outcome": outcome, "engine": args.engine})
    shutdown_observability()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
