#!/usr/bin/env python3
"""Render HYWorld pure-scene multiviews while loading the blend once.

Normal use:
    python scripts/render_hyworld_scene_multiview.py --scene-template scene_template.json --output-dir out

Blender worker mode is internal:
    blender -b scene.blend -P scripts/render_hyworld_scene_multiview.py -- --worker-config config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode
except Exception:  # pragma: no cover - Blender/remote render env may not include OTel.
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

    trace = _NoopTrace()
    metrics = _NoopMetrics()
    Status = _NoopStatus
    StatusCode = _NoopStatusCode


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

try:
    from observability import init_observability, shutdown_observability
except Exception:  # pragma: no cover
    def init_observability(*_args, **_kwargs):
        return None

    def shutdown_observability():
        return None


logger = logging.getLogger("dataevolver.hyworld_scene_multiview")
tracer = trace.get_tracer("dataevolver.hyworld_scene_multiview")
meter = metrics.get_meter("dataevolver.hyworld_scene_multiview")
scene_multiview_runs = meter.create_counter("dataevolver.hyworld_scene_multiview.runs", unit="1")
scene_multiview_duration = meter.create_histogram("dataevolver.hyworld_scene_multiview.duration", unit="s")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_views(value: str) -> list[int]:
    views = [int(item.strip()) % 360 for item in value.split(",") if item.strip()]
    if not views or len(set(views)) != len(views):
        raise ValueError("Views must be a non-empty unique comma-separated list")
    return views


def wrapper_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render HYWorld pure-scene multiviews")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene-template", help="Template JSON produced by create_hyworld_scene_blend.py")
    source.add_argument("--blend-path", help="HYWorld Blender scene path")
    parser.add_argument("--contract", default=None, help="HYWorld scene contract; required with --blend-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--blender-bin", default=os.environ.get("BLENDER_BIN"))
    parser.add_argument("--views", default="0,45,90,135,180,225,270,315")
    parser.add_argument("--render-width", type=int, default=None)
    parser.add_argument("--render-height", type=int, default=None)
    parser.add_argument("--engine", choices=["EEVEE", "CYCLES"], default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--xvfb", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def worker_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal Blender worker")
    parser.add_argument("--worker-config", required=True)
    return parser.parse_args(argv)


def scene_config_from_args(args: argparse.Namespace) -> dict:
    if args.scene_template:
        template_path = Path(args.scene_template).resolve()
        template = load_json(template_path)
        blend_path = Path(template["blend_path"]).resolve()
        contract_path = Path(template["scene_contract_path"]).resolve()
    else:
        if not args.contract:
            raise SystemExit("--contract is required with --blend-path")
        template_path = None
        template = {}
        blend_path = Path(args.blend_path).resolve()
        contract_path = Path(args.contract).resolve()

    contract = load_json(contract_path)
    if contract.get("schema") != "dataevolver.hyworld_scene.v1" or contract.get("status") != "ready":
        raise SystemExit(f"Invalid HYWorld scene contract: {contract_path}")
    support = next(
        (
            item
            for item in contract.get("support_surfaces", [])
            if item.get("id") == contract.get("selected_support_surface_id")
        ),
        None,
    )
    if not support or support.get("source") != "hyworld_reconstruction" or support.get("artificial") is not False:
        raise SystemExit("Scene contract must contain a reconstructed non-artificial support surface")

    render_width = args.render_width or int(template.get("render_width", template.get("render_resolution", 832)))
    render_height = args.render_height or int(template.get("render_height", template.get("render_resolution", 480)))
    engine = args.engine or str(template.get("render_engine", "EEVEE")).upper()
    samples = args.samples or int(template.get("eevee_render_samples", 32))
    return {
        "schema": "dataevolver.hyworld_scene_multiview_config.v1",
        "template_path": str(template_path) if template_path else None,
        "blend_path": str(blend_path),
        "contract_path": str(contract_path),
        "scene_id": contract.get("scene_id") or template.get("scene_id") or contract_path.parent.name,
        "output_dir": str(Path(args.output_dir).resolve()),
        "views": parse_views(args.views),
        "render_width": render_width,
        "render_height": render_height,
        "engine": engine,
        "samples": samples,
        "support_anchor_xyz": support.get("anchor_xyz"),
        "support_bounds_xy": support.get("bounds_xy"),
        "support_source": support.get("source"),
    }


def build_command(args: argparse.Namespace, config_path: Path, blend_path: Path) -> list[str]:
    if not args.blender_bin:
        raise SystemExit("--blender-bin or BLENDER_BIN is required")
    command = [
        args.blender_bin,
        "-b",
        str(blend_path),
        "-P",
        str(Path(__file__).resolve()),
        "--",
        "--worker-config",
        str(config_path),
    ]
    if args.xvfb == "on" or (args.xvfb == "auto" and not os.environ.get("DISPLAY")):
        command = ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24", *command]
    return command


def run_wrapper(argv: list[str]) -> int:
    args = wrapper_args(argv)
    config = scene_config_from_args(args)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "scene_multiview_config.json"
    summary_path = output_dir / "summary.json"
    log_path = output_dir / "blender.log"
    config["summary_output"] = str(summary_path)
    write_json(config_path, config)
    command = build_command(args, config_path, Path(config["blend_path"]))

    if args.dry_run:
        print(json.dumps({"success": True, "command": command, "config": config}, indent=2, ensure_ascii=False))
        return 0

    init_observability("dataevolver.hyworld_scene_multiview")
    started = time.perf_counter()
    outcome = "failure"
    exit_code = 1
    with tracer.start_as_current_span("hyworld_scene.render_multiview") as span:
        span.set_attributes({
            "scene.id": str(config["scene_id"]),
            "views.count": len(config["views"]),
            "render.engine": config["engine"],
        })
        try:
            logger.info("starting HYWorld pure-scene multiview render", extra={"scene_id": config["scene_id"]})
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            log_path.write_text(completed.stdout, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"Blender scene multiview failed with rc={completed.returncode}")
            summary = load_json(summary_path)
            if not summary.get("success") or summary.get("scene_load_count") != 1:
                raise RuntimeError("Scene multiview did not preserve one scene load")
            if int(summary.get("view_count", 0)) != len(config["views"]):
                raise RuntimeError("Scene multiview summary view count mismatch")
            outcome = "success"
            exit_code = 0
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            logger.exception("HYWorld pure-scene multiview render failed")
        finally:
            elapsed = time.perf_counter() - started
            span.set_attributes({"outcome": outcome, "duration.seconds": elapsed})
            scene_multiview_runs.add(1, {"outcome": outcome, "engine": str(config["engine"])})
            scene_multiview_duration.record(elapsed, {"outcome": outcome, "engine": str(config["engine"])})
    shutdown_observability()
    return exit_code


def run_worker(argv: list[str]) -> int:
    import bpy
    from mathutils import Matrix, Vector

    args = worker_args(argv)
    config = load_json(Path(args.worker_config))
    output_dir = Path(config["output_dir"]).resolve()
    views_dir = output_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    if scene.camera is None:
        raise RuntimeError("HYWorld blend has no active camera")

    engine = str(config["engine"]).upper()
    scene.render.engine = "BLENDER_EEVEE_NEXT" if engine == "EEVEE" and bpy.app.version[0] >= 4 else (
        "BLENDER_EEVEE" if engine == "EEVEE" else "CYCLES"
    )
    scene.render.resolution_x = int(config["render_width"])
    scene.render.resolution_y = int(config["render_height"])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    if "EEVEE" in scene.render.engine and hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = int(config["samples"])

    camera = scene.camera
    base_matrix = camera.matrix_world.copy()
    anchor_values = config.get("support_anchor_xyz") or [0.0, 0.0, 0.0]
    anchor = Vector((float(anchor_values[0]), float(anchor_values[1]), float(anchor_values[2])))
    base_location = base_matrix.translation.copy()
    offset = base_location - anchor
    if offset.length < 1e-6:
        offset = Vector((0.0, -4.0, 1.6))

    renders = []
    for yaw in config["views"]:
        rotation = Matrix.Rotation(float(yaw) * 3.141592653589793 / 180.0, 4, "Z")
        camera.location = anchor + (rotation @ offset)
        direction = anchor - camera.location
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        filename = f"view_yaw{int(yaw):03d}.png"
        output = views_dir / filename
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
        renders.append({
            "view_yaw_deg": int(yaw),
            "rgb_path": str(output.relative_to(output_dir)),
            "camera_location": [round(float(v), 6) for v in camera.location],
            "camera_target": [round(float(v), 6) for v in anchor],
        })

    camera.matrix_world = base_matrix
    summary = {
        "schema": "dataevolver.hyworld_scene_multiview.v1",
        "success": True,
        "scene_id": config["scene_id"],
        "scene_load_count": 1,
        "view_count": len(renders),
        "views": renders,
        "render_width": int(config["render_width"]),
        "render_height": int(config["render_height"]),
        "engine": engine,
        "scene_contract_path": config["contract_path"],
        "blend_path": config["blend_path"],
        "support_source": config.get("support_source"),
    }
    write_json(Path(config["summary_output"]), summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    if "--worker-config" in argv:
        return run_worker(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run_wrapper(argv)


if __name__ == "__main__":
    raise SystemExit(main())
