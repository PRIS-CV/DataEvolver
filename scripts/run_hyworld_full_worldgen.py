#!/usr/bin/env python3
"""Run HYWorld full worldgen and validate real-3D scene outputs.

This wrapper is intentionally strict: panorama-shell outputs and constant depth
are treated as failures, not as production scenes.
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
from typing import Any

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode
except ModuleNotFoundError:  # pragma: no cover - local CLI checks may not install OTel.
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


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from hyworld_scene_contract import build_scene_contract, write_contract
from hyworld_intermediate_store import HYWorldIntermediateStore
try:
    from observability import init_observability, shutdown_observability
except ModuleNotFoundError:  # pragma: no cover - optional production telemetry.
    def init_observability(*args, **kwargs):
        return None

    def shutdown_observability(*args, **kwargs):
        return None


logger = logging.getLogger("dataevolver.hyworld_full")
tracer = trace.get_tracer("dataevolver.hyworld_full")
meter = metrics.get_meter("dataevolver.hyworld_full")
worldgen_runs = meter.create_counter("dataevolver.hyworld_full.runs", unit="1")
worldgen_duration = meter.create_histogram("dataevolver.hyworld_full.duration", unit="s")

REQUIRED_LOCAL_MODEL_ENV = {
    "HYWORLD_MOGE_MODEL_PATH": "MoGe metric-depth model",
    "HYWORLD_ZIM_MODEL_PATH": "ZIM sky-mask model",
    "HYWORLD_GROUNDING_DINO_MODEL_PATH": "GroundingDINO detector",
    "HYWORLD_SAM3_MODEL_PATH": "SAM3 source or checkpoint directory",
    "HYWORLD_WORLDSTEREO_PATH": "WorldStereo weights root",
    "HYWORLD_WAN_BASE_MODEL": "Wan I2V base Diffusers model",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HYWorld full worldgen with production geometry gates")
    parser.add_argument("--profile", default=os.environ.get("DATAEVOLVER_PRODUCTION_PROFILE"))
    parser.add_argument("--scene-dir", required=True, help="Scene directory containing panorama.png and meta_info.json")
    parser.add_argument("--result-dir", default=None, help="3DGS/WorldMirror result directory")
    parser.add_argument("--python", dest="python_bin", default=os.environ.get("HYWORLD_PYTHON") or sys.executable)
    parser.add_argument("--hyworld-src", default=os.environ.get("HYWORLD_SRC"))
    parser.add_argument("--hyworld-weights", default=os.environ.get("HYWORLD_WEIGHTS"))
    parser.add_argument("--support-object-name", default="HYWorldSceneMesh")
    parser.add_argument("--contract-output", default=None)
    parser.add_argument("--llm-addr", default=os.environ.get("LLM_ADDR", "localhost"))
    parser.add_argument("--llm-port", type=int, default=int(os.environ.get("LLM_PORT", "8000")))
    parser.add_argument("--llm-name", default=os.environ.get("LLM_NAME", "Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--nproc", type=int, default=None, help="torchrun process count; defaults to number of --gpus")
    parser.add_argument("--worldstereo-model-type", default="worldstereo-memory-dmd")
    parser.add_argument("--gs-max-steps", type=int, default=1500)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument(
        "--intermediate-root",
        default=None,
        help="Durable snapshot root; defaults to <scene-dir>/intermediates",
    )
    parser.add_argument("--intermediate-run-id", default=None)
    parser.add_argument(
        "--no-preserve-intermediates",
        action="store_true",
        help="Disable durable per-stage snapshots (enabled by default)",
    )
    return parser.parse_args()


def load_runtime(args: argparse.Namespace) -> tuple[dict[str, str], Path | None, dict[str, Any] | None]:
    env = os.environ.copy()
    profile_path = None
    profile = None
    if args.profile:
        try:
            from production_runtime import build_runtime_environment, load_profile, resolve_profile_path
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "--profile requires pipeline/production_runtime.py or an installed production runtime module"
            ) from exc
        profile_path, profile = load_profile(args.profile)
        env.update(build_runtime_environment(profile_path, profile))
        runtime = profile.get("runtime") or {}
        paths = profile.get("paths") or {}
        args.python_bin = str(resolve_profile_path(profile_path, runtime.get("hyworld_python")) or args.python_bin)
        args.hyworld_src = str(resolve_profile_path(profile_path, paths.get("hyworld_source")) or args.hyworld_src)
        args.hyworld_weights = str(resolve_profile_path(profile_path, paths.get("hyworld_weights")) or args.hyworld_weights)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("DIFFUSERS_OFFLINE", "1")
    env["HYWORLD_NO_MODEL_DOWNLOADS"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    if env.get("HYWORLD_MOGE_MODEL_PATH"):
        env.setdefault("MOGE_MODEL_PATH", env["HYWORLD_MOGE_MODEL_PATH"])
    src = Path(args.hyworld_src or "").resolve()
    if src:
        extra_pythonpath = [
            str(src),
            str(src / "hyworld2" / "panogen"),
            str(src / "hyworld2" / "worldgen"),
        ]
        env["PYTHONPATH"] = os.pathsep.join(extra_pythonpath + [env.get("PYTHONPATH", "")])
    return env, profile_path, profile


def resolve_path_from_env(env: dict[str, str], key: str) -> Path | None:
    raw = env.get(key)
    if not raw:
        return None
    return Path(os.path.expandvars(os.path.expanduser(raw))).resolve()


def preflight(args: argparse.Namespace, env: dict[str, str]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, success: bool, detail: str) -> None:
        checks.append({"name": name, "success": bool(success), "detail": detail})

    scene_dir = Path(args.scene_dir).resolve()
    src = Path(args.hyworld_src or "").resolve()
    weights = Path(args.hyworld_weights or "").resolve()
    add("scene_dir", scene_dir.is_dir(), str(scene_dir))
    add("scene.panorama", (scene_dir / "panorama.png").is_file(), str(scene_dir / "panorama.png"))
    add("hyworld_src", (src / "hyworld2" / "worldgen" / "traj_generate.py").is_file(), str(src))
    add("hyworld_weights", weights.is_dir(), str(weights))
    add(
        "hyworld_worldmirror",
        (weights / "HY-WorldMirror-2.0" / "model.safetensors").is_file(),
        str(weights / "HY-WorldMirror-2.0"),
    )
    add("hyworld_no_model_downloads", env.get("HYWORLD_NO_MODEL_DOWNLOADS") == "0", env.get("HYWORLD_NO_MODEL_DOWNLOADS", ""))
    add("hf_hub_offline", env.get("HF_HUB_OFFLINE") == "1", env.get("HF_HUB_OFFLINE", ""))
    for env_name, label in REQUIRED_LOCAL_MODEL_ENV.items():
        candidate = resolve_path_from_env(env, env_name)
        add(f"local_model.{env_name}", bool(candidate and candidate.exists()), f"{label}: {candidate}")
    import_code = (
        "import json;"
        "mods=['moge.model.v2','hyworld2.worldrecon.pipeline','gsplat'];"
        "out={};"
        "\nfor m in mods:\n"
        "    try:\n"
        "        __import__(m); out[m]='ok'\n"
        "    except Exception as e:\n"
        "        out[m]=type(e).__name__+': '+str(e)[:200]\n"
        "print(json.dumps(out))"
    )
    completed = subprocess.run(
        [args.python_bin, "-c", import_code],
        cwd=str(src) if src.exists() else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
    )
    import_output = completed.stdout[-2000:]
    try:
        import_result = json.loads(import_output.strip().splitlines()[-1])
    except Exception:
        import_result = {}
    add(
        "imports.hyworld_real3d",
        completed.returncode == 0 and import_result and all(value == "ok" for value in import_result.values()),
        import_output,
    )
    return {"success": all(item["success"] for item in checks), "checks": checks}


def torchrun_command(args: argparse.Namespace, env: dict[str, str], script: Path, script_args: list[str]) -> list[str]:
    nproc = args.nproc
    if nproc is None:
        nproc = len([item for item in args.gpus.split(",") if item.strip()])
    if nproc > 1:
        return [args.python_bin, "-m", "torch.distributed.run", "--nproc_per_node", str(nproc), str(script), *script_args]
    return [args.python_bin, str(script), *script_args]


def stage_commands(args: argparse.Namespace, env: dict[str, str]) -> list[dict[str, Any]]:
    scene_dir = Path(args.scene_dir).resolve()
    src = Path(args.hyworld_src or "").resolve()
    worldgen_dir = src / "hyworld2" / "worldgen"
    result_dir = Path(args.result_dir).resolve() if args.result_dir else scene_dir / "worldgen_real3d_results"
    common_llm = [
        "--llm_addr", args.llm_addr,
        "--llm_port", str(args.llm_port),
        "--llm_name", args.llm_name,
    ]
    skip_flag = ["--skip_exist"] if args.skip_existing else []
    gs_steps = str(args.gs_max_steps)
    return [
        {
            "id": "traj_generate",
            "cwd": str(worldgen_dir),
            "command": [
                args.python_bin,
                str(worldgen_dir / "traj_generate.py"),
                "--target_path", str(scene_dir),
                *common_llm,
                "--apply_nav_traj",
                "--apply_up_route",
                "--apply_recon_iteration",
                "--force_vlm",
                *skip_flag,
            ],
        },
        {
            "id": "traj_render",
            "cwd": str(worldgen_dir),
            "command": torchrun_command(
                args,
                env,
                worldgen_dir / "traj_render.py",
                ["--target_path", str(scene_dir), *common_llm, *skip_flag],
            ),
        },
        {
            "id": "video_gen",
            "cwd": str(worldgen_dir),
            "command": torchrun_command(
                args,
                env,
                worldgen_dir / "video_gen.py",
                [
                    "--target_path", str(scene_dir),
                    "--model_type", args.worldstereo_model_type,
                    "--fsdp",
                    "--local_files_only",
                    *skip_flag,
                ],
            ),
        },
        {
            "id": "gen_gs_data",
            "cwd": str(worldgen_dir),
            "command": torchrun_command(
                args,
                env,
                worldgen_dir / "gen_gs_data.py",
                ["--root_path", str(scene_dir), "--save_normal", "--split_sky"],
            ),
        },
        {
            "id": "world_gs_trainer",
            "cwd": str(worldgen_dir),
            "command": [
                args.python_bin,
                "-m",
                "world_gs_trainer",
                "default",
                "--data_dir", str(scene_dir / "gs_data"),
                "--result_dir", str(result_dir),
                "--max_steps", gs_steps,
                "--save_steps", gs_steps,
                "--eval_steps", gs_steps,
                "--ply_steps", gs_steps,
                "--save_ply",
                "--convert_to_spz",
                "--disable_video",
                "--use_scale_regularization",
                "--antialiased",
                "--depth_loss",
                "--normal_loss",
                "--sky_depth_from_pcd",
                "--use_mask_gaussian",
                "--mask_export_stochastic",
                "--no-mask-export-anchor-protection",
                "--use_anchor_protection",
                "--export_mesh",
            ],
        },
        {
            "id": "worldmirror_recon",
            "cwd": str(src),
            "command": [
                args.python_bin,
                "-m",
                "hyworld2.worldrecon.pipeline",
                "--input_path", str(worldmirror_input_path(scene_dir, args.worldstereo_model_type)),
                "--strict_output_path", str(scene_dir / "worldmirror_recon"),
                "--pretrained_model_name_or_path", str(Path(args.hyworld_weights).resolve()),
                "--subfolder", "HY-WorldMirror-2.0",
                "--enable_bf16",
                "--save_rendered",
                "--no_sky_mask",
                "--no_interactive",
            ],
        },
    ]


def worldmirror_input_path(scene_dir: Path, model_type: str) -> Path:
    candidates = [
        scene_dir / "render_results" / "pano_bank" / "images",
        scene_dir / "render_results" / f"generation_bank_{model_type}" / "images",
        scene_dir / "render_results",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def resolved_result_dir(args: argparse.Namespace) -> Path:
    scene_dir = Path(args.scene_dir).resolve()
    return Path(args.result_dir).resolve() if args.result_dir else scene_dir / "worldgen_real3d_results"


def intermediate_sources(args: argparse.Namespace, *, source_only: bool = False) -> list[tuple[str, Path, list[str]]]:
    scene_dir = Path(args.scene_dir).resolve()
    if source_only:
        return [("scene", scene_dir, ["panorama.png", "meta_info.json", "source_scene.json"])]
    sources = [("scene", scene_dir, ["**/*"])]
    result_dir = resolved_result_dir(args)
    try:
        result_dir.relative_to(scene_dir)
    except ValueError:
        sources.append(("result", result_dir, ["**/*"]))
    if args.contract_output:
        contract_path = Path(args.contract_output).resolve()
        try:
            contract_path.relative_to(scene_dir)
        except ValueError:
            sources.append(("contract", contract_path.parent, [contract_path.name]))
    return sources


def create_intermediate_store(args: argparse.Namespace) -> HYWorldIntermediateStore:
    scene_dir = Path(args.scene_dir).resolve()
    root = Path(args.intermediate_root).resolve() if args.intermediate_root else scene_dir / "intermediates"
    return HYWorldIntermediateStore(root, scene_id=scene_dir.name, run_id=args.intermediate_run_id)


def run_command(step: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        step["command"],
        cwd=step["cwd"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return {
        "id": step["id"],
        "return_code": completed.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "stdout_tail": completed.stdout[-4000:],
    }


def validate_outputs(args: argparse.Namespace) -> dict[str, Any]:
    scene_dir = Path(args.scene_dir).resolve()
    mesh = scene_dir / "render_results" / "global_mesh.ply"
    depth = scene_dir / "render_results" / "full_depth_prediction.pt"
    panorama = scene_dir / "panorama.png"
    camera = scene_dir / "render_results" / "pano_bank" / "cameras.json"
    contract_path = Path(args.contract_output).resolve() if args.contract_output else scene_dir / "scene_contract.json"
    payload = build_scene_contract(
        mesh,
        args.support_object_name,
        scene_id=scene_dir.name,
        depth_path=depth,
        panorama_path=panorama,
        camera_path=camera if camera.exists() else None,
    )
    write_contract(contract_path, payload)
    recon_dir = scene_dir / "worldmirror_recon"
    recon_artifacts = sorted(
        str(path.relative_to(recon_dir))
        for path in recon_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".ply", ".json", ".png", ".npy", ".npz", ".mp4"}
    ) if recon_dir.exists() else []
    if not recon_artifacts:
        raise RuntimeError(f"WorldMirror output is missing or empty: {recon_dir}")
    return {
        "success": True,
        "contract": str(contract_path),
        "worldmirror_recon": str(recon_dir),
        "worldmirror_artifact_count": len(recon_artifacts),
        "worldmirror_artifact_sample": recon_artifacts[:20],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_observability("dataevolver.hyworld_full_worldgen")
    started = time.perf_counter()
    outcome = "failure"
    response: dict[str, Any] = {"success": False}
    exit_code = 1
    intermediate_store: HYWorldIntermediateStore | None = None
    with tracer.start_as_current_span("hyworld.full_worldgen") as span:
        span.set_attributes({
            "hyworld.scene_dir": str(Path(args.scene_dir).resolve()),
            "hyworld.dry_run": bool(args.dry_run),
            "hyworld.validate_only": bool(args.validate_only),
        })
        try:
            env, _, _ = load_runtime(args)
            pf = preflight(args, env)
            commands = stage_commands(args, env)
            if args.dry_run:
                intermediate_root = (
                    Path(args.intermediate_root).resolve()
                    if args.intermediate_root
                    else Path(args.scene_dir).resolve() / "intermediates"
                )
                response = {
                    "success": pf["success"],
                    "preflight": pf,
                    "commands": commands,
                    "intermediates": {
                        "enabled": not args.no_preserve_intermediates,
                        "root": str(intermediate_root),
                    },
                }
                exit_code = 0
                outcome = "success" if pf["success"] else "blocked"
            else:
                if not args.no_preserve_intermediates:
                    intermediate_store = create_intermediate_store(args)
                    intermediate_store.capture(
                        "00_source_panorama",
                        intermediate_sources(args, source_only=True),
                        required=[("panorama", Path(args.scene_dir).resolve() / "panorama.png")],
                    )
                if not pf["success"]:
                    response = {"success": False, "preflight": pf, "error": "HYWorld real-3D preflight failed"}
                    exit_code = 1
                    raise RuntimeError(response["error"])
                results = []
                if not args.validate_only:
                    for step in commands:
                        logger.info("running HYWorld stage %s", step["id"])
                        result = run_command(step, env)
                        results.append(result)
                        if intermediate_store is not None:
                            intermediate_store.capture(step["id"], intermediate_sources(args))
                        if result["return_code"] != 0:
                            raise RuntimeError(f"HYWorld stage failed: {step['id']}")
                validation = validate_outputs(args)
                if intermediate_store is not None:
                    intermediate_store.capture(
                        "validation",
                        intermediate_sources(args),
                        required=[
                            (
                                "scene_contract",
                                Path(args.contract_output).resolve()
                                if args.contract_output
                                else Path(args.scene_dir).resolve() / "scene_contract.json",
                            )
                        ],
                    )
                response = {"success": True, "preflight": pf, "results": results, "validation": validation}
                exit_code = 0
                outcome = "success"
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            logger.exception("HYWorld full worldgen failed")
            response.setdefault("success", False)
            response["error"] = str(exc)
            exit_code = 1
        finally:
            elapsed = time.perf_counter() - started
            if intermediate_store is not None:
                intermediate_store.finish("success" if exit_code == 0 else "failure", error=response.get("error"))
                response["intermediates"] = intermediate_store.summary()
            span.set_attributes({"outcome": outcome, "hyworld.duration_seconds": elapsed})
            worldgen_runs.add(1, {"outcome": outcome, "dry_run": str(bool(args.dry_run)).lower()})
            worldgen_duration.record(elapsed, {"outcome": outcome})
            response["elapsed_seconds"] = round(elapsed, 3)
            if args.summary_output:
                write_json(Path(args.summary_output), response)
    print(json.dumps(response, indent=2, ensure_ascii=False))
    shutdown_observability()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
