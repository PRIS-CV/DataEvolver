#!/usr/bin/env python3
"""Generate HYWorld panorama scene directories from DataEvolver scene images."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from dataevolver.paths import runtime_path

from dataevolver.runtime.hyworld_intermediate_store import HYWorldIntermediateStore
try:
    from observability import init_observability, shutdown_observability
except ModuleNotFoundError:  # pragma: no cover - optional production telemetry.
    def init_observability(*args, **kwargs):
        return None

    def shutdown_observability(*args, **kwargs):
        return None


DEFAULT_SCENE_IMAGES_DIR = runtime_path("scenes", "images")
DEFAULT_OUTPUT_ROOT = runtime_path("hyworld_scenes")
PANO_INSTRUCTION = "Expand this image to a 360-degree equirectangular panorama."
logger = logging.getLogger("dataevolver.hyworld_pano")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HYWorld panorama generation for DataEvolver scene images")
    parser.add_argument("--scene-prompts-path", required=True)
    parser.add_argument("--scene-images-dir", default=str(DEFAULT_SCENE_IMAGES_DIR))
    parser.add_argument("--scene-image", default=None, help="Single input image override; use with one scene id")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--ids", default=None, help="Comma-separated scene IDs to process")
    parser.add_argument("--python", dest="python_bin", default=os.environ.get("HYWORLD_PYTHON") or sys.executable)
    parser.add_argument("--hyworld-src", default=os.environ.get("HYWORLD_SRC"))
    parser.add_argument("--hyworld-weights", default=os.environ.get("HYWORLD_WEIGHTS") or os.environ.get("HY_PANO_MODEL_PATH"))
    parser.add_argument("--subfolder", default="HY-Pano-2.0")
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=1952)
    parser.add_argument("--diff-infer-steps", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--bot-task", choices=["image", "auto", "recaption", "think_recaption"], default=None)
    parser.add_argument("--use-system-prompt", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--attn-impl", choices=["sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--moe-impl", choices=["eager", "flashinfer"], default="eager")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without writing outputs")
    parser.add_argument("--no-skip", action="store_true", help="Regenerate existing panorama.png")
    parser.add_argument(
        "--intermediate-root",
        default=None,
        help="Durable snapshot parent; defaults to each <scene-dir>/intermediates",
    )
    parser.add_argument("--intermediate-run-id", default=None)
    parser.add_argument("--no-preserve-intermediates", action="store_true")
    return parser.parse_args()


def load_scene_records(path: Path, ids: set[str] | None) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise SystemExit("--scene-prompts-path must contain a JSON list or object")
    records = [item for item in payload if isinstance(item, dict)]
    for item in records:
        if not str(item.get("scene_id") or "").strip():
            raise SystemExit("Every scene record must include scene_id")
    if ids:
        records = [item for item in records if str(item.get("scene_id")).strip() in ids]
    if not records:
        raise SystemExit("No matching scene records found")
    return records


def build_pano_prompt(record: dict) -> str:
    explicit = str(record.get("pano_prompt") or "").strip()
    if explicit:
        return explicit if PANO_INSTRUCTION in explicit else f"{PANO_INSTRUCTION} {explicit}"
    scene_desc = str(record.get("scene_description") or record.get("t2i_prompt") or "").strip()
    scene_type = str(record.get("scene_type") or "outdoor").strip()
    style_tags = ", ".join(str(x).strip() for x in record.get("style_tags") or [] if str(x).strip())
    layout_hints = "; ".join(str(x).strip() for x in record.get("layout_hints") or [] if str(x).strip())
    prompt = (
        f"{PANO_INSTRUCTION} Preserve the DataEvolver generated {scene_type} scene, "
        "extend it into a coherent, navigable 360-degree environment with a believable ground or floor plane, "
        "consistent lighting, and no logos, text, or named copyrighted locations."
    )
    if scene_desc:
        prompt += f" Scene description: {scene_desc}"
    if style_tags:
        prompt += f" Visual style: {style_tags}."
    if layout_hints:
        prompt += f" Layout constraints: {layout_hints}."
    return prompt


def resolve_scene_image(record: dict, args: argparse.Namespace) -> Path:
    if args.scene_image:
        return Path(args.scene_image).resolve()
    raw_path = record.get("scene_image_path") or record.get("image_path")
    if raw_path:
        return Path(str(raw_path)).expanduser().resolve()
    return (Path(args.scene_images_dir) / f"{record['scene_id']}.png").resolve()


def env_for_hyworld(args: argparse.Namespace) -> dict:
    if not args.hyworld_src:
        raise SystemExit("HYWORLD_SRC or --hyworld-src is required")
    if not args.hyworld_weights:
        raise SystemExit("HYWORLD_WEIGHTS, HY_PANO_MODEL_PATH, or --hyworld-weights is required")
    src = Path(args.hyworld_src).resolve()
    panogen = src / "hyworld2" / "panogen"
    if not (panogen / "pipeline.py").exists():
        raise SystemExit(f"HYWorld panogen pipeline not found: {panogen / 'pipeline.py'}")
    if not Path(args.hyworld_weights).exists():
        raise SystemExit(f"HYWorld weights path not found: {args.hyworld_weights}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(src), str(panogen), env.get("PYTHONPATH", "")])
    return env


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def create_intermediate_store(scene_id: str, scene_dir: Path, args: argparse.Namespace) -> HYWorldIntermediateStore:
    root = (
        Path(args.intermediate_root).resolve() / scene_id
        if args.intermediate_root
        else scene_dir / "intermediates"
    )
    return HYWorldIntermediateStore(root, scene_id=scene_id, run_id=args.intermediate_run_id)


def run_one(record: dict, args: argparse.Namespace, env: dict) -> dict:
    scene_id = str(record["scene_id"]).strip()
    scene_image = resolve_scene_image(record, args)
    if not scene_image.exists():
        raise FileNotFoundError(f"Scene image not found for {scene_id}: {scene_image}")

    scene_dir = Path(args.output_root).resolve() / scene_id
    pano_path = scene_dir / "panorama.png"
    prompt = build_pano_prompt(record)
    src = Path(args.hyworld_src).resolve()
    panogen_dir = src / "hyworld2" / "panogen"
    cmd = [
        args.python_bin,
        str(panogen_dir / "pipeline.py"),
        "--image",
        str(scene_image),
        "--prompt",
        prompt,
        "--save",
        str(pano_path),
        "--pretrained-model-name-or-path",
        str(Path(args.hyworld_weights).resolve()),
        "--subfolder",
        args.subfolder,
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--diff-infer-steps",
        str(args.diff_infer_steps),
        "--attn-impl",
        args.attn_impl,
        "--moe-impl",
        args.moe_impl,
    ]
    if args.max_new_tokens is not None:
        cmd.extend(["--max_new_tokens", str(args.max_new_tokens)])
    if args.bot_task:
        cmd.extend(["--bot-task", args.bot_task])
    if args.use_system_prompt:
        cmd.extend(["--use-system-prompt", args.use_system_prompt])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed), "--reproduce"])

    if args.dry_run:
        return {"scene_id": scene_id, "dry_run": True, "command": cmd, "scene_image": str(scene_image), "scene_dir": str(scene_dir)}
    if pano_path.exists() and not args.no_skip:
        result = {"scene_id": scene_id, "skipped": True, "panorama_path": str(pano_path)}
        if not args.no_preserve_intermediates:
            store = create_intermediate_store(scene_id, scene_dir, args)
            store.capture(
                "01_panorama",
                [("scene", scene_dir, ["panorama.png", "meta_info.json", "source_scene.json"])],
                required=[("panorama", pano_path)],
            )
            store.finish("success")
            result["intermediates"] = store.summary()
        return result

    scene_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        scene_dir / "meta_info.json",
        {
            "scene_id": scene_id,
            "scene_type": str(record.get("scene_type") or "outdoor").strip().lower(),
            "source": "dataevolver_scene_prompt",
        },
    )
    save_json(
        scene_dir / "source_scene.json",
        {
            "scene_record": record,
            "scene_image_path": str(scene_image),
            "panorama_prompt": prompt,
            "hyworld_weights": str(Path(args.hyworld_weights).resolve()),
        },
    )

    started = time.time()
    completed = subprocess.run(
        cmd,
        cwd=str(panogen_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    result = {
        "scene_id": scene_id,
        "return_code": completed.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "scene_image": str(scene_image),
        "scene_dir": str(scene_dir),
        "panorama_path": str(pano_path),
        "stdout_tail": completed.stdout[-4000:],
    }
    if completed.returncode == 0 and (not pano_path.exists() or pano_path.stat().st_size < 1024):
        result["return_code"] = -1
        result["error"] = "panorama_missing_or_empty"
    if not args.no_preserve_intermediates:
        store = create_intermediate_store(scene_id, scene_dir, args)
        try:
            store.capture(
                "01_panorama",
                [
                    ("scene", scene_dir, ["panorama.png", "meta_info.json", "source_scene.json"]),
                    ("source_image", scene_image.parent, [scene_image.name]),
                ],
                required=[("panorama", pano_path)] if result["return_code"] == 0 else (),
            )
        except Exception as exc:
            result["return_code"] = -2
            result["error"] = f"intermediate_preservation_failed: {exc}"
            logger.exception("HYWorld panorama intermediate preservation failed scene=%s", scene_id)
        store.finish("success" if result["return_code"] == 0 else "failure", error=result.get("error"))
        result["intermediates"] = store.summary()
    return result


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_observability("dataevolver.hyworld_scene_pano")
    ids = {x.strip() for x in args.ids.split(",") if x.strip()} if args.ids else None
    records = load_scene_records(Path(args.scene_prompts_path), ids)
    if args.scene_image and len(records) != 1:
        raise SystemExit("--scene-image can only be used when exactly one scene record is selected")
    env = env_for_hyworld(args)
    try:
        results = []
        for record in records:
            result = run_one(record, args, env)
            results.append(result)
            status = "dry-run" if result.get("dry_run") else ("skipped" if result.get("skipped") else result.get("return_code"))
            print(f"[HYWorld pano] {record['scene_id']}: {status}")
            if result.get("dry_run"):
                print(" ".join(str(x) for x in result["command"]))
        if not args.dry_run:
            summary_path = Path(args.output_root).resolve() / "hyworld_scene_pano_summary.json"
            save_json(summary_path, {"results": results})
            if any(int(item.get("return_code", 0) or 0) != 0 for item in results if not item.get("skipped")):
                raise SystemExit(1)
    finally:
        shutdown_observability()


if __name__ == "__main__":
    main()
