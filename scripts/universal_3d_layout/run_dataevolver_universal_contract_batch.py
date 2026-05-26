#!/usr/bin/env python3
"""Run a complete DataEvolver-real universal 3D layout dataset batch.

This is the production batch wrapper for the six-paper common contract. It
launches Blender once per accepted sample, using the DataEvolver real scene pool and
existing GLB meshes, while the Blender-side script renders target, OSCR,
per-object masks, depth/order proxy, normal/orientation proxy and record
fragments.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATAEVOLVER_ROOT = Path(os.environ.get("DATAEVOLVER_ROOT") or os.environ.get("ARIS_ROOT") or "/home/jiazhuangzhuang/ARIS")
DEFAULT_RUNTIME_ROOT = Path(os.environ.get("DATAEVOLVER_RUNTIME_ROOT") or os.environ.get("ARIS_RUNTIME_ROOT") or "/aaaidata/jiazhuangzhuang/ARIS_runtime")
DEFAULT_BLENDER = Path(os.environ.get("DATAEVOLVER_BLENDER") or os.environ.get("ARIS_BLENDER") or "/aaaidata/zhangqisong/blender-4.24/blender")
DEFAULT_RECORD_SCRIPT = SCRIPT_DIR / "render_dataevolver_universal_contract_record.py"
DEFAULT_EXCLUDE_SCENES = "indoor_4blend,outdoor16"
OBJ_RE = re.compile(r"obj_(\d+)\.glb$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataevolver-root", type=Path, default=DEFAULT_DATAEVOLVER_ROOT)
    parser.add_argument("--aris-root", dest="dataevolver_root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--record-script", type=Path, default=DEFAULT_RECORD_SCRIPT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--engine", choices=["EEVEE", "CYCLES"], default="EEVEE")
    parser.add_argument("--mesh-tail", type=int, default=260)
    parser.add_argument("--max-attempts", type=int, default=300)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--max-object-ratio", type=float, default=0.08)
    parser.add_argument("--exclude-scenes", default=DEFAULT_EXCLUDE_SCENES)
    return parser.parse_args()


def parse_excluded_scenes(raw: str) -> set[str]:
    return {name.strip() for name in raw.split(",") if name.strip()}


def load_scene_pool(dataevolver_root: Path, excluded_names: set[str]) -> list[dict]:
    pool_path = dataevolver_root / "configs" / "scene_pool.json"
    data = json.loads(pool_path.read_text(encoding="utf-8"))
    usable = []
    for scene in data.get("scenes", []):
        if scene["name"] in excluded_names:
            continue
        blend_path = Path(scene["blend_path"])
        if blend_path.exists():
            usable.append(scene)
    if not usable:
        raise RuntimeError(f"No usable scenes found in {pool_path}; excluded={sorted(excluded_names)}")
    return usable


def collect_meshes(runtime_root: Path, tail: int) -> list[dict]:
    best_by_id: dict[int, Path] = {}
    for path in runtime_root.glob("new_asset_force_rot8_*/stage1_assets/meshes/obj_*.glb"):
        match = OBJ_RE.match(path.name)
        if not match:
            continue
        best_by_id[int(match.group(1))] = path
    meshes = [{"obj_id": obj_id, "path": str(path)} for obj_id, path in sorted(best_by_id.items())]
    if len(meshes) < 2:
        raise RuntimeError(f"Need at least two meshes under {runtime_root}")
    return meshes[-tail:]


def choose_pair(rng: random.Random, meshes: list[dict], attempt_idx: int) -> tuple[dict, dict]:
    high = meshes[-min(len(meshes), 80):]
    if attempt_idx % 4 == 0 and len(high) >= 2:
        return tuple(rng.sample(high, 2))
    return tuple(rng.sample(meshes, 2))


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, obj: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for name in ["target", "structure", "mask", "depth", "normal", "metadata", "logs", "fragments"]:
        (out / name).mkdir(parents=True, exist_ok=True)

    excluded = parse_excluded_scenes(args.exclude_scenes)
    scenes = load_scene_pool(args.dataevolver_root, excluded)
    meshes = collect_meshes(args.runtime_root, args.mesh_tail)

    metadata_dir = out / "metadata"
    records_path = metadata_dir / "records.jsonl"
    compat_path = metadata_dir / "st3d_compat.jsonl"
    failures_path = metadata_dir / "failures.jsonl"
    progress_path = metadata_dir / "progress.json"
    manifest_path = metadata_dir / "render_manifest.json"
    for path in [records_path, compat_path, failures_path]:
        path.write_text("", encoding="utf-8")

    manifest = {
        "schema": "dataevolver_real_universal_3d_layout_contract_batch_v1",
        "out": str(out),
        "num_samples_requested": args.num_samples,
        "seed": args.seed,
        "resolution": args.resolution,
        "engine": args.engine,
        "blender": str(args.blender),
        "record_script": str(args.record_script),
        "dataevolver_root": str(args.dataevolver_root),
        "excluded_scenes": sorted(excluded),
        "scene_pool": scenes,
        "mesh_count_available_tail": len(meshes),
        "mesh_id_min": meshes[0]["obj_id"],
        "mesh_id_max": meshes[-1]["obj_id"],
        "artifact_contract": ["target", "structure", "mask", "depth", "normal", "metadata/records.jsonl", "metadata/st3d_compat.jsonl"],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "records": [],
    }
    write_json(manifest_path, manifest)

    completed = 0
    attempts = 0
    durations: list[float] = []

    while completed < args.num_samples and attempts < args.max_attempts:
        attempts += 1
        scene = scenes[(attempts - 1) % len(scenes)]
        mesh_a, mesh_b = choose_pair(rng, meshes, attempts)
        seed = rng.randint(1, 2_000_000_000)
        record_id = f"datauniv_{completed:04d}_obj{mesh_a['obj_id']}__obj{mesh_b['obj_id']}__{scene['name']}"
        fragment_path = out / "fragments" / f"{record_id}.json"
        log_path = out / "logs" / f"{record_id}.log"
        cmd = [
            str(args.blender),
            scene["blend_path"],
            "--background",
            "--python",
            str(args.record_script),
            "--",
            "--dataevolver-root",
            str(args.dataevolver_root),
            "--obj1",
            mesh_a["path"],
            "--obj2",
            mesh_b["path"],
            "--out",
            str(out),
            "--record-id",
            record_id,
            "--scene-name",
            scene["name"],
            "--scene-type",
            scene.get("type", "unknown"),
            "--scene-template",
            scene.get("template", ""),
            "--scene-blend-path",
            scene["blend_path"],
            "--seed",
            str(seed),
            "--resolution",
            str(args.resolution),
            "--engine",
            args.engine,
            "--max-object-ratio",
            str(args.max_object_ratio),
            "--fragment-output",
            str(fragment_path),
        ]
        start = time.time()
        timed_out = False
        try:
            proc = subprocess.run(cmd, cwd=str(args.dataevolver_root), text=True, capture_output=True, timeout=args.timeout_sec)
            stdout = proc.stdout
            stderr = proc.stderr
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            returncode = 124
        duration = time.time() - start
        log_path.write_text("CMD: " + " ".join(cmd) + f"\nTIMED_OUT: {timed_out}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}", encoding="utf-8")

        base = {
            "attempt": attempts,
            "record_id": record_id,
            "scene": {"name": scene["name"], "type": scene.get("type"), "template": scene.get("template"), "blend_path": scene["blend_path"]},
            "objects": [
                {"object_id": "obj_a", "mesh_id": mesh_a["obj_id"], "mesh_path": mesh_a["path"]},
                {"object_id": "obj_b", "mesh_id": mesh_b["obj_id"], "mesh_path": mesh_b["path"]},
            ],
            "seed": seed,
            "engine": args.engine,
            "resolution": args.resolution,
            "duration_sec": round(duration, 3),
            "log": str(log_path.relative_to(out)),
        }
        ok = returncode == 0 and fragment_path.exists() and fragment_path.stat().st_size > 0
        if ok:
            try:
                fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
                record = fragment["record"]
                compat = fragment["st3d_compat"]
                for rel_path in [
                    record["render_artifacts"]["target_image"],
                    record["render_artifacts"]["structure_image"],
                    record["render_artifacts"]["depth"],
                    record["render_artifacts"]["normal"],
                    *record["render_artifacts"]["masks"],
                ]:
                    if rel_path and not (out / rel_path).exists():
                        raise FileNotFoundError(rel_path)
                append_jsonl(records_path, record)
                append_jsonl(compat_path, compat)
                completed += 1
                durations.append(duration)
                manifest["records"].append(
                    {
                        **base,
                        "sample_index": completed - 1,
                        "target_image": record["render_artifacts"]["target_image"],
                        "structure_image": record["render_artifacts"]["structure_image"],
                        "masks": record["render_artifacts"]["masks"],
                        "depth": record["render_artifacts"]["depth"],
                        "normal": record["render_artifacts"]["normal"],
                        "status": "ok",
                    }
                )
                if completed % 10 == 0 or completed == 1:
                    write_json(manifest_path, manifest)
            except Exception as exc:
                ok = False
                returncode = 125
                stderr = stderr + f"\nFRAGMENT_VALIDATION_ERROR: {exc}"

        if not ok:
            failure = {
                **base,
                "status": "failed",
                "returncode": returncode,
                "timed_out": timed_out,
                "fragment": str(fragment_path.relative_to(out)),
                "stdout_tail": stdout[-2400:],
                "stderr_tail": stderr[-2400:],
            }
            append_jsonl(failures_path, failure)

        avg = sum(durations) / len(durations) if durations else None
        progress = {
            "completed": completed,
            "requested": args.num_samples,
            "attempts": attempts,
            "failures": attempts - completed,
            "avg_success_duration_sec": round(avg, 3) if avg else None,
            "estimated_remaining_sec": round((args.num_samples - completed) * avg, 1) if avg else None,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(progress_path, progress)
        print(json.dumps(progress, ensure_ascii=False), flush=True)

    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["completed"] = completed
    manifest["attempts"] = attempts
    manifest["status"] = "complete" if completed >= args.num_samples else "incomplete"
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
