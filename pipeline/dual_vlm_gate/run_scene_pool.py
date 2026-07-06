from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GATE_LOOP = THIS_DIR / "dual_gate_loop.py"
DEFAULT_POOL = THIS_DIR / "dual_scene_pool.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Run dual VLM gate over every scene in a scene pool")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--sample-prefix", required=True)
    parser.add_argument("--mesh-a", required=True)
    parser.add_argument("--mesh-b", required=True)
    parser.add_argument("--ref-a", default=None)
    parser.add_argument("--ref-b", default=None)
    parser.add_argument("--obj-a-id", default="object_a")
    parser.add_argument("--obj-b-id", default="object_b")
    parser.add_argument("--scene-pool", default=str(DEFAULT_POOL))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--max-rounds", type=int, default=9)
    parser.add_argument("--init-candidates", type=int, default=8)
    parser.add_argument("--use-vlm-init-review", action="store_true")
    parser.add_argument("--vlm-init-weight", type=float, default=0.55)
    parser.add_argument("--vlm-init-min-score", type=float, default=0.35)
    parser.add_argument("--skip-init-search", action="store_true")
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument("--engine", default="CYCLES", choices=["EEVEE", "CYCLES"])
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    return parser.parse_args()


def load_pool(path: str | Path) -> list[dict]:
    pool_path = Path(path).resolve()
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    scenes = payload.get("scenes") or []
    if not scenes:
        raise ValueError(f"No scenes in pool: {path}")
    for scene in scenes:
        placement_state_path = scene.get("placement_state_path")
        if placement_state_path:
            placement_path = Path(str(placement_state_path))
            if not placement_path.is_absolute():
                placement_path = (pool_path.parent / placement_path).resolve()
            scene["placement_state_path"] = str(placement_path)
    return scenes


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def main():
    args = parse_args()
    root = Path(args.root_dir).resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    scenes = load_pool(args.scene_pool)
    (root / "scene_pool_snapshot.json").write_text(
        json.dumps(scenes, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    results = []
    for idx, scene in enumerate(scenes):
        name = str(scene.get("name") or f"scene{idx:02d}")
        blend_path = str(scene.get("blend_path") or "")
        placement_state_path = str(scene.get("placement_state_path") or "")
        sample_id = f"{args.sample_prefix}_{idx:02d}_{name}"
        scene_control = scene.get("initial_control_state") or scene.get("control_state") or None
        scene_control_path = None
        if isinstance(scene_control, dict) and scene_control:
            control_state = deep_merge({}, scene_control)
            scene_control_path = root / "initial_states" / f"{sample_id}.json"
            scene_control_path.parent.mkdir(parents=True, exist_ok=True)
            scene_control_path.write_text(json.dumps(control_state, indent=2, ensure_ascii=False), encoding="utf-8")
        cmd = [
            sys.executable,
            str(GATE_LOOP),
            "--root-dir",
            str(root / "gate_loop"),
            "--sample-id",
            sample_id,
            "--obj-a-id",
            args.obj_a_id,
            "--obj-b-id",
            args.obj_b_id,
            "--mesh-a",
            args.mesh_a,
            "--mesh-b",
            args.mesh_b,
            "--scene",
            blend_path,
            "--device",
            args.device,
            "--threshold",
            str(args.threshold),
            "--max-rounds",
            str(args.max_rounds),
            "--seed",
            str(args.seed + idx),
            "--engine",
            args.engine,
            "--resolution",
            str(args.resolution),
            "--init-candidates",
            str(args.init_candidates),
        ]
        if scene_control_path:
            cmd += ["--initial-control-state", str(scene_control_path)]
        if placement_state_path:
            cmd += ["--placement-state", placement_state_path, "--skip-init-search"]
        if args.use_vlm_init_review:
            cmd += [
                "--use-vlm-init-review",
                "--vlm-init-weight",
                str(args.vlm_init_weight),
                "--vlm-init-min-score",
                str(args.vlm_init_min_score),
            ]
        if args.skip_init_search:
            cmd += ["--skip-init-search"]
        if args.ref_a:
            cmd += ["--ref-a", args.ref_a]
        if args.ref_b:
            cmd += ["--ref-b", args.ref_b]
        log_path = logs / f"{sample_id}.log"
        print(f"[scene-pool] start {sample_id}: {blend_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        result = {"sample_id": sample_id, "scene": scene, "returncode": proc.returncode, "log_path": str(log_path)}
        results.append(result)
        (root / "scene_pool_run_summary.json").write_text(
            json.dumps({"results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[scene-pool] done {sample_id}: exit={proc.returncode}", flush=True)
        if proc.returncode != 0 and not args.continue_on_error:
            raise SystemExit(proc.returncode)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
