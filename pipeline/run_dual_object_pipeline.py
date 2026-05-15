#!/usr/bin/env python3
"""
End-to-end dual object pipeline:
  1. Stage 1: Generate prompts for two objects via Claude API
  2. Stage 2: T2I generate images
  3. Stage 3: Image-to-3D mesh generation
  4. Stage 4: Random scene selection + dual placement + render

Usage:
    python pipeline/run_dual_object_pipeline.py \
        --concepts "park_bench" "fire_hydrant" \
        --gpu 0 \
        --output-dir pipeline/data/dual_renders

    # Or with random concept pair from seed list:
    python pipeline/run_dual_object_pipeline.py --random-pair --gpu 0
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
BLENDER_BIN = "blender"

SCENE_POOL = [
    "<large-file-root>",
    "<large-file-root>",
    "<large-file-root>",
    "<large-file-root>",
    "<large-file-root>",
    "<large-file-root>",
    "<large-file-root>",
    "<large-file-root>",
    "<large-file-root>",
    "<large-file-root>",
    "<large-file-root>",
    "<large-file-root>",
    "<large-file-root>",
    "<large-file-root>",
    "<large-file-root>",
    "<large-file-root>",
    "<large-file-root>",
    "<large-file-root>",
]

# PLACEHOLDER_CONTINUE

CONCEPT_PAIRS = [
    ("park_bench", "fire_hydrant"),
    ("wooden_chair", "ceramic_mug"),
    ("bicycle", "traffic_cone"),
    ("potted_plant", "garden_gnome"),
    ("mailbox", "street_lamp"),
    ("motorcycle", "helmet"),
    ("suitcase", "umbrella"),
    ("guitar", "amplifier"),
    ("telescope", "tripod"),
    ("lantern", "backpack"),
]


def run_cmd(cmd, desc="", check=True):
    print(f"\n{'='*60}")
    print(f"[DUAL] {desc}")
    print(f"  CMD: {cmd[:200]}...")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-10:]:
            print(f"  {line}")
    if result.returncode != 0 and check:
        print(f"  STDERR: {result.stderr[-500:]}")
        raise RuntimeError(f"Command failed: {desc}")
    return result


def stage1_generate_prompts(concept_a, concept_b, output_path, api_base_url=None, template_only=False):
    """Generate T2I prompts for two concepts using stage1 logic."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from stage1_text_expansion import build_scene_conditioned_prompt, SCENE_PROFILE_4BLEND

    prompts = []
    for i, concept in enumerate([concept_a, concept_b]):
        obj_id = f"dual_{i+1:03d}"
        print(f"[Stage 1] Generating prompt for: {concept} (id={obj_id})")

        if template_only:
            result = build_scene_conditioned_prompt(concept, SCENE_PROFILE_4BLEND)
        else:
            try:
                from stage1_text_expansion import call_claude_api
                result = call_claude_api(
                    concept_name=concept,
                    model="claude-sonnet-4-20250514",
                    api_base_url=api_base_url,
                )
            except Exception as e:
                print(f"[Stage 1] API call failed ({e}), using template fallback")
                result = build_scene_conditioned_prompt(concept, SCENE_PROFILE_4BLEND)

        prompts.append({
            "id": obj_id,
            "name": concept,
            "prompt": result["prompt"],
            "features": result.get("features", {}),
        })
        print(f"  Prompt: {result['prompt'][:100]}...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"[Stage 1] Saved prompts to {output_path}")
    return prompts


def stage2_generate_images(prompts_path, output_dir, device="cuda:0"):
    """Generate T2I images using stage2."""
    cmd = (
        f"cd {REPO_ROOT} && python pipeline/stage2_t2i_generate.py "
        f"--prompts {prompts_path} "
        f"--output-dir {output_dir} "
        f"--device {device}"
    )
    run_cmd(cmd, "Stage 2: T2I image generation")
    return output_dir


def stage3_generate_meshes(images_dir, output_dir, device="cuda:0"):
    """Generate 3D meshes from images using stage3."""
    cmd = (
        f"cd {REPO_ROOT} && python pipeline/stage3_image_to_3d.py "
        f"--images-dir {images_dir} "
        f"--output-dir {output_dir} "
        f"--device {device}"
    )
    run_cmd(cmd, "Stage 3: Image-to-3D mesh generation")
    return output_dir


def stage4_dual_render(mesh1_path, mesh2_path, scene_blend, output_path, seed=None):
    """Render two objects in a scene using dual placement demo."""
    if seed is None:
        seed = random.randint(0, 99999)
    cmd = (
        f"{BLENDER_BIN} {scene_blend} --background "
        f"--python {SCRIPT_DIR}/demo_dual_object_placement.py -- "
        f"--obj1 {mesh1_path} --obj2 {mesh2_path} "
        f"--output {output_path} --seed {seed}"
    )
    run_cmd(cmd, f"Stage 4: Dual render in {Path(scene_blend).stem}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end dual object pipeline")
    parser.add_argument("--concepts", nargs=2, metavar=("A", "B"),
                        help="Two object concepts to generate")
    parser.add_argument("--random-pair", action="store_true",
                        help="Pick a random concept pair")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--output-dir", type=str,
                        default=str(DATA_DIR / "dual_renders"),
                        help="Output directory for renders")
    parser.add_argument("--scene", type=str, default=None,
                        help="Specific scene blend file (random if not set)")
    parser.add_argument("--skip-stage1", action="store_true",
                        help="Skip prompt generation (use existing prompts)")
    parser.add_argument("--skip-stage2", action="store_true",
                        help="Skip T2I (use existing images)")
    parser.add_argument("--skip-stage3", action="store_true",
                        help="Skip mesh generation (use existing meshes)")
    parser.add_argument("--meshes-dir", type=str, default=None,
                        help="Path to directory with pre-existing meshes (for skip-stage3)")
    parser.add_argument("--api-base-url", type=str, default=None,
                        help="Relay API base URL for stage1")
    parser.add_argument("--template-only", action="store_true",
                        help="Use template prompts instead of API calls in stage1")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for placement")
    return parser.parse_args()


def main():
    args = parse_args()
    device = f"cuda:{args.gpu}"
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Pick concepts
    if args.concepts:
        concept_a, concept_b = args.concepts
    elif args.random_pair:
        concept_a, concept_b = random.choice(CONCEPT_PAIRS)
    else:
        print("ERROR: specify --concepts A B or --random-pair")
        sys.exit(1)

    print(f"[DUAL] Pipeline start: {concept_a} + {concept_b}")
    print(f"[DUAL] GPU: {device}, timestamp: {timestamp}")

    # Working directories for this run
    run_dir = DATA_DIR / f"dual_run_{timestamp}"
    prompts_path = run_dir / "prompts.json"
    images_dir = run_dir / "images"
    meshes_dir = run_dir / "meshes"
    output_dir = Path(args.output_dir)

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Stage 1: Prompt generation
    if not args.skip_stage1:
        prompts = stage1_generate_prompts(
            concept_a, concept_b, str(prompts_path),
            api_base_url=args.api_base_url,
            template_only=args.template_only,
        )
    else:
        with open(prompts_path) as f:
            prompts = json.load(f)
        print(f"[DUAL] Skipped stage 1, loaded {len(prompts)} prompts")

    # Stage 2: T2I
    if not args.skip_stage2:
        os.makedirs(images_dir, exist_ok=True)
        stage2_generate_images(str(prompts_path), str(images_dir), device=device)
    else:
        print(f"[DUAL] Skipped stage 2")

    # Stage 3: Image-to-3D
    if not args.skip_stage3:
        os.makedirs(meshes_dir, exist_ok=True)
        stage3_generate_meshes(str(images_dir), str(meshes_dir), device=device)
    else:
        print(f"[DUAL] Skipped stage 3")

    # Find generated meshes
    if args.meshes_dir:
        meshes_dir = Path(args.meshes_dir)
    mesh1 = meshes_dir / "dual_001.glb"
    mesh2 = meshes_dir / "dual_002.glb"
    if not mesh1.exists() or not mesh2.exists():
        # Try alternative naming
        glbs = sorted(meshes_dir.glob("*.glb"))
        if len(glbs) >= 2:
            mesh1, mesh2 = glbs[0], glbs[1]
        else:
            print(f"[DUAL] ERROR: Need 2 meshes, found {len(glbs)} in {meshes_dir}")
            sys.exit(1)

    print(f"[DUAL] Meshes: {mesh1.name}, {mesh2.name}")

    # Stage 4: Scene render
    if args.scene:
        scene_blend = args.scene
    else:
        scene_blend = random.choice(SCENE_POOL)
    print(f"[DUAL] Selected scene: {Path(scene_blend).stem}")

    render_name = f"{concept_a}__{concept_b}__{Path(scene_blend).stem}_{timestamp}.png"
    render_path = output_dir / render_name

    stage4_dual_render(
        str(mesh1), str(mesh2), scene_blend,
        str(render_path), seed=args.seed,
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"[DUAL] Pipeline complete!")
    print(f"  Concepts: {concept_a} + {concept_b}")
    print(f"  Scene: {Path(scene_blend).stem}")
    print(f"  Render: {render_path}")
    print(f"  Run dir: {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
