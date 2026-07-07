"""
Stage 3: Image-to-3D — Generate 3D meshes using Hunyuan3D-2.1.
Reads .dataevolver/runtime/images/*.png and writes .dataevolver/runtime/meshes_raw/{id}.glb.
In the standard stage1-5 pipeline, these raw meshes are then promoted directly
to .dataevolver/runtime/meshes/ for downstream scene rendering. Stage 3.5 is no longer
part of the mainline flow.

Two-stage pipeline:
  1. Shape: Hunyuan3DDiTFlowMatchingPipeline → geometry .obj
  2. Paint: Hunyuan3DPaintPipeline → textured .glb (UV + PBR textures)

Use --shape-only to skip paint (debug / fallback to shape-only GLB).

Supports --ids for parallel multi-GPU runs (each worker handles a subset).

Prerequisites:
  mkdir -p .dataevolver/local/vendor
  cd .dataevolver/local/vendor
  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
  cd Hunyuan3D-2.1 && pip install -r requirements.txt

  # Compile custom_rasterizer (CUDA extension for paint):
  export PATH=/usr/local/cuda-12.1/bin:$PATH
  export CUDA_HOME=/usr/local/cuda-12.1
  export TORCH_CUDA_ARCH_LIST="8.0"
  cd hy3dpaint/custom_rasterizer && pip install -e .

  # Compile DifferentiableRenderer (C++ pybind11 extension):
  cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh
"""

import argparse
import ctypes
import json
import os
import shutil
import sys
import sysconfig
import tempfile
import traceback
from pathlib import Path
import torch

from dataevolver.paths import DATAEVOLVER_ROOT, runtime_path

LOCAL_ROOT = Path(os.environ.get("DATAEVOLVER_LOCAL_ROOT", os.fspath(DATAEVOLVER_ROOT / "local")))
LOCAL_VENDOR_ROOT = Path(os.environ.get("DATAEVOLVER_VENDOR_ROOT", os.fspath(LOCAL_ROOT / "vendor")))
LOCAL_MODEL_ROOT = Path(os.environ.get("DATAEVOLVER_MODEL_ROOT", os.fspath(LOCAL_ROOT / "model_hub")))

HUNYUAN3D_REPO = os.environ.get("HUNYUAN3D_REPO", os.fspath(LOCAL_VENDOR_ROOT / "Hunyuan3D-2.1-src"))
MODEL_HUB = os.environ.get("MODEL_HUB", os.fspath(LOCAL_MODEL_ROOT / "Hunyuan3D-2.1"))
# Paint model: hunyuan3d-paintpbr-v2-1 lives under the model hub root
# multiview_utils expects parent dir; it appends "hunyuan3d-paintpbr-v2-1" itself
PAINT_MODEL_HUB = os.environ.get("PAINT_MODEL_HUB", MODEL_HUB)
DINO_MODEL_PATH = os.environ.get("DINO_MODEL_PATH", os.fspath(LOCAL_MODEL_ROOT / "dinov2-giant"))
REALESRGAN_CKPT = os.environ.get(
    "REALESRGAN_CKPT",
    os.path.join(HUNYUAN3D_REPO, "hy3dpaint", "ckpt", "RealESRGAN_x4plus.pth"),
)
DATA_DIR = os.fspath(runtime_path())
IMAGES_DIR = os.path.join(DATA_DIR, "images")
IMAGES_RGBA_DIR = os.path.join(DATA_DIR, "images_rgba")
MESHES_DIR = os.path.join(DATA_DIR, "meshes_raw")  # Stage 4 consumes meshes/ after raw->canonical sync.
PROMPTS_PATH = os.path.join(DATA_DIR, "prompts.json")
HOME_BUILD_FIX_ROOT = Path.home() / "hunyuan3d_buildfix"


def _prepend_sys_path(path_str: str):
    if path_str and os.path.exists(path_str) and path_str not in sys.path:
        sys.path.insert(0, path_str)


def _iter_candidate_dirs(env_var: str, default_path: str, home_fallback: Path):
    seen = set()
    candidates = []
    env_value = os.environ.get(env_var)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(home_fallback)
    candidates.append(Path(default_path))
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        yield candidate


def _has_extension(module_dir: Path, module_name: str) -> bool:
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
    if ext_suffix and (module_dir / f"{module_name}{ext_suffix}").exists():
        return True
    patterns = [f"{module_name}*.so", f"{module_name}*.pyd", f"{module_name}*.dll"]
    return any(module_dir.glob(pattern) for pattern in patterns)


def _resolve_extension_dir(env_var: str, default_path: str, home_fallback: Path, module_name: str) -> str | None:
    for candidate in _iter_candidate_dirs(env_var, default_path, home_fallback):
        if candidate.exists() and _has_extension(candidate, module_name):
            return str(candidate)
    candidate = Path(default_path)
    if candidate.exists():
        return str(candidate)
    return None


def preload_torch_shared_libs():
    """Preload torch shared libs so custom CUDA extensions can resolve libc10/libtorch."""
    if os.name != "posix":
        return []

    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    loaded = []
    for lib_name in ["libc10.so", "libtorch_cpu.so", "libtorch_python.so", "libc10_cuda.so"]:
        lib_path = torch_lib_dir / lib_name
        if not lib_path.exists():
            continue
        ctypes.CDLL(str(lib_path), mode=mode)
        loaded.append(str(lib_path))
    if loaded:
        print(f"[Stage 3] Preloaded torch shared libs from {torch_lib_dir}")
    return loaded


def relax_transformers_local_load_guard():
    """Allow trusted local .bin checkpoints in this controlled offline pipeline."""
    try:
        import transformers.modeling_utils as modeling_utils
        import transformers.utils.import_utils as import_utils
    except ImportError:
        return False

    def _allow_local_trusted_torch_load():
        return None

    modeling_utils.check_torch_load_is_safe = _allow_local_trusted_torch_load
    import_utils.check_torch_load_is_safe = _allow_local_trusted_torch_load
    print("[Stage 3] Relaxed transformers torch.load safety gate for trusted local Hunyuan weights")
    return True


def resolve_rgb_images_dir(images_dir=None, rgb_images_dir=None):
    """Pick the RGB directory paired with an RGBA input directory."""
    if rgb_images_dir:
        return rgb_images_dir
    if not images_dir or images_dir == IMAGES_DIR:
        return IMAGES_DIR
    candidate = Path(images_dir)
    if candidate.name == "images_rgba":
        sibling = candidate.parent / "images"
        if sibling.exists():
            return str(sibling)
    return IMAGES_DIR


def check_model_ready():
    """Check if Hunyuan3D-2.1 weights are fully downloaded (no .aria2 files)."""
    aria2_files = []
    for root, _, files in os.walk(MODEL_HUB):
        for f in files:
            if f.endswith(".aria2"):
                aria2_files.append(os.path.join(root, f))
    if aria2_files:
        print(f"[Stage 3] ERROR: Model still downloading ({len(aria2_files)} .aria2 files remain):")
        for f in aria2_files[:5]:
            print(f"    {f}")
        print("[Stage 3] Wait for download to complete or resume with huggingface-cli")
        return False
    return True


def setup_paths():
    """Add Hunyuan3D-2.1 sub-packages to Python path."""
    for subdir in ["hy3dshape", "hy3dpaint", "."]:
        path = os.path.join(HUNYUAN3D_REPO, subdir)
        _prepend_sys_path(path)

    # Prefer caller overrides or per-user writable build outputs before the
    # read-only repo copy, so Python 3.10 workers can pick up freshly built .so files.
    diffrender_path = _resolve_extension_dir(
        env_var="HUNYUAN3D_DIFFRENDER_DIR",
        default_path=os.path.join(HUNYUAN3D_REPO, "hy3dpaint", "DifferentiableRenderer"),
        home_fallback=HOME_BUILD_FIX_ROOT / "DifferentiableRenderer",
        module_name="mesh_inpaint_processor",
    )
    if diffrender_path:
        _prepend_sys_path(str(Path(diffrender_path).parent))
        _prepend_sys_path(diffrender_path)
        print(f"[Stage 3] Using DifferentiableRenderer from {diffrender_path}")

    # custom_rasterizer package dir (so `import custom_rasterizer_kernel` works)
    # The .so is in the package dir alongside the __init__.py
    cr_path = _resolve_extension_dir(
        env_var="HUNYUAN3D_CUSTOM_RASTERIZER_DIR",
        default_path=os.path.join(HUNYUAN3D_REPO, "hy3dpaint", "custom_rasterizer"),
        home_fallback=HOME_BUILD_FIX_ROOT / "custom_rasterizer",
        module_name="custom_rasterizer_kernel",
    )
    if cr_path:
        _prepend_sys_path(cr_path)
        print(f"[Stage 3] Using custom_rasterizer from {cr_path}")


def build_paint_pipeline(device):
    """Load Hunyuan3DPaintPipeline with local model weights.

    Key path overrides (all local, no HF download):
      - multiview_pretrained_path -> .dataevolver/local/model_hub/Hunyuan3D-2.1
        (multiview_utils.py will resolve -> .../hunyuan3d-paintpbr-v2-1/)
      - dino_ckpt_path -> .dataevolver/local/model_hub/dinov2-giant
      - realesrgan_ckpt_path → hy3dpaint/ckpt/RealESRGAN_x4plus.pth
    """
    preload_torch_shared_libs()
    relax_transformers_local_load_guard()
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    paint_cfg = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)

    # Override ALL paths to use local storage (avoid HF download)
    paint_cfg.multiview_pretrained_path = PAINT_MODEL_HUB
    paint_cfg.dino_ckpt_path = DINO_MODEL_PATH
    paint_cfg.realesrgan_ckpt_path = REALESRGAN_CKPT
    paint_cfg.device = device

    # multiview_cfg_path is relative in original code ("hy3dpaint/cfgs/..."),
    # patch to absolute path so it works regardless of cwd
    paint_cfg.multiview_cfg_path = os.path.join(
        HUNYUAN3D_REPO, "hy3dpaint", "cfgs", "hunyuan-paint-pbr.yaml"
    )

    print(f"[Stage 3] Loading paint pipeline from {PAINT_MODEL_HUB} on {device}...")
    paint_pipe = Hunyuan3DPaintPipeline(paint_cfg)
    if hasattr(paint_pipe, "to"):
        paint_pipe = paint_pipe.to(device)
    print(f"[Stage 3] Paint pipeline loaded on {device}.")
    return paint_pipe


def _device_slug(device: str) -> str:
    return str(device).replace(":", "_").replace("/", "_").replace("\\", "_")


def _safe_unlink(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def analyze_rgba_quality(image_rgba) -> dict:
    """Lightweight quality gate for RGBA masks before shape/paint."""
    import numpy as np

    alpha = np.array(image_rgba.getchannel("A"), dtype=np.uint8)
    h, w = alpha.shape[:2]
    if h == 0 or w == 0:
        return {
            "usable": False,
            "reason": "empty_image",
            "foreground_ratio": 0.0,
            "bbox_ratio_w": 0.0,
            "bbox_ratio_h": 0.0,
            "touches_border": False,
        }

    mask = alpha > 8
    foreground_ratio = float(mask.mean())
    if foreground_ratio <= 0.0:
        return {
            "usable": False,
            "reason": "empty_alpha",
            "foreground_ratio": 0.0,
            "bbox_ratio_w": 0.0,
            "bbox_ratio_h": 0.0,
            "touches_border": False,
        }

    ys, xs = np.where(mask)
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    bbox_ratio_w = float((max_x - min_x + 1) / max(w, 1))
    bbox_ratio_h = float((max_y - min_y + 1) / max(h, 1))
    touches_border = bool(min_x == 0 or min_y == 0 or max_x == w - 1 or max_y == h - 1)

    component_count = None
    main_component_ratio = None
    fragment_ratio = None
    try:
        import cv2

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        component_areas = [int(area) for area in stats[1:, cv2.CC_STAT_AREA]] if num_labels > 1 else []
        component_count = len(component_areas)
        if component_areas:
            total_fg = float(sum(component_areas))
            main_area = float(max(component_areas))
            main_component_ratio = main_area / max(total_fg, 1.0)
            fragment_ratio = max(0.0, (total_fg - main_area) / max(total_fg, 1.0))
    except Exception:
        component_count = None
        main_component_ratio = None
        fragment_ratio = None

    issues = []
    if foreground_ratio < 0.01:
        issues.append("foreground_too_small")
    if foreground_ratio > 0.92:
        issues.append("foreground_too_large")
    if bbox_ratio_w < 0.08:
        issues.append("bbox_too_narrow")
    if bbox_ratio_h < 0.08:
        issues.append("bbox_too_short")
    if fragment_ratio is not None and fragment_ratio > 0.25:
        issues.append("foreground_fragmented")
    if component_count is not None and component_count > 12:
        issues.append("too_many_components")

    return {
        "usable": not issues,
        "reason": "ok" if not issues else ",".join(issues),
        "foreground_ratio": round(foreground_ratio, 6),
        "bbox_ratio_w": round(bbox_ratio_w, 6),
        "bbox_ratio_h": round(bbox_ratio_h, 6),
        "touches_border": touches_border,
        "alpha_extrema": list(image_rgba.getchannel("A").getextrema()),
        "component_count": component_count,
        "main_component_ratio": round(main_component_ratio, 6) if main_component_ratio is not None else None,
        "fragment_ratio": round(fragment_ratio, 6) if fragment_ratio is not None else None,
    }


def _material_has_image_texture(material) -> bool:
    if material is None:
        return False
    if isinstance(material, (list, tuple, set)):
        return any(_material_has_image_texture(item) for item in material)
    for attr in (
        "image",
        "baseColorTexture",
        "emissiveTexture",
        "normalTexture",
        "occlusionTexture",
        "metallicRoughnessTexture",
    ):
        value = getattr(material, attr, None)
        if value is not None:
            return True
    return False


def inspect_glb_output(glb_path: str) -> dict:
    """Best-effort textured GLB validation using trimesh metadata."""
    result = {
        "path": glb_path,
        "exists": os.path.exists(glb_path),
        "size_kb": round(os.path.getsize(glb_path) / 1024.0, 3) if os.path.exists(glb_path) else 0.0,
        "has_image_texture": None,
        "geometry_count": None,
        "reason": None,
    }
    if not result["exists"]:
        result["reason"] = "missing_output"
        return result

    try:
        import trimesh
    except ImportError:
        result["reason"] = "trimesh_unavailable"
        return result

    try:
        loaded = trimesh.load(glb_path, force="scene")
    except Exception as exc:
        result["reason"] = f"trimesh_load_failed:{exc}"
        return result

    if isinstance(loaded, trimesh.Trimesh):
        geometries = [loaded]
    else:
        geometries = list(getattr(loaded, "geometry", {}).values())

    result["geometry_count"] = len(geometries)
    has_texture = False
    for geom in geometries:
        visual = getattr(geom, "visual", None)
        if visual is None:
            continue
        if getattr(visual, "kind", None) == "texture":
            has_texture = True
            break
        material = getattr(visual, "material", None)
        if _material_has_image_texture(material):
            has_texture = True
            break

    result["has_image_texture"] = bool(has_texture)
    if not has_texture:
        result["reason"] = "no_image_texture_detected"
    return result


def save_stage3_summary(output_dir: str, device: str, obj_ids: list[str], records: list[dict], verification: dict):
    summary_path = os.path.join(output_dir, f"stage3_summary_{_device_slug(device)}.json")
    default_summary_path = os.path.join(output_dir, "stage3_summary.json")
    textured_success_count = sum(1 for item in records if item.get("textured_success"))
    shape_only_count = sum(1 for item in records if item.get("used_shape_only_fallback"))
    failed_count = sum(
        1
        for item in records
        if not item.get("textured_success") and not item.get("used_shape_only_fallback")
    )
    payload = {
        "device": device,
        "object_count": len(obj_ids),
        "objects": obj_ids,
        "stats": {
            "textured_success_count": textured_success_count,
            "shape_only_count": shape_only_count,
            "failed_count": failed_count,
        },
        "records": records,
        "verification": verification,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(default_summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[Stage 3] Wrote summary -> {summary_path}")
    return summary_path


def _resolve_paint_glb(tmp_dir: str, temp_textured_obj: str) -> str:
    temp_glb = temp_textured_obj.replace(".obj", ".glb")
    if os.path.exists(temp_glb):
        return temp_glb
    glb_candidates = [f for f in os.listdir(tmp_dir) if f.endswith(".glb")]
    if glb_candidates:
        return os.path.join(tmp_dir, glb_candidates[0])
    raise FileNotFoundError(f"Paint pipeline did not produce a .glb in {tmp_dir}")


def _attempt_textured_paint(*, paint_pipe, temp_obj: str, img_path_rgb: str, temp_textured_obj: str, use_remesh: bool) -> dict:
    paint_pipe(
        mesh_path=temp_obj,
        image_path=img_path_rgb,
        output_mesh_path=temp_textured_obj,
        use_remesh=use_remesh,
        save_glb=True,
    )
    glb_path = _resolve_paint_glb(os.path.dirname(temp_textured_obj), temp_textured_obj)
    inspection = inspect_glb_output(glb_path)
    return {
        "use_remesh": bool(use_remesh),
        "glb_path": glb_path,
        "inspection": inspection,
        "success": bool(inspection.get("has_image_texture")),
        "reason": inspection.get("reason"),
    }


def generate_meshes(obj_ids, device="cuda:0", skip_existing=True, shape_only=False,
                    images_dir=None, rgb_images_dir=None, output_dir=None, seed_base=42):
    """Generate textured 3D meshes from images using Hunyuan3D-2.1.

    Pipeline:
        image → rembg(RGBA) → shape pipeline → temp .obj
                                              ↓ (unless --shape-only)
                            paint pipeline (projects image color onto mesh)
                                              ↓
                                    textured .glb (UV + PBR)
    """
    output_dir = output_dir or MESHES_DIR
    os.makedirs(output_dir, exist_ok=True)
    setup_paths()

    # Auto-detect input directory: prefer pre-segmented RGBA if available
    if images_dir is None:
        if os.path.isdir(IMAGES_RGBA_DIR):
            images_dir = IMAGES_RGBA_DIR
        else:
            images_dir = IMAGES_DIR
    use_precut_rgba = (images_dir != IMAGES_DIR)
    rgb_images_dir = resolve_rgb_images_dir(images_dir=images_dir, rgb_images_dir=rgb_images_dir)
    if use_precut_rgba:
        print(f"[Stage 3] Using pre-segmented RGBA images from {images_dir}")
        print(f"[Stage 3] Using paired RGB images from {rgb_images_dir}")

    if not check_model_ready():
        sys.exit(1)

    # Filter already processed
    if skip_existing:
        remaining = [oid for oid in obj_ids
                     if not os.path.exists(os.path.join(output_dir, f"{oid}.glb"))]
        if len(remaining) < len(obj_ids):
            print(f"[Stage 3] Skipping {len(obj_ids) - len(remaining)} existing meshes")
        obj_ids = remaining

    if not obj_ids:
        print("[Stage 3] All meshes already exist, skipping model load")
        return

    print(f"[Stage 3] Loading shape pipeline from {MODEL_HUB} on {device}...")

    try:
        from PIL import Image
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        if not use_precut_rgba:
            from hy3dshape.rembg import BackgroundRemover
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Run: cd Hunyuan3D-2.1 && pip install -r requirements.txt")
        sys.exit(1)

    # Load shape pipeline (DiT, ~12GB VRAM)
    try:
        shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL_HUB,
            torch_dtype=torch.float16,
            device=device,
        )
    except TypeError:
        shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_HUB)
        if hasattr(shape_pipe, "to"):
            shape_pipe = shape_pipe.to(device)
        if hasattr(shape_pipe, "half"):
            shape_pipe = shape_pipe.half()

    rembg = BackgroundRemover() if not use_precut_rgba else None
    print(f"[Stage 3] Shape pipeline loaded on {device}. Processing {len(obj_ids)} objects...")

    # Load paint pipeline once (saves ~29GB VRAM re-alloc per object)
    paint_pipe = None
    if not shape_only:
        try:
            paint_pipe = build_paint_pipeline(device)
        except Exception as e:
            print(f"[WARN] Failed to load paint pipeline: {e}")
            print("[WARN] Falling back to shape-only mode (no UV texture).")
            import traceback
            traceback.print_exc()

    for i, obj_id in enumerate(obj_ids):
        img_path_input = os.path.join(images_dir, f"{obj_id}.png")
        img_path_rgb = os.path.join(rgb_images_dir, f"{obj_id}.png")
        out_path = os.path.join(output_dir, f"{obj_id}.glb")

        if not os.path.exists(img_path_input):
            print(f"[WARN] Image not found: {img_path_input}, skipping")
            continue

        print(f"[Stage 3][{device}] ({i+1}/{len(obj_ids)}) Processing: {obj_id} ...")

        try:
            if use_precut_rgba:
                image_rgba = Image.open(img_path_input).convert("RGBA")
                # Validate alpha channel is meaningful
                alpha = image_rgba.split()[-1]
                extrema = alpha.getextrema()
                if extrema == (0, 0) or extrema == (255, 255):
                    print(f"[WARN] {obj_id}: RGBA alpha invalid (extrema={extrema}), fallback to rembg")
                    image = Image.open(img_path_rgb).convert("RGB")
                    from hy3dshape.rembg import BackgroundRemover
                    _rembg = BackgroundRemover()
                    image_rgba = _rembg(image)
                    del _rembg
            else:
                image = Image.open(img_path_input).convert("RGB")
                image_rgba = rembg(image)

            # ── Stage 3a: Shape generation ─────────────────────────────
            with torch.no_grad():
                outputs = shape_pipe(
                    image=image_rgba,
                    num_inference_steps=50,
                    guidance_scale=5.0,
                    seed=seed_base + i,
                )
            mesh = outputs[0]

            if paint_pipe is None:
                # shape-only fallback: export directly to GLB (no UV texture)
                mesh.export(out_path)
                size_kb = os.path.getsize(out_path) / 1024
                print(f"    [shape-only] Saved → {out_path} ({size_kb:.1f} KB)")
                if size_kb < 10:
                    print(f"[WARN] Mesh suspiciously small ({size_kb:.1f} KB)")
                continue

            # ── Stage 3b: Texture painting ────────────────────────────
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Export geometry as .obj (paint pipeline requires .obj input)
                temp_obj = os.path.join(tmp_dir, f"{obj_id}.obj")
                mesh.export(temp_obj)

                temp_textured_obj = os.path.join(tmp_dir, f"{obj_id}_textured.obj")
                try:
                    paint_pipe(
                        mesh_path=temp_obj,
                        image_path=img_path_rgb,      # original RGB for color projection
                        output_mesh_path=temp_textured_obj,
                        use_remesh=True,
                        save_glb=True,
                    )
                    # paint pipeline writes .glb alongside the .obj
                    temp_glb = temp_textured_obj.replace(".obj", ".glb")
                    if not os.path.exists(temp_glb):
                        # some versions name it differently — look for any .glb in tmp
                        glb_candidates = [
                            f for f in os.listdir(tmp_dir) if f.endswith(".glb")
                        ]
                        if glb_candidates:
                            temp_glb = os.path.join(tmp_dir, glb_candidates[0])
                        else:
                            raise FileNotFoundError(
                                f"Paint pipeline did not produce a .glb in {tmp_dir}"
                            )
                    shutil.move(temp_glb, out_path)
                    size_kb = os.path.getsize(out_path) / 1024
                    print(f"    [textured] Saved → {out_path} ({size_kb:.1f} KB)")
                    if size_kb < 50:
                        print(f"[WARN] Textured mesh suspiciously small ({size_kb:.1f} KB)")
                except Exception as paint_err:
                    print(f"[WARN] Paint failed for {obj_id}: {paint_err}")
                    print("[WARN] Falling back to shape-only GLB for this object.")
                    import traceback
                    traceback.print_exc()
                    # fallback: export shape-only .glb so pipeline can continue
                    mesh.export(out_path)
                    size_kb = os.path.getsize(out_path) / 1024
                    print(f"    [shape-only fallback] Saved → {out_path} ({size_kb:.1f} KB)")

        except Exception as e:
            print(f"[WARN] Failed to generate mesh for {obj_id}: {e}")
            import traceback
            traceback.print_exc()

    del shape_pipe
    if rembg is not None:
        del rembg
    if paint_pipe is not None:
        del paint_pipe
    torch.cuda.empty_cache()
    print(f"[Stage 3][{device}] GPU memory released.")


def verify_outputs(obj_ids, output_dir=None):
    """Check all expected meshes exist and have reasonable size."""
    output_dir = output_dir or MESHES_DIR
    missing = []
    for oid in obj_ids:
        path = os.path.join(output_dir, f"{oid}.glb")
        if not os.path.exists(path) or os.path.getsize(path) < 10 * 1024:
            missing.append(oid)

    if missing:
        print(f"[Stage 3] WARN: Missing/small meshes: {missing}")
    else:
        print(f"[Stage 3] ✓ All {len(obj_ids)} meshes generated successfully")
    return len(missing) == 0


def generate_meshes(obj_ids, device="cuda:0", skip_existing=True, shape_only=False,
                    images_dir=None, rgb_images_dir=None, output_dir=None, seed_base=42):
    """Generate textured 3D meshes from images using Hunyuan3D-2.1."""
    output_dir = output_dir or MESHES_DIR
    os.makedirs(output_dir, exist_ok=True)
    setup_paths()

    if images_dir is None:
        if os.path.isdir(IMAGES_RGBA_DIR):
            images_dir = IMAGES_RGBA_DIR
        else:
            images_dir = IMAGES_DIR
    use_precut_rgba = (images_dir != IMAGES_DIR)
    rgb_images_dir = resolve_rgb_images_dir(images_dir=images_dir, rgb_images_dir=rgb_images_dir)
    if use_precut_rgba:
        print(f"[Stage 3] Using pre-segmented RGBA images from {images_dir}")
        print(f"[Stage 3] Using paired RGB images from {rgb_images_dir}")

    if not check_model_ready():
        sys.exit(1)

    if skip_existing:
        remaining = [oid for oid in obj_ids if not os.path.exists(os.path.join(output_dir, f"{oid}.glb"))]
        if len(remaining) < len(obj_ids):
            print(f"[Stage 3] Skipping {len(obj_ids) - len(remaining)} existing meshes")
        obj_ids = remaining

    if not obj_ids:
        print("[Stage 3] All meshes already exist, skipping model load")
        return []

    print(f"[Stage 3] Loading shape pipeline from {MODEL_HUB} on {device}...")

    try:
        from PIL import Image
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        if not use_precut_rgba:
            from hy3dshape.rembg import BackgroundRemover
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Run: cd Hunyuan3D-2.1 && pip install -r requirements.txt")
        sys.exit(1)

    try:
        shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL_HUB,
            torch_dtype=torch.float16,
            device=device,
        )
    except TypeError:
        shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_HUB)
        if hasattr(shape_pipe, "to"):
            shape_pipe = shape_pipe.to(device)
        if hasattr(shape_pipe, "half"):
            shape_pipe = shape_pipe.half()

    rembg = BackgroundRemover() if not use_precut_rgba else None
    print(f"[Stage 3] Shape pipeline loaded on {device}. Processing {len(obj_ids)} objects...")

    paint_pipe = None
    if not shape_only:
        try:
            paint_pipe = build_paint_pipeline(device)
        except Exception as e:
            print(f"[WARN] Failed to load paint pipeline: {e}")
            print("[WARN] Falling back to shape-only mode (no UV texture).")
            import traceback
            traceback.print_exc()

    records = []
    for i, obj_id in enumerate(obj_ids):
        img_path_input = os.path.join(images_dir, f"{obj_id}.png")
        img_path_rgb = os.path.join(rgb_images_dir, f"{obj_id}.png")
        out_path = os.path.join(output_dir, f"{obj_id}.glb")
        record = {
            "obj_id": obj_id,
            "device": device,
            "input_rgba_path": img_path_input,
            "input_rgb_path": img_path_rgb,
            "used_precut_rgba": bool(use_precut_rgba),
            "rgba_source": "precut_rgba" if use_precut_rgba else "rembg",
            "rgba_quality": None,
            "shape_success": False,
            "paint_pipeline_loaded": paint_pipe is not None,
            "paint_attempts": [],
            "textured_success": False,
            "used_shape_only_fallback": False,
            "final_output_inspection": None,
            "status": None,
            "error": None,
        }
        _safe_unlink(out_path)

        if not os.path.exists(img_path_input):
            print(f"[WARN] Image not found: {img_path_input}, skipping")
            record["status"] = "missing_input_image"
            records.append(record)
            continue

        print(f"[Stage 3][{device}] ({i+1}/{len(obj_ids)}) Processing: {obj_id} ...")

        try:
            if use_precut_rgba:
                image_rgba = Image.open(img_path_input).convert("RGBA")
                alpha = image_rgba.split()[-1]
                extrema = alpha.getextrema()
                if extrema == (0, 0) or extrema == (255, 255):
                    print(f"[WARN] {obj_id}: RGBA alpha invalid (extrema={extrema}), fallback to rembg")
                    image = Image.open(img_path_rgb).convert("RGB")
                    from hy3dshape.rembg import BackgroundRemover
                    _rembg = BackgroundRemover()
                    image_rgba = _rembg(image)
                    del _rembg
                    record["rgba_source"] = "rembg_fallback_invalid_alpha"
            else:
                image = Image.open(img_path_input).convert("RGB")
                image_rgba = rembg(image)

            rgba_quality = analyze_rgba_quality(image_rgba)
            record["rgba_quality"] = rgba_quality
            if not rgba_quality.get("usable") and os.path.exists(img_path_rgb):
                print(f"[WARN] {obj_id}: RGBA quality={rgba_quality.get('reason')}, retrying rembg")
                image = Image.open(img_path_rgb).convert("RGB")
                from hy3dshape.rembg import BackgroundRemover
                _rembg = BackgroundRemover()
                image_rgba = _rembg(image)
                del _rembg
                record["rgba_source"] = "rembg_fallback_quality"
                rgba_quality = analyze_rgba_quality(image_rgba)
                record["rgba_quality"] = rgba_quality

            if not rgba_quality.get("usable"):
                print(f"[WARN] {obj_id}: unusable RGBA after fallback ({rgba_quality.get('reason')})")
                record["status"] = "rgba_unusable"
                records.append(record)
                continue

            with torch.no_grad():
                outputs = shape_pipe(
                    image=image_rgba,
                    num_inference_steps=50,
                    guidance_scale=5.0,
                    seed=seed_base + i,
                )
            mesh = outputs[0]
            record["shape_success"] = True

            if paint_pipe is None:
                mesh.export(out_path)
                size_kb = os.path.getsize(out_path) / 1024
                print(f"    [shape-only] Saved -> {out_path} ({size_kb:.1f} KB)")
                if size_kb < 10:
                    print(f"[WARN] Mesh suspiciously small ({size_kb:.1f} KB)")
                record["used_shape_only_fallback"] = True
                record["status"] = "paint_pipeline_unavailable_shape_only"
                record["final_output_inspection"] = inspect_glb_output(out_path)
                records.append(record)
                continue

            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_obj = os.path.join(tmp_dir, f"{obj_id}.obj")
                mesh.export(temp_obj)
                temp_textured_obj = os.path.join(tmp_dir, f"{obj_id}_textured.obj")
                textured_success = False
                for use_remesh in (True, False):
                    attempt_info = {"use_remesh": bool(use_remesh), "success": False}
                    try:
                        attempt_result = _attempt_textured_paint(
                            paint_pipe=paint_pipe,
                            temp_obj=temp_obj,
                            img_path_rgb=img_path_rgb,
                            temp_textured_obj=temp_textured_obj,
                            use_remesh=use_remesh,
                        )
                        attempt_info.update(attempt_result)
                        record["paint_attempts"].append(attempt_info)
                        if attempt_result["success"]:
                            shutil.move(attempt_result["glb_path"], out_path)
                            size_kb = os.path.getsize(out_path) / 1024
                            print(f"    [textured] Saved -> {out_path} ({size_kb:.1f} KB) | use_remesh={use_remesh}")
                            if size_kb < 50:
                                print(f"[WARN] Textured mesh suspiciously small ({size_kb:.1f} KB)")
                            record["textured_success"] = True
                            record["status"] = "textured_success"
                            textured_success = True
                            break
                        print(f"[WARN] {obj_id}: paint output without texture (use_remesh={use_remesh})")
                    except Exception as paint_err:
                        attempt_info["error"] = str(paint_err)
                        record["paint_attempts"].append(attempt_info)
                        print(f"[WARN] Paint failed for {obj_id} (use_remesh={use_remesh}): {paint_err}")
                        import traceback
                        traceback.print_exc()

                if not textured_success:
                    print("[WARN] Falling back to shape-only GLB for this object.")
                    mesh.export(out_path)
                    size_kb = os.path.getsize(out_path) / 1024
                    print(f"    [shape-only fallback] Saved -> {out_path} ({size_kb:.1f} KB)")
                    record["used_shape_only_fallback"] = True
                    record["status"] = "paint_failed_shape_only_fallback"

                record["final_output_inspection"] = inspect_glb_output(out_path)
                if record["textured_success"] and not record["final_output_inspection"].get("has_image_texture"):
                    record["textured_success"] = False
                    record["status"] = "textured_output_failed_validation"

        except Exception as e:
            print(f"[WARN] Failed to generate mesh for {obj_id}: {e}")
            import traceback
            traceback.print_exc()
            record["status"] = "generation_failed"
            record["error"] = str(e)
            _safe_unlink(out_path)

        if record["final_output_inspection"] is None and os.path.exists(out_path):
            record["final_output_inspection"] = inspect_glb_output(out_path)
        if record["status"] is None:
            record["status"] = "unknown"
        records.append(record)

    del shape_pipe
    if rembg is not None:
        del rembg
    if paint_pipe is not None:
        del paint_pipe
    torch.cuda.empty_cache()
    print(f"[Stage 3][{device}] GPU memory released.")
    return records


def verify_outputs(obj_ids, output_dir=None, require_image_texture=True):
    """Check all expected meshes exist, have reasonable size, and optionally carry image textures."""
    output_dir = output_dir or MESHES_DIR
    missing = []
    too_small = []
    missing_texture = []
    inspections = {}
    for oid in obj_ids:
        path = os.path.join(output_dir, f"{oid}.glb")
        inspection = inspect_glb_output(path)
        inspections[oid] = inspection
        if not inspection["exists"]:
            missing.append(oid)
            continue
        if inspection["size_kb"] < 10.0:
            too_small.append(oid)
        if require_image_texture and inspection.get("has_image_texture") is not True:
            missing_texture.append(oid)

    success = not missing and not too_small and not missing_texture
    if success:
        print(f"[Stage 3] All {len(obj_ids)} meshes generated successfully")
    else:
        if missing:
            print(f"[Stage 3] WARN: Missing meshes: {missing}")
        if too_small:
            print(f"[Stage 3] WARN: Suspiciously small meshes: {too_small}")
        if missing_texture:
            print(f"[Stage 3] WARN: Meshes without image texture: {missing_texture}")
    return {
        "success": success,
        "missing": missing,
        "too_small": too_small,
        "missing_texture": missing_texture,
        "require_image_texture": bool(require_image_texture),
        "inspections": inspections,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Image-to-3D mesh generation with texture")
    parser.add_argument("--device", default="cuda:0",
                        help="CUDA device (e.g. cuda:0, cuda:1)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Force regeneration even if output .glb already exists")
    parser.add_argument("--ids", default=None,
                        help="Comma-separated list of object IDs to process (e.g. obj_001,obj_002). "
                             "Default: all objects in prompts.json")
    parser.add_argument("--shape-only", action="store_true",
                        help="Skip texture painting, output shape-only GLB (faster, for debugging)")
    parser.add_argument("--images-dir", default=None,
                        help="Override input images directory. "
                             "Default: auto-detect (images_rgba/ if exists, else images/)")
    parser.add_argument("--rgb-images-dir", default=None,
                        help="Optional RGB directory paired with --images-dir when using RGBA inputs")
    parser.add_argument("--output-dir", default=MESHES_DIR,
                        help="Where to write textured/raw GLBs")
    parser.add_argument("--seed-base", type=int, default=42,
                        help="Base seed for deterministic regeneration runs")
    parser.add_argument("--prompts-path", default=PROMPTS_PATH,
                        help="Prompt file used to enumerate valid object IDs")
    args = parser.parse_args()

    with open(args.prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    all_ids = [p["id"] for p in prompts]

    if args.ids:
        obj_ids = [x.strip() for x in args.ids.split(",") if x.strip()]
        invalid = [oid for oid in obj_ids if oid not in all_ids]
        if invalid:
            print(f"[WARN] Unknown object IDs: {invalid}")
        obj_ids = [oid for oid in obj_ids if oid in all_ids]
    else:
        obj_ids = all_ids

    mode = "shape-only" if args.shape_only else "shape+paint (textured)"
    print(f"[Stage 3] Mode: {mode}")
    print(f"[Stage 3] Processing {len(obj_ids)} objects on {args.device}: {obj_ids}")

    records = generate_meshes(
        obj_ids,
        device=args.device,
        skip_existing=not args.no_skip,
        shape_only=args.shape_only,
        images_dir=args.images_dir,
        rgb_images_dir=args.rgb_images_dir,
        output_dir=args.output_dir,
        seed_base=args.seed_base,
    )
    verification = verify_outputs(
        obj_ids,
        output_dir=args.output_dir,
        require_image_texture=not args.shape_only,
    )
    save_stage3_summary(args.output_dir, args.device, obj_ids, records, verification)
    if not verification.get("success", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
