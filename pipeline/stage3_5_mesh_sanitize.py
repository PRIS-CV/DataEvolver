"""
Stage 3.5: Mesh Sanitize — Remove pseudo-floor planes from Hunyuan3D-generated GLBs.

Root cause: T2I reference images with residual ground → rembg imperfect removal
→ Hunyuan3D reconstructs the residual as 3D geometry (a flat slab at the base).

Algorithm (RANSAC + multi-condition scoring):
  1. Find seed faces in the bottom 5% height band with up-facing normals (|n·up| >= 0.97)
  2. Connected-component analysis on seeds
  3. Candidate filter (ALL must hold):
       area_frac    >= 0.08   (>=8% of total surface area)
       footprint    >= 0.45   (XY footprint >= 45% of mesh footprint)
       thickness    <= 0.03   (Z extent <= 3% of total mesh height)
       overhang     >= 1.15   (XY footprint >= 1.15x median object XY extent)
       ransac_inlier >= 0.85  (>=85% of seed faces fit a plane within 2mm)
  4. Weighted confidence:
       0.30*area_score + 0.20*footprint_score + 0.15*overhang_score
     + 0.20*ransac_score + 0.15*thinness_score
  5. Decision:
       conf >= 0.78 → auto-remove
       0.55 <= conf < 0.78 → flag for review (skip removal, mark in QC JSON)
       conf < 0.55 → keep (no floor detected)
  6. Removal: prefer connected-component deletion; fallback to half-space cut
  7. Protection rules:
       - Flat objects (Z_extent / XY_extent <= 0.15): skip
       - Height loss after removal > 8%: abort removal
       - Remaining faces < 200: abort removal

Input:  data/meshes_raw/{obj_id}.glb  (original Hunyuan3D output)
Output: data/meshes/{obj_id}.glb      (sanitized, ready for Stage 4)
        data/mesh_qc/{obj_id}.json    (QC report with floor_confidence and details)

Usage:
    python pipeline/stage3_5_mesh_sanitize.py                    # all objects
    python pipeline/stage3_5_mesh_sanitize.py --dry-run          # QC only, no deletion
    python pipeline/stage3_5_mesh_sanitize.py --ids obj_001 obj_003
    python pipeline/stage3_5_mesh_sanitize.py --conf-auto 0.78 --conf-review 0.55
"""

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import numpy as np

# trimesh is required; install with: pip install trimesh
try:
    import trimesh
    import trimesh.grouping
except ImportError:
    print("[Stage 3.5] ERROR: trimesh not installed. Run: pip install trimesh")
    sys.exit(1)


# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MESHES_RAW_DIR = os.path.join(DATA_DIR, "meshes_raw")   # Stage 3 output (original)
MESHES_OUT_DIR = os.path.join(DATA_DIR, "meshes")       # Stage 3.5 output (cleaned)
QC_DIR = os.path.join(DATA_DIR, "mesh_qc")

# ── Thresholds ─────────────────────────────────────────────────────────────────
CONF_AUTO   = 0.78   # >= this → auto remove
CONF_REVIEW = 0.55   # >= this → flag for review (no removal)

# Filter thresholds
BOTTOM_BAND_FRAC  = 0.05   # Bottom 5% of mesh height
NORMAL_UP_DOT     = 0.97   # |n·up| threshold for horizontal faces
MIN_AREA_FRAC     = 0.08   # Candidate must cover >= 8% of total surface area
MIN_FOOTPRINT     = 0.45   # XY footprint ratio >= 45%
MAX_THICKNESS     = 0.03   # Z extent <= 3% of total height
MIN_OVERHANG      = 1.15   # XY footprint >= 1.15x median object XY side
RANSAC_INLIER_THR = 0.85   # RANSAC plane inlier ratio
RANSAC_DIST_TOL   = 0.02   # Plane fit distance tolerance (in normalized units)

# Protection rules
MIN_REMAIN_FACES  = 200    # Don't leave mesh with fewer than this many faces
MAX_HEIGHT_LOSS   = 0.08   # Abort if height loss > 8%
FLAT_RATIO_THR    = 0.15   # Z_extent/XY_extent <= this → object is flat, skip


@dataclass
class FloorCandidate:
    """Detected floor component with scoring details."""
    component_id: int
    face_indices: List[int]
    area: float
    area_frac: float
    z_min: float
    z_max: float
    thickness: float          # (z_max - z_min) / total_height
    xy_footprint_area: float  # convex hull area in XY
    footprint_ratio: float    # xy_footprint / mesh_xy_footprint
    overhang_ratio: float     # xy_footprint / (median_xy_extent ** 2)  approx
    ransac_inlier_ratio: float
    # Scores (0-1 each)
    area_score: float = 0.0
    footprint_score: float = 0.0
    overhang_score: float = 0.0
    ransac_score: float = 0.0
    thinness_score: float = 0.0
    confidence: float = 0.0


@dataclass
class QCReport:
    obj_id: str
    n_faces_original: int
    n_faces_cleaned: int
    total_height: float
    z_extent_frac: float        # Z_extent / XY_extent (flat object check)
    floor_candidates: List[dict] = field(default_factory=list)
    decision: str = "keep"      # "removed", "review", "keep", "skip_flat"
    floor_confidence: float = 0.0
    height_loss_frac: float = 0.0
    error: Optional[str] = None


# ── Core geometry helpers ──────────────────────────────────────────────────────

def load_mesh(glb_path: str) -> Optional[trimesh.Trimesh]:
    """Load GLB and merge all submeshes into a single Trimesh."""
    try:
        scene = trimesh.load(glb_path, force="scene")
    except Exception as e:
        print(f"  [load] ERROR loading {glb_path}: {e}")
        return None

    if isinstance(scene, trimesh.Trimesh):
        return scene

    meshes = []
    for name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
            # Apply scene transform if available
            if name in scene.graph.nodes:
                T = scene.graph.get(name)[0]
                geom = geom.copy()
                geom.apply_transform(T)
            meshes.append(geom)

    if not meshes:
        print(f"  [load] No valid meshes found in {glb_path}")
        return None

    merged = trimesh.util.concatenate(meshes)
    return merged


def ransac_plane_fit(points: np.ndarray, dist_tol: float = RANSAC_DIST_TOL,
                     n_iter: int = 100) -> Tuple[np.ndarray, float]:
    """
    Simple RANSAC plane fitting.
    Returns (plane_normal, inlier_ratio).
    Points should have shape (N, 3).
    """
    if len(points) < 3:
        return np.array([0, 0, 1]), 0.0

    best_inliers = 0
    best_normal = np.array([0.0, 0.0, 1.0])
    n = len(points)

    for _ in range(n_iter):
        # Sample 3 points
        idx = np.random.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        v1, v2 = p1 - p0, p2 - p0
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-8:
            continue
        normal = normal / norm_len
        # Distance from all points to plane
        dists = np.abs((points - p0) @ normal)
        inliers = int((dists < dist_tol).sum())
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal

    return best_normal, best_inliers / n


def convex_hull_area_xy(vertices: np.ndarray) -> float:
    """2D convex hull area of the XY projection."""
    from scipy.spatial import ConvexHull
    pts_2d = vertices[:, :2]
    # Remove duplicates
    pts_2d = np.unique(pts_2d, axis=0)
    if len(pts_2d) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts_2d)
        return float(hull.volume)  # In 2D, .volume = area
    except Exception:
        return 0.0


# ── Floor detection ───────────────────────────────────────────────────────────

def detect_floor(mesh: trimesh.Trimesh) -> Tuple[List[FloorCandidate], dict]:
    """
    Detect floor candidates in the mesh.
    Returns list of FloorCandidate objects and a dict of mesh statistics.
    """
    vertices = mesh.vertices
    faces = mesh.faces
    face_normals = mesh.face_normals
    face_areas = mesh.area_faces

    v_min_z = vertices[:, 2].min()
    v_max_z = vertices[:, 2].max()
    total_height = v_max_z - v_min_z
    if total_height < 1e-6:
        return [], {"total_height": 0, "flat": True}

    # XY extents for flat-object check
    xy_extent = max(
        vertices[:, 0].max() - vertices[:, 0].min(),
        vertices[:, 1].max() - vertices[:, 1].min()
    )
    z_extent_frac = total_height / (xy_extent + 1e-8)

    mesh_xy_footprint = convex_hull_area_xy(vertices)
    total_area = float(face_areas.sum())

    stats = {
        "total_height": float(total_height),
        "z_extent_frac": float(z_extent_frac),
        "mesh_xy_footprint": float(mesh_xy_footprint),
        "total_area": float(total_area),
        "flat": z_extent_frac <= FLAT_RATIO_THR,
    }

    # Step 1: Find seed faces — bottom 5% + horizontal normal
    bottom_z_thr = v_min_z + total_height * BOTTOM_BAND_FRAC
    face_z_centers = vertices[faces].mean(axis=1)[:, 2]  # (F,)
    up = np.array([0, 0, 1])
    normal_dot = np.abs(face_normals @ up)  # (F,)

    seed_mask = (face_z_centers <= bottom_z_thr) & (normal_dot >= NORMAL_UP_DOT)
    seed_face_ids = np.where(seed_mask)[0]

    if len(seed_face_ids) == 0:
        return [], stats

    # Step 2: Connected-component analysis on seed faces
    # Build face adjacency (shared edges)
    # Use trimesh's face adjacency (pairs of adjacent face indices)
    adj_pairs = mesh.face_adjacency  # (E, 2)

    # Filter adjacency to only seed faces
    seed_set = set(seed_face_ids.tolist())
    seed_adj = [(a, b) for a, b in adj_pairs if a in seed_set and b in seed_set]

    # Union-Find for connected components
    parent = {fid: fid for fid in seed_face_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a, b in seed_adj:
        union(a, b)

    # Group faces by component root
    components = {}
    for fid in seed_face_ids:
        root = find(fid)
        components.setdefault(root, []).append(fid)

    # Step 3: Score each component as a floor candidate
    candidates = []
    # Approximate median XY side for overhang computation
    median_xy_side = math.sqrt(mesh_xy_footprint + 1e-8)

    for comp_id, (root, face_list) in enumerate(components.items()):
        face_list = np.array(face_list)
        comp_area = float(face_areas[face_list].sum())
        area_frac = comp_area / (total_area + 1e-8)

        # Z extent of component
        comp_verts = vertices[faces[face_list].flatten()]
        comp_z_min = float(comp_verts[:, 2].min())
        comp_z_max = float(comp_verts[:, 2].max())
        comp_thickness = (comp_z_max - comp_z_min) / (total_height + 1e-8)

        # XY footprint
        comp_fp_area = convex_hull_area_xy(comp_verts)
        footprint_ratio = comp_fp_area / (mesh_xy_footprint + 1e-8)

        # Overhang: fp_area relative to median XY cross-section area of object
        # Approximated as footprint / (median_xy_side ** 2)
        overhang_ratio = comp_fp_area / (median_xy_side ** 2 + 1e-8)

        # RANSAC on face centroids (fast approximation)
        face_centers = face_z_centers[face_list]
        comp_normals = face_normals[face_list]
        _, ransac_inlier = ransac_plane_fit(
            vertices[faces[face_list].flatten()]
        )

        # Step 3: Candidate filter (all must hold)
        passes_filter = (
            area_frac >= MIN_AREA_FRAC and
            footprint_ratio >= MIN_FOOTPRINT and
            comp_thickness <= MAX_THICKNESS and
            overhang_ratio >= MIN_OVERHANG and
            ransac_inlier >= RANSAC_INLIER_THR
        )
        if not passes_filter:
            continue

        # Step 4: Weighted confidence score
        # Normalize each dimension to [0, 1] with soft clipping
        area_score     = min(area_frac / 0.40, 1.0)           # saturates at 40%
        footprint_score = min(footprint_ratio / 0.90, 1.0)    # saturates at 90%
        overhang_score  = min((overhang_ratio - MIN_OVERHANG) / 0.60, 1.0)
        ransac_score    = min((ransac_inlier - RANSAC_INLIER_THR) / (1.0 - RANSAC_INLIER_THR), 1.0)
        thinness_score  = min((MAX_THICKNESS - comp_thickness) / MAX_THICKNESS, 1.0)

        confidence = (
            0.30 * area_score +
            0.20 * footprint_score +
            0.15 * overhang_score +
            0.20 * ransac_score +
            0.15 * thinness_score
        )

        cand = FloorCandidate(
            component_id=comp_id,
            face_indices=face_list.tolist(),
            area=comp_area,
            area_frac=float(area_frac),
            z_min=comp_z_min,
            z_max=comp_z_max,
            thickness=float(comp_thickness),
            xy_footprint_area=float(comp_fp_area),
            footprint_ratio=float(footprint_ratio),
            overhang_ratio=float(overhang_ratio),
            ransac_inlier_ratio=float(ransac_inlier),
            area_score=float(area_score),
            footprint_score=float(footprint_score),
            overhang_score=float(overhang_score),
            ransac_score=float(ransac_score),
            thinness_score=float(thinness_score),
            confidence=float(confidence),
        )
        candidates.append(cand)

    # Sort by confidence descending
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates, stats


# ── Mesh removal ──────────────────────────────────────────────────────────────

def remove_floor_faces(mesh: trimesh.Trimesh,
                       candidate: FloorCandidate,
                       total_height: float) -> Tuple[Optional[trimesh.Trimesh], str]:
    """
    Remove floor faces from mesh.
    Returns (cleaned_mesh, status_message).
    Status: "ok", "abort_height_loss", "abort_min_faces"
    """
    face_ids_to_remove = set(candidate.face_indices)

    # Try to expand to entire connected component at the same Z level
    # (handles cases where seed component is slightly under-estimated)
    all_face_ids = np.array(list(face_ids_to_remove))

    # Protection: check remaining face count
    n_remain = len(mesh.faces) - len(face_ids_to_remove)
    if n_remain < MIN_REMAIN_FACES:
        return None, "abort_min_faces"

    # Build cleaned mesh by keeping non-floor faces
    keep_mask = np.ones(len(mesh.faces), dtype=bool)
    keep_mask[all_face_ids] = False

    cleaned = mesh.copy()
    cleaned.update_faces(keep_mask)
    cleaned.remove_unreferenced_vertices()

    # Protection: check height loss
    if len(cleaned.vertices) == 0:
        return None, "abort_empty"

    orig_height = total_height
    new_height = float(cleaned.vertices[:, 2].max() - cleaned.vertices[:, 2].min())
    height_loss = (orig_height - new_height) / (orig_height + 1e-8)

    if height_loss > MAX_HEIGHT_LOSS:
        return None, f"abort_height_loss_{height_loss:.3f}"

    return cleaned, "ok"


# ── Export ────────────────────────────────────────────────────────────────────

def export_glb(mesh: trimesh.Trimesh, out_path: str) -> bool:
    """Export mesh as GLB. Returns True on success."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Export to GLB
        export_bytes = trimesh.exchange.gltf.export_glb(mesh)
        with open(out_path, "wb") as f:
            f.write(export_bytes)
        return True
    except Exception as e:
        print(f"  [export] ERROR: {e}")
        # Fallback: use trimesh's own export
        try:
            mesh.export(out_path)
            return True
        except Exception as e2:
            print(f"  [export] Fallback also failed: {e2}")
            return False


# ── Per-object processing ─────────────────────────────────────────────────────

def process_object(obj_id: str,
                   dry_run: bool = False,
                   conf_auto: float = CONF_AUTO,
                   conf_review: float = CONF_REVIEW) -> QCReport:
    """
    Process a single object: detect floor, optionally remove, export.
    """
    raw_path = os.path.join(MESHES_RAW_DIR, f"{obj_id}.glb")
    out_path = os.path.join(MESHES_OUT_DIR, f"{obj_id}.glb")
    qc_path  = os.path.join(QC_DIR, f"{obj_id}.json")

    report = QCReport(
        obj_id=obj_id,
        n_faces_original=0,
        n_faces_cleaned=0,
        total_height=0.0,
        z_extent_frac=0.0,
    )

    if not os.path.exists(raw_path):
        report.error = f"raw mesh not found: {raw_path}"
        print(f"  [{obj_id}] SKIP: {report.error}")
        return report

    # Load mesh
    mesh = load_mesh(raw_path)
    if mesh is None:
        report.error = "failed to load mesh"
        return report

    report.n_faces_original = len(mesh.faces)
    print(f"  [{obj_id}] Loaded: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    # Flat object check
    candidates, stats = detect_floor(mesh)
    report.total_height = float(stats.get("total_height", 0))
    report.z_extent_frac = float(stats.get("z_extent_frac", 0))
    report.floor_candidates = [asdict(c) for c in candidates]

    if stats.get("flat", False):
        report.decision = "skip_flat"
        report.n_faces_cleaned = report.n_faces_original
        print(f"  [{obj_id}] SKIP: flat object (z/xy={report.z_extent_frac:.2f})")
        # Still copy to output dir
        if not dry_run:
            os.makedirs(MESHES_OUT_DIR, exist_ok=True)
            shutil.copy2(raw_path, out_path)
        return report

    if not candidates:
        report.decision = "keep"
        report.floor_confidence = 0.0
        report.n_faces_cleaned = report.n_faces_original
        print(f"  [{obj_id}] No floor candidates → keep")
        if not dry_run:
            os.makedirs(MESHES_OUT_DIR, exist_ok=True)
            shutil.copy2(raw_path, out_path)
        return report

    # Use the highest-confidence candidate
    best = candidates[0]
    report.floor_confidence = best.confidence

    conf_str = f"conf={best.confidence:.3f} area={best.area_frac:.2%} fp={best.footprint_ratio:.2%}"
    print(f"  [{obj_id}] Floor candidate: {conf_str}")

    if best.confidence >= conf_auto:
        # Auto remove
        if dry_run:
            report.decision = "removed_dry"
            report.n_faces_cleaned = report.n_faces_original - len(best.face_indices)
            print(f"  [{obj_id}] DRY-RUN: would remove {len(best.face_indices)} faces")
        else:
            cleaned, status = remove_floor_faces(mesh, best, stats["total_height"])
            if status == "ok" and cleaned is not None:
                report.n_faces_cleaned = len(cleaned.faces)
                report.height_loss_frac = float(
                    (stats["total_height"] - float(cleaned.vertices[:, 2].max() - cleaned.vertices[:, 2].min()))
                    / (stats["total_height"] + 1e-8)
                )
                success = export_glb(cleaned, out_path)
                if success:
                    report.decision = "removed"
                    print(f"  [{obj_id}] REMOVED floor: {report.n_faces_original}→{report.n_faces_cleaned} faces, height_loss={report.height_loss_frac:.2%}")
                else:
                    report.decision = "keep"
                    report.error = "export failed"
                    shutil.copy2(raw_path, out_path)
            else:
                report.decision = "keep"
                report.error = status
                report.n_faces_cleaned = report.n_faces_original
                print(f"  [{obj_id}] ABORT removal: {status} → keep original")
                shutil.copy2(raw_path, out_path)

    elif best.confidence >= conf_review:
        report.decision = "review"
        report.n_faces_cleaned = report.n_faces_original
        print(f"  [{obj_id}] FLAG for review (conf={best.confidence:.3f})")
        if not dry_run:
            os.makedirs(MESHES_OUT_DIR, exist_ok=True)
            shutil.copy2(raw_path, out_path)

    else:
        report.decision = "keep"
        report.n_faces_cleaned = report.n_faces_original
        print(f"  [{obj_id}] KEEP (low confidence={best.confidence:.3f})")
        if not dry_run:
            os.makedirs(MESHES_OUT_DIR, exist_ok=True)
            shutil.copy2(raw_path, out_path)

    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global MESHES_RAW_DIR, MESHES_OUT_DIR, QC_DIR
    parser = argparse.ArgumentParser(description="Stage 3.5: Mesh Floor Sanitize")
    parser.add_argument("--ids", nargs="+", default=None,
                        help="Process only these object IDs (e.g. obj_001 obj_003)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only compute QC scores; do not write cleaned meshes")
    parser.add_argument("--conf-auto", type=float, default=CONF_AUTO,
                        help=f"Confidence threshold for auto removal (default {CONF_AUTO})")
    parser.add_argument("--conf-review", type=float, default=CONF_REVIEW,
                        help=f"Confidence threshold for review flag (default {CONF_REVIEW})")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True,
                        help="Skip objects where cleaned GLB already exists")
    parser.add_argument("--no-skip", dest="skip_existing", action="store_false",
                        help="Reprocess objects even if cleaned GLB already exists")
    parser.add_argument("--input-dir", default=MESHES_RAW_DIR,
                        help="Input raw mesh directory")
    parser.add_argument("--output-dir", default=MESHES_OUT_DIR,
                        help="Output cleaned mesh directory")
    parser.add_argument("--qc-dir", default=QC_DIR,
                        help="Output QC report directory")
    args = parser.parse_args()

    MESHES_RAW_DIR = args.input_dir
    MESHES_OUT_DIR = args.output_dir
    QC_DIR = args.qc_dir

    os.makedirs(QC_DIR, exist_ok=True)
    if not args.dry_run:
        os.makedirs(MESHES_OUT_DIR, exist_ok=True)

    # Collect object IDs
    if args.ids:
        obj_ids = args.ids
    elif os.path.isdir(MESHES_RAW_DIR):
        obj_ids = sorted(
            fn.replace(".glb", "")
            for fn in os.listdir(MESHES_RAW_DIR)
            if fn.endswith(".glb")
        )
    else:
        print(f"[Stage 3.5] ERROR: meshes_raw dir not found: {MESHES_RAW_DIR}")
        sys.exit(1)

    if not obj_ids:
        print("[Stage 3.5] No objects to process.")
        return

    print(f"[Stage 3.5] Processing {len(obj_ids)} objects")
    print(f"[Stage 3.5] conf_auto={args.conf_auto}, conf_review={args.conf_review}")
    print(f"[Stage 3.5] dry_run={args.dry_run}")
    print()

    summary = {"removed": 0, "review": 0, "keep": 0, "skip_flat": 0, "error": 0}

    for i, obj_id in enumerate(obj_ids):
        # Skip existing cleaned mesh
        out_path = os.path.join(MESHES_OUT_DIR, f"{obj_id}.glb")
        qc_path  = os.path.join(QC_DIR, f"{obj_id}.json")
        if args.skip_existing and not args.dry_run and os.path.exists(out_path):
            print(f"[{i+1}/{len(obj_ids)}] {obj_id}: SKIP (already exists)")
            summary["keep"] += 1
            continue

        print(f"[{i+1}/{len(obj_ids)}] {obj_id}")
        try:
            report = process_object(
                obj_id,
                dry_run=args.dry_run,
                conf_auto=args.conf_auto,
                conf_review=args.conf_review,
            )
        except Exception as e:
            import traceback
            print(f"  [{obj_id}] EXCEPTION: {e}")
            traceback.print_exc()
            report = QCReport(
                obj_id=obj_id,
                n_faces_original=0,
                n_faces_cleaned=0,
                total_height=0.0,
                z_extent_frac=0.0,
                decision="error",
                error=str(e),
            )

        # Save QC report
        qc_data = asdict(report)
        with open(qc_path, "w") as f:
            json.dump(qc_data, f, indent=2)

        dec = report.decision
        if "removed" in dec:
            summary["removed"] += 1
        elif dec == "review":
            summary["review"] += 1
        elif dec == "skip_flat":
            summary["skip_flat"] += 1
        elif report.error:
            summary["error"] += 1
        else:
            summary["keep"] += 1

    # Print summary
    print()
    print("=" * 50)
    print("[Stage 3.5] Summary:")
    print(f"  Removed (auto):  {summary['removed']}")
    print(f"  Flagged (review): {summary['review']}")
    print(f"  Kept (no floor): {summary['keep']}")
    print(f"  Skipped (flat):  {summary['skip_flat']}")
    print(f"  Errors:          {summary['error']}")
    print(f"  Total:           {sum(summary.values())}")
    print()
    print(f"  QC reports: {QC_DIR}/")
    if not args.dry_run:
        print(f"  Cleaned meshes: {MESHES_OUT_DIR}/")
    print("=" * 50)

    # If there are objects flagged for review, list them
    review_list = []
    for obj_id in obj_ids:
        qc_path = os.path.join(QC_DIR, f"{obj_id}.json")
        if os.path.exists(qc_path):
            with open(qc_path) as f:
                qc = json.load(f)
            if qc.get("decision") == "review":
                review_list.append((obj_id, qc.get("floor_confidence", 0)))

    if review_list:
        print(f"\n[Stage 3.5] Objects needing manual review ({len(review_list)}):")
        for oid, conf in sorted(review_list, key=lambda x: -x[1]):
            print(f"  {oid}  conf={conf:.3f}")


if __name__ == "__main__":
    main()
