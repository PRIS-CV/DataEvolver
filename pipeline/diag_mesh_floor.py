"""
Stage 3.5 Diagnostic: Full floor-feature scan for all meshes.

Runs a multi-threshold parameter sweep to characterise the floor signal
in each mesh and drive parameter decisions for stage3_5_mesh_sanitize.py.

Output (JSON + console):
  data/mesh_diag/{obj_id}.json  — per-object feature report
  data/mesh_diag/summary.json  — cross-object summary table

Usage:
    python pipeline/diag_mesh_floor.py
    python pipeline/diag_mesh_floor.py --ids obj_003 obj_007
    python pipeline/diag_mesh_floor.py --output-dir /tmp/diag
"""

import argparse
import json
import math
import os
import sys

import numpy as np

try:
    import trimesh
    import trimesh.grouping
except ImportError:
    print("[diag] ERROR: trimesh not installed. Run: pip install trimesh")
    sys.exit(1)

try:
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, "data")
MESHES_RAW   = os.path.join(DATA_DIR, "meshes_raw")
DEFAULT_DIAG = os.path.join(DATA_DIR, "mesh_diag")

# ── Parameter sweep grid ──────────────────────────────────────────────────────
BAND_FRACS    = [0.05, 0.10, 0.15, 0.20]   # Bottom-band height fraction
NORMAL_DOTS   = [0.90, 0.95, 0.97]          # |n·up| thresholds


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_mesh(glb_path):
    """Load GLB → merged Trimesh."""
    try:
        scene = trimesh.load(glb_path, force="scene")
    except Exception as e:
        return None, f"load error: {e}"

    if isinstance(scene, trimesh.Trimesh):
        return scene, None

    meshes = []
    for name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
            if name in scene.graph.nodes:
                T = scene.graph.get(name)[0]
                geom = geom.copy()
                geom.apply_transform(T)
            meshes.append(geom)

    if not meshes:
        return None, "no valid meshes"

    return trimesh.util.concatenate(meshes), None


def hull_area_xy(verts):
    """2-D convex hull area of the XY projection; 0 if fewer than 3 pts."""
    if not HAS_SCIPY:
        # Fallback: axis-aligned bounding box area
        return float((verts[:, 0].max() - verts[:, 0].min()) *
                     (verts[:, 1].max() - verts[:, 1].min()))
    pts = np.unique(verts[:, :2], axis=0)
    if len(pts) < 3:
        return 0.0
    try:
        return float(ConvexHull(pts).volume)
    except Exception:
        return 0.0


def component_analysis(mesh, band_frac, normal_dot):
    """
    Run one (band_frac, normal_dot) combination and return per-component metrics.
    """
    verts        = mesh.vertices
    faces        = mesh.faces
    face_normals = mesh.face_normals
    face_areas   = mesh.area_faces

    v_min_z = verts[:, 2].min()
    v_max_z = verts[:, 2].max()
    total_h  = v_max_z - v_min_z
    if total_h < 1e-6:
        return []

    total_area     = float(face_areas.sum())
    mesh_fp_area   = hull_area_xy(verts)
    bottom_z_thr   = v_min_z + total_h * band_frac
    face_z_centers = verts[faces].mean(axis=1)[:, 2]
    up_dot         = np.abs(face_normals @ np.array([0, 0, 1.0]))

    seed_mask    = (face_z_centers <= bottom_z_thr) & (up_dot >= normal_dot)
    seed_ids     = np.where(seed_mask)[0]
    n_seed_faces = int(seed_ids.sum())

    if len(seed_ids) == 0:
        return []

    # Union-Find connected components on seed faces
    adj_pairs = mesh.face_adjacency
    seed_set  = set(seed_ids.tolist())
    parent    = {fid: fid for fid in seed_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a, b in adj_pairs:
        if a in seed_set and b in seed_set:
            union(a, b)

    comps = {}
    for fid in seed_ids:
        comps.setdefault(find(fid), []).append(fid)

    result = []
    for root, flist in comps.items():
        flist    = np.array(flist)
        comp_area = float(face_areas[flist].sum())
        area_frac  = comp_area / (total_area + 1e-8)

        comp_verts = verts[faces[flist].flatten()]
        z_min      = float(comp_verts[:, 2].min())
        z_max      = float(comp_verts[:, 2].max())
        thickness  = (z_max - z_min) / (total_h + 1e-8)
        fp_area    = hull_area_xy(comp_verts)
        fp_ratio   = fp_area / (mesh_fp_area + 1e-8)

        result.append({
            "n_faces":      int(len(flist)),
            "area_frac":    round(float(area_frac),    4),
            "thickness":    round(float(thickness),    4),
            "fp_ratio":     round(float(fp_ratio),     4),
            "z_min":        round(float(z_min),        4),
            "z_max":        round(float(z_max),        4),
        })

    return sorted(result, key=lambda c: -c["area_frac"])


def z_histogram_gap(mesh, n_bins=50):
    """
    Build Z-axis face-density histogram and detect the largest gap.
    Returns {"largest_gap_frac": ..., "gap_z_range": [lo, hi], "bins": [...]}
    """
    verts        = mesh.vertices
    faces        = mesh.faces
    face_z       = verts[faces].mean(axis=1)[:, 2]

    v_min_z = float(verts[:, 2].min())
    v_max_z = float(verts[:, 2].max())
    total_h  = v_max_z - v_min_z
    if total_h < 1e-6:
        return {}

    counts, edges = np.histogram(face_z, bins=n_bins)
    # Find the largest run of zero-count bins
    zero_runs = []
    i = 0
    while i < len(counts):
        if counts[i] == 0:
            j = i
            while j < len(counts) and counts[j] == 0:
                j += 1
            lo = float(edges[i])
            hi = float(edges[j])
            zero_runs.append((lo, hi, (hi - lo) / (total_h + 1e-8)))
            i = j
        else:
            i += 1

    if not zero_runs:
        largest_gap = 0.0
        gap_range   = [None, None]
    else:
        zero_runs.sort(key=lambda x: -x[2])
        largest_gap = round(zero_runs[0][2], 4)
        gap_range   = [round(zero_runs[0][0], 4), round(zero_runs[0][1], 4)]

    return {
        "largest_gap_frac": largest_gap,
        "gap_z_range":      gap_range,
        "n_empty_bins":     len(zero_runs),
    }


def topology_analysis(mesh):
    """
    Split mesh into connected components (trimesh.split).
    Return per-component bbox and bottom-fraction stats.
    """
    verts   = mesh.vertices
    v_min_z = float(verts[:, 2].min())
    v_max_z = float(verts[:, 2].max())
    total_h  = v_max_z - v_min_z

    try:
        parts = trimesh.graph.split(mesh, only_watertight=False)
    except Exception as e:
        return {"error": str(e)}

    comp_reports = []
    for part in parts:
        pv       = part.vertices
        pz_min   = float(pv[:, 2].min())
        pz_max   = float(pv[:, 2].max())
        ph       = pz_max - pz_min
        p_area   = float(part.area)
        p_frac   = p_area / (mesh.area + 1e-8)
        p_bottom = (pz_min - v_min_z) / (total_h + 1e-8)   # 0 = at global bottom
        p_top    = (pz_max - v_min_z) / (total_h + 1e-8)

        comp_reports.append({
            "n_faces":    int(len(part.faces)),
            "area_frac":  round(float(p_frac),   4),
            "z_bottom":   round(float(p_bottom), 4),
            "z_top":      round(float(p_top),    4),
            "z_height":   round(float(ph),       4),
            "is_bottom_only":  p_top < 0.15,   # entire component in bottom 15%
            "is_thin":         ph / (total_h + 1e-8) < 0.05,
        })

    comp_reports.sort(key=lambda c: -c["area_frac"])
    return {
        "n_components":        len(parts),
        "components":          comp_reports,
        "has_detached_bottom": any(
            c["is_bottom_only"] and c["is_thin"] and c["area_frac"] > 0.02
            for c in comp_reports
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-object diagnosis
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_object(obj_id, meshes_raw_dir):
    raw_path = os.path.join(meshes_raw_dir, f"{obj_id}.glb")
    if not os.path.exists(raw_path):
        return {"obj_id": obj_id, "error": "file not found"}

    mesh, err = load_mesh(raw_path)
    if mesh is None:
        return {"obj_id": obj_id, "error": err}

    verts    = mesh.vertices
    v_min_z  = float(verts[:, 2].min())
    v_max_z  = float(verts[:, 2].max())
    total_h   = v_max_z - v_min_z
    xy_extent = max(verts[:, 0].max() - verts[:, 0].min(),
                    verts[:, 1].max() - verts[:, 1].min())

    report = {
        "obj_id":     obj_id,
        "n_verts":    int(len(verts)),
        "n_faces":    int(len(mesh.faces)),
        "total_h":    round(float(total_h), 4),
        "xy_extent":  round(float(xy_extent), 4),
        "z_xy_ratio": round(float(total_h / (xy_extent + 1e-8)), 4),
    }

    # 1. Multi-threshold sweep
    sweep = {}
    for band in BAND_FRACS:
        for dot in NORMAL_DOTS:
            key   = f"band{int(band*100):02d}_dot{int(dot*100)}"
            comps = component_analysis(mesh, band, dot)
            sweep[key] = {
                "n_seed_components": len(comps),
                "top_area_frac":     comps[0]["area_frac"] if comps else 0.0,
                "top_thickness":     comps[0]["thickness"] if comps else 1.0,
                "top_fp_ratio":      comps[0]["fp_ratio"]  if comps else 0.0,
                "top_passes_soft":   (
                    comps[0]["area_frac"]  >= 0.02 and
                    comps[0]["thickness"]  <= 0.10
                ) if comps else False,
            }
    report["param_sweep"] = sweep

    # 2. Z-axis gap histogram
    report["z_gap"] = z_histogram_gap(mesh)

    # 3. Topology / connected-component analysis
    report["topology"] = topology_analysis(mesh)

    # 4. Quick summary flags
    best_sweep = max(sweep.values(), key=lambda s: s["top_area_frac"])
    report["summary"] = {
        "max_top_area_frac":   round(best_sweep["top_area_frac"], 4),
        "min_top_thickness":   round(min(s["top_thickness"] for s in sweep.values()), 4),
        "any_sweep_passes":    any(s["top_passes_soft"] for s in sweep.values()),
        "has_z_gap":           (report["z_gap"].get("largest_gap_frac", 0) > 0.05),
        "has_detached_bottom": report["topology"].get("has_detached_bottom", False),
        "floor_signal_score":  (
            (1 if best_sweep["top_area_frac"] > 0.05 else 0) +
            (1 if best_sweep["top_thickness"] < 0.10 else 0) +
            (1 if best_sweep["top_fp_ratio"]  > 0.30 else 0) +
            (1 if report["z_gap"].get("largest_gap_frac", 0) > 0.05 else 0) +
            (1 if report["topology"].get("has_detached_bottom") else 0)
        ),
    }

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mesh floor diagnostics")
    parser.add_argument("--ids", nargs="+", default=None)
    parser.add_argument("--meshes-dir", default=MESHES_RAW)
    parser.add_argument("--output-dir", default=DEFAULT_DIAG)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.ids:
        obj_ids = args.ids
    elif os.path.isdir(args.meshes_dir):
        obj_ids = sorted(
            fn.replace(".glb", "")
            for fn in os.listdir(args.meshes_dir)
            if fn.endswith(".glb")
        )
    else:
        print(f"[diag] ERROR: meshes dir not found: {args.meshes_dir}")
        sys.exit(1)

    if not obj_ids:
        print("[diag] No objects found.")
        return

    print(f"[diag] Diagnosing {len(obj_ids)} objects …")
    all_reports = []

    for i, obj_id in enumerate(obj_ids, 1):
        print(f"  [{i}/{len(obj_ids)}] {obj_id} …", end="", flush=True)
        rpt = diagnose_object(obj_id, args.meshes_dir)
        all_reports.append(rpt)

        # Save individual JSON
        out_path = os.path.join(args.output_dir, f"{obj_id}.json")
        with open(out_path, "w") as f:
            json.dump(rpt, f, indent=2)

        sig = rpt.get("summary", {})
        flags = []
        if sig.get("has_detached_bottom"):  flags.append("DETACHED")
        if sig.get("has_z_gap"):            flags.append("Z-GAP")
        if sig.get("any_sweep_passes"):     flags.append("SWEEP-PASS")
        score = sig.get("floor_signal_score", 0)
        print(f" score={score}/5  {' '.join(flags) if flags else 'clean'}")

    # Cross-object summary
    summary = []
    for r in all_reports:
        if "error" in r:
            summary.append({"obj_id": r["obj_id"], "error": r["error"]})
            continue
        s = r.get("summary", {})
        summary.append({
            "obj_id":              r["obj_id"],
            "floor_signal_score":  s.get("floor_signal_score", 0),
            "max_top_area_frac":   s.get("max_top_area_frac", 0),
            "min_top_thickness":   s.get("min_top_thickness", 1),
            "has_z_gap":           s.get("has_z_gap", False),
            "has_detached_bottom": s.get("has_detached_bottom", False),
            "any_sweep_passes":    s.get("any_sweep_passes", False),
            "n_faces":             r.get("n_faces", 0),
        })

    summary.sort(key=lambda x: -x.get("floor_signal_score", 0))

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print()
    print("=" * 75)
    print(f"{'OBJ':12s}  {'SCORE':5s}  {'AREA%':7s}  {'THICK%':7s}  "
          f"{'Z-GAP':6s}  {'DETACH':7s}  {'SWEEP':5s}")
    print("-" * 75)
    for row in summary:
        if "error" in row:
            print(f"  {row['obj_id']:10s}  ERROR: {row['error']}")
            continue
        print(
            f"  {row['obj_id']:10s}  {row['floor_signal_score']:5d}  "
            f"{row['max_top_area_frac']*100:6.1f}%  "
            f"{row['min_top_thickness']*100:6.1f}%  "
            f"{'YES':6s}" if row['has_z_gap']           else f"  {row['obj_id']:10s}  {row['floor_signal_score']:5d}  "
            f"{row['max_top_area_frac']*100:6.1f}%  "
            f"{row['min_top_thickness']*100:6.1f}%  "
            f"{'no':6s}",
            end=""
        )
        print(f"  {'YES':7s}" if row['has_detached_bottom'] else f"  {'no':7s}", end="")
        print(f"  {'YES':5s}" if row['any_sweep_passes']    else f"  {'no':5s}")

    print("=" * 75)
    print(f"\n[diag] Per-object JSONs → {args.output_dir}/")
    print(f"[diag] Summary        → {summary_path}")

    # Recommendation
    n_floor = sum(1 for r in summary if r.get("floor_signal_score", 0) >= 2)
    print(f"\n[diag] Objects with floor signal (score>=2): {n_floor}/{len(obj_ids)}")
    if n_floor:
        print("[diag] Recommendation: run stage3_5 with --conf-auto 0.65 --conf-review 0.40")


if __name__ == "__main__":
    main()
