#!/usr/bin/env python3
"""
Dual Pipeline: 共享 VLM gate → 分组 rotation export → 合并 trainready。

架构：
  Phase 1: 所有 N+M 物体共享 seed → T2I → SAM → 3D → VLM gate (steps 01-04)
  Phase 2: 按 seed 顺序将 accepted 物体分为 Group A (前 N) 和 Group B (后 M)
  Phase 3A: Group A → rotation8 export (全角度) → angle gate → trainready (全目标角度)
  Phase 3B: Group B → rotation8 export (0° + 弱势角度) → angle gate → VLM 质量过滤 → trainready (仅弱势角度)
  Phase 4: 合并 A + B + 可选 baseline
  Phase 5: object-disjoint split
  Phase 6: 验证

用法:
    python scripts/run_dual_pipeline.py \
        --full-objects 3 \
        --weak-objects 3 \
        --weak-angles 225 \
        --skip-training

完整参数见 --help。
"""
import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_ROOT = Path(os.environ.get(
    "ARIS_RUNTIME_ROOT",
    "<runtime-root>",
))
PYTHON = os.environ.get(
    "ARIS_PYTHON",
    "python",
)
BLENDER = os.environ.get(
    "ARIS_BLENDER",
    "blender",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 0: 弱势角度分析
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_weak_angles(csv_path: str, threshold: float = 0.98) -> list[int]:
    angle_metrics: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = re.search(r"angle(\d+)", row.get("image_name", ""))
            if not m:
                continue
            angle_idx = int(m.group(1))
            for k in ("psnr", "ssim", "lpips", "dino_similarity"):
                val = row.get(k, "")
                if val:
                    angle_metrics[angle_idx][k].append(float(val))

    composites: dict[int, float] = {}
    for idx, metrics in angle_metrics.items():
        deg = idx * 45
        if deg == 0:
            continue
        psnr = _mean(metrics["psnr"])
        ssim = _mean(metrics["ssim"])
        lpips = _mean(metrics["lpips"])
        dino = _mean(metrics["dino_similarity"])
        composites[deg] = psnr / 20 + ssim + (1 - lpips) + dino

    avg = sum(composites.values()) / len(composites) if composites else 0
    weak = sorted(deg for deg, s in composites.items() if s < avg * threshold)
    return weak


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def _find_next_seed_index() -> int:
    max_idx = 1100
    for p in RUNTIME_ROOT.glob("**/seed_concepts.json"):
        try:
            data = json.loads(p.read_text())
            for item in data if isinstance(data, list) else [data]:
                obj_id = item.get("id", "")
                m = re.search(r"obj_(\d+)", obj_id)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
        except Exception:
            continue
    return max_idx + 1


def run_cmd(cmd: list[str], label: str, log_path: Path | None = None, **kwargs):
    print(f"[{label}] CMD: {' '.join(str(c) for c in cmd)}")
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as fh:
            result = subprocess.run(
                cmd, stdout=fh, stderr=subprocess.STDOUT, **kwargs,
            )
    else:
        result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        msg = f"[{label}] 失败 (exit={result.returncode})"
        if log_path:
            msg += f"，日志: {log_path}"
        raise RuntimeError(msg)
    print(f"[{label}] 完成")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: 共享 VLM gate（所有 N+M 物体一起跑 steps 01-04）
# ═══════════════════════════════════════════════════════════════════════════════

def run_shared_gate(
    total_count: int,
    seed_start_index: int,
    gate_max_rounds: int,
    gate_threshold: float,
    log_dir: Path,
    no_force_accept: bool = False,
) -> dict:
    script = REPO_ROOT / "scripts" / "launch_new_asset_force_rot8.sh"
    cmd = [
        "bash", str(script),
        "--seed-count", str(total_count),
        "--seed-start-index", str(seed_start_index),
        "--gate-max-rounds", str(gate_max_rounds),
        "--gate-threshold", str(gate_threshold),
        "--no-augment",
        "--stop-after-gate",
    ]
    if no_force_accept:
        cmd.append("--no-force-accept")

    env = os.environ.copy()
    log_file = log_dir / "phase1_shared_gate.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Phase 1] 共享 gate: {total_count} 物体, start_index={seed_start_index}")
    print(f"[Phase 1] 日志: {log_file}")

    with open(log_file, "w") as fh:
        result = subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT,
            env=env, cwd=str(REPO_ROOT),
        )

    if result.returncode != 0:
        raise RuntimeError(f"[Phase 1] 共享 gate 失败 (exit={result.returncode})，日志: {log_file}")

    log_text = log_file.read_text(errors="ignore")
    m = re.search(r"(?:运行目录|运行根目录):\s*(\S+)", log_text)
    if not m:
        raise RuntimeError("[Phase 1] 无法从日志中找到运行目录")

    gate_run_root = Path(m.group(1))
    loop_state_path = gate_run_root / "LOOP_STATE.json"
    if not loop_state_path.exists():
        raise RuntimeError(f"[Phase 1] LOOP_STATE.json 不存在: {loop_state_path}")

    loop_state = json.loads(loop_state_path.read_text())
    print(f"[Phase 1] 完成: gate_run_root={gate_run_root}")
    return loop_state


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: 拆分 accepted 物体为 Group A 和 Group B
# ═══════════════════════════════════════════════════════════════════════════════

def split_accepted_objects(
    gate_state: dict,
    n_full: int,
    n_weak: int,
    run_root: Path,
    group_ratio: str = None,
) -> tuple[Path, Path, list[str], list[str]]:
    seed_file = Path(gate_state["seed_concepts_file"])
    accepted_root = Path(gate_state["accepted_pair_root"])

    seeds = json.loads(seed_file.read_text())
    all_obj_ids = [s["id"] for s in seeds]

    accepted_dirs = set()
    if accepted_root.exists():
        for d in accepted_root.iterdir():
            if d.is_dir():
                m = re.match(r"(obj_\d+)", d.name)
                if m:
                    accepted_dirs.add(m.group(1))

    accepted_obj_ids = [oid for oid in all_obj_ids if oid in accepted_dirs]
    rejected_obj_ids = [oid for oid in all_obj_ids if oid not in accepted_dirs]

    if rejected_obj_ids:
        print(f"[Phase 2] 被 VLM gate 淘汰的物体: {rejected_obj_ids}")

    if group_ratio:
        parts = group_ratio.split(":")
        ra, rb = int(parts[0]), int(parts[1])
        n_accepted = len(accepted_obj_ids)
        n_a = math.ceil(n_accepted * ra / (ra + rb))
        n_b = n_accepted - n_a
        group_a_ids = accepted_obj_ids[:n_a]
        group_b_ids = accepted_obj_ids[n_a:n_a + n_b]
        print(f"[Phase 2] 按比例 {ra}:{rb} 分组 → A={n_a}, B={n_b} (accepted={n_accepted})")
    else:
        group_a_ids = accepted_obj_ids[:n_full]
        group_b_ids = accepted_obj_ids[n_full:n_full + n_weak]

    print(f"[Phase 2] 实际 accepted: {len(accepted_obj_ids)}/{len(all_obj_ids)}")
    print(f"[Phase 2] Group A (全角度): {group_a_ids}")
    print(f"[Phase 2] Group B (弱角度): {group_b_ids}")

    accepted_a = run_root / "accepted_pairs_A"
    accepted_b = run_root / "accepted_pairs_B"

    for group_dir, ids in [(accepted_a, group_a_ids), (accepted_b, group_b_ids)]:
        group_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(accepted_root.iterdir()):
            if not src.is_dir():
                continue
            obj_match = re.match(r"(obj_\d+)", src.name)
            if obj_match and obj_match.group(1) in ids:
                dst = group_dir / src.name
                if not dst.exists():
                    dst.symlink_to(src.resolve())

    return accepted_a, accepted_b, group_a_ids, group_b_ids


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: rotation8 export + angle gate + trainready
# ═══════════════════════════════════════════════════════════════════════════════

def run_rotation8_export(
    source_root: Path,
    output_dir: Path,
    meshes_dir: str,
    rotations: str,
    gpus: str,
    log_dir: Path,
    label: str,
):
    cmd = [
        PYTHON,
        str(REPO_ROOT / "scripts" / "export_rotation8_from_best_object_state.py"),
        "--source-root", str(source_root),
        "--output-dir", str(output_dir),
        "--python", PYTHON,
        "--blender", BLENDER,
        "--meshes-dir", meshes_dir,
        "--gpus", gpus,
        "--fallback-to-best-any-angle",
        "--rotations", rotations,
    ]
    run_cmd(cmd, label, log_path=log_dir / f"{label}_rotation8_export.log")


def run_angle_gate(
    source_root: Path,
    output_dir: Path,
    meshes_dir: str,
    profile: str,
    device: str,
    angle_threshold: float,
    max_enhance_rounds: int,
    log_dir: Path,
    label: str,
    gpus: str = None,
):
    cmd = [
        PYTHON,
        str(REPO_ROOT / "scripts" / "run_rotation8_angle_gate.py"),
        "--source-root", str(source_root),
        "--output-dir", str(output_dir),
        "--meshes-dir", meshes_dir,
        "--blender", BLENDER,
        "--profile", profile,
        "--device", device,
        "--angle-threshold", str(angle_threshold),
        "--max-enhance-rounds", str(max_enhance_rounds),
        "--copy-mode", "symlink",
    ]
    if gpus:
        cmd += ["--gpus", gpus]
    run_cmd(cmd, label, log_path=log_dir / f"{label}_angle_gate.log")


def run_build_trainready(
    source_root: Path,
    output_dir: Path,
    target_rotations: str,
    prompts_json: str,
    log_dir: Path,
    label: str,
):
    cmd = [
        PYTHON,
        str(REPO_ROOT / "scripts" / "build_rotation8_trainready_dataset.py"),
        "--source-root", str(source_root),
        "--output-dir", str(output_dir),
        "--copy-mode", "symlink",
        "--target-rotations", target_rotations,
        "--prompts-json", prompts_json,
    ]
    run_cmd(cmd, label, log_path=log_dir / f"{label}_trainready.log")


def filter_by_angle_gate_scores(
    enhanced_root: Path,
    output_dir: Path,
    weak_angles: list[int],
    reject_threshold: float,
) -> tuple[Path, list[str], list[str]]:
    """读取 angle gate 输出的 per-angle VLM 分数，过滤掉弱角度分数低于阈值的物体。"""
    summary_path = enhanced_root / "summary.json"
    if not summary_path.exists():
        print(f"[weak_gate] summary.json 不存在，跳过过滤")
        return enhanced_root, [], []

    summary = json.loads(summary_path.read_text())
    per_angle_scores = summary.get("per_angle_scores", {})

    passed_ids = []
    rejected_ids = []
    reject_details = {}

    for obj_id, scores in sorted(per_angle_scores.items()):
        obj_passed = True
        obj_scores = {}
        for angle in weak_angles:
            score = float(scores.get(str(angle), 0))
            obj_scores[angle] = score
            if score < reject_threshold:
                obj_passed = False

        if obj_passed:
            passed_ids.append(obj_id)
            tag = "PASS"
        else:
            rejected_ids.append(obj_id)
            reject_details[obj_id] = obj_scores
            tag = "REJECT"

        score_str = ", ".join(f"{a}°={obj_scores[a]:.3f}" for a in weak_angles)
        print(f"  [weak_gate] {obj_id}: [{tag}] {score_str} (threshold={reject_threshold})")

    if not rejected_ids:
        print(f"[weak_gate] 全部 {len(passed_ids)} 物体通过")
        return enhanced_root, passed_ids, rejected_ids

    output_dir.mkdir(parents=True, exist_ok=True)
    objects_dir = output_dir / "objects"
    objects_dir.mkdir(exist_ok=True)

    manifest_path = enhanced_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        filtered_objects = {
            k: v for k, v in manifest.get("objects", {}).items()
            if k in passed_ids
        }
        manifest["objects"] = filtered_objects
        manifest["weak_gate_applied"] = True
        manifest["weak_gate_rejected"] = rejected_ids
        manifest["weak_gate_reject_details"] = {
            oid: {str(a): s for a, s in scores.items()}
            for oid, scores in reject_details.items()
        }
        manifest["weak_gate_threshold"] = reject_threshold
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False)
        )

    src_objects = enhanced_root / "objects"
    for obj_id in passed_ids:
        src = src_objects / obj_id
        dst = objects_dir / obj_id
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())

    for f in enhanced_root.iterdir():
        if f.name in ("objects", "manifest.json", "summary.json", "_enhance_work"):
            continue
        dst = output_dir / f.name
        if not dst.exists():
            if f.is_file():
                dst.symlink_to(f.resolve())
            elif f.is_dir():
                dst.symlink_to(f.resolve())

    print(f"[weak_gate] {len(passed_ids)} 通过, {len(rejected_ids)} 淘汰 → {output_dir}")
    return output_dir, passed_ids, rejected_ids


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: 合并 trainready
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_obj_id(pair: dict) -> str:
    obj_id = pair.get("obj_id", "") or pair.get("object_id", "")
    if not obj_id:
        src = pair.get("source_image", "")
        m = re.search(r"(obj_\d+)", src)
        if m:
            obj_id = m.group(1)
    return obj_id


def merge_trainready_roots(
    roots: list[str],
    output_dir: str,
    existing_baseline: str | None = None,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "views").mkdir(exist_ok=True)
    (out / "objects").mkdir(exist_ok=True)
    (out / "pairs").mkdir(exist_ok=True)

    all_pairs = []
    linked_view_ids = set()
    obj_ids_seen = set()
    sources = []

    if existing_baseline:
        sources.append(("baseline", Path(existing_baseline)))
    for i, root in enumerate(roots):
        sources.append((f"pipeline_{i}", Path(root)))

    for label, root in sources:
        views = root / "views"
        if views.exists():
            for obj_dir in sorted(views.iterdir()):
                if obj_dir.is_dir() and obj_dir.name not in linked_view_ids:
                    dst = out / "views" / obj_dir.name
                    if not dst.exists():
                        dst.symlink_to(obj_dir.resolve())
                    linked_view_ids.add(obj_dir.name)

        objects = root / "objects"
        if objects.exists():
            for obj_dir in sorted(objects.iterdir()):
                if obj_dir.is_dir():
                    dst = out / "objects" / obj_dir.name
                    if not dst.exists():
                        dst.symlink_to(obj_dir.resolve())

        pairs_jsonl = root / "pairs" / "train_pairs.jsonl"
        if pairs_jsonl.exists():
            with open(pairs_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        pair = json.loads(line)
                        pair["_source"] = label
                        all_pairs.append(pair)
                        obj_id = _extract_obj_id(pair)
                        if obj_id:
                            obj_ids_seen.add(obj_id)

    merged_jsonl = out / "pairs" / "train_pairs.jsonl"
    with open(merged_jsonl, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    objects_dict = {}
    for obj_id in sorted(obj_ids_seen):
        obj_pairs = [p for p in all_pairs if _extract_obj_id(p) == obj_id]
        objects_dict[obj_id] = {
            "obj_id": obj_id,
            "pair_count": len(obj_pairs),
        }

    manifest = {
        "schema": "merged_trainready_v1",
        "created": datetime.now().isoformat(),
        "sources": [str(s[1]) for s in sources],
        "total_objects": len(obj_ids_seen),
        "total_pairs": len(all_pairs),
        "summary": {
            "total_objects": len(obj_ids_seen),
            "total_pairs": len(all_pairs),
        },
        "objects": objects_dict,
        "pairs": all_pairs,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )

    print(f"[merge] 合并完成: {len(obj_ids_seen)} 物体, {len(all_pairs)} 训练对 → {out}")
    return manifest


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: Split
# ═══════════════════════════════════════════════════════════════════════════════

def build_split(
    trainready_root: str,
    output_dir: str,
    seed: int = 42,
) -> str:
    pairs_jsonl = Path(trainready_root) / "pairs" / "train_pairs.jsonl"
    obj_ids = set()
    with open(pairs_jsonl, "r") as f:
        for line in f:
            if line.strip():
                pair = json.loads(line)
                obj_id = _extract_obj_id(pair)
                if obj_id:
                    obj_ids.add(obj_id)

    n = len(obj_ids)
    if n == 0:
        raise RuntimeError("No objects found in trainready pairs")

    if n == 1:
        n_train, n_val, n_test = 1, 0, 0
    elif n == 2:
        n_train, n_val, n_test = 1, 1, 0
    else:
        n_val = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
        n_train = n - n_val - n_test

    cmd = [
        PYTHON,
        str(REPO_ROOT / "scripts" / "build_object_split_for_rotation_dataset.py"),
        "--source-root", trainready_root,
        "--output-dir", output_dir,
        "--train-objects", str(n_train),
        "--val-objects", str(n_val),
        "--test-objects", str(n_test),
        "--seed", str(seed),
        "--asset-mode", "symlink",
    ]
    print(f"[split] {n} 物体: train={n_train}, val={n_val}, test={n_test}")
    subprocess.run(cmd, check=True)
    return output_dir


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6: 验证
# ═══════════════════════════════════════════════════════════════════════════════

def validate_dataset(split_root: str) -> bool:
    report_path = Path(split_root) / "validation_report.json"
    cmd = [
        PYTHON,
        str(REPO_ROOT / "scripts" / "feedback_loop" / "validate_dataset.py"),
        split_root,
        "--output", str(report_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[validate] 验证失败:\n{result.stderr}")
        return False

    if report_path.exists():
        report = json.loads(report_path.read_text())
        passed = report.get("passed", report.get("status") == "passed")
        print(f"[validate] {'通过' if passed else '未通过'}: {report_path}")
        return passed
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 训练（可选）
# ═══════════════════════════════════════════════════════════════════════════════

def launch_training(
    dataset_root: str,
    output_dir: str,
    epochs: int,
    lr: str,
    lora_rank: int,
    gpus: str,
    tmux_session: str = "dual-pipeline-train",
) -> str:
    train_script = REPO_ROOT / "scripts" / "train_clockwise_externalfb.sh"

    env_vars = {
        "DATASET_ROOT": dataset_root,
        "TRAIN_OUTPUT_DIR": output_dir,
        "NUM_EPOCHS": str(epochs),
        "LEARNING_RATE": lr,
        "LORA_RANK": str(lora_rank),
        "CUDA_VISIBLE_DEVICES": gpus,
    }
    env_exports = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in env_vars.items()
    )
    output_log = shlex.quote(str(Path(output_dir) / "train.log"))

    inner_cmd = (
        f"export {env_exports}; "
        f"bash {shlex.quote(str(train_script))} 2>&1 | tee {output_log}"
    )
    tmux_cmd = (
        f"tmux new -d -s {shlex.quote(tmux_session)} {shlex.quote(inner_cmd)}"
    )

    os.makedirs(output_dir, exist_ok=True)
    print(f"[train] 启动训练: tmux={tmux_session}, epochs={epochs}, lr={lr}, rank={lora_rank}")
    subprocess.run(tmux_cmd, shell=True, check=True)
    return tmux_session


# ═══════════════════════════════════════════════════════════════════════════════
# 状态持久化
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineState:
    def __init__(self, run_root: Path):
        self.path = run_root / "DUAL_PIPELINE_STATE.json"
        self.run_root = run_root
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {
                "schema": "dual_pipeline_v2",
                "created": datetime.now().isoformat(),
            }

    def update(self, key: str, value):
        self.data[key] = value
        self.data["updated"] = datetime.now().isoformat()
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False))

    def get(self, key: str, default=None):
        return self.data.get(key, default)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="双管线：共享 VLM gate → 分组 rotation export → 合并 trainready",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g_data = p.add_argument_group("数据集生成")
    g_data.add_argument("--full-objects", type=int, required=True,
                        help="Group A：全量8角度物体数")
    g_data.add_argument("--weak-objects", type=int, required=True,
                        help="Group B：弱势角度补强物体数")
    g_data.add_argument("--group-ratio", type=str, default=None,
                        help="A:B 分组比例（如 3:2），应用于 accepted 后动态分组。设置后忽略 full/weak-objects 的分组含义，仅用其总和作为种子数")
    g_data.add_argument("--eval-metrics", type=str, default=None,
                        help="训练反馈 CSV 文件路径（自动检测弱势角度）")
    g_data.add_argument("--weak-angles", type=str, default=None,
                        help="手动指定弱势角度（逗号分隔，如 225,135）")
    g_data.add_argument("--weak-threshold", type=float, default=0.98,
                        help="弱势角度阈值 (default: 0.98)")
    g_data.add_argument("--existing-baseline", type=str, default=None,
                        help="已有 trainready 数据集路径，合并到新数据集中")

    g_gate = p.add_argument_group("VLM 质量门控")
    g_gate.add_argument("--gate-max-rounds", type=int, default=3,
                        help="VLM gate 最大轮数 (default: 3)")
    g_gate.add_argument("--gate-threshold", type=float, default=0.78,
                        help="VLM gate 接受阈值 (default: 0.78)")
    g_gate.add_argument("--angle-gate-threshold", type=float, default=0.50,
                        help="Angle gate 阈值 (default: 0.50)")
    g_gate.add_argument("--angle-gate-max-rounds", type=int, default=2,
                        help="Angle gate 最大增强轮数 (default: 2)")
    g_gate.add_argument("--weak-gate-reject-threshold", type=float, default=0.45,
                        help="Group B 弱角度质量门控淘汰阈值 (default: 0.45)")
    g_gate.add_argument("--no-force-accept", action="store_true",
                        help="不强制接受 VLM gate 拒绝的物体（淘汰不合格物体）")

    g_train = p.add_argument_group("训练")
    g_train.add_argument("--epochs", type=int, default=30)
    g_train.add_argument("--lr", type=str, default="1e-4")
    g_train.add_argument("--lora-rank", type=int, default=32)
    g_train.add_argument("--skip-training", action="store_true",
                         help="跳过训练阶段，只构建数据集")

    g_misc = p.add_argument_group("其他")
    g_misc.add_argument("--gpus", type=str, default="0,1,2",
                        help="GPU 列表 (default: 0,1,2)")
    g_misc.add_argument("--device", type=str, default="cuda:0")
    g_misc.add_argument("--seed", type=int, default=42)
    g_misc.add_argument("--dry-run", action="store_true")
    g_misc.add_argument("--resume", type=str, default=None,
                        help="从已有的 run 目录恢复")

    return p.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_objects = args.full_objects + args.weak_objects

    # ── Phase 0: 弱势角度分析 ──
    if args.weak_angles:
        weak_angles = [int(a.strip()) for a in args.weak_angles.split(",")]
        print(f"[Phase 0] 手动指定弱势角度: {weak_angles}")
    elif args.eval_metrics:
        weak_angles = analyze_weak_angles(args.eval_metrics, args.weak_threshold)
        print(f"[Phase 0] 自动检测弱势角度 (threshold={args.weak_threshold}): {weak_angles}")
        if not weak_angles:
            print("[Phase 0] 未检测到弱势角度，默认 225°")
            weak_angles = [225]
    else:
        print("[Phase 0] 未提供 --eval-metrics 或 --weak-angles，默认补强 225°")
        weak_angles = [225]

    weak_export_rotations = ",".join(["0"] + [str(a) for a in weak_angles])
    weak_target_rotations = ",".join(str(a) for a in weak_angles)
    full_export_rotations = "0,45,90,135,180,225,270,315"
    full_target_rotations = "45,90,135,180,225,270,315"

    # ── 创建运行目录 ──
    if args.resume:
        run_root = Path(args.resume)
    else:
        run_root = RUNTIME_ROOT / f"dual_pipeline_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir = run_root / "logs"
    log_dir.mkdir(exist_ok=True)

    state = PipelineState(run_root)
    state.update("params", {
        "full_objects": args.full_objects,
        "weak_objects": args.weak_objects,
        "weak_angles": weak_angles,
        "gate_max_rounds": args.gate_max_rounds,
        "gate_threshold": args.gate_threshold,
    })

    print(f"\n{'='*60}")
    print(f"  Dual Pipeline v2 (共享 gate → 分组 export)")
    print(f"  运行目录: {run_root}")
    print(f"  总物体数: {total_objects} ({args.full_objects} full + {args.weak_objects} weak)")
    print(f"  弱势角度: {weak_angles}")
    print(f"  Group A export: {full_export_rotations} → target {full_target_rotations}")
    print(f"  Group B export: {weak_export_rotations} → target {weak_target_rotations}")
    if args.existing_baseline:
        print(f"  Baseline: {args.existing_baseline}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("[dry-run] 以上为执行计划，退出。")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1: 共享 VLM gate（所有 N+M 物体一起 steps 01-04）
    # ══════════════════════════════════════════════════════════════════════════
    if state.get("phase1_status") == "completed":
        print("[Phase 1] 已完成，跳过")
        gate_state = state.get("gate_state")
    else:
        state.update("phase1_status", "running")
        base_idx = _find_next_seed_index()
        state.update("seed_start_index", base_idx)

        gate_state = run_shared_gate(
            total_count=total_objects,
            seed_start_index=base_idx,
            gate_max_rounds=args.gate_max_rounds,
            gate_threshold=args.gate_threshold,
            log_dir=log_dir,
            no_force_accept=args.no_force_accept,
        )
        state.update("phase1_status", "completed")
        state.update("gate_state", gate_state)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2: 拆分为 Group A 和 Group B
    # ══════════════════════════════════════════════════════════════════════════
    if state.get("phase2_status") == "completed":
        print("[Phase 2] 已完成，跳过")
        accepted_a = Path(state.get("accepted_a"))
        accepted_b = Path(state.get("accepted_b"))
        group_a_ids = state.get("group_a_ids")
        group_b_ids = state.get("group_b_ids")
    else:
        state.update("phase2_status", "running")
        accepted_a, accepted_b, group_a_ids, group_b_ids = split_accepted_objects(
            gate_state, args.full_objects, args.weak_objects, run_root,
            group_ratio=args.group_ratio,
        )
        state.update("phase2_status", "completed")
        state.update("accepted_a", str(accepted_a))
        state.update("accepted_b", str(accepted_b))
        state.update("group_a_ids", group_a_ids)
        state.update("group_b_ids", group_b_ids)

    meshes_dir = str(Path(gate_state["stage1_assets_root"]) / "meshes")
    prompts_json = str(Path(gate_state["stage1_assets_root"]) / "prompts.json")
    gate_profile = str(Path(gate_state["gate_root"]).parent / "scene_v7_gate_profile.json")
    if not Path(gate_profile).exists():
        gate_profile = str(REPO_ROOT / "configs" / "dataset_profiles" / "scene_v7.json")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3A: Group A → rotation8 (全角度) → angle gate → trainready
    # ══════════════════════════════════════════════════════════════════════════
    rot8_a = run_root / "rotation8_A"
    rot8_enhanced_a = run_root / "rotation8_enhanced_A"
    trainready_a = run_root / "trainready_A"

    if state.get("phase3a_status") == "completed":
        print("[Phase 3A] 已完成，跳过")
    elif not group_a_ids:
        print("[Phase 3A] Group A 为空（无物体通过 VLM gate），跳过")
        state.update("phase3a_status", "completed")
        state.update("trainready_a", None)
    else:
        state.update("phase3a_status", "running")

        print(f"\n[Phase 3A] Group A: {len(group_a_ids)} 物体 × 全角度")
        run_rotation8_export(
            source_root=accepted_a,
            output_dir=rot8_a,
            meshes_dir=meshes_dir,
            rotations=full_export_rotations,
            gpus=args.gpus,
            log_dir=log_dir,
            label="3A",
        )

        run_angle_gate(
            source_root=rot8_a,
            output_dir=rot8_enhanced_a,
            meshes_dir=meshes_dir,
            profile=gate_profile,
            device=args.device,
            angle_threshold=args.angle_gate_threshold,
            max_enhance_rounds=args.angle_gate_max_rounds,
            log_dir=log_dir,
            label="3A",
            gpus=args.gpus,
        )

        run_build_trainready(
            source_root=rot8_enhanced_a,
            output_dir=trainready_a,
            target_rotations=full_target_rotations,
            prompts_json=prompts_json,
            log_dir=log_dir,
            label="3A",
        )

        state.update("phase3a_status", "completed")
        state.update("trainready_a", str(trainready_a))

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3B: Group B → rotation8 (0° + 弱角度) → VLM 质量门控 → trainready
    # ══════════════════════════════════════════════════════════════════════════
    rot8_b = run_root / "rotation8_B"
    rot8_enhanced_b = run_root / "rotation8_enhanced_B"
    rot8_filtered_b = run_root / "rotation8_filtered_B"
    trainready_b = run_root / "trainready_B"

    if state.get("phase3b_status") == "completed":
        print("[Phase 3B] 已完成，跳过")
    elif not group_b_ids:
        print("[Phase 3B] Group B 为空（无物体分配到弱角度组），跳过")
        state.update("phase3b_status", "completed")
        state.update("trainready_b", None)
    else:
        state.update("phase3b_status", "running")

        print(f"\n[Phase 3B] Group B: {len(group_b_ids)} 物体 × 弱角度 {weak_angles}")
        run_rotation8_export(
            source_root=accepted_b,
            output_dir=rot8_b,
            meshes_dir=meshes_dir,
            rotations=weak_export_rotations,
            gpus=args.gpus,
            log_dir=log_dir,
            label="3B",
        )

        run_angle_gate(
            source_root=rot8_b,
            output_dir=rot8_enhanced_b,
            meshes_dir=meshes_dir,
            profile=gate_profile,
            device=args.device,
            angle_threshold=args.angle_gate_threshold,
            max_enhance_rounds=args.angle_gate_max_rounds,
            log_dir=log_dir,
            label="3B",
            gpus=args.gpus,
        )

        print(f"\n[Phase 3B] VLM 弱角度质量门控 (reject < {args.weak_gate_reject_threshold})")
        filtered_root, passed, rejected = filter_by_angle_gate_scores(
            enhanced_root=rot8_enhanced_b,
            output_dir=rot8_filtered_b,
            weak_angles=weak_angles,
            reject_threshold=args.weak_gate_reject_threshold,
        )
        state.update("phase3b_weak_gate_passed", passed)
        state.update("phase3b_weak_gate_rejected", rejected)

        if not passed and rejected:
            print("[Phase 3B] 所有物体被淘汰，Group B 无训练对产出")
            state.update("phase3b_status", "completed")
            state.update("trainready_b", None)
        else:
            run_build_trainready(
                source_root=filtered_root,
                output_dir=trainready_b,
                target_rotations=weak_target_rotations,
                prompts_json=prompts_json,
                log_dir=log_dir,
                label="3B",
            )
            state.update("phase3b_status", "completed")
            state.update("trainready_b", str(trainready_b))

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 4: 合并 A + B + baseline
    # ══════════════════════════════════════════════════════════════════════════
    if state.get("phase4_status") == "completed":
        print("[Phase 4] 已完成，跳过")
        merged_root = state.get("merged_root")
    else:
        state.update("phase4_status", "running")
        merged_root = str(run_root / "merged_trainready")
        trainready_a_str = state.get("trainready_a")
        trainready_b_str = state.get("trainready_b")
        merge_roots = []
        if trainready_a_str:
            merge_roots.append(trainready_a_str)
        if trainready_b_str:
            merge_roots.append(trainready_b_str)
        if not merge_roots and not args.existing_baseline:
            print("[Phase 4] 无新训练对且无 baseline，跳过合并")
            state.update("phase4_status", "completed")
            state.update("merged_root", None)
            merged_root = None
        else:
            manifest = merge_trainready_roots(
                roots=merge_roots,
                output_dir=merged_root,
                existing_baseline=args.existing_baseline,
            )
            state.update("phase4_status", "completed")
            state.update("merged_root", merged_root)
            state.update("merged_manifest", manifest)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 5: Split
    # ══════════════════════════════════════════════════════════════════════════
    if not merged_root:
        print("[Phase 5] 无合并数据集，跳过 split/validate/training")
        state.update("phase5_status", "skipped")
        state.update("phase6_status", "skipped")
        state.update("training_status", "skipped")
        split_root = None
    elif state.get("phase5_status") == "completed":
        print("[Phase 5] 已完成，跳过")
        split_root = state.get("split_root")
    else:
        state.update("phase5_status", "running")
        split_root = str(run_root / "final_split")
        build_split(merged_root, split_root, seed=args.seed)
        state.update("phase5_status", "completed")
        state.update("split_root", split_root)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 6: 验证
    # ══════════════════════════════════════════════════════════════════════════
    if split_root and state.get("phase6_status") != "completed":
        state.update("phase6_status", "running")
        valid = validate_dataset(split_root)
        state.update("phase6_status", "completed")
        state.update("phase6_valid", valid)
        if not valid:
            print("[WARN] 数据集验证未通过，请检查")

    # ── 可选训练 ──
    if not split_root:
        print("[训练] 跳过（无 split 数据集）")
        state.update("training_status", "skipped")
    elif args.skip_training:
        print("[训练] 跳过（--skip-training）")
        state.update("training_status", "skipped")
    elif state.get("training_status") not in ("completed", "running"):
        state.update("training_status", "running")
        train_output = str(run_root / "train_output")
        tmux = launch_training(
            dataset_root=split_root,
            output_dir=train_output,
            epochs=args.epochs,
            lr=args.lr,
            lora_rank=args.lora_rank,
            gpus=args.gpus,
            tmux_session=f"dual-train-{timestamp[:8]}",
        )
        state.update("training_tmux", tmux)

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"  Dual Pipeline v2 完成")
    print(f"  运行目录:      {run_root}")
    print(f"  Group A 物体:  {group_a_ids}")
    print(f"  Group B 物体:  {group_b_ids}")
    print(f"  Trainready A:  {trainready_a}")
    print(f"  Trainready B:  {trainready_b}")
    print(f"  合并数据集:    {merged_root}")
    print(f"  Split 数据集:  {split_root}")
    print(f"  状态文件:      {state.path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
