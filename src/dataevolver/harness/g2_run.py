"""Freeze, execute and independently validate the first real G2 batch."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile
from dataevolver.dataset.contracts import ArtifactStore, canonical_json, file_sha256, payload_sha256
from dataevolver.dataset.g2_contracts import G2Task, validate_g2_config
from dataevolver.dataset.g2_pipeline import load_pair_manifest
from .state_store import atomic_write_json


def inventory(root):
    root = Path(root).resolve()
    paths = sorted(p for p in root.rglob("*") if p.is_file() and not any(s.startswith(".") for s in p.relative_to(root).parts)
                   and p.suffix in {".json", ".safetensors", ".txt", ".model", ".jinja"})
    if not any(p.suffix == ".safetensors" for p in paths):
        raise ValueError("model inventory has no checkpoint shards")
    files = {p.relative_to(root).as_posix(): file_sha256(p) for p in paths}
    for index in root.rglob("*.safetensors.index.json"):
        value = json.loads(index.read_text())
        for shard in set(value["weight_map"].values()):
            relative = (index.parent/shard).relative_to(root).as_posix()
            if relative not in files:
                raise ValueError("checkpoint index references a missing or unbound shard")
    return files


def freeze(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out/"config.json").exists():
        raise ValueError("refusing to overwrite a frozen G2 configuration")
    source = json.loads(Path(args.source_manifest).read_text())
    tasks = [G2Task.from_dict(x) for x in source["tasks"]]
    if json.loads((Path(args.edit_model)/"model_index.json").read_text()).get("_class_name") != "QwenImage21Pipeline":
        raise ValueError("edit cache architecture differs from the selected Qwen backend")
    if json.loads((Path(args.review_model)/"config.json").read_text()).get("model_type") != "qwen3_5":
        raise ValueError("review cache architecture differs from the selected Qwen3.8 backend (qwen3_5 architecture)")
    runtime = {"python": platform.python_version(), **{n: importlib.metadata.version(n) for n in
               ("torch", "transformers", "diffusers", "accelerate", "numpy", "pillow", "tokenizers", "safetensors")}}
    edit_files, review_files = inventory(args.edit_model), inventory(args.review_model)
    workers = {"edit": {"kind": "qwen_edit_gpu", "model_path_env": "DATAEVOLVER_G2_EDIT_MODEL", "python_env": "DATAEVOLVER_G2_PYTHON",
                           "file_hashes": edit_files, "runtime_versions": runtime, "threads": 8},
               "review": {"kind": "qwen_vl_cpu", "model_path_env": "DATAEVOLVER_G2_REVIEW_MODEL", "python_env": "DATAEVOLVER_G2_PYTHON",
                           "file_hashes": review_files, "runtime_versions": runtime, "threads": 8}}
    profiles = [ModelProfile("edit", "local_python", "Qwen/Qwen-Image-2.1", "sha256:"+payload_sha256(edit_files), ("json", "vision")).to_dict(),
                ModelProfile("review", "local_python", "Qwen/Qwen3.8-27B", "sha256:"+payload_sha256(review_files), ("json", "vision")).to_dict()]
    allowed, recipes = {}, {}
    for task in tasks:
        values = {"target_description": [task.target, "the specified "+task.target],
                  "spatial_expression": ["", source["spatial_expressions"][task.task_id]],
                  "preservation_instruction": ["Keep everything else unchanged.",
                      "Preserve the target geometry, all other objects, the background, lighting and framing exactly."]}
        allowed[task.task_id] = values
        recipes[task.task_id] = {k: v[0] for k, v in values.items()}
    count = len(tasks)
    config = {"schema_version": "dataevolver.g2_config.v1", "tasks": [t.to_dict() for t in tasks],
              "allowed_recipe_values": allowed, "initial_recipe": recipes,
              "inference": {"width": 512, "height": 512, "steps": 40, "true_cfg_scale": 4.0, "guidance_scale": 1.0,
                            "negative_prompt": " ", "timeout_seconds": 1800},
              "evaluation": {"max_outside_mae": .10, "min_target_mae": .02, "review_max_tokens": 700,
                             "review_timeout_seconds": 1200, "calibration_status": "pilot_uncalibrated"},
              "budget": {"max_model_calls": 2*count, "max_edit_calls": count, "max_probe_calls": 0,
                         "max_wall_seconds": 10800, "max_output_bytes": 256*1024*1024},
              "models": {"profiles": profiles, "workers": workers, "policy": ModelPolicy({
                  "visual.edit": "edit", "visual.inspect_region": "review", "visual.review_constraints": "review"}, 2*count).to_dict()},
              "stores": {"source": "DATAEVOLVER_G2_SOURCE_ROOT", "dataset": "DATAEVOLVER_G2_DATASET_ROOT"}, "dataset_store": "dataset",
              "method": {"id": "g2-fixed-m-v1", "mutable": False, "knowledge_snapshot": None},
              "split_policy": {"selection_splits": ["pilot", "experience_validation"], "benefit_split": "reuse_test", "freeze_k_during_test": True}}
    validate_g2_config(config)
    evaluator = {"schema_version": "dataevolver.harness.evaluator.v1", "evaluator_id": "g2-production-v1", "revision": "1",
                 "hard_constraints": {"artifact_valid": {"required": True}, "all_tasks_recorded": {"required": True}},
                 "target_metrics": {"completion_ratio": {"required": True, "minimum": 1.0}},
                 "protected_metrics": {}}
    atomic_write_json(out/"evaluator.json", evaluator)
    config_hash = payload_sha256(config)
    goal = {"schema_version": "dataevolver.harness.goal.v1", "campaign_id": args.campaign_id,
            "objective": {"kind": "real_g2_pilot", "real_data_execution": True, "initial_trial_key": {
                 "input_artifact_hash": payload_sha256([t.source.to_dict() for t in tasks]),
                 "parent_artifact_hash": "", "seed": tasks[0].seed, "operator_id": "data.visual_g2", "operator_version": "1",
                 "model_config_hash": payload_sha256(config["models"]), "preprocessing_hash": "none",
                 "evaluator_hash": file_sha256(out/"evaluator.json"), "environment_contract_hash": payload_sha256(runtime)}},
            "initial_operator_id": "data.visual_g2", "route_candidates": ["data.visual_g2"],
            "budgets": {"max_rounds": 1, "max_operator_calls": 1, "max_experiments": 0, "max_retries": 0,
                        "max_wall_seconds": 10800, "max_planner_calls": 0},
            "resource_policy": {"allow_gpu": True, "allow_network": False, "allowed_operators": ["data.visual_g2"]},
            "frozen_evaluator": {"path": str((out/"evaluator.json").resolve()), "sha256": file_sha256(out/"evaluator.json")},
            "planner": {"kind": "deterministic"}, "initial_operator_parameters": config, "interventions": [], "lease_seconds": 10860}
    atomic_write_json(out/"config.json", config)
    atomic_write_json(out/"goal.json", goal)
    atomic_write_json(out/"frozen.json", {"config_sha256": file_sha256(out/"config.json"), "config_content_hash": config_hash,
                      "goal_sha256": file_sha256(out/"goal.json"), "source_manifest_sha256": file_sha256(Path(args.source_manifest)),
                      "model_asset_inventory_hashes": {"edit": payload_sha256(edit_files), "review": payload_sha256(review_files)},
                      "runtime": runtime, "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      "code_status": subprocess.check_output(["git", "status", "--porcelain=v1"], text=True)})
    print(canonical_json({"status": "FROZEN", "config_hash": config_hash, "tasks": count, "max_model_calls": 2*count}))


def gpu_snapshot():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=20, check=True)
    rows = []
    for line in result.stdout.splitlines():
        index, uuid, free, utilization = [x.strip() for x in line.split(",")]
        rows.append({"physical_index": int(index), "uuid": uuid, "free_mib": int(free), "utilization": int(utilization)})
    return rows


def run(args):
    from .cli import main as harness_main
    config_dir = Path(args.config_dir).resolve()
    frozen = json.loads((config_dir/"frozen.json").read_text())
    for name in ("config", "goal"):
        if file_sha256(config_dir/f"{name}.json") != frozen[f"{name}_sha256"]:
            raise ValueError("frozen G2 files changed")
    config = json.loads((config_dir/"config.json").read_text())
    validate_g2_config(config)
    goal = json.loads((config_dir/"goal.json").read_text())
    if (Path(args.campaign_root)/goal["campaign_id"]/"final_report.json").exists():
        return _finalize_run(args, config_dir, config, goal, None, 0)
    # Resource waiting occurs before the Harness wall-time/call budget starts.
    status_path = Path(args.status)
    deadline, stable, selected = time.monotonic()+args.max_wait_seconds, 0, None
    while True:
        rows = gpu_snapshot()
        available = sorted([x for x in rows if x["free_mib"] >= args.min_free_mib and x["utilization"] <= args.max_utilization],
                           key=lambda x: (-x["free_mib"], x["utilization"]))
        candidate = available[0] if available else None
        stable = stable+1 if candidate and selected and selected["uuid"] == candidate["uuid"] else int(candidate is not None)
        selected = candidate
        atomic_write_json(status_path, {"state": "WAITING_GPU", "gpu_snapshot": rows, "stable_observations": stable,
                          "required_free_mib": args.min_free_mib, "max_utilization": args.max_utilization,
                          "updated_at": time.time(), "config_hash": frozen["config_content_hash"]})
        if stable >= 2:
            break
        if time.monotonic() >= deadline:
            atomic_write_json(status_path, {"state": "WAIT_EXPIRED", "gpu_snapshot": rows, "updated_at": time.time()})
            return 75
        time.sleep(min(60, max(1, deadline-time.monotonic())))
    # Recheck immediately before the actual model route starts.
    live = next(x for x in gpu_snapshot() if x["uuid"] == selected["uuid"])
    if live["free_mib"] < args.min_free_mib or live["utilization"] > args.max_utilization:
        raise RuntimeError("selected GPU became busy before launch; no inference submitted")
    os.environ["CUDA_VISIBLE_DEVICES"] = selected["uuid"]
    atomic_write_json(status_path, {"state": "RUNNING", "selected_gpu": selected, "process_device": "cuda:0",
                      "updated_at": time.time(), "config_hash": frozen["config_content_hash"]})
    rc = harness_main(["evolve", "--goal", str(config_dir/"goal.json"), "--campaign-root", args.campaign_root,
                       "--idempotency-key", "g2:"+frozen["config_content_hash"]])
    return _finalize_run(args, config_dir, config, goal, selected, rc)


def _finalize_run(args, config_dir, config, goal, selected, rc):
    try:
        validation = validate_campaign(config_dir, Path(args.campaign_root)/goal["campaign_id"])
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        validation = {"step1_passed": False, "reason": str(exc), "quality_benefit_verified": False}
    atomic_write_json(Path(args.validation), validation)
    pairs_path = Path(args.campaign_root)/goal["campaign_id"]/"rounds/r000/execute/pairs.json"
    if pairs_path.exists():
        from dataevolver.dataset.g2_export import export_pairs, load_bundle
        store = ArtifactStore({name: os.environ[env] for name, env in config["stores"].items()})
        try:
            if Path(args.export_dir).exists():
                bundle = Path(args.export_dir)
                loaded = load_bundle(bundle, include_rejected=True)
                if (file_sha256(bundle/"pairs.json") != file_sha256(pairs_path) or
                        file_sha256(bundle/"config.json") != file_sha256(config_dir/"config.json") or
                        file_sha256(bundle/"frozen.json") != file_sha256(config_dir/"frozen.json")):
                    raise ValueError("existing portable export belongs to different execution/configuration")
                validation["portable_export"] = {"bundle": str(bundle), "reused": True,
                    "manifest_sha256": file_sha256(pairs_path), "loadable_pairs": len(loaded)}
            else:
                validation["portable_export"] = export_pairs(pairs_path, store, args.export_dir,
                    metadata_files={"config.json": config_dir/"config.json", "frozen.json": config_dir/"frozen.json"})
        except (ValueError, OSError, KeyError) as exc:
            validation.update(step1_passed=False, export_error=str(exc))
        atomic_write_json(Path(args.validation), validation)
    atomic_write_json(Path(args.status), {"state": "STEP1_PASSED" if validation["step1_passed"] else "STEP1_FAILED",
                      "selected_gpu": selected, "validation": str(Path(args.validation).resolve()), "updated_at": time.time()})
    return 0 if rc == 0 and validation["step1_passed"] else 1


def validate_campaign(config_dir, campaign):
    config = json.loads((Path(config_dir)/"config.json").read_text())
    roots = {name: os.environ[env] for name, env in config["stores"].items()}
    store = ArtifactStore(roots)
    pairs_path = Path(campaign)/"rounds/r000/execute/pairs.json"
    final_path = Path(campaign)/"final_report.json"
    if not pairs_path.exists() or not final_path.exists():
        return {"step1_passed": False, "reason": "missing pair manifest or Harness final report"}
    pairs = load_pair_manifest(pairs_path, store, include_rejected=True)
    value = json.loads(pairs_path.read_text())
    from .composition import open_controller
    controller = open_controller(Path(campaign).parent, Path(campaign).name)
    replay = controller.replay().to_dict() == controller.status().to_dict()
    from .models import OperatorCall
    from .operators import _load_receipt, _receipt_path
    # Validate through the owning Operator as well as the independent loader.
    planned = [e for e in controller.store.events.read() if e.event_type == "ACTION_PLANNED"]
    effect_verified = len(planned) == 1
    if effect_verified:
        call = OperatorCall.from_dict(planned[0].payload["operator_call"])
        effect_verified = controller.registry.get(call.operator_id).verify_effect(call, _load_receipt(_receipt_path(call))).status.value == "VERIFIED"
    contract_valid, changed_fields = controller._frozen()
    all_rows = value["summary"]["tasks"] == len(config["tasks"])
    all_reviewed = value["summary"]["reviewed_pairs"] == len(config["tasks"])
    all_produced = len(pairs) == len(config["tasks"])
    within_budget = value["usage"]["model_calls"] <= config["budget"]["max_model_calls"]
    state = controller.status()
    passed = all_rows and all_reviewed and all_produced and within_budget and replay and effect_verified and contract_valid and state.state.value == "COMPLETE"
    return {"step1_passed": passed, "production_acceptance_only": True, "quality_benefit_verified": False,
            "pairs_manifest": str(pairs_path), "pairs_sha256": file_sha256(pairs_path), "summary": value["summary"],
            "usage": value["usage"], "harness_state": state.state.value, "replay_matches_snapshot": replay,
            "effect_verified": effect_verified, "frozen_contract_valid": contract_valid, "contract_drift": list(changed_fields),
            "all_tasks_recorded": all_rows, "independent_loader_pairs": len(pairs), "within_budget": within_budget,
            "calibration_status": config["evaluation"]["calibration_status"],
            "formal_comparison_authorized_next": passed, "formal_comparison_config_frozen": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    for name in ("source-manifest", "edit-model", "review-model", "out", "campaign-id"):
        f.add_argument("--"+name, required=True)
    r = sub.add_parser("run")
    for name in ("config-dir", "campaign-root", "status", "validation", "export-dir"):
        r.add_argument("--"+name, required=True)
    r.add_argument("--min-free-mib", type=int, default=100000)
    r.add_argument("--max-utilization", type=int, default=30)
    r.add_argument("--max-wait-seconds", type=int, default=21600)
    v = sub.add_parser("validate")
    v.add_argument("--config-dir", required=True); v.add_argument("--campaign", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze(args); return 0
    if args.command == "run":
        if args.min_free_mib < 100000 or not 0 <= args.max_utilization <= 30 or not 60 <= args.max_wait_seconds <= 21600:
            raise ValueError("GPU wait exceeds the bounded first pilot resource policy")
        return run(args)
    value = validate_campaign(args.config_dir, args.campaign)
    print(canonical_json(value)); return 0 if value["step1_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
