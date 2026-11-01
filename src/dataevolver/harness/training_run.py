"""Compile an accepted media plan into one frozen existing-Harness batch.

Private deployment paths enter through environment variables. This module
does not choose models, fabricate planning results, or drive a second loop.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tarfile
from pathlib import Path

from dataevolver.paths import REPO_ROOT
from dataevolver.dataset.contracts import file_sha256, payload_sha256
from dataevolver.dataset.supervision import load_dataset
from dataevolver.dataset.training_pipeline import validate_production_contract
from .models import GoalSpec, OperatorCall
from .operators import _load_receipt, _receipt_path
from .state_store import atomic_write_json


def trial_key(contract, evaluator_hash, roots):
    return {"input_artifact_hash":payload_sha256(contract["planning"]), "parent_artifact_hash":payload_sha256(contract["planning"]),
        "seed":contract["production"]["tasks"][0]["seed"], "operator_id":"data.training_supervision", "operator_version":"1",
        "model_config_hash":payload_sha256({"config":contract, "roots":roots}),
        "preprocessing_hash":payload_sha256(contract["production"]["recipe"]), "evaluator_hash":evaluator_hash,
        "environment_contract_hash":payload_sha256({"roots":roots, "models":contract["models"]})}


def freeze_contract(contract, out, campaign_id, *, wall_seconds):
    from .adapters.training_supervision import TrainingSupervisionOperator
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("refusing to overwrite a frozen training run")
    validate_production_contract(contract)
    if contract["production"]["evidence_kind"] != "real_execution":
        raise ValueError("production freeze does not accept fixture evidence")
    if wall_seconds <= contract["production"]["resource_wait_seconds"]:
        raise ValueError("campaign time must also allow inference after resource waiting")
    # Validate actual store and selected worker identities before creating files.
    operator=TrainingSupervisionOperator(allowed_write_root=out/"campaigns"/campaign_id, contract=contract)
    dirty = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, text=True)
    if dirty.strip():
        raise ValueError("freeze requires a committed clean production checkout")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    out.mkdir(parents=True)
    archive = out/"source.tar"
    subprocess.run(["git", "archive", "--format=tar", "--output", str(archive), commit,
                    "src", "configs", "pyproject.toml"], cwd=REPO_ROOT, check=True)
    with tarfile.open(archive) as handle:
        handle.extractall(out/"source", filter="data")
    source_hashes = {p.relative_to(out/"source").as_posix():file_sha256(p)
                     for p in sorted((out/"source").rglob("*")) if p.is_file()}
    evaluator = {"schema_version":"dataevolver.harness.evaluator.v1", "evaluator_id":"training-production-v1", "revision":"1",
        "hard_constraints":{"artifact_valid":{"required":True}, "all_tasks_recorded":{"required":True},"all_qualified":{"required":True}},
        "target_metrics":{"qualified_ratio":{"required":True,"minimum":1.0}}}
    atomic_write_json(out/"evaluator.json", evaluator)
    goal = {"schema_version":"dataevolver.harness.goal.v1", "campaign_id":campaign_id,
        "objective":{"real_data_execution":True, "type":"training_supervision", "contract_hash":payload_sha256(contract),
            "initial_trial_key":trial_key(contract,file_sha256(out/"evaluator.json"),{k:str(v) for k,v in operator.store.roots.items()})},
        "initial_operator_id":"data.training_supervision", "route_candidates":["data.training_supervision"],
        "budgets":{"max_rounds":1, "max_operator_calls":1, "max_experiments":0, "max_retries":0,
                   "max_planner_calls":0, "max_wall_seconds":wall_seconds},
        "resource_policy":{"allow_gpu":True, "allow_network":False, "allowed_operators":["data.training_supervision"]},
        "frozen_evaluator":{"path":str(out/"evaluator.json"), "sha256":file_sha256(out/"evaluator.json")},
        "planner":{"kind":"deterministic"}, "initial_operator_parameters":contract, "interventions":[],
        "lease_seconds":wall_seconds+600}
    GoalSpec.from_dict(goal)
    atomic_write_json(out/"config.json", contract)
    atomic_write_json(out/"goal.json", goal)
    atomic_write_json(out/"frozen.json", {"source_commit":commit, "source_status":dirty, "source_hashes":source_hashes,
        "source_archive_sha256":file_sha256(archive), "config_sha256":file_sha256(out/"config.json"),
        "goal_sha256":file_sha256(out/"goal.json"), "roots":{k:str(Path(os.environ[v]).resolve()) for k,v in contract["stores"].items()}})
    return {"status":"FROZEN", "campaign_id":campaign_id, "config_hash":payload_sha256(contract),
            "accepted_target":contract["production"]["count"], "model_usage_mode":"accounting", "model_calls":0}


def read_frozen(directory, *, require_snapshot=True):
    directory = Path(directory).resolve()
    frozen = json.loads((directory/"frozen.json").read_text())
    if require_snapshot and REPO_ROOT != directory/"source":
        raise ValueError("run from the frozen source snapshot, not the mutable checkout")
    for name in ("config", "goal"):
        if file_sha256(directory/(name+".json")) != frozen[name+"_sha256"]:
            raise ValueError("frozen training configuration changed")
    for name,digest in frozen["source_hashes"].items():
        if file_sha256(directory/"source"/name) != digest:
            raise ValueError("frozen production source changed: "+name)
    contract = json.loads((directory/"config.json").read_text())
    validate_production_contract(contract)
    roots = {k:str(Path(os.environ[v]).resolve()) for k,v in contract["stores"].items()}
    if roots != frozen["roots"]:
        raise ValueError("frozen artifact store mapping changed")
    goal = GoalSpec.from_json((directory/"goal.json").read_text())
    if goal.initial_operator_parameters != contract or goal.objective["contract_hash"] != payload_sha256(contract):
        raise ValueError("GoalSpec compiler lost the independent media contract")
    return contract,goal


def validate_campaign(directory):
    from .composition import open_controller
    contract,goal = read_frozen(directory)
    campaign = Path(directory)/"campaigns"/goal.campaign_id
    controller = open_controller(campaign.parent, campaign.name)
    planned = [e for e in controller.store.events.read() if e.event_type == "ACTION_PLANNED"]
    if len(planned) != 1:
        raise ValueError("production did not preserve its one owning Harness action")
    call = OperatorCall.from_dict(planned[0].payload["operator_call"])
    operator = controller.registry.get(call.operator_id)
    verification = operator.verify_effect(call, _load_receipt(_receipt_path(call)))
    frozen,drift = controller._frozen()
    replay = controller.replay().to_dict() == controller.status().to_dict()
    production = json.loads((Path(call.work_dir)/"production.json").read_text())
    loaded = load_dataset(operator.bundle_root) if production["complete"] else []
    expected = contract["production"]["count"]
    passed = (controller.status().state.value == "COMPLETE" and production["complete"] and len(loaded) == expected and
              len(production["slots"]) == expected and verification.status.value == "VERIFIED" and frozen and replay)
    return {"production_acceptance_passed":passed, "harness_state":controller.status().state.value,
        "effect_verified":verification.status.value == "VERIFIED", "effect_reason":verification.reason,
        "frozen_contract_valid":frozen, "contract_drift":list(drift), "replay_matches_snapshot":replay,
        "all_tasks_recorded":len(production["slots"]) == expected, "accepted_records":len(loaded), "expected_records":expected,
        "split_counts":{k:sum(r["split"]==k for r,_ in loaded) for k in contract["production"]["split_counts"]},
        "usage":production["usage"], "stop_reason":production["stop_reason"], "bundle":str(operator.bundle_root),
        "quality_benefit_verified":False, "training_benefit_verified":False, "rsi_benefit_verified":False}


def run(directory):
    from .cli import main as harness_main
    directory = Path(directory).resolve()
    contract,goal = read_frozen(directory)
    atomic_write_json(directory/"status.json", {"state":"RUNNING", "config_hash":payload_sha256(contract)})
    try:
        rc = harness_main(["evolve", "--goal", str(directory/"goal.json"), "--campaign-root", str(directory/"campaigns"),
                           "--idempotency-key", "training:"+payload_sha256(contract)])
        validation = validate_campaign(directory)
        atomic_write_json(directory/"validation.json", validation)
        passed = rc == 0 and validation["production_acceptance_passed"]
        atomic_write_json(directory/"status.json", {"state":"PASSED" if passed else "FAILED", "validation":str(directory/"validation.json")})
        return 0 if passed else 1
    except BaseException as exc:
        atomic_write_json(directory/"status.json", {"state":"FAILED", "error_code":type(exc).__name__, "error":str(exc)[:4000]})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--contract", required=True); f.add_argument("--out", required=True); f.add_argument("--campaign-id", required=True)
    f.add_argument("--wall-seconds", type=int, required=True)
    for name in ("run", "validate"):
        sub.add_parser(name).add_argument("--config-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(json.dumps(freeze_contract(json.loads(Path(args.contract).read_text()), args.out, args.campaign_id,
                                        wall_seconds=args.wall_seconds), ensure_ascii=False)); return 0
    if args.command == "run":
        return run(args.config_dir)
    value = validate_campaign(args.config_dir)
    print(json.dumps(value, ensure_ascii=False)); return 0 if value["production_acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
