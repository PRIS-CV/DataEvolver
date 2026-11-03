"""Contract builders and offline inspection for the two image routes."""
from __future__ import annotations

import json
import os
from pathlib import Path

from dataevolver.dataset.collection_spec import CollectionSpec, resolve_hf_endpoint
from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, file_sha256, payload_sha256
from dataevolver.dataset.layout_quality import RUBRIC_HASH, compile_criteria
from .data_contracts import DataInput, DataNodeGraph, DataNodeSpec, DataRunBudget
from .task_contracts import DatasetIntent, PROTECTED_GATES, QualityPolicy
from .models import GoalSpec


def is_image_pipeline(goal):
    return goal.initial_operator_id == "data.image_collection" or (
        goal.initial_operator_id == "data.image_curation"
        and goal.initial_operator_parameters.get("intent", {}).get("output_schema") == "curated_images.v2")


def collection_contract(spec_ref, *, mode, stores, dataset_store, budget):
    node = DataNodeSpec("collect", "images.collect", "1", "data", {"spec": DataInput(artifact=spec_ref)})
    return {
        "intent": DatasetIntent("image-collection", "curate", "image_collection", "SourceBundle.v1",
                               "Acquire a bounded immutable source bundle.", (spec_ref,), {"execution_mode": mode}).to_dict(),
        "quality_policy": QualityPolicy("image-collection", tuple(sorted(PROTECTED_GATES)), ("images.collect",), {}, 1).to_dict(),
        "graph": DataNodeGraph((node,)).to_dict(), "budget": budget.to_dict(),
        "stores": stores, "dataset_store": dataset_store, "models": {},
    }


def layout_contract(bundle_ref, *, task_criteria, models, stores, dataset_store, budget, candidate_limit=5000):
    if type(candidate_limit) is not int or not 1 <= candidate_limit <= 5000:
        raise ValueError("layout candidate limit must be between 1 and 5000")
    mode = task_criteria["execution_mode"]
    criteria = compile_criteria(task_criteria)
    references = [bundle_ref] + [ArtifactRef.from_dict(criteria[name]) for name in ("calibration", "protected_evidence", "deployments")
                                 if criteria.get(name)]
    nodes, previous = [], None
    for name in ("audit", "families", "review", "repair", "rereview", "export"):
        role = "annotator" if name == "repair" else "reviewer" if name in {"review", "rereview"} else ""
        inputs = {"bundle": DataInput(artifact=bundle_ref)} if previous is None else {"catalog": DataInput(node=previous, output="catalog")}
        nodes.append(DataNodeSpec(name, f"layout.{name}", "1", "model" if role else "data", inputs,
            {"max_tokens": 2048, "timeout_seconds": 60} if role else {}, role,
            candidate_limit if role else 0, candidate_limit if role else 0))
        previous = name
    return {
        "intent": DatasetIntent("layout-curation", "curate", "multi_subject_layout", "curated_images.v2",
            "Curate traceable layout records with independent evidence.", tuple(references), {"execution_mode": mode}).to_dict(),
        "quality_policy": QualityPolicy("layout-curation", tuple(sorted(PROTECTED_GATES | {"deletion"})),
            tuple(node.operator for node in nodes), task_criteria, budget.max_model_calls).to_dict(),
        "graph": DataNodeGraph(tuple(nodes)).to_dict(), "budget": budget.to_dict(), "stores": stores,
        "dataset_store": dataset_store, "models": models,
    }


def offline_plan(goal):
    if isinstance(goal, (str, Path)):
        from .models import GoalSpec
        goal = GoalSpec.from_json(Path(goal).read_text())
    if not is_image_pipeline(goal):
        raise ValueError("offline plan currently supports image collection v1 and layout curation v3")
    contract = goal.initial_operator_parameters
    intent = DatasetIntent.from_dict(contract["intent"])
    policy = QualityPolicy.from_dict(contract["quality_policy"])
    graph = DataNodeGraph.from_dict(contract["graph"])
    budget = DataRunBudget.from_dict(contract["budget"])
    roles = set(contract.get("models", {}).get("policy", {}).get("bindings", {}))
    graph.validate_policy(policy, budget, roles)
    missing = [value for value in contract["stores"].values() if not os.environ.get(value)]
    result = {"operator_id": goal.initial_operator_id, "mode": intent.parameters.get("execution_mode"),
        "network_performed": False, "inference_performed": False, "campaign_created": False,
        "capability_level": "designed", "nodes": [node.to_dict() for node in graph.ordered()],
        "budget": budget.to_dict(), "missing_store_environment": missing, "blockers": []}
    if goal.initial_operator_id == "data.image_collection":
        result["operator_version"], result["hf_endpoint"] = "1", resolve_hf_endpoint()
        result["hub_transport"] = "sdk_http"
        if result["mode"] == "server" and os.environ.get("HF_HUB_DISABLE_XET") != "1":
            result["blockers"].append("set_HF_HUB_DISABLE_XET_before_start")
        if not missing:
            store = ArtifactStore({key: os.environ[value] for key, value in contract["stores"].items()})
            source = store.resolve(graph.nodes[0].inputs["spec"].artifact)
            spec = CollectionSpec.from_dict(json.loads(source.read_text()))
            result.update(collection_spec=spec.to_dict(), hf_endpoint=resolve_hf_endpoint(spec.hf_endpoint))
    else:
        result["operator_version"] = "3"
        result["review_rubric_hash"] = RUBRIC_HASH
        result["resolved_criteria"] = compile_criteria(policy.task_criteria)
        if result["mode"] == "server" and not policy.task_criteria.get("deployments"):
            result["blockers"].append("missing_server_model_deployments")
    if missing:
        result["blockers"].append("missing_artifact_store_environment")
    result["preflight_status"] = "BLOCKED" if result["blockers"] else "OFFLINE_PLAN_VALIDATED"
    return result


def image_pipeline_goal(contract, *, campaign_id, evaluator_path):
    evaluator_path = Path(evaluator_path).resolve()
    mode = contract["intent"]["parameters"]["execution_mode"]
    if mode not in {"fixture", "server"}:
        raise ValueError("image goal requires fixture or server mode")
    collection = contract["intent"]["output_schema"] == "SourceBundle.v1"
    operator_id, version = ("data.image_collection", "1") if collection else ("data.image_curation", "3")
    key = {"input_artifact_hash": payload_sha256(contract["intent"]), "parent_artifact_hash": "none",
        "seed": contract["quality_policy"]["task_criteria"].get("overrides", {}).get("seed", 20261103),
        "operator_id": operator_id, "operator_version": version, "model_config_hash": payload_sha256(contract),
        "preprocessing_hash": payload_sha256(contract["quality_policy"]), "evaluator_hash": file_sha256(evaluator_path),
        "environment_contract_hash": f"image-pipeline-{mode}-v1"}
    return GoalSpec.from_dict({"schema_version": "dataevolver.harness.goal.v1", "campaign_id": campaign_id,
        "objective": {"kind": "image_collection" if collection else "layout_curation", "real_data_execution": mode == "server",
                      "initial_trial_key": key, "canary_trial_key": key},
        "initial_operator_id": operator_id, "initial_operator_parameters": contract, "route_candidates": ["image_pipeline"],
        "budgets": {"max_rounds": 1, "max_operator_calls": 1, "max_experiments": 0, "max_retries": 0,
                    "max_wall_seconds": contract["budget"]["max_wall_seconds"]},
        "resource_policy": {"allow_gpu": False, "allow_network": mode == "server", "allowed_operators": [operator_id]},
        "frozen_evaluator": {"path": str(evaluator_path), "sha256": file_sha256(evaluator_path)},
        "planner": {"kind": "deterministic"}, "interventions": []})
