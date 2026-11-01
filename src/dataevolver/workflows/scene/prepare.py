"""Freeze host-owned scene execution into the existing Goal/Operator contract."""
import json
import shutil
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, identifier, payload_sha256
from dataevolver.harness.adapters.native_scene import NativeSceneOperator, worker_environment
from dataevolver.harness.state_store import atomic_write_json
from dataevolver.paths import CONFIGS_ROOT


def prepare(profile_path, output, campaign, campaign_root):
    identifier(campaign, "campaign")
    profile_path = Path(profile_path).resolve()
    profile = json.loads(profile_path.read_text())
    required = {"stage", "argv", "files", "inputs", "outputs", "timeout_seconds",
                "allowed_env_keys", "allow_gpu", "allow_network"}
    if set(profile) != required or any(type(profile[k]) is not bool for k in ("allow_gpu", "allow_network")):
        raise ValueError("scene profile requires explicit worker, inputs, outputs and resource permissions")

    def local(name):
        path = Path(name).expanduser()
        return path.resolve() if path.is_absolute() else (profile_path.parent / path).resolve()

    binary = shutil.which(profile["argv"][0])
    binary = Path(binary).resolve() if binary else local(profile["argv"][0])
    files = {str(binary): file_sha256(binary)}
    for name in profile["files"]:
        path = local(name)
        files[str(path)] = file_sha256(path)
    # Package workers import these files in the host's Python or Blender Python.
    package = Path(__file__).resolve().parent
    for path in package.rglob("*.py"):
        files[str(path)] = file_sha256(path)
    from dataevolver.dataset import contracts
    files[str(Path(contracts.__file__).resolve())] = file_sha256(Path(contracts.__file__))
    inputs = {role: {"path": str(local(name)), "sha256": file_sha256(local(name))}
              for role, name in profile["inputs"].items()}
    bound_paths = set(files) | {record["path"] for record in inputs.values()}
    argv = [str(binary)] + [str(local(x)) if str(local(x)) in bound_paths else x for x in profile["argv"][1:]]
    config = {key: profile[key] for key in ("stage", "outputs", "timeout_seconds", "allowed_env_keys")}
    environment = worker_environment(profile["allowed_env_keys"])
    config.update(schema_version="dataevolver.native_scene.v2", argv=argv, files=files, inputs=inputs,
                  environment_sha256=payload_sha256(environment))
    campaign_root = Path(campaign_root).resolve()
    if profile["stage"] == "ue5":
        from .ue5.render_batch import package_contract, validate_manifest
        plan = validate_manifest(json.loads(Path(inputs["render_manifest"]["path"]).read_text()))
        if not (campaign_root / campaign).is_relative_to(Path(plan["root"]).resolve() / "experiments"):
            raise ValueError("native UE5 campaigns must be inside the render root's experiments directory")
        files.update(package_contract(plan)["file_hashes"])
    operator = NativeSceneOperator(output_root=campaign_root / campaign, configuration=config)
    evaluator = (CONFIGS_ROOT / "harness/native_scene_evaluator.json").resolve()
    key = {"input_artifact_hash": payload_sha256(inputs), "parent_artifact_hash": payload_sha256(config),
           "seed": 0, "operator_id": operator.spec.operator_id, "operator_version": operator.spec.version,
           "model_config_hash": "no-implicit-model", "preprocessing_hash": payload_sha256(files),
           "evaluator_hash": file_sha256(evaluator), "environment_contract_hash": config["environment_sha256"]}
    goal = {"schema_version": "dataevolver.harness.goal.v1", "campaign_id": campaign,
        "objective": {"host_workload": {"kind": "native_scene", "configuration": {
            "output_root": str(campaign_root / campaign), "configuration": config}},
            "required_hard_constraints": ["artifact_valid"], "target_metrics": {"completion_ratio": 1.0},
            "initial_trial_key": key, "canary_trial_key": key},
        "initial_operator_id": operator.spec.operator_id, "initial_operator_parameters": {},
        "planner": {"kind": "deterministic"}, "route_candidates": [operator.spec.operator_id],
        "budgets": {"max_rounds": 1, "max_operator_calls": 1, "max_experiments": 0, "max_retries": 0,
                    "max_planner_calls": 0, "max_wall_seconds": config["timeout_seconds"] + 60},
        "resource_policy": {"allowed_operators": [operator.spec.operator_id],
                            "allow_gpu": profile["allow_gpu"], "allow_network": profile["allow_network"]},
        "frozen_evaluator": {"path": str(evaluator), "sha256": file_sha256(evaluator)}}
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(root / "goal.json", goal)
    return {"compiled": True, "goal": str(root / "goal.json"), "operator": operator.spec.operator_id,
            "execution_performed": False, "inference_performed": False}
