import hashlib
import json
import os
import subprocess
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataevolver.harness.adapters.three_d_asset import ThreeDAssetEvolutionOperator
from dataevolver.harness.cli import _planner, _production_preflight, _registry, _validate_creation_mode
from dataevolver.harness.models import BudgetSpec, GoalSpec
from dataevolver.harness.planner import AgenticRSIPlanner


ROOT = Path(__file__).resolve().parents[3]
MODULE = "dataevolver.harness.cli"


def run_cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", MODULE, *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def write_goal(tmp_path, campaign_id="cli-demo"):
    evaluator = tmp_path / "evaluator.json"
    evaluator.write_text(
        json.dumps(
            {
                "schema_version": "dataevolver.harness.evaluator.v1",
                "evaluator_id": "fake-quality-v1",
                "revision": "1",
                "hard_constraints": {"artifact_valid": {"required": True}},
                "target_metrics": {
                    "quality": {"required": True, "minimum": 0.8, "minimum_improvement": 0.2}
                },
                "protected_metrics": {"safety": {"required": True, "max_regression": 0.05}},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    evaluator_hash = hashlib.sha256(evaluator.read_bytes()).hexdigest()
    trial = {
        "input_artifact_hash": "input-a",
        "parent_artifact_hash": "parent-a",
        "seed": 17,
        "operator_id": "fake.baseline",
        "operator_version": "1",
        "model_config_hash": "model-a",
        "preprocessing_hash": "pre-a",
        "evaluator_hash": evaluator_hash,
        "environment_contract_hash": "env-a",
    }
    goal = tmp_path / "goal.json"
    goal.write_text(
        json.dumps(
            {
                "schema_version": "dataevolver.harness.goal.v1",
                "campaign_id": campaign_id,
                "objective": {
                    "kind": "fake_quality",
                    "initial_trial_key": trial,
                    "canary_trial_key": trial,
                    "dry_run_fixture": {
                        "baseline": {
                            "status": "FAILED",
                            "metrics": {"quality": 0.4, "safety": 1.0},
                            "hard_constraints": {"artifact_valid": True},
                            "failure_code": "LOW_QUALITY",
                        },
                        "candidate": {
                            "status": "SUCCESS",
                            "metrics": {"quality": 0.9, "safety": 1.0},
                            "hard_constraints": {"artifact_valid": True},
                        },
                    },
                },
                "initial_operator_id": "fake.baseline",
                "route_candidates": ["fake"],
                "budgets": {
                    "max_rounds": 2,
                    "max_operator_calls": 3,
                    "max_experiments": 1,
                    "max_retries": 0,
                    "max_wall_seconds": 60,
                },
                "resource_policy": {
                    "allow_gpu": False,
                    "allowed_operators": ["fake.baseline", "fake.candidate"],
                },
                "frozen_evaluator": {"path": str(evaluator), "sha256": evaluator_hash},
                "interventions": [
                    {
                        "intervention_id": "fix-low-quality",
                        "failure_code": "LOW_QUALITY",
                        "operator_id": "fake.candidate",
                        "level": "L0",
                        "parameters": {"quality_patch": "improve"},
                        "candidate_delta": {"quality_patch": "improve"},
                        "rollback_plan": {"kind": "retain_baseline"},
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return goal


def write_t2i_goal(tmp_path, campaign_id="t2i-cli"):
    evaluator = tmp_path / "t2i_evaluator.json"
    evaluator.write_text(
        json.dumps(
            {
                "schema_version": "dataevolver.harness.evaluator.v1",
                "evaluator_id": "t2i-dryrun-v1",
                "revision": "1",
                "hard_constraints": {"artifact_valid": {"required": True}},
                "target_metrics": {
                    "completion_ratio": {"required": True, "minimum": 1.0, "minimum_improvement": 0.0}
                },
                "protected_metrics": {
                    "resource_policy_respected": {"required": True, "max_regression": 0.0}
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    evaluator_hash = hashlib.sha256(evaluator.read_bytes()).hexdigest()
    trial = {
        "input_artifact_hash": "request-a",
        "parent_artifact_hash": "none",
        "seed": 17,
        "operator_id": "multimodal.t2i_dryrun",
        "operator_version": "1",
        "model_config_hash": "multimodal-t2i-dryrun-v1",
        "preprocessing_hash": "dryrun-v1",
        "evaluator_hash": evaluator_hash,
        "environment_contract_hash": "cpu-only-v1",
    }
    goal = tmp_path / "t2i_goal.json"
    goal.write_text(
        json.dumps(
            {
                "schema_version": "dataevolver.harness.goal.v1",
                "campaign_id": campaign_id,
                "objective": {"kind": "t2i_dryrun", "initial_trial_key": trial, "canary_trial_key": trial},
                "initial_operator_id": "multimodal.t2i_dryrun",
                "initial_operator_parameters": {
                    "user_request": "desk objects",
                    "num_samples": 2,
                    "height": 96,
                    "width": 128,
                },
                "route_candidates": ["t2i"],
                "budgets": {
                    "max_rounds": 1,
                    "max_operator_calls": 1,
                    "max_experiments": 0,
                    "max_retries": 0,
                    "max_wall_seconds": 60,
                },
                "resource_policy": {"allow_gpu": False, "allowed_operators": ["multimodal.t2i_dryrun"]},
                "frozen_evaluator": {"path": str(evaluator), "sha256": evaluator_hash},
                "interventions": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return goal


def parse_output(result):
    return json.loads(result.stdout)


def test_cli_evolve_is_terminal_and_same_request_is_idempotent(tmp_path):
    goal = write_goal(tmp_path)
    campaigns = tmp_path / "campaigns"
    args = (
        "evolve",
        "--goal",
        str(goal),
        "--campaign-root",
        str(campaigns),
        "--idempotency-key",
        "request-1",
        "--dry-run",
    )

    first = parse_output(run_cli(*args))
    event_path = campaigns / "cli-demo" / "events.jsonl"
    event_count = len(event_path.read_text(encoding="utf-8").splitlines())
    second = parse_output(run_cli(*args))

    assert first["state"] == "COMPLETE"
    assert second["state"] == "COMPLETE"
    assert len(event_path.read_text(encoding="utf-8").splitlines()) == event_count
    assert (campaigns / "cli-demo" / "final_report.json").is_file()


def test_cli_step_pause_resume_status_explain_and_replay(tmp_path):
    goal = write_goal(tmp_path, campaign_id="step-demo")
    campaigns = tmp_path / "campaigns"
    common = ("--campaign", "step-demo", "--campaign-root", str(campaigns))

    stepped = parse_output(
        run_cli(
            "step",
            *common,
            "--goal",
            str(goal),
            "--idempotency-key",
            "request-step",
            "--max-actions",
            "1",
            "--format",
            "json",
            "--dry-run",
        )
    )
    assert stepped["previous_state"] == "INIT"
    assert stepped["new_state"] == "PLAN"
    assert stepped["action_count"] == 1

    paused = parse_output(run_cli("pause", *common))
    assert paused["state"] == "PAUSED"
    assert parse_output(run_cli("status", *common))["state"] == "PAUSED"

    resumed = parse_output(run_cli("resume", *common))
    assert resumed["state"] == "COMPLETE"
    explained = parse_output(run_cli("explain", *common, "--limit", "3"))
    assert len(explained["events"]) == 3
    replayed = parse_output(run_cli("replay", *common))
    assert replayed["matches_snapshot"] is True
    assert replayed["state"]["state"] == "COMPLETE"


def test_cli_runs_registered_t2i_dryrun_adapter_and_replays(tmp_path):
    goal = write_t2i_goal(tmp_path)
    campaigns = tmp_path / "campaigns"

    final = parse_output(
        run_cli(
            "evolve",
            "--goal",
            str(goal),
            "--campaign-root",
            str(campaigns),
            "--idempotency-key",
            "t2i-request",
            "--dry-run",
        )
    )
    replayed = parse_output(
        run_cli(
            "replay",
            "--campaign",
            "t2i-cli",
            "--campaign-root",
            str(campaigns),
        )
    )

    assert final["state"] == "COMPLETE"
    assert final["operator_calls"] == 1
    assert replayed["matches_snapshot"] is True
    assert (campaigns / "t2i-cli" / "rounds" / "r000" / "execute" / "dataset" / "manifest.jsonl").is_file()


def test_cli_agentic_selection_is_closed_world_and_missing_backend_fails_before_creation(tmp_path, monkeypatch):
    planner_config = tmp_path / "planner.json"
    planner_config.write_text(
        json.dumps(
            {
                "backend": "openai_compatible",
                "base_url_env": "DATAEVOLVER_AGENT_BASE_URL",
                "api_key_env": "DATAEVOLVER_AGENT_API_KEY",
                "model_env": "DATAEVOLVER_AGENT_MODEL",
                "timeout_seconds": 120,
            }
        ),
        encoding="utf-8",
    )
    scene_template = tmp_path / "scene.json"
    scene_template.write_text('{"blend_path":"fixture.blend"}\n', encoding="utf-8")
    monkeypatch.setenv("DATAEVOLVER_3D_SCENE_TEMPLATE", str(scene_template))
    goal = GoalSpec(
        schema_version="dataevolver.harness.goal.v1",
        campaign_id="agentic-cli",
        objective={
            "kind": "agentic_3d_rsi",
            "asset_ids": ["mug"],
            "initial_trial_key": {
                "input_artifact_hash": "input",
                "parent_artifact_hash": "parent",
                "seed": 17,
                "operator_id": "dataevolver.3d_asset_evolution",
                "operator_version": "1",
                "model_config_hash": "models",
                "preprocessing_hash": "pre",
                "evaluator_hash": "eval",
                "environment_contract_hash": "env",
            },
            "allowed_scene_actions": ["L_KEY_UP"],
        },
        initial_operator_id="dataevolver.3d_asset_evolution",
        route_candidates=("agentic_3d",),
        budgets=BudgetSpec(4, 5, 4, 0, 600),
        resource_policy={"allow_gpu": True, "allowed_operators": ["dataevolver.3d_asset_evolution"]},
        frozen_evaluator={"path": "evaluator.json", "sha256": "a" * 64},
        planner={"kind": "agentic_rsi", "config_path": str(planner_config)},
    )

    assert isinstance(_planner(goal), AgenticRSIPlanner)
    assert isinstance(_registry(goal, tmp_path / "campaigns").get(goal.initial_operator_id), ThreeDAssetEvolutionOperator)
    with pytest.raises(ValueError, match="DATAEVOLVER_AGENT_BASE_URL"):
        _validate_creation_mode(goal, dry_run=False)

    monkeypatch.setenv("DATAEVOLVER_AGENT_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("DATAEVOLVER_AGENT_API_KEY", "test-key")
    monkeypatch.setenv("DATAEVOLVER_AGENT_MODEL", "test-model")
    for name in (
        "QWEN_IMAGE_MODEL_PATH", "SAM3_CKPT", "SAM3_DIR", "HUNYUAN3D_REPO", "MODEL_HUB",
        "PAINT_MODEL_HUB", "DINO_MODEL_PATH", "VLM_MODEL_PATH", "BLENDER_BIN",
    ):
        monkeypatch.setenv(name, str(scene_template))
    monkeypatch.setenv("DATAEVOLVER_3D_GPU_ID", "1")
    _validate_creation_mode(goal, dry_run=False)

    invalid = goal.with_updates(planner={"kind": "arbitrary_import", "config_path": str(planner_config)})
    with pytest.raises(ValueError, match="planner kind is not registered"):
        _planner(invalid)


def test_agentic_production_preflight_verifies_endpoint_gpu_scene_and_dependencies(tmp_path, monkeypatch):
    goal_path = ROOT / "configs" / "harness" / "agentic_3d_rsi_real_canary.json"
    goal = GoalSpec.from_json(goal_path.read_text(encoding="utf-8"))
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"BLENDER")
    scene_template = tmp_path / "scene_template.json"
    scene_template.write_text(
        json.dumps({"blend_path": str(blend), "require_scene_contract": True}),
        encoding="utf-8",
    )
    resource = tmp_path / "resource"
    resource.mkdir()
    blender = tmp_path / "blender"
    blender.write_text("binary", encoding="utf-8")
    sam_checkpoint = tmp_path / "sam3.pt"
    sam_checkpoint.write_bytes(b"weights")
    values = {
        "DATAEVOLVER_AGENT_BASE_URL": "http://127.0.0.1:9000/v1",
        "DATAEVOLVER_AGENT_API_KEY": "test-key",
        "DATAEVOLVER_AGENT_MODEL": "planner-model",
        "QWEN_IMAGE_MODEL_PATH": str(resource),
        "SAM3_CKPT": str(sam_checkpoint),
        "SAM3_DIR": str(resource),
        "HUNYUAN3D_REPO": str(resource),
        "MODEL_HUB": str(resource),
        "PAINT_MODEL_HUB": str(resource),
        "DINO_MODEL_PATH": str(resource),
        "VLM_MODEL_PATH": str(resource),
        "BLENDER_BIN": str(blender),
        "DATAEVOLVER_3D_SCENE_TEMPLATE": str(scene_template),
        "DATAEVOLVER_3D_GPU_ID": "1",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"data": [{"id": "planner-model"}]}).encode("utf-8")

    calls = []

    def run(argv, **kwargs):
        calls.append(tuple(argv))
        if argv[0] == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="1, 1, 81559, 0\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    report = _production_preflight(goal, urlopen=lambda request, timeout: Response(), run=run)
    real_operator = _registry(goal, tmp_path / "campaigns").get("dataevolver.3d_asset_evolution")

    assert report["status"] == "VERIFIED"
    assert report["agent_model"] == "planner-model"
    assert report["gpu"]["index"] == 1
    assert report["scene_template_hash"]
    assert real_operator.resource_config["t2i_height"] == 512
    assert real_operator.resource_config["t2i_width"] == 512
    assert real_operator.resource_config["t2i_steps"] == 20
    assert any(call[0] == "nvidia-smi" for call in calls)
    assert all("test-key" not in json.dumps(value) for value in report.values())

    def unavailable(request, timeout):
        raise urllib.error.URLError("offline")

    with pytest.raises(ValueError, match="Agent endpoint unavailable"):
        _production_preflight(goal, urlopen=unavailable, run=run)
