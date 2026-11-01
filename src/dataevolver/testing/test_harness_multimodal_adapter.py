import json
import hashlib
from pathlib import Path

import pytest

from dataevolver.harness.adapters.multimodal_dryrun import MultimodalT2IDryRunOperator
from dataevolver.harness.controller import EvolutionController
from dataevolver.harness.evaluator import EvaluatorContract
from dataevolver.harness.models import (
    BudgetSpec,
    ControllerState,
    EvidenceStatus,
    GoalSpec,
    MatchedTrialKey,
    OperatorCall,
    RecoveryStatus,
    StageStatus,
)
from dataevolver.harness.operators import OperatorRegistry
from dataevolver.harness.planner import DeterministicPlanner
from dataevolver.workflows.multimodal.router import build_request
from dataevolver.workflows.multimodal.run_multimodal_dataset import run as run_multimodal
from dataevolver.workflows.multimodal.router import build_request
from dataevolver.workflows.multimodal.run_multimodal_dataset import run as run_multimodal


def make_call(tmp_path, *, call_id="t2i-call", work_dir=None, parameters=None):
    return OperatorCall(
        campaign_id="t2i-campaign",
        call_id=call_id,
        idempotency_key=call_id,
        operator_id="multimodal.t2i_dryrun",
        operator_version="1",
        round_index=0,
        work_dir=str(work_dir or (tmp_path / "run")),
        parameters=parameters
        or {
            "user_request": "generate two desk objects",
            "num_samples": 2,
            "height": 96,
            "width": 128,
            "seed": 17,
            "sample_prefix": "desk",
            "skip_existing": True,
            "validate_outputs": True,
        },
    )


def test_t2i_dryrun_operator_executes_real_workflow_and_is_idempotent(tmp_path):
    operator = MultimodalT2IDryRunOperator(allowed_write_root=tmp_path)
    call = make_call(tmp_path)

    first = operator.execute(call)
    first_verification = operator.verify_effect(call, first)
    second = operator.execute(call)

    dataset = Path(call.work_dir) / "dataset"
    request = json.loads((dataset / "request.json").read_text(encoding="utf-8"))
    summary = json.loads((dataset / "dataset_summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]

    assert first.status is StageStatus.SUCCESS
    assert first_verification.status is EvidenceStatus.VERIFIED
    assert second.reused is True
    assert request["route"] == "t2i"
    assert request["dry_run"] is True
    assert request["allow_model_inference"] is False
    assert request["generator"] == "dryrun"
    assert request["device"] == "cpu"
    assert request["vlm_eval"] is False
    assert summary["sample_count"] == 2
    assert len(rows) == 2
    assert all((dataset / row["output_path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for row in rows)
    assert first.stage_result["metrics"] == {
        "completion_ratio": 1.0,
        "resource_policy_respected": 1.0,
    }


def test_t2i_adapter_matches_direct_multimodal_run_semantics(tmp_path):
    parameters = {
        "user_request": "generate two desk objects",
        "num_samples": 2,
        "height": 96,
        "width": 128,
        "seed": 17,
        "sample_prefix": "desk",
        "skip_existing": True,
        "validate_outputs": True,
    }
    adapter = MultimodalT2IDryRunOperator(allowed_write_root=tmp_path / "adapter")
    call = make_call(tmp_path, work_dir=tmp_path / "adapter" / "run", parameters=parameters)
    adapter.execute(call)
    adapter_root = Path(call.work_dir) / "dataset"

    direct_root = tmp_path / "direct"
    request = build_request(
        route="t2i",
        user_request=parameters["user_request"],
        output_root=str(direct_root),
        dry_run=True,
        allow_model_inference=False,
        num_samples=parameters["num_samples"],
        input_image_dir=None,
        prompt_file=None,
        generator="dryrun",
        model_name=None,
        model_path=None,
        device="cpu",
        height=parameters["height"],
        width=parameters["width"],
        steps=1,
        num_frames=1,
        fps=1,
        true_cfg_scale=1.0,
        guidance_scale=1.0,
        negative_prompt="",
        seed=parameters["seed"],
        sample_prefix=parameters["sample_prefix"],
        skip_existing=parameters["skip_existing"],
        validate_outputs=parameters["validate_outputs"],
        vlm_eval=False,
        vlm_device="cpu",
        vlm_model_path=None,
        vlm_max_samples=None,
        vlm_max_retries=1,
        vlm_review_mode="studio",
        vlm_use_freeform_first=False,
        vlm_enable_thinking=False,
    )
    run_multimodal(request)

    adapter_request = json.loads((adapter_root / "request.json").read_text(encoding="utf-8"))
    direct_request = json.loads((direct_root / "request.json").read_text(encoding="utf-8"))
    request_fields = {
        "route",
        "dry_run",
        "allow_model_inference",
        "num_samples",
        "generator",
        "model_name",
        "device",
        "height",
        "width",
        "seed",
        "sample_prefix",
        "skip_existing",
        "validate_outputs",
        "vlm_eval",
    }
    assert {key: adapter_request[key] for key in request_fields} == {
        key: direct_request[key] for key in request_fields
    }

    adapter_summary = json.loads((adapter_root / "dataset_summary.json").read_text(encoding="utf-8"))
    direct_summary = json.loads((direct_root / "dataset_summary.json").read_text(encoding="utf-8"))
    summary_fields = {
        "schema",
        "route",
        "dry_run",
        "generator",
        "sample_count",
        "status_counts",
        "validation_counts",
        "vlm_counts",
        "outputs",
    }
    assert {key: adapter_summary[key] for key in summary_fields} == {
        key: direct_summary[key] for key in summary_fields
    }

    adapter_rows = [json.loads(line) for line in (adapter_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    direct_rows = [json.loads(line) for line in (direct_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]

    def semantic_rows(rows):
        normalized = json.loads(json.dumps(rows))
        for row in normalized:
            row.get("validation", {}).pop("path", None)
        return normalized

    assert semantic_rows(adapter_rows) == semantic_rows(direct_rows)
    assert all(
        (adapter_root / row["output_path"]).read_bytes() == (direct_root / row["output_path"]).read_bytes()
        for row in adapter_rows
    )


def test_t2i_adapter_matches_direct_multimodal_api_semantics(tmp_path):
    parameters = {
        "user_request": "generate two desk objects",
        "num_samples": 2,
        "height": 96,
        "width": 128,
        "seed": 17,
        "sample_prefix": "desk",
        "skip_existing": True,
        "validate_outputs": True,
    }
    direct_root = tmp_path / "direct"
    direct_request = build_request(
        route="t2i",
        user_request=parameters["user_request"],
        output_root=str(direct_root),
        dry_run=True,
        allow_model_inference=False,
        num_samples=parameters["num_samples"],
        input_image_dir=None,
        prompt_file=None,
        generator="dryrun",
        model_name=None,
        model_path=None,
        device="cpu",
        height=parameters["height"],
        width=parameters["width"],
        steps=1,
        num_frames=1,
        fps=1,
        true_cfg_scale=1.0,
        guidance_scale=1.0,
        negative_prompt="",
        seed=parameters["seed"],
        sample_prefix=parameters["sample_prefix"],
        skip_existing=parameters["skip_existing"],
        validate_outputs=parameters["validate_outputs"],
        vlm_eval=False,
        vlm_device="cpu",
        vlm_model_path=None,
        vlm_max_samples=None,
        vlm_max_retries=1,
        vlm_review_mode="studio",
        vlm_use_freeform_first=False,
        vlm_enable_thinking=False,
    )
    run_multimodal(direct_request)

    operator = MultimodalT2IDryRunOperator(allowed_write_root=tmp_path / "adapter")
    call = make_call(
        tmp_path,
        call_id="equivalence",
        work_dir=tmp_path / "adapter" / "run",
        parameters=parameters,
    )
    operator.execute(call)
    adapter_root = Path(call.work_dir) / "dataset"

    direct_request_json = json.loads((direct_root / "request.json").read_text(encoding="utf-8"))
    adapter_request_json = json.loads((adapter_root / "request.json").read_text(encoding="utf-8"))
    direct_request_json.pop("output_root")
    adapter_request_json.pop("output_root")
    assert adapter_request_json == direct_request_json

    direct_summary = json.loads((direct_root / "dataset_summary.json").read_text(encoding="utf-8"))
    adapter_summary = json.loads((adapter_root / "dataset_summary.json").read_text(encoding="utf-8"))
    direct_summary.pop("created_at")
    adapter_summary.pop("created_at")
    assert adapter_summary == direct_summary

    direct_rows = [
        json.loads(line)
        for line in (direct_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    adapter_rows = [
        json.loads(line)
        for line in (adapter_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for direct_row, adapter_row in zip(direct_rows, adapter_rows, strict=True):
        direct_validation_path = Path(direct_row["validation"].pop("path")).resolve()
        adapter_validation_path = Path(adapter_row["validation"].pop("path")).resolve()
        direct_validation_path.relative_to(direct_root.resolve())
        adapter_validation_path.relative_to(adapter_root.resolve())
    assert adapter_rows == direct_rows
    assert [
        (adapter_root / row["output_path"]).read_bytes() for row in adapter_rows
    ] == [
        (direct_root / row["output_path"]).read_bytes() for row in direct_rows
    ]


def test_t2i_dryrun_operator_recovers_complete_outputs_without_receipt(tmp_path):
    operator = MultimodalT2IDryRunOperator(allowed_write_root=tmp_path)
    call = make_call(tmp_path)
    receipt = operator.execute(call)
    Path(receipt.receipt_path).unlink()

    recovered = operator.recover(call)

    assert recovered.status is RecoveryStatus.RECOVERED
    assert recovered.receipt is not None
    assert recovered.receipt.reused is True
    assert operator.verify_effect(call, recovered.receipt).status is EvidenceStatus.VERIFIED


def test_t2i_dryrun_operator_missing_outputs_are_retry_safe(tmp_path):
    operator = MultimodalT2IDryRunOperator(allowed_write_root=tmp_path)

    recovered = operator.recover(make_call(tmp_path))

    assert recovered.status is RecoveryStatus.RETRY_SAFE
    assert recovered.receipt is None


def test_t2i_dryrun_operator_rejects_unknown_parameters_and_path_escape(tmp_path):
    operator = MultimodalT2IDryRunOperator(allowed_write_root=tmp_path / "allowed")
    unknown = make_call(
        tmp_path,
        work_dir=tmp_path / "allowed" / "run",
        parameters={"user_request": "x", "route": "edit"},
    )
    escaped = make_call(tmp_path, call_id="escape", work_dir=tmp_path / "outside")

    with pytest.raises(ValueError, match="unsupported parameters"):
        operator.execute(unknown)
    with pytest.raises(ValueError, match="outside allowed write root"):
        operator.execute(escaped)


def test_t2i_dryrun_operator_corrupted_manifest_fails_closed(tmp_path):
    operator = MultimodalT2IDryRunOperator(allowed_write_root=tmp_path)
    call = make_call(tmp_path)
    receipt = operator.execute(call)
    Path(receipt.receipt_path).unlink()
    manifest = Path(call.work_dir) / "dataset" / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[0]["output_path"] = "../../escape.png"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    verification = operator.verify_effect(call, None)
    recovered = operator.recover(call)

    assert verification.status is EvidenceStatus.CONTRADICTED
    assert "escapes dataset root" in verification.reason
    assert recovered.status is RecoveryStatus.AMBIGUOUS


def test_controller_runs_t2i_dryrun_goal_to_complete_without_canary(tmp_path):
    evaluator_path = tmp_path / "evaluator.json"
    evaluator_path.write_text(
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
    evaluator_hash = hashlib.sha256(evaluator_path.read_bytes()).hexdigest()
    key = MatchedTrialKey(
        input_artifact_hash="request-a",
        parent_artifact_hash="none",
        seed=17,
        operator_id="multimodal.t2i_dryrun",
        operator_version="1",
        model_config_hash="multimodal-t2i-dryrun-v1",
        preprocessing_hash="dryrun-v1",
        evaluator_hash=evaluator_hash,
        environment_contract_hash="cpu-only-v1",
    )
    goal = GoalSpec(
        schema_version="dataevolver.harness.goal.v1",
        campaign_id="t2i-controller",
        objective={"kind": "t2i_dryrun", "initial_trial_key": key.to_dict(), "canary_trial_key": key.to_dict()},
        initial_operator_id="multimodal.t2i_dryrun",
        initial_operator_parameters={"user_request": "desk objects", "num_samples": 2, "height": 96, "width": 128},
        route_candidates=("t2i",),
        budgets=BudgetSpec(1, 1, 0, 0, 60),
        resource_policy={"allow_gpu": False, "allowed_operators": ["multimodal.t2i_dryrun"]},
        frozen_evaluator={"path": str(evaluator_path), "sha256": evaluator_hash},
    )
    campaigns = tmp_path / "campaigns"
    registry = OperatorRegistry()
    registry.register(MultimodalT2IDryRunOperator(allowed_write_root=campaigns / goal.campaign_id))
    controller = EvolutionController.create(
        repo_root=tmp_path,
        campaigns_root=campaigns,
        goal=goal,
        registry=registry,
        planner=DeterministicPlanner(()),
        evaluator_contract=EvaluatorContract.from_path(evaluator_path),
        execution_paths=(Path(__file__).resolve(),),
        idempotency_key="t2i-request",
    )

    final = controller.run()
    event_types = [event.event_type for event in controller.store.events.read()]

    assert final.state is ControllerState.COMPLETE
    assert final.operator_calls == 1
    assert "INTERVENTION_PROPOSED" not in event_types
    assert "CAPABILITY_CANDIDATE_CREATED" not in event_types
    assert (controller.layout.campaign_dir / "rounds" / "r000" / "execute" / "dataset" / "manifest.jsonl").is_file()
