import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from dataevolver.adapters.model_gateway import MockModelProvider
from dataevolver.harness.adapters.visual_mock import VisualContractMockOperator
from dataevolver.harness.cli import _controller, _registry
from dataevolver.harness.models import ControllerState, GoalSpec, OperatorCall, RecoveryStatus


ROOT = Path(__file__).resolve().parents[1]


def goal():
    return GoalSpec.from_dict(json.loads((ROOT / "configs/harness/visual_contract_mock.json").read_text()))


def test_visual_contract_graph_uses_existing_controller_and_terminal_replay(tmp_path):
    controller = _controller(goal(), tmp_path, create=True, idempotency_key="mock-test")
    assert controller.run().state is ControllerState.COMPLETE
    assert controller.replay().to_dict() == controller.status().to_dict()
    campaign = tmp_path / goal().campaign_id
    nodes = list(campaign.glob("rounds/*/execute/node_runs/*/node_run.json"))
    assert len(nodes) == 2
    assert all(json.loads(x.read_text())["schema_version"] == "dataevolver.node_run.v1" for x in nodes)
    before = (campaign / "events.jsonl").read_bytes()
    reopened = _controller(goal(), tmp_path, create=False)
    assert reopened.run().state is ControllerState.COMPLETE
    assert (campaign / "events.jsonl").read_bytes() == before


def test_controller_recovers_a_crash_between_model_effect_and_node_commit(tmp_path, monkeypatch):
    controller = _controller(goal(), tmp_path, create=True, idempotency_key="interrupted")
    controller.tick(2)
    assert controller.status().state is ControllerState.EXECUTE
    call = OperatorCall.from_dict(controller.status().data["pending_call"])
    original = MockModelProvider.invoke
    calls = []
    def crash_once(self, request, digest):
        calls.append(request.request_id)
        response = original(self, request, digest)
        if calls == ["plan"]:
            raise RuntimeError("crash after durable provider effect")
        return response
    monkeypatch.setattr(MockModelProvider, "invoke", crash_once)
    with pytest.raises(RuntimeError, match="crash"):
        controller.tick()
    reopened = _controller(goal(), tmp_path, create=False)
    assert reopened.run().state is ControllerState.COMPLETE
    assert calls == ["plan", "review"]
    intents = list(Path(call.work_dir).glob("model_receipts/*.intent.json"))
    assert len(intents) == 2 and all(json.loads(x.read_text())["attempts"] == 1 for x in intents)
    for path in Path(call.work_dir).glob(".receipts/*.json"):
        path.unlink()
    operator = _registry(goal(), tmp_path).get("visual.contract_mock")
    assert operator.recover(call).status is RecoveryStatus.RECOVERED
    assert calls == ["plan", "review"]


def test_output_tampering_and_path_escape_fail_closed(tmp_path):
    controller = _controller(goal(), tmp_path, create=True, idempotency_key="integrity")
    controller.tick(2)
    call = OperatorCall.from_dict(controller.status().data["pending_call"])
    controller.run()
    operator = _registry(goal(), tmp_path).get("visual.contract_mock")
    (Path(call.work_dir) / "node_runs/plan/output.json").write_text('{"plan":"tampered"}')
    assert operator.recover(call).status is RecoveryStatus.AMBIGUOUS
    with pytest.raises(ValueError, match="escapes"):
        operator.execute(replace(call, work_dir=str(tmp_path / "outside")))


def test_invalid_graph_or_model_policy_is_rejected_before_execution(tmp_path):
    contract = goal().initial_operator_parameters
    contract["graph"]["nodes"][0]["parameters"]["shell"] = "not-executable"
    with pytest.raises(ValueError, match="unsupported"):
        VisualContractMockOperator(allowed_write_root=tmp_path / "not-created", contract=contract)
    assert not (tmp_path / "not-created").exists()
    contract = goal().initial_operator_parameters
    contract["model_profiles"][0]["backend"] = "external_api"
    with pytest.raises(ValueError, match="mock"):
        VisualContractMockOperator(allowed_write_root=tmp_path / "not-created", contract=contract)


def run_cli(*arguments):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, "-m", "dataevolver.harness.cli", *arguments],
                          cwd=ROOT, env=env, capture_output=True, text=True, check=False)


def test_visual_cli_evolve_resume_replay_and_no_inference_without_dry_run(tmp_path):
    config = ROOT / "configs/harness/visual_contract_mock.json"
    base = ["--campaign-root", str(tmp_path)]
    result = run_cli("evolve", "--goal", str(config), "--idempotency-key", "cli", "--dry-run", *base)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["state"] == "COMPLETE"
    resume = run_cli("resume", "--campaign", goal().campaign_id, *base)
    assert resume.returncode == 0 and json.loads(resume.stdout)["state"] == "COMPLETE"
    replay = run_cli("replay", "--campaign", goal().campaign_id, *base)
    assert replay.returncode == 0 and json.loads(replay.stdout)["matches_snapshot"]
    rejected = run_cli("evolve", "--goal", str(config), "--idempotency-key", "no-dry-run", "--campaign-root", str(tmp_path / "reject"))
    assert rejected.returncode == 1 and not (tmp_path / "reject").exists()


def test_model_inspection_is_configuration_only_unless_health_is_requested(tmp_path):
    config = tmp_path / "models.json"
    contract = goal().initial_operator_parameters
    config.write_text(json.dumps({"profiles": contract["model_profiles"], "policy": contract["model_policy"]}))
    result = run_cli("models", "--config", str(config))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["inference_performed"] is False and payload["external_api_enabled"] is False
    assert payload["roles"]["planner"]["health"]["status"] == "NOT_CHECKED"
    assert not (tmp_path / ".unused-model-inspection").exists()


def test_gateway_implementation_drift_pauses_before_a_side_effect(tmp_path, monkeypatch):
    from dataevolver.harness import evaluator
    controller = _controller(goal(), tmp_path, create=True, idempotency_key="drift")
    original = evaluator.sha256_file
    monkeypatch.setattr(evaluator, "sha256_file", lambda path: "changed" if str(path).endswith("adapters/model_gateway.py") else original(path))
    controller.tick()
    assert controller.status().state is ControllerState.PAUSED_CONTRACT_DRIFT
    assert not list((tmp_path / goal().campaign_id).glob("**/effects/*.json"))
