"""Recovery fixtures preserve actual native effects; no new model invocation."""
import json

import pytest

from dataevolver.adapters.model_gateway import ModelError, validate_response_schema
from dataevolver.adapters.codex_planner_recovery import reconcile
from dataevolver.harness.runtime_recovery import recover_planner
from dataevolver.testing.test_codex_planner_provider import setup, request


def test_cli_completed_effect_is_reconciled_after_local_validator_dependency_failure(tmp_path, monkeypatch):
    provider, gateway, config = setup(tmp_path, monkeypatch)
    def missing(*args):
        raise ModelError('DEPENDENCY_MISSING', 'fixture schema dependency was absent')
    monkeypatch.setattr('dataevolver.adapters.codex_planner.validate_response_schema', missing)
    with pytest.raises(ModelError):
        gateway.invoke(request())
    assert json.loads((tmp_path / 'calls.json').read_text()) == 1
    failure = (tmp_path / 'native/fixture-request/native.failure.json').read_bytes()
    monkeypatch.setattr('dataevolver.adapters.codex_planner.validate_response_schema', validate_response_schema)
    cfg = {'root': str(tmp_path), 'worker': config, 'profile': provider.profile.to_dict()}
    proof = reconcile(cfg, request())
    assert proof['new_model_calls'] == 0 and proof['decision_edited'] is False
    recovered = gateway.recover(request())
    assert recovered.output == {'ok': True} and recovered.reused
    assert json.loads((tmp_path / 'calls.json').read_text()) == 1
    assert (tmp_path / 'native/fixture-request/native.failure.json').read_bytes() == failure
    assert gateway.audit_usage()['model_calls'] == 1


def test_reconciler_rejects_different_prompt_without_rewriting_output(tmp_path, monkeypatch):
    provider, gateway, config = setup(tmp_path, monkeypatch)
    def missing(*args):
        raise ModelError('DEPENDENCY_MISSING', 'fixture dependency')
    monkeypatch.setattr('dataevolver.adapters.codex_planner.validate_response_schema', missing)
    with pytest.raises(ModelError): gateway.invoke(request())
    from dataclasses import replace
    changed = replace(request(), messages=({'role': 'user', 'content': 'changed'},))
    with pytest.raises(ValueError, match='binding'):
        reconcile({'root': str(tmp_path), 'worker': config, 'profile': provider.profile.to_dict()}, changed)
    assert not (tmp_path / 'native/fixture-request/native.json').exists()


def test_controller_recovery_uses_existing_receipt_and_durable_event(tmp_path):
    from dataevolver.adapters.codex_planner import GatewayPlannerBackend
    from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile
    from dataevolver.adapters.model_gateway import ModelGateway, MockModelProvider
    from dataevolver.harness.models import BudgetSpec, ControllerState
    from dataevolver.harness.runtime_rsi import RuntimeRSIPlanner
    from dataevolver.harness.controller import EvolutionController
    from dataevolver.testing.test_harness_controller import build_controller
    from dataevolver.testing.test_runtime_rsi import FixtureDomain, action
    old, source = build_controller(tmp_path)
    goal = old.goal.with_updates(campaign_id='recover-runtime', planner={'kind': 'runtime_rsi'}, interventions=(),
        budgets=BudgetSpec(4, 6, 2, 0, 60, 5))
    profile = ModelProfile('fixture', 'mock', 'fixture-model', '1', mock_output=action('OBSERVE', tool='inspect'))
    calls = []
    class LostAfterEffect(MockModelProvider):
        def invoke(self, req, digest):
            calls.append(req.request_id); super().invoke(req, digest)
            raise RuntimeError('fixture crash after actual stored output')
    gateway = ModelGateway(profiles=(profile,), policy=ModelPolicy({'agent.runtime_rsi': 'fixture'}, None, usage_mode='accounting'),
        receipt_root=tmp_path/'ledger', providers={'fixture': lambda: LostAfterEffect(profile,tmp_path/'native')})
    backend = GatewayPlannerBackend(gateway, model_id='fixture-model')
    planner = RuntimeRSIPlanner(backend=backend, domain=FixtureDomain(), trace_root=tmp_path/'campaigns/recover-runtime/planner')
    c = EvolutionController.create(repo_root=tmp_path,campaigns_root=tmp_path/'campaigns',goal=goal,
        registry=old.registry,planner=planner,evaluator_contract=old.evaluator.contract,
        execution_paths=(source,),idempotency_key='fixture-recovery')
    assert c.run().state is ControllerState.ABSTAIN
    assert len(calls) == 1
    original = (planner.trace_root/'p000/trace.json').read_bytes()
    result = recover_planner(c)
    assert result['new_model_calls'] == 0 and result['status'] == 'PROPOSE'
    assert c.status().planner_calls == 1 and len(calls) == 1
    assert c.status().data['rsi_pending_decision']['action'] == 'OBSERVE'
    assert c.status().terminal_reason == ''
    assert c.replay().to_dict() == c.status().to_dict()
    assert (planner.trace_root/'p000/trace.json').read_bytes() == original
    assert recover_planner(c)['status'] == 'ALREADY_RECOVERED'
