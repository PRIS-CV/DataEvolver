"""Fixture control-plane evidence; these tests do not validate image quality."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import ValidationError

from dataevolver.harness.capability_requests import validate_capability_request
from dataevolver.harness.models import ControllerState
from dataevolver.harness.runtime_rsi import validate_decision
from dataevolver.testing.test_runtime_rsi import make_runtime, action
from dataevolver.testing.test_training_rsi_domain import fixture_domain, observe


def request():
    return {'problem': 'Fixture: registered operations cannot meet the fixed quality gate.',
            'proposed_capability': 'A reusable evidence-inspection operation.',
            'implementation_kind': 'registered_operator',
            'required_inputs': ['committed source and rejected candidate'],
            'required_outputs': ['evidence with uncertainty'],
            'acceptance_checks': ['preserve original constraints', 'recover without another inference'],
            'validation_design': 'Compare on predeclared independent fixture sources under equal budgets.'}


def decision():
    return action('REQUEST_CAPABILITY', layer='M', patch='capability_gap', parameters=request())


def test_request_ends_same_controller_without_an_experiment_or_knowledge_promotion(tmp_path):
    c, backend = make_runtime(tmp_path, [decision()])
    original = c.planner.domain.context
    c.planner.domain.context = lambda s: {**original(s), 'capability_requests_allowed': True}
    c.planner.domain.compile_capability_request = lambda d, s: {
        'status': 'IMPLEMENTATION_REQUIRED', 'proposal_actor': 'runtime_agent', 'request': request(),
        'method_improvement_verified': False}
    end = c.run()
    assert end.state is ControllerState.ABSTAIN and end.terminal_reason == 'CAPABILITY_REQUESTED'
    assert end.planner_calls == 1 and end.experiments_used == 0 and end.operator_calls == 1
    assert not c.capabilities.list_capabilities()
    artifact = Path(end.data['rsi_capability_request']['path'])
    assert json.loads(artifact.read_text())['status'] == 'IMPLEMENTATION_REQUIRED'
    old = c.layout.events_path.read_bytes()
    c.run()
    assert c.layout.events_path.read_bytes() == old and len(backend.calls) == 1


def test_crash_after_request_event_recovers_without_another_model_decision(tmp_path):
    c, backend = make_runtime(tmp_path, [decision()])
    original_context = c.planner.domain.context
    c.planner.domain.context = lambda s: {**original_context(s), 'capability_requests_allowed': True}
    c.planner.domain.compile_capability_request = lambda d, s: {'status': 'IMPLEMENTATION_REQUIRED'}
    original_abstain = c._abstain
    def interrupt(state, reason):
        if reason == 'CAPABILITY_REQUESTED':
            raise RuntimeError('fixture process interruption after committed request')
        return original_abstain(state, reason)
    c._abstain = interrupt
    with pytest.raises(RuntimeError, match='process interruption'):
        c.run()
    assert c.status().data['rsi_capability_request']
    c._abstain = original_abstain
    assert c.run().terminal_reason == 'CAPABILITY_REQUESTED' and len(backend.calls) == 1


def test_domain_request_binds_fresh_evidence_and_keeps_requirements_unmodified(tmp_path):
    domain, state, trial = fixture_domain(tmp_path)
    domain.contract['models'] = {'fixture_model': 'fixed'}
    d = decision(); d['evidence_ids'] = ['probe']
    with pytest.raises(ValueError, match='committed current-source'):
        domain.compile_capability_request(d, state)
    observe(domain, state, trial)
    d['evidence_ids'] = [state.data['rsi_observations'][0]['call_id']]
    value = domain.compile_capability_request(d, state)
    assert value['source'] == trial['source'] and value['rejected_candidate'] == trial['candidate']
    assert value['method_improvement_verified'] is False and value['knowledge_promoted'] is False
    assert value['implementation_authorized_by_request'] is False and value['production_effects'] == 0
    path = Path(state.data['rsi_observations'][0]['stage_result']['evidence_refs'][0])
    observed = json.loads(path.read_text()); observed['candidate']['sha256'] = '0'*64
    path.write_text(json.dumps(observed))
    with pytest.raises(ValueError, match='committed current-source'):
        domain.compile_capability_request(d, state)


def test_frozen_review_mode_cannot_execute_a_recipe(tmp_path):
    domain, state, _ = fixture_domain(tmp_path)
    domain.value['capability_review_only'] = True
    context = domain.context(state)
    assert context['allowed_actions'] == ['REQUEST_CAPABILITY', 'ABSTAIN']
    assert not context['allowed_patches'] and not context['allowed_observations']
    d = action('RUN_EXPERIMENT', layer='R', patch='background_recipe', parameters={}, evidence=['current_trial'])
    with pytest.raises(ValueError, match='frozen execution mode'):
        validate_decision(d, context)
    with pytest.raises(ValueError, match='cannot execute'):
        domain.compile_intervention(d, state)


def test_review_only_trial_operator_rejects_direct_production_bypass(tmp_path):
    from dataevolver.harness.adapters.training_rsi import TrainingRSITrialOperator
    domain, _, _ = fixture_domain(tmp_path)
    domain.value['capability_review_only'] = True
    operator = TrainingRSITrialOperator(domain)
    call = SimpleNamespace(parameters={'scope': 'candidate-000', 'patch': {}, 'experience_id': ''})
    with pytest.raises(ValueError, match='unchanged cached baseline'):
        operator._prepare(call)
    call.parameters = {'scope': 'baseline', 'patch': {'kind': 'background_recipe'}, 'experience_id': ''}
    with pytest.raises(ValueError, match='unchanged cached baseline'):
        operator._prepare(call)


@pytest.mark.parametrize('field,value', [('regions', []), ('budget', 10), ('source_edit', 'fixture.py'),
                                       ('quality_threshold', 0)])
def test_implementation_request_cannot_smuggle_executable_patch_fields(field, value):
    invalid = {**request(), field: value}
    with pytest.raises(ValidationError):
        validate_capability_request(invalid)


def test_unregistered_domain_cannot_request_capabilities():
    with pytest.raises(ValueError, match='non-execution authority'):
        validate_decision(decision(), {'available_evidence_ids': ['baseline']})
