"""Feedback/admission invariants, including preserved refusal and byte drift."""
import copy
import json
from pathlib import Path

import pytest

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from dataevolver.harness.training_candidate_feedback import (
    summarize_execution, ProductionFeedbackReviewDomain, FrozenTrainingCandidateFeedbackAuthority)
from dataevolver.harness.adapters.training_candidate_feedback import TrainingCandidateFeedbackOperator
from dataevolver.harness.models import OperatorCall, EvidenceStatus, RecoveryStatus
from dataevolver.harness.runtime_rsi import validate_decision
from dataevolver.testing.test_runtime_capability_requests import decision


def fixture_summary():
    output = {'status': 'unsupported', 'authorized_support': {}, 'image_submission_allowed': False,
        'candidate_adoption_allowed': False, 'unresolved': [
            {'reason': 'boundary', 'pixels': [[2, 3]]}, {'reason': 'boundary', 'pixels': [[2, 3]]}],
        'conflicts': [{'reason': 'required_coverage_intersects_preservation'}]}
    report = {'production_context_scope': 'registered_production_runtime', 'registered_evidence_used': True,
        'production_qualification_passed': False, 'image_request_submitted': False, 'new_accepted_records': 0,
        'unsupported_submission_block_verified': True,
        'lifecycle_total': {'model_calls': 4, 'planner_calls': 2, 'operator_calls': 13,
                            'wall_seconds': 257, 'output_bytes': 16184832, 'image_experiments': 0},
        'lifecycle_usage': {'model_calls': 0}}
    execution = {'status': 'COMPLETED', 'isolation_verified': True, 'return_code': 0,
        'new_model_calls': 0, 'stdout': json.dumps(output), 'output': output}
    projection = {'registered_status': 'unsupported', 'boundary_envelope_added_authority_pixels': 0}
    origin = {'resource_deadline_epoch': 1792328957.7055008, 'execution_sha256': '1'*64}
    request = {'request': {'problem': 'Missing independently registered boundary authority'}}
    return report, execution, projection, origin, request


def summary():
    r, e, p, o, q = fixture_summary()
    return summarize_execution(r, e, p, origin=o, request=q)


class FixtureAuthority:
    def __init__(self, tmp_path):
        self.value = summary(); self.changed = False
        self.path = tmp_path/'original.json'; self.path.write_text('original')
        self.digest = file_sha256(self.path)
    def identity(self): return {'fixture': not self.changed}
    def read(self):
        if file_sha256(self.path) != self.digest: raise ValueError('historical artifact changed')
        return copy.deepcopy(self.value), {'artifact_hashes': {str(self.path): self.digest}}


def admission(value):
    prior = value['lifecycle_total']
    inc = {'model_calls': 1, 'planner_calls': 1, 'operator_calls': 1, 'wall_seconds': 960,
           'output_bytes': 1048576, 'image_experiments': 0}
    return {'schema_version': 'dataevolver.production_feedback_review_admission.v1',
        'feedback_sha256': payload_sha256(value), 'max_planner_seconds': 900,
        'new_model_calls': 1, 'new_planner_calls': 1, 'new_candidate_executions': 0, 'new_image_calls': 0,
        'budget': {'schema_version': 'dataevolver.operator_development_budget.v1',
                   'limits': {k: prior[k]+inc[k] for k in prior}, 'prior_usage': dict(prior),
                   'resource_deadline_epoch': value['origin']['resource_deadline_epoch']}}


def test_bounded_view_preserves_counts_hashes_and_refusal_without_pixels():
    value = summary()
    assert value['unresolved']['reason_row_counts'] == {'boundary': 2}
    assert value['unresolved']['counts_may_overlap']
    assert value['unresolved']['rows_payload_sha256'] == payload_sha256(fixture_summary()[1]['output']['unresolved'])
    assert '"pixels"' not in json.dumps(value)
    assert not value['algorithm_defect_established'] and not value['production_qualification_passed']
    assert value['image_submission_allowed'] is False and value['candidate_adoption_allowed'] is False


@pytest.mark.parametrize('field,new', [('production_context_scope', 'synthetic_authority_contract_engineering'),
    ('registered_evidence_used', False), ('production_qualification_passed', True),
    ('image_request_submitted', True), ('new_accepted_records', 1), ('unsupported_submission_block_verified', False)])
def test_synthetic_qualification_and_submission_cannot_be_relabelled(field, new):
    r, e, p, o, q = fixture_summary(); r[field] = new
    with pytest.raises(ValueError): summarize_execution(r, e, p, origin=o, request=q)


@pytest.mark.parametrize('field,new', [('isolation_verified', False), ('status', 'STARTED'),
    ('return_code', 1), ('new_model_calls', 1), ('stdout', '{}')])
def test_uncommitted_or_contradictory_child_cannot_supply_feedback(field, new):
    r, e, p, o, q = fixture_summary(); e[field] = new
    with pytest.raises(ValueError): summarize_execution(r, e, p, origin=o, request=q)


def test_large_free_form_reasons_and_requests_are_rejected():
    r, e, p, o, q = fixture_summary()
    e['output']['unresolved'][0]['reason'] = 'x'*129; e['stdout'] = json.dumps(e['output'])
    with pytest.raises(ValueError, match='vocabulary'): summarize_execution(r, e, p, origin=o, request=q)
    r, e, p, o, q = fixture_summary(); q['request']['problem'] = 'x'*65536
    with pytest.raises(ValueError, match='64KiB'): summarize_execution(r, e, p, origin=o, request=q)


@pytest.mark.parametrize('mutation', ['deadline', 'prior', 'image', 'candidate', 'models', 'no_admission', 'extra_wall'])
def test_zero_headroom_requires_explicit_bounded_prior_accounted_review(tmp_path, mutation):
    auth = FixtureAuthority(tmp_path); a = admission(auth.value)
    if mutation == 'deadline': a['budget']['resource_deadline_epoch'] += 1
    if mutation == 'prior': a['budget']['prior_usage']['model_calls'] -= 1
    if mutation == 'image': a['new_image_calls'] = 1
    if mutation == 'candidate': a['new_candidate_executions'] = 1
    if mutation == 'models': a['budget']['limits']['model_calls'] += 1
    if mutation == 'no_admission': a.pop('budget')
    if mutation == 'extra_wall': a['budget']['limits']['wall_seconds'] += 1
    with pytest.raises((ValueError, KeyError)): ProductionFeedbackReviewDomain(authority=auth, admission=a)


def test_runtime_receives_only_actual_diagnostic_and_can_request_or_abstain(tmp_path):
    auth = FixtureAuthority(tmp_path); domain = ProductionFeedbackReviewDomain(authority=auth, admission=admission(auth.value))
    context = domain.context(None)
    assert context['allowed_actions'] == ['REQUEST_CAPABILITY', 'ABSTAIN']
    assert context['allowed_patches'] == context['allowed_observations'] == []
    d = decision(); d['evidence_ids'] = ['registered_production_diagnostic']
    result = domain.compile_capability_request(d, None)
    assert result['decision'] == d and not result['implementation_authorized_by_request']
    assert not result['candidate_adoption_allowed'] and result['production_effects'] == 0
    for action in ('RUN_EXPERIMENT', 'OBSERVE', 'COMPLETE'):
        d['action'] = action
        with pytest.raises(ValueError): validate_decision(d, context)
    with pytest.raises(ValueError): domain.compile_observation({}, None)
    with pytest.raises(ValueError): domain.compile_intervention({}, None)
    auth.value['candidate_status'] = 'supported'
    with pytest.raises(ValueError, match='feedback changed'): domain.context(None)


def test_registered_capture_recovers_without_child_and_rejects_drift(tmp_path):
    auth = FixtureAuthority(tmp_path)
    op = TrainingCandidateFeedbackOperator(authority=auth, output_root=tmp_path/'output')
    call = OperatorCall(campaign_id='fixture', call_id='fixture:feedback', idempotency_key='feedback',
        operator_id=op.spec.operator_id, operator_version='1', round_index=0,
        parameters={}, work_dir=str(tmp_path/'output/call'))
    receipt = op.execute(call)
    assert receipt.stage_result['metrics']['production_feedback_captured'] == 1
    assert op.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    assert op.recover(call).status is RecoveryStatus.RECOVERED
    auth.path.write_text('tampered')
    assert op.verify_effect(call, receipt).status is EvidenceStatus.CONTRADICTED
    assert op.recover(call).status is RecoveryStatus.AMBIGUOUS


def test_frozen_cached_authority_rehashes_captured_files(tmp_path):
    # A populated cache cannot bypass checks performed after the expensive
    # original frozen verifier. This tests persisted byte drift, not its reader.
    auth = object.__new__(FrozenTrainingCandidateFeedbackAuthority)
    auth.identity = lambda: {'fixture': True}
    r, e, p, o, q = fixture_summary(); path = tmp_path/'execution.json'; path.write_text('captured')
    auth._cached = {'report': r, 'execution': e, 'projection': p, 'origin': o, 'request': q,
                    'artifact_hashes': {str(path): file_sha256(path)}}
    value, _ = auth.read(); assert value['candidate_status'] == 'unsupported'
    path.write_text('changed')
    with pytest.raises(ValueError, match='artifact changed'): auth.read()


def test_feedback_cannot_create_new_files_inside_original_config(tmp_path):
    auth = FixtureAuthority(tmp_path)
    original = auth.read
    def read():
        value, proof = original()
        proof.update(config=str(tmp_path/'historical'), descriptor={
            'request_config': str(tmp_path/'request'), 'trial_config': str(tmp_path/'trial'),
            'observation_config': str(tmp_path/'observation'), 'stores': {'candidate': str(tmp_path/'candidate')}})
        return value, proof
    auth.read = read
    with pytest.raises(ValueError, match='overlap'):
        TrainingCandidateFeedbackOperator(authority=auth, output_root=tmp_path/'historical/new-file')
