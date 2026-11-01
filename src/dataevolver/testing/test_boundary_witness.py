"""Local findings cannot become pixel authority through a witness contract."""
import copy

import pytest
from jsonschema import Draft202012Validator

from dataevolver.dataset.contracts import ArtifactRef, payload_sha256
from dataevolver.harness.boundary_feedback import summarize_boundary_probe
from dataevolver.harness.boundary_witness import (
    WITNESS_SCHEMA, compile_boundary_witness, validate_boundary_witness)
from dataevolver.harness.adapters.boundary_witness import BoundaryWitnessOperator
from dataevolver.harness.production_feedback_request import FrozenProductionFeedbackRequestAuthority
from dataevolver.harness.models import OperatorCall, EvidenceStatus, RecoveryStatus
from dataevolver.testing.test_boundary_feedback import native_value, fixture_authority


def evidence():
    value = native_value()
    value['origin'].update({'report_sha256': '1'*64, 'frozen_contract_sha256': '2'*64,
                            'receipt_sha256': '3'*64})
    old = ArtifactRef('original', 'mask.png', '4'*64, 'image/png').to_dict()
    proposed = ArtifactRef('proposals', 'mask.png', '5'*64, 'image/png').to_dict()
    value['report']['mask_diagnostics'][0] = {
        'subject_id': 'one', 'target_instance_id': 'original-one', 'immutable_mask': old,
        'native_proposal_mask': proposed, 'proposal_unique_pixels': 10,
        'added_vs_immutable_pixels': 3, 'removed_vs_immutable_pixels': 2,
        'immutable_background_included_pixels': 3, 'all_immutable_background_was_preserved': False,
        'proposal_is_edit_authority': False, 'boundary_witness_qualification': 'NOT_ASSESSED',
        'aperture_truth_established_by_mask_comparison': False}
    feedback = summarize_boundary_probe(value)
    source = ArtifactRef('original', 'source.png', '6'*64, 'image/png').to_dict()
    context = {'source': source, 'media': [{'label': 'instance mask one', 'artifact': old},
        {'label': 'instance mask two', 'artifact': old}], 'original_segment_parameters': {'subjects': [
            {'subject_id': 'one', 'target_instance_id': 'original-one'},
            {'subject_id': 'two', 'target_instance_id': 'original-two'}]},
        'selected_subject_ids': ['one', 'two'], 'original_task_constraints': [{'preserve': 'visible gaps'}],
        'original_contract_sha256': '7'*64, 'original_snapshot': {'sha256': '8'*64},
        'resource_deadline_epoch': 1000}
    request = {'feedback_sha256': payload_sha256(feedback), 'request_sha256': '9'*64,
        'request': {'request': {'implementation_kind': 'artifact_contract'}},
        'frozen': True, 'replay': True, 'registered_effect_verified': True, 'native_and_receipt_verified': True,
        'artifact_hashes': {}}
    receipts = {feedback[k+'_request']['request_id']: feedback[k+'_receipt_sha256']
                for k in ('segmentation', 'visual')}
    return value, feedback, context, request, receipts


def test_original_rejection_uncertainty_and_mask_comparison_remain_distinct():
    _, feedback, context, request, receipts = evidence()
    before = copy.deepcopy((feedback, context, request))
    value = compile_boundary_witness(feedback, context, request_binding=request)
    Draft202012Validator(WITNESS_SCHEMA).validate(value)
    finding = value['findings'][0]
    assert finding['assessment']['visible_apertures'] == 'rejected'
    assert finding['assessment']['boundary_coverage'] == 'uncertain'
    assert finding['assessment_reference']['row_payload_sha256'] == payload_sha256(finding['assessment'])
    assert finding['comparison_reference']['json_pointer'] == '/mask_diagnostics/0'
    assert value['unassessed_subject_ids'] == ['two']
    result = validate_boundary_witness(value, feedback, context, request_binding=request, available_receipts=receipts)
    assert result['contract_conformant'] and result['status'] == 'INSUFFICIENT_EVIDENCE'
    assert result['observed_preservation_conflicts'] == ['one']
    assert value['pixel_evidence'] == dict.fromkeys(['object', 'background', 'uncertain'], 'NOT_ASSESSED')
    assert not result['production_qualification_passed'] and not result['candidate_adoption_allowed']
    assert not result['algorithm_defect_established'] and (feedback, context, request) == before


@pytest.mark.parametrize('mutation', ['source', 'identity', 'aperture', 'reason', 'scope', 'mask',
    'row_pointer', 'row_hash', 'receipt', 'truth', 'effect', 'qualification', 'deadline', 'pixel_labels', 'unknown'])
def test_tampering_identity_provenance_findings_or_authority_never_validates(mutation):
    _, f, c, r, receipts = evidence(); value = compile_boundary_witness(f, c, request_binding=r)
    row = value['findings'][0]
    if mutation == 'source': value['source']['sha256'] = 'a'*64
    if mutation == 'identity': value['immutable_correspondence'][0]['target_instance_id'] = 'different'
    if mutation == 'aperture': row['assessment']['visible_apertures'] = 'supported'
    if mutation == 'reason': row['assessment']['reason'] = 'fully solved'
    if mutation == 'scope': value['runtime_scope']['question'] = 'new scope'
    if mutation == 'mask': row['mask_comparison']['native_proposal_mask']['sha256'] = 'a'*64
    if mutation == 'row_pointer': row['assessment_reference']['json_pointer'] = '/different/0'
    if mutation == 'row_hash': row['assessment_reference']['row_payload_sha256'] = 'a'*64
    if mutation == 'receipt': row['comparison_reference']['receipt_sha256'] = 'a'*64
    if mutation == 'truth': value['mask_comparison_establishes_pixel_truth'] = True
    if mutation == 'effect': value['qualification']['effect_ownership'] = 'SUPPORTED'
    if mutation == 'qualification': value['qualification']['image_submission_allowed'] = True
    if mutation == 'deadline': value['provenance']['resource_deadline_epoch'] += 1
    if mutation == 'pixel_labels': value['pixel_evidence']['object'] = [[1, 2]]
    if mutation == 'unknown': value['grant_edit'] = True
    result = validate_boundary_witness(value, f, c, request_binding=r, available_receipts=receipts)
    assert not result['contract_conformant'] and result['status'] == 'INVALID_EVIDENCE'
    assert not result['image_submission_allowed'] and result['new_model_calls'] == 0


def test_missing_native_receipt_is_an_explicit_gap_without_an_algorithm_diagnosis():
    _, f, c, r, receipts = evidence(); v = compile_boundary_witness(f, c, request_binding=r)
    receipts.pop(f['visual_request']['request_id'])
    result = validate_boundary_witness(v, f, c, request_binding=r, available_receipts=receipts)
    assert result['status'] == 'INSUFFICIENT_EVIDENCE' and 'visual_receipt' in result['missing_evidence']
    assert not result['algorithm_defect_established']


@pytest.mark.parametrize('mutation', ['mapping', 'native', 'request', 'duplicate', 'mask_identity', 'deadline'])
def test_conflicting_trusted_inputs_cannot_compile(mutation):
    _, f, c, r, _ = evidence()
    if mutation == 'mapping': c['original_segment_parameters']['subjects'][0]['target_instance_id'] = 'wrong'
    if mutation == 'native': r['native_and_receipt_verified'] = False
    if mutation == 'request': r['feedback_sha256'] = 'a'*64
    if mutation == 'duplicate': f['runtime_visual_assessment']['assessments'] *= 2; r['feedback_sha256'] = payload_sha256(f)
    if mutation == 'mask_identity': c['media'][0]['artifact']['sha256'] = 'a'*64
    if mutation == 'deadline': c['resource_deadline_epoch'] += 1
    with pytest.raises(ValueError): compile_boundary_witness(f, c, request_binding=r)


def test_registered_cpu_capture_recovers_without_dispatch_and_detects_original_drift(tmp_path, monkeypatch):
    authority, original = fixture_authority(tmp_path, monkeypatch)
    raw, f, context, request, _ = evidence()
    authority._cached.update(raw); authority._cached['input_context'] = context
    auth = object.__new__(FrozenProductionFeedbackRequestAuthority)
    binding = {'config': str(tmp_path/'review'), 'reader_sha256': 'b'*64}
    monkeypatch.setattr(auth, 'identity', lambda: copy.deepcopy(binding))
    monkeypatch.setattr(auth, 'read', lambda: copy.deepcopy(request))
    with pytest.raises(ValueError, match='overlap'):
        BoundaryWitnessOperator(authority=authority, request_authority=auth, output_root=tmp_path/'review'/'new')
    op = BoundaryWitnessOperator(authority=authority, request_authority=auth, output_root=tmp_path/'output')
    call = OperatorCall('cpu', 'capture', 'key', op.spec.operator_id, '1', 0, str(op.output_root/'capture'), {})
    receipt = op.execute(call)
    assert op.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    assert op.recover(call).status is RecoveryStatus.RECOVERED
    assert not op._result()['image_submission_allowed']
    original.write_text('changed old native bytes')
    assert op.verify_effect(call, receipt).status is EvidenceStatus.CONTRADICTED


def test_witness_feedback_identity_is_replayable_despite_readback_timing_and_rejects_later_drift(tmp_path, monkeypatch):
    import json, sys
    from dataevolver.dataset.contracts import file_sha256
    from dataevolver.harness import boundary_witness_readback as module
    from dataevolver.harness.operator_development import DevelopmentBudget
    from dataevolver.harness.adapters.boundary_witness_feedback import BoundaryWitnessFeedbackOperator
    cfg=tmp_path/'capture';(cfg/'protected').mkdir(parents=True);(cfg/'output').mkdir()
    _, f, c, r, receipts = evidence(); witness=compile_boundary_witness(f,c,request_binding=r)
    validation=validate_boundary_witness(witness,f,c,request_binding=r,available_receipts=receipts)
    artifact=cfg/'output/witness.json';artifact.write_text(json.dumps(witness))
    zero=dict.fromkeys(['model_calls','planner_calls','operator_calls','wall_seconds','output_bytes','image_experiments'],0)
    usage={**zero,'operator_calls':1,'wall_seconds':1,'output_bytes':artifact.stat().st_size}
    budget=DevelopmentBudget(usage,zero,1000)
    (cfg/'protected/admission.json').write_text(json.dumps({'budget':budget.to_dict()}))
    (cfg/'protected/feedback.json').write_text(json.dumps(f))
    (cfg/'validation.json').write_text(json.dumps({'artifact_file_sha256':file_sha256(artifact),
        'witness_sha256':payload_sha256(witness),'source_commit':'a'*40,'lifecycle_usage':usage,'lifecycle_total':usage}))
    clock=iter([10,20])
    def verified(*args,**kwargs):
        return {'artifact_sha256':file_sha256(artifact),'witness_sha256':payload_sha256(witness),
            'source_commit':'a'*40,'witness':witness,'validation':validation,
            'artifact_hashes':{str(artifact):file_sha256(artifact)},'input_store_roots':{'capture':str(cfg)},
            'readback_seconds':next(clock),'new_model_calls':0}
    monkeypatch.setattr(module,'verify_frozen_boundary_witness',verified)
    a=module.FrozenBoundaryWitnessFeedbackAuthority(config=cfg,python=sys.executable)
    b=module.FrozenBoundaryWitnessFeedbackAuthority(config=cfg,python=sys.executable)
    first=a.read();assert first==b.read() and first[0]['lifecycle_total']==usage
    op=BoundaryWitnessFeedbackOperator(authority=a,output_root=tmp_path/'next-output')
    call=OperatorCall('review','capture','key',op.spec.operator_id,'1',0,str(op.output_root/'capture'),{})
    receipt=op.execute(call);assert op.verify_effect(call,receipt).status is EvidenceStatus.VERIFIED
    assert op.recover(call).status is RecoveryStatus.RECOVERED
    artifact.write_text('drift')
    with pytest.raises(ValueError,match='changed'):a.read()
    assert op.verify_effect(call,receipt).status is EvidenceStatus.CONTRADICTED
