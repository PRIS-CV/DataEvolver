"""Runtime query/proposal boundaries and interrupted-effect no-resend safety."""
import copy
import json
from pathlib import Path

import pytest
from PIL import Image
from jsonschema.exceptions import ValidationError

from dataevolver.dataset.contracts import ArtifactStore, ArtifactRef, file_sha256, payload_sha256
from dataevolver.harness.boundary_evidence import compile_boundary_query, boundary_mask_diagnostics, BoundaryProbeDomain
from dataevolver.harness.adapters.boundary_evidence import BoundaryEvidenceProbeOperator
from dataevolver.harness.models import OperatorSpec, OperatorCall, EvidenceStatus, RecoveryStatus
from dataevolver.harness.state_store import atomic_write_json


def fixture_inputs():
    subject = {'subject_id': 'one', 'target_instance_id': 'observed-one', 'description': 'original observed item',
               'box_xyxy_normalized': [.1, .2, .6, .8]}
    params = {'subjects': [subject], 'threshold': .5, 'mask_threshold': .5, 'min_box_iou': .25}
    return {'selected_subject_ids': ['one'], 'original_segment_parameters': params,
        'original_segment_record': {'request': {'seed': 123}},
        'contract': {'production': {'recipe': {'model_timeout_seconds': 2400}, 'resource_deadline_epoch': 1000}},
        'trial': {'source': ArtifactRef('fixture', 'source.png', '1'*64, 'image/png').to_dict(),
                  'candidate': ArtifactRef('fixture', 'candidate.png', '2'*64, 'image/png').to_dict(), 'media': []},
        'feedback_sha256': '3'*64}


def query():
    return {'question': 'Does the native proposal preserve observed boundary gaps?', 'subjects': [
        {'subject_id': 'one', 'description': 'runtime observed item boundary',
         'box_xyxy_normalized': [.12, .2, .61, .79], 'reason': 'fresh registered visual evidence'}]}


def test_only_runtime_geometry_changes_and_frozen_thresholds_are_retained():
    inputs = fixture_inputs(); before = copy.deepcopy(inputs)
    result = compile_boundary_query(query(), inputs)
    assert result['subjects'][0]['target_instance_id'] == 'observed-one'
    assert result['subjects'][0]['box_xyxy_normalized'] == query()['subjects'][0]['box_xyxy_normalized']
    assert result['threshold'] == .5 and result['mask_threshold'] == .5 and result['min_box_iou'] == .25
    assert inputs == before


def test_identical_completed_parameters_are_rejected_even_with_a_new_question():
    inputs = fixture_inputs(); q = query(); old = inputs['original_segment_parameters']['subjects'][0]
    q['subjects'][0].update(description=old['description'], box_xyxy_normalized=old['box_xyxy_normalized'])
    with pytest.raises(ValueError, match='identical completed'): compile_boundary_query(q, inputs)


@pytest.mark.parametrize('change', ['unselected', 'duplicate', 'nan', 'reversed', 'threshold', 'identity'])
def test_queries_cannot_change_selection_thresholds_or_mapping(change):
    q = query()
    if change == 'unselected': q['subjects'][0]['subject_id'] = 'other'
    if change == 'duplicate': q['subjects'].append(copy.deepcopy(q['subjects'][0]))
    if change == 'nan': q['subjects'][0]['box_xyxy_normalized'][0] = float('nan')
    if change == 'reversed': q['subjects'][0]['box_xyxy_normalized'] = [.7, .2, .3, .8]
    if change == 'threshold': q['threshold'] = .1
    if change == 'identity': q['subjects'][0]['target_instance_id'] = 'other'
    with pytest.raises((ValueError, ValidationError)): compile_boundary_query(q, fixture_inputs())


def test_native_hole_or_boundary_difference_is_recorded_without_granting_authority(tmp_path):
    store = ArtifactStore({'fixture': tmp_path})
    def image(name, hole):
        im = Image.new('L', (8, 8), 255)
        if hole: im.putpixel((3, 3), 0)
        p = tmp_path/name; im.save(p)
        return ArtifactRef('fixture', name, file_sha256(p), 'image/png').to_dict()
    old, native = image('immutable.png', True), image('native.png', False)
    inputs = fixture_inputs(); inputs['trial']['media'] = [{'label': 'instance mask one', 'artifact': old}]
    result = {'instances': [{'subject_id': 'one', 'target_instance_id': 'observed-one', 'mask': native}]}
    rows = boundary_mask_diagnostics(inputs, query(), result, store)
    assert rows[0]['added_vs_immutable_pixels'] == 1
    assert rows[0]['immutable_background_included_pixels'] == 1
    assert not rows[0]['proposal_is_edit_authority']
    assert not rows[0]['aperture_truth_established_by_mask_comparison']
    assert rows[0]['boundary_witness_qualification'] == 'NOT_ASSESSED'
    result['instances'][0]['target_instance_id'] = 'wrong'
    with pytest.raises(ValueError, match='identity'): boundary_mask_diagnostics(inputs, query(), result, store)


def fixture_operator(tmp_path):
    op = object.__new__(BoundaryEvidenceProbeOperator)
    op.output_root = tmp_path/'out'
    op.spec = OperatorSpec('data.probe_instance_boundaries', '1', allowed_write_roots=(str(op.output_root),))
    from dataevolver.harness.operator_contracts import OperatorContract
    from dataevolver.harness.boundary_evidence import QUERY_SCHEMA
    op.operator_contract = OperatorContract(op.spec.operator_id, '1', op.spec.sha256(), 'basic', {},
        {'proposal': {'type': 'object'}}, QUERY_SCHEMA, ('image', 'json'), 'model', 'verify_then_reuse', ('fixture only',))
    op.inputs = fixture_inputs(); op.origin = {'fixture': True}
    class Authority:
        def identity(self): return {'fixture': True}
    op.authority = Authority()
    op._binding = lambda call: {'call_sha256': payload_sha256(call.to_dict()),
        'operator_spec_sha256': op.spec.sha256(), 'origin_sha256': payload_sha256(op.origin)}
    call = OperatorCall('fixture', 'fixture:p000:observe', 'fixture', op.spec.operator_id, '1', 0,
                        str(op.output_root/'observations/p000'), query())
    return op, call


def test_missing_native_stage_after_an_intent_is_ambiguous_and_never_resubmitted(tmp_path):
    op, call = fixture_operator(tmp_path); submits = []
    path = op._path(call)
    atomic_write_json(path.parent/'probe-intent.json', {'binding': op._binding(call), 'at': 'fixture'})
    def finish(call, *, submit):
        submits.append(submit)
        raise ValueError('no committed native receipt')
    op._finish = finish
    assert op.recover(call).status is RecoveryStatus.AMBIGUOUS
    assert submits == [False]
    with pytest.raises(ValueError, match='never resend'): op.execute(call)
    assert submits == [False, False]


def test_original_request_is_hash_bound_and_never_accepts_developer_parameters_outside_schema(tmp_path):
    op, call = fixture_operator(tmp_path)
    request = op._segment_request(call)
    assert request.artifacts[0].to_dict() == op.inputs['trial']['source']
    assert request.seed == 123 and request.max_attempts == 1 and request.timeout_seconds == 900
    assert json.loads(request.messages[0]['content']) == compile_boundary_query(call.parameters, op.inputs)
    assert op.recover(call).status is RecoveryStatus.RETRY_SAFE
    path = Path(call.work_dir); path.mkdir(parents=True); (path/'boundary-probe.json').symlink_to(tmp_path/'outside')
    with pytest.raises(ValueError, match='symlink|escaped'): op.recover(call)


def test_readback_never_asks_gateway_to_materialize_a_missing_receipt(tmp_path):
    class Gateway:
        root = tmp_path
        def recover(self, request): raise AssertionError('recovery would be a new effect')
        def _recover(self, *args): raise AssertionError('no receipt to read')
    op, call = fixture_operator(tmp_path)
    assert op._existing(Gateway(), op._segment_request(call)) is None
