"""Typed insufficiency and declared-set conflicts preserve all authority gates."""

import pytest

from dataevolver.dataset.contracts import payload_sha256
from dataevolver.harness.training_candidate_feedback import _collect_diagnostic_records
from dataevolver.harness.evidence_sufficiency import compile_sufficiency_receipt, validate_evidence_declarations
from dataevolver.harness.adapters.evidence_sufficiency import EvidenceSufficiencyOperator
from dataevolver.testing.test_training_candidate_feedback import fixture_summary, summary


def fixture():
    feedback = summary(); execution = fixture_summary()[1]
    payload = {'evidence': {'instances': [{'required_boundary': [[1, 2], [1, 2], [3, 4]],
        'effects': [{'pixels': [[5, 6]]}]}], 'protected': {'pixels': [[1, 2]]}}}
    records = _collect_diagnostic_records(execution, payload, execution_sha256='1'*64,
        input_ref={'store': 'fixture', 'path': 'input.json', 'sha256': '2'*64})
    registry = {'schema_version': 'dataevolver.evidence_requirement_registry.v1', 'entries': [
        {'reason': 'boundary', 'evidence_type': 'registered_boundary_witness',
         'provenance_rules': ['exact committed source/candidate/mask/reference and witness bindings'],
         'declaration_status': 'DECLARED', 'implementation_qualification': 'NOT_ASSESSED',
         'grants_authority': False}]}
    request = {'feedback_sha256': payload_sha256(feedback),
        'request': {'request': {'implementation_kind': 'artifact_contract'}}, 'request_sha256': '3'*64,
        'frozen': True, 'replay': True, 'registered_effect_verified': True, 'native_and_receipt_verified': True}
    return feedback, records, registry, request


def test_cpu_conflict_is_exact_for_supplied_sets_without_inferring_task_infeasibility():
    feedback, records, registry, request = fixture()
    finding = records['cpu_findings'][0]
    assert finding['declared_coverage_unique_pixels'] == 3
    assert finding['intersection_unique_pixels'] == 1
    receipt = compile_sufficiency_receipt(feedback, records, registry, request_binding=request)
    assert receipt['status'] == 'conflicting' and receipt['task_feasibility'] == 'indeterminate'
    assert not receipt['actual_task_infeasibility_established']
    assert len(receipt['evidence_gaps']) == 2 and receipt['row_counts_may_overlap']
    assert not receipt['image_submission_allowed'] and not receipt['candidate_adoption_allowed']
    assert not receipt['positive_production_behavior_verified'] and not receipt['new_permissions']
    assert receipt['captured_conflicts'][0]['declaration_missing']


def test_original_rows_have_hash_bound_exact_pointers_and_no_emitted_geometry():
    _, records, _, _ = fixture()
    assert records['records'][0]['reference']['json_pointer'] == '/output/unresolved/0'
    assert records['records'][1]['reference']['json_pointer'] == '/output/unresolved/1'
    assert records['records'][0]['reference']['row_payload_sha256'] == payload_sha256(fixture_summary()[1]['output']['unresolved'][0])
    assert 'pixels' not in records['records'][0]
    assert not records['geometry_emitted'] and not records['new_permissions']


@pytest.mark.parametrize('mutation', ['source', 'counts', 'pointer', 'section', 'request', 'kind', 'native', 'feasibility', 'pixel_count'])
def test_stale_rows_requests_or_unjustified_feasibility_are_rejected(mutation):
    f, records, reg, request = fixture()
    if mutation == 'source': records['candidate_output_payload_sha256'] = '4'*64
    if mutation == 'counts': records['records'].pop(0)
    if mutation == 'pointer': records['records'][0]['reference']['json_pointer'] = '/output/conflicts/0'
    if mutation == 'section': records['records'][0]['section'] = 'inferred'
    if mutation == 'request': request['feedback_sha256'] = '4'*64
    if mutation == 'kind': request['request']['request']['implementation_kind'] = 'registered_operator'
    if mutation == 'native': request['native_and_receipt_verified'] = False
    if mutation == 'feasibility': records['cpu_findings'][0]['task_feasibility'] = 'impossible'
    if mutation == 'pixel_count': records['cpu_findings'][0]['intersection_unique_pixels'] = 20
    with pytest.raises(ValueError): compile_sufficiency_receipt(f, records, reg, request_binding=request)


@pytest.mark.parametrize('key,new', [('grants_authority', True), ('declaration_status', 'WITNESS'),
    ('implementation_qualification', 'QUALIFIED'), ('provenance_rules', [])])
def test_declarations_cannot_masquerade_as_verified_witnesses(key, new):
    _, _, registry, _ = fixture(); registry['entries'][0][key] = new
    with pytest.raises(ValueError): validate_evidence_declarations(registry)


def test_missing_declarations_are_kept_and_never_used_to_grant_permissions():
    f, records, registry, request = fixture(); registry['entries'] = []
    records['cpu_findings'][0]['intersection_unique_pixels'] = 0
    # With captured conflicts still present the receipt reports conflicting
    # evidence, not that the real task itself is unsatisfiable.
    receipt = compile_sufficiency_receipt(f, records, registry, request_binding=request)
    assert all(row['declaration_missing'] for row in receipt['evidence_gaps'])
    assert all(not row['witness_satisfied'] for row in receipt['evidence_gaps'])
    assert not receipt['algorithm_defect_established']


def test_a_package_or_fixture_authority_cannot_authorize_the_production_operator(tmp_path):
    with pytest.raises(ValueError, match='registered frozen'):
        EvidenceSufficiencyOperator(authority=object(), request_authority=object(), declarations={}, output_root=tmp_path)
