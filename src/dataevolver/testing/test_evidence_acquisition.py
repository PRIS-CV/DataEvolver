"""Provider import fixtures: no actual provider, pixel truth or method benefit."""
import copy
import json

import pytest

from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256
from dataevolver.harness.boundary_qualification import RegisteredBoundaryQualificationEvidence
from dataevolver.harness.evidence_acquisition import (
    RegisteredIndependentEvidenceProvider, _select_provider_claims)
from dataevolver.testing.test_boundary_qualification import fixture


def provenance_fixture():
    witness, consumed = fixture(); entry = consumed['records'][0]
    audit = {'schema_version': 'dataevolver.independent_evidence_provider.v1',
        'provider_id': 'fixture-external-provider', 'trust_profile_sha256': 'd'*64,
        'execution_contract_sha256': 'e'*64,
        **{k: entry['record'][k] for k in ('source', 'witness_sha256', 'original_contract_sha256',
            'original_constraints_sha256', 'immutable_correspondence_sha256')},
        'record': entry['artifact'], 'host_receipt_sha256': entry['host_receipt_sha256'],
        'claims': [{'claim_sha256': payload_sha256(row), 'independence': 'independent',
            'authority_scope': 'authorized', 'reason': 'Fixture host independently assessed this scope.',
            'production_receipts': [ArtifactRef('host', 'independent-job.json', 'f'*64).to_dict()],
            'assessment_dependencies': []} for row in entry['record']['claims']]}
    return witness, entry, audit


def select(witness, entry, audit):
    return _select_provider_claims(witness, entry, audit,
        execution_sha256='e'*64, trust_profile_sha256='d'*64)


def test_complete_provenance_fixture_preserves_scope_and_uncertainty():
    witness, entry, audit = provenance_fixture(); before = copy.deepcopy((witness, entry, audit))
    rows, gaps, receipts = select(witness, entry, audit)
    assert rows == entry['record']['claims'] and not gaps and receipts
    assert before == (witness, entry, audit)


@pytest.mark.parametrize('key', ['source', 'witness_sha256', 'original_contract_sha256',
    'original_constraints_sha256', 'immutable_correspondence_sha256', 'record',
    'host_receipt_sha256', 'execution_contract_sha256', 'trust_profile_sha256'])
def test_provider_cannot_relabel_scope_or_external_trust(key):
    witness, entry, audit = provenance_fixture()
    audit[key] = {**audit[key], 'sha256': '0'*64} if isinstance(audit[key], dict) else '0'*64
    with pytest.raises(ValueError, match='provenance changed'):
        select(witness, entry, audit)


@pytest.mark.parametrize('kind', ['dependent', 'unassessed', 'prior_mask', 'prior_assessment', 'receipt', 'source_receipt', 'authority'])
def test_missing_authority_or_dependent_assessment_cannot_supply_claims(kind):
    witness, entry, audit = provenance_fixture(); row = audit['claims'][0]
    if kind in ('dependent', 'unassessed'): row['independence'] = kind
    if kind == 'prior_mask': row['assessment_dependencies'] = [witness['immutable_correspondence'][0]['immutable_mask']['sha256']]
    if kind == 'prior_assessment': row['assessment_dependencies'] = [witness['findings'][0]['assessment_reference']['row_payload_sha256']]
    if kind == 'receipt': row['production_receipts'] = [witness['findings'][0]['mask_comparison']['native_proposal_mask']]
    if kind == 'source_receipt': row['production_receipts'] = [witness['source']]
    if kind == 'authority': row['authority_scope'] = 'unavailable'
    rows, gaps, _ = select(witness, entry, audit)
    assert len(rows) == len(entry['record']['claims'])-1 and len(gaps) == 1


@pytest.mark.parametrize('kind', ['duplicate', 'omit', 'replace', 'unknown'])
def test_provider_claim_inventory_is_closed(kind):
    witness, entry, audit = provenance_fixture()
    if kind == 'duplicate': audit['claims'].append(copy.deepcopy(audit['claims'][0]))
    if kind == 'omit': audit['claims'].pop()
    if kind == 'replace': audit['claims'][0]['claim_sha256'] = '0'*64
    if kind == 'unknown': audit['independent'] = True
    with pytest.raises(ValueError): select(witness, entry, audit)


def test_no_authorized_provider_returns_all_claim_gaps_without_dispatch(monkeypatch):
    from dataevolver.adapters.model_gateway import ModelGateway
    monkeypatch.setattr(ModelGateway, 'invoke', lambda *a, **kw: pytest.fail('cannot dispatch'))
    witness, _ = fixture()
    provider = RegisteredIndependentEvidenceProvider(evidence_authority=RegisteredBoundaryQualificationEvidence())
    result = provider.acquire(witness)
    assert result['provider_availability'] == 'UNAVAILABLE' and result['accepted_evidence_records'] == 0
    assert result['qualification']['status'] == 'INSUFFICIENT_EVIDENCE'
    assert len(result['qualification']['findings']) == len(witness['immutable_correspondence'])*4
    assert all('authorized_independent_evidence_provider' in r['missing_prerequisites']
               for r in result['qualification']['findings'])
    assert result['qualification']['prior_observations'] == witness['findings']
    assert not result['new_model_calls'] and not result['new_probe_executions']
    denied = provider.acquire(witness, import_allowed=False)
    assert all('evidence_import_permission' in r['missing_prerequisites'] for r in denied['qualification']['findings'])
    with pytest.raises(ValueError): RegisteredIndependentEvidenceProvider(evidence_authority={})


def qualified_fixture(tmp_path, *, mode=None):
    from dataevolver.testing.test_production_qualification import setup, stage_controller
    from dataevolver.testing.test_operator_development import write_ref
    f, execution, parent, declarations, _ = setup(tmp_path)
    witness, consumed = fixture(); store = f[4]
    source = execution.design_protocol['sources'][0]['artifact']; witness['source'] = source
    record = consumed['records'][0]['record']; record['source'] = source
    record['witness_sha256'] = payload_sha256(witness)
    from dataevolver.harness.boundary_qualification import required_scope_sha256
    subjects = {s['subject_id']: s for s in witness['immutable_correspondence']}
    extra = []
    for row in record['claims']:
        ref = write_ref(store.roots['protocol'], 'protocol', row['subject_id']+'/'+row['claim']+'.json',
                        {'fixture': 'independent assessment', 'subject': row['subject_id'], 'claim': row['claim']})
        row['evidence'] = [ref]; row['scope_sha256'] = required_scope_sha256(witness, subjects[row['subject_id']])
        extra.append(ref)
    if mode == 'effect':
        record['claims'][-1]['evidence'] = record['claims'][0]['evidence']
    if mode == 'prior_conflict':
        for row in record['claims']: row['resolves_prior_assessments'] = []
    record_ref = write_ref(store.roots['protocol'], 'protocol', 'provider/claims.json', record); extra.append(record_ref)
    entry = {**consumed['records'][0], 'record': record, 'artifact': record_ref}
    _, _, audit = provenance_fixture()
    audit.update({k: record[k] for k in ('source', 'witness_sha256', 'original_contract_sha256',
        'original_constraints_sha256', 'immutable_correspondence_sha256')})
    audit.update(record=record_ref, execution_contract_sha256=execution.compute_hash(),
                 trust_profile_sha256=execution.runtime['trust_profile_sha256'])
    receipt = write_ref(store.roots['protocol'], 'protocol', 'provider/job.json', {'fixture': 'new independent job'})
    extra.append(receipt)
    audit['claims'] = [{**row, 'claim_sha256': payload_sha256(claim), 'production_receipts': [receipt]}
                      for row, claim in zip(audit['claims'], record['claims'])]
    if mode == 'dependent': audit['claims'][0]['independence'] = 'dependent'
    audit_ref = write_ref(store.roots['protocol'], 'protocol', 'provider/audit.json', audit); extra.append(audit_ref)
    authorities = {}
    for stage in execution.stages:
        def modify(abi):
            sources = [s for s in execution.design_protocol['sources'] if s['family_id'] in execution.execution_protocol().families(stage)]
            for role, ref in (('quality_checks', record_ref), ('native_audit', audit_ref)):
                manifest = {'role': role, 'cases': [{'family_id': s['family_id'], 'artifact': ref,
                    'receipt_sha256': entry['host_receipt_sha256']} for s in sources]}
                path = store.resolve(ArtifactRef.from_dict(abi['artifacts'][role]))
                path.write_text(json.dumps(manifest)); abi['artifacts'][role]['sha256'] = file_sha256(path)
        authorities[stage] = stage_controller(tmp_path, f, execution, parent, declarations, stage,
            abi_change=modify, extra_refs=extra if mode != 'missing_closure' else ())
    return witness, RegisteredIndependentEvidenceProvider(evidence_authority=
        RegisteredBoundaryQualificationEvidence(execution=execution, authorities=authorities))


def test_real_fixture_controllers_import_provenance_and_use_original_qualifier(tmp_path):
    witness, provider = qualified_fixture(tmp_path)
    before = {s: a.base.controller.layout.events_path.read_bytes()
              for s, a in provider.evidence_authority.authorities.items()}
    result = provider.acquire(witness)
    assert result['provider_availability'] == 'AVAILABLE' and result['accepted_evidence_records'] == 1
    assert result['qualification']['status'] == 'SUPPORTED'
    assert all(not result[k] for k in ('image_submission_allowed', 'candidate_adoption_allowed', 'production_qualification_passed'))
    assert before == {s: a.base.controller.layout.events_path.read_bytes()
                      for s, a in provider.evidence_authority.authorities.items()}


@pytest.mark.parametrize('mode', ['missing_closure', 'effect', 'dependent', 'prior_conflict'])
def test_real_fixture_authorities_do_not_bypass_receipt_independence_effect_or_prior_conflicts(tmp_path, mode):
    witness, provider = qualified_fixture(tmp_path, mode=mode)
    if mode in ('missing_closure', 'effect'):
        with pytest.raises(ValueError): provider.acquire(witness)
    else:
        value = provider.acquire(witness)
        assert value['qualification']['status'] == 'INSUFFICIENT_EVIDENCE'
        assert value['qualification']['prior_observations'] == witness['findings']


def test_registered_acquisition_binds_actual_request_and_recovers_without_dispatch(tmp_path, monkeypatch):
    from dataevolver.harness.boundary_qualification_readback import FrozenBoundaryQualificationFeedbackAuthority
    from dataevolver.harness.boundary_witness_readback import FrozenBoundaryWitnessFeedbackAuthority
    from dataevolver.harness.production_feedback_request import FrozenProductionFeedbackRequestAuthority
    from dataevolver.harness.adapters.evidence_acquisition import IndependentEvidenceAcquisitionOperator
    from dataevolver.harness.boundary_qualification import qualify_boundary_witness
    from dataevolver.harness.models import OperatorCall, EvidenceStatus, RecoveryStatus
    old = tmp_path/'old'; old.mkdir(); raw = old/'original.json'; raw.write_text('original')
    witness, _ = fixture(); provider = RegisteredIndependentEvidenceProvider(
        evidence_authority=RegisteredBoundaryQualificationEvidence())
    summary = {'qualification': qualify_boundary_witness(witness, provider.evidence_authority.read(witness))}
    proof = {'artifact_hashes': {str(raw): file_sha256(raw)}, 'input_store_roots': {'old': str(old)}}
    authority = object.__new__(FrozenBoundaryQualificationFeedbackAuthority); authority.config = old
    wa = object.__new__(FrozenBoundaryWitnessFeedbackAuthority); wa.config = old
    ra = object.__new__(FrozenProductionFeedbackRequestAuthority)
    request = {'feedback_sha256': payload_sha256(summary), 'request_sha256': 'a'*64,
        'request': {'request': {'implementation_kind': 'registered_operator'}},
        'frozen': True, 'replay': True, 'registered_effect_verified': True,
        'native_and_receipt_verified': True, 'artifact_hashes': {}}
    for auth, value in ((authority, (summary, proof)), (wa, ({'witness': witness}, proof))):
        monkeypatch.setattr(auth, 'identity', lambda: {'config': str(old)})
        monkeypatch.setattr(auth, 'read', lambda v=value: copy.deepcopy(v))
    monkeypatch.setattr(ra, 'identity', lambda: {'config': str(old)})
    monkeypatch.setattr(ra, 'read', lambda: copy.deepcopy(request))
    with pytest.raises(ValueError, match='overlap'):
        IndependentEvidenceAcquisitionOperator(authority=authority, witness_authority=wa,
            request_authority=ra, provider=provider, output_root=old/'bad')
    op = IndependentEvidenceAcquisitionOperator(authority=authority, witness_authority=wa,
        request_authority=ra, provider=provider, output_root=tmp_path/'output')
    call = OperatorCall('cpu', 'import', 'key', op.spec.operator_id, '1', 0, str(op.output_root/'import'), {})
    receipt = op.execute(call)
    assert op.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    assert op.recover(call).status is RecoveryStatus.RECOVERED
    assert op._result()['qualification']['status'] == 'INSUFFICIENT_EVIDENCE'
    request['request_sha256'] = 'b'*64
    assert op.verify_effect(call, receipt).status is EvidenceStatus.CONTRADICTED
