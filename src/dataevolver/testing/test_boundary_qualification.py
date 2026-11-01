"""Independent qualification semantics; positives here are engineering fixtures."""
import copy

import pytest

from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256
from dataevolver.harness.boundary_qualification import (
    CLAIMS, RegisteredBoundaryQualificationEvidence, qualify_boundary_witness, required_scope_sha256)
from dataevolver.harness.boundary_witness import compile_boundary_witness, validate_boundary_witness
from dataevolver.harness.boundary_witness_readback import FrozenBoundaryWitnessFeedbackAuthority
from dataevolver.harness.production_feedback_request import FrozenProductionFeedbackRequestAuthority
from dataevolver.harness.adapters.boundary_qualification import BoundaryOperationalQualificationOperator
from dataevolver.harness.models import OperatorCall, EvidenceStatus, RecoveryStatus
from dataevolver.testing.test_boundary_witness import evidence


def fixture():
    _, f, context, request, _ = evidence()
    witness = compile_boundary_witness(f, context, request_binding=request)
    chain = {'production_engineering_qualification': 'VERIFIED',
        'production_independent_qualification': 'VERIFIED', 'production_adoption_allowed': False,
        'qualification_evidence': {'engineering': {'campaign_id': 'engineering'},
                                 'independent': {'campaign_id': 'independent'}}}
    records = {'schema_version': 'dataevolver.boundary_operational_evidence.v1', 'source': witness['source'],
        'witness_sha256': payload_sha256(witness),
        'original_contract_sha256': witness['provenance']['original_contract_sha256'],
        'original_constraints_sha256': witness['provenance']['original_constraints_sha256'],
        'immutable_correspondence_sha256': payload_sha256(witness['immutable_correspondence']), 'claims': []}
    old = {r['assessment']['subject_id']: r['assessment_reference']['row_payload_sha256'] for r in witness['findings']}
    for subject in witness['immutable_correspondence']:
        for kind in CLAIMS:
            records['claims'].append({'subject_id': subject['subject_id'], 'target_instance_id': subject['target_instance_id'],
                'claim': kind, 'finding': 'supported', 'scope': 'full_required_scope',
                'scope_sha256': required_scope_sha256(witness, subject),
                'basis': 'independent_effect_evidence' if kind == 'effect_ownership' else 'independent_pixel_evidence',
                'reason': 'Independent fixture evidence for the complete frozen scope.',
                'evidence': [ArtifactRef('independent', kind+'.json', 'a'*64).to_dict()],
                'resolves_prior_assessments': [old[subject['subject_id']]] if subject['subject_id'] in old else []})
    consumed = {'qualified_chain': chain, 'records': [{'record': records, 'stage': 'engineering',
        'family_id': 'regression', 'artifact': ArtifactRef('host', 'claims.json', 'b'*64).to_dict(),
        'host_receipt_sha256': 'c'*64}], 'artifact_hashes': {}, 'missing_prerequisites': [],
        'new_model_calls': 0, 'new_operator_executions': 0}
    return witness, consumed


def test_actual_local_diagnostic_without_additional_authority_remains_insufficient(monkeypatch):
    from dataevolver.adapters.model_gateway import ModelGateway
    def forbidden(*a, **kw): raise AssertionError('qualification must not infer or probe')
    monkeypatch.setattr(ModelGateway, 'invoke', forbidden)
    witness, _ = fixture(); before = copy.deepcopy(witness)
    holder = RegisteredBoundaryQualificationEvidence()
    consumed = holder.read(witness); result = qualify_boundary_witness(witness, consumed)
    assert result['status'] == 'INSUFFICIENT_EVIDENCE'
    assert all(r['finding'] == 'unresolved' and 'independent_production_qualification_authority'
               in r['missing_prerequisites'] for r in result['findings'])
    assert all(r['original_observation_status'] == 'NOT_ASSESSED' for r in result['findings'] if r['subject_id'] == 'two')
    assert result['prior_observations'] == witness['findings'] and witness == before
    assert not result['image_submission_allowed'] and not result['candidate_adoption_allowed']
    assert not result['algorithm_defect_established'] and not result['production_qualification_passed']


def test_complete_qualified_fixture_can_support_scoped_evidence_without_task_or_edit_permission():
    witness, consumed = fixture(); before=copy.deepcopy((witness,consumed))
    result=qualify_boundary_witness(witness,consumed)
    assert result['status']=='SUPPORTED' and all(r['finding']=='supported' for r in result['findings'])
    assert all(not r['missing_prerequisites'] for r in result['findings'])
    assert result['prior_observations']==witness['findings']
    assert not result['full_task_completion'] and not result['production_qualification_passed']
    assert not result['candidate_adoption_allowed'] and (witness,consumed)==before


@pytest.mark.parametrize('mutation',['source','witness','constraints','correspondence','target','scope','basis','immutable','native','raw_source','chain','same_campaign'])
def test_wrong_identity_or_unqualified_derived_evidence_cannot_qualify(mutation):
    witness, consumed=fixture();record=consumed['records'][0]['record'];row=record['claims'][0]
    if mutation=='source':record['source']={**record['source'],'sha256':'d'*64}
    if mutation=='witness':record['witness_sha256']='d'*64
    if mutation=='constraints':record['original_constraints_sha256']='d'*64
    if mutation=='correspondence':record['immutable_correspondence_sha256']='d'*64
    if mutation=='target':row['target_instance_id']='different'
    if mutation=='scope':row['scope_sha256']='d'*64
    if mutation=='basis':row['basis']='independent_effect_evidence'
    if mutation=='immutable':row['evidence']=[witness['immutable_correspondence'][0]['immutable_mask']]
    if mutation=='native':row['evidence']=[witness['findings'][0]['mask_comparison']['native_proposal_mask']]
    if mutation=='raw_source':row['evidence']=[witness['source']]
    if mutation=='chain':consumed['qualified_chain']=None
    if mutation=='same_campaign':consumed['qualified_chain']['qualification_evidence']['independent']['campaign_id']='engineering'
    with pytest.raises(ValueError):qualify_boundary_witness(witness,consumed)


def test_uncertain_and_rejected_observations_need_explicit_independent_resolution():
    witness, consumed=fixture()
    for row in consumed['records'][0]['record']['claims']:row['resolves_prior_assessments']=[]
    result=qualify_boundary_witness(witness,consumed)
    unresolved={r['claim'] for r in result['findings'] if r['subject_id']=='one' and r['finding']=='unresolved'}
    assert unresolved=={'boundary_coverage','visible_apertures','background_preservation'}
    assert result['status']=='INSUFFICIENT_EVIDENCE'
    assert result['prior_observations'][0]['assessment']['visible_apertures']=='rejected'


def test_missing_partial_and_conflicting_independent_findings_remain_unresolved():
    witness, consumed=fixture();claims=consumed['records'][0]['record']['claims']
    claims[0]['scope']='partial';claims.pop()
    conflicting=copy.deepcopy(claims[1]);conflicting['finding']='rejected';claims.append(conflicting)
    result=qualify_boundary_witness(witness,consumed)
    assert result['status']=='INSUFFICIENT_EVIDENCE'
    by_key={(r['subject_id'],r['claim']):r for r in result['findings']}
    assert 'complete_required_scope' in by_key[('one','boundary_coverage')]['missing_prerequisites']
    assert 'conflicting_independent_evidence' in by_key[('one','visible_apertures')]['missing_prerequisites']
    assert by_key[('two','effect_ownership')]['finding']=='unresolved'


def test_independent_rejection_is_retained_and_fixture_or_json_authorities_are_refused():
    witness, consumed=fixture();consumed['records'][0]['record']['claims'][0]['finding']='rejected'
    assert qualify_boundary_witness(witness,consumed)['status']=='REJECTED'
    with pytest.raises(ValueError,match='Controller authorities'):
        RegisteredBoundaryQualificationEvidence(execution=object(),authorities={'engineering':object(),'independent':object()})
    with pytest.raises(ValueError,match='both qualified'):
        RegisteredBoundaryQualificationEvidence(execution=object())


def test_real_controller_fixture_stage_proofs_do_not_turn_generic_quality_reports_into_pixel_evidence(tmp_path):
    from dataevolver.testing.test_production_qualification import setup, stage_controller
    from dataevolver.harness.production_qualification import assess_production_qualification
    f, execution, parent, declarations, _=setup(tmp_path)
    authorities={stage:stage_controller(tmp_path,f,execution,parent,declarations,stage)
                 for stage in execution.stages}
    before={stage:a.base.controller.layout.events_path.read_bytes() for stage,a in authorities.items()}
    # These are actual committed fixture Controllers, not production observations.
    assert assess_production_qualification(execution,authorities)['production_independent_qualification']=='VERIFIED'
    witness,_=fixture()
    reader=RegisteredBoundaryQualificationEvidence(execution=execution,authorities=authorities)
    with pytest.raises(ValueError,match='registered production role manifest'):
        reader.read(witness)
    assert before=={stage:a.base.controller.layout.events_path.read_bytes() for stage,a in authorities.items()}


def test_registered_cpu_operator_binds_request_refuses_overlap_and_recovers_without_model(tmp_path,monkeypatch):
    _, f, c, r, receipts=evidence();witness=compile_boundary_witness(f,c,request_binding=r)
    old=tmp_path/'old';old.mkdir();evidence_file=old/'evidence.json';evidence_file.write_text('original')
    authority=object.__new__(FrozenBoundaryWitnessFeedbackAuthority);authority.config=old
    summary={'schema_version':'dataevolver.boundary_witness_feedback.v1','witness':witness,
        'validation':validate_boundary_witness(witness,f,c,request_binding=r,available_receipts=receipts)}
    proof={'artifact_hashes':{str(evidence_file):file_sha256(evidence_file)},'input_store_roots':{'old':str(old)}}
    monkeypatch.setattr(authority,'identity',lambda:{'config':str(old),'hash':'e'*64})
    monkeypatch.setattr(authority,'read',lambda:(copy.deepcopy(summary),copy.deepcopy(proof)))
    request=object.__new__(FrozenProductionFeedbackRequestAuthority)
    binding={'config':str(tmp_path/'request')}
    value={**r,'feedback_sha256':payload_sha256(summary),'artifact_hashes':{},
           'request':{'request':{'implementation_kind':'registered_operator'}}}
    monkeypatch.setattr(request,'identity',lambda:copy.deepcopy(binding))
    monkeypatch.setattr(request,'read',lambda:copy.deepcopy(value))
    holder=RegisteredBoundaryQualificationEvidence()
    with pytest.raises(ValueError,match='overlap'):
        BoundaryOperationalQualificationOperator(authority=authority,request_authority=request,
            evidence_authority=holder,output_root=old/'new')
    op=BoundaryOperationalQualificationOperator(authority=authority,request_authority=request,
        evidence_authority=holder,output_root=tmp_path/'output')
    call=OperatorCall('cpu','capture','key',op.spec.operator_id,'1',0,str(op.output_root/'capture'),{})
    receipt=op.execute(call)
    assert op.verify_effect(call,receipt).status is EvidenceStatus.VERIFIED
    assert op.recover(call).status is RecoveryStatus.RECOVERED
    assert op._result()['qualification']['status']=='INSUFFICIENT_EVIDENCE'
    value['request_sha256']='f'*64
    assert op.verify_effect(call,receipt).status is EvidenceStatus.CONTRADICTED


def test_completed_qualification_feedback_is_stable_across_readbacks_and_checks_original_bytes(tmp_path,monkeypatch):
    import json,sys
    from dataevolver.harness import boundary_qualification_readback as module
    from dataevolver.harness.operator_development import DevelopmentBudget
    from dataevolver.harness.adapters.boundary_qualification_feedback import BoundaryQualificationFeedbackOperator
    cfg=tmp_path/'qualification';(cfg/'protected').mkdir(parents=True);(cfg/'output').mkdir()
    witness,_=fixture();value=qualify_boundary_witness(witness,RegisteredBoundaryQualificationEvidence().read(witness))
    artifact=cfg/'output/qualification.json';artifact.write_text(json.dumps(value))
    zero=dict.fromkeys(['model_calls','planner_calls','operator_calls','wall_seconds','output_bytes','image_experiments'],0)
    usage={**zero,'operator_calls':1,'wall_seconds':1,'output_bytes':artifact.stat().st_size}
    (cfg/'protected/admission.json').write_text(json.dumps({'budget':DevelopmentBudget(usage,zero,1000).to_dict()}))
    (cfg/'protected/feedback.json').write_text(json.dumps({'origin':{'package_sha256':'a'*64}}))
    (cfg/'protected/origin-proof.json').write_text(json.dumps({'input_store_roots':{'original':str(cfg)}}))
    (cfg/'validation.json').write_text(json.dumps({'artifact_file_sha256':file_sha256(artifact),
        'qualification_sha256':payload_sha256(value),'source_commit':'b'*40,'lifecycle_usage':usage,'lifecycle_total':usage}))
    clock=iter([1,2])
    def verified(*args,**kwargs):
        return {'artifact_sha256':file_sha256(artifact),'qualification_sha256':payload_sha256(value),
            'qualification':value,'source_commit':'b'*40,'artifact_hashes':{str(artifact):file_sha256(artifact)},
            'readback_seconds':next(clock)}
    monkeypatch.setattr(module,'verify_frozen_boundary_qualification',verified)
    a=module.FrozenBoundaryQualificationFeedbackAuthority(config=cfg,python=sys.executable)
    b=module.FrozenBoundaryQualificationFeedbackAuthority(config=cfg,python=sys.executable)
    assert a.read()==b.read()
    op=BoundaryQualificationFeedbackOperator(authority=a,output_root=tmp_path/'next')
    call=OperatorCall('review','capture','key',op.spec.operator_id,'1',0,str(op.output_root/'capture'),{})
    receipt=op.execute(call);assert op.verify_effect(call,receipt).status is EvidenceStatus.VERIFIED
    assert op.recover(call).status is RecoveryStatus.RECOVERED
    artifact.write_text('changed original')
    assert op.verify_effect(call,receipt).status is EvidenceStatus.CONTRADICTED
