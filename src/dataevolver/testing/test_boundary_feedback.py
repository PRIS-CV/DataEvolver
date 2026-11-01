"""Native probe feedback preserves uncertainty and never upgrades authority."""
import copy,json
from pathlib import Path
import pytest
from dataevolver.dataset.contracts import file_sha256,payload_sha256
from dataevolver.harness.boundary_feedback import FrozenBoundaryFeedbackAuthority,summarize_boundary_probe
from dataevolver.harness.adapters.boundary_feedback import BoundaryProbeFeedbackOperator
from dataevolver.harness.training_candidate_feedback import ProductionFeedbackReviewDomain
from dataevolver.testing.test_training_candidate_feedback import admission
from dataevolver.harness.models import OperatorCall,EvidenceStatus,RecoveryStatus


def native_value():
    params={'question':'inspect original local boundaries','subjects':[{'subject_id':'one','description':'observed object',
        'reason':'actual native result','box_xyxy_normalized':[.1,.2,.5,.6]}]}
    return {'report':{'proposal_status':'OBSERVED_UNQUALIFIED','boundary_witness_qualification':'NOT_ASSESSED',
        'effect_attribution':'NOT_ASSESSED','image_submission_allowed':False,'candidate_adoption_allowed':False,
        'new_accepted_records':0,'new_permissions':False,'immutable_masks_modified':False,'generation_model_calls':0,
        'runtime_query':params,'mask_diagnostics':[{'subject_id':'one','target_instance_id':'original-one','added_vs_immutable_pixels':3}],
        'visual_review':{'request':{'request_id':'actual-visual'},'receipt_hash':'2'*64,'receipt':{'output':{
            'assessments':[{'subject_id':'one','target_instance_id':'original-one','identity_match':'supported',
            'boundary_coverage':'uncertain','visible_apertures':'rejected','preservation_conflict':'observed','reason':'visible gaps remain'}],
            'reason':'No editing authority'}}},'segmentation':{'request':{'request_id':'actual-segment'},'receipt_hash':'3'*64}},
        'origin':{'resource_deadline_epoch':1000,'package_sha256':'4'*64},
        'lifecycle_total':{'model_calls':9,'planner_calls':4,'operator_calls':19,'wall_seconds':419,'output_bytes':1000,'image_experiments':0},
        'lifecycle_usage':{'model_calls':2,'planner_calls':0,'operator_calls':2,'wall_seconds':25,'output_bytes':100,'image_experiments':0}}


def test_real_visual_refusal_and_original_query_remain_unqualified_and_cannot_grant_pixel_truth():
    value=native_value();before=copy.deepcopy(value);summary=summarize_boundary_probe(value)
    assert summary['runtime_query']==value['report']['runtime_query']
    assert summary['runtime_visual_assessment']['assessments'][0]['visible_apertures']=='rejected'
    assert not summary['mask_differences_establish_pixel_truth'] and not summary['local_visual_support_is_full_task_completion']
    assert not summary['image_submission_allowed'] and not summary['candidate_adoption_allowed'] and value==before
    assert summary['lifecycle_total']['model_calls']==9


@pytest.mark.parametrize('field,new', [('proposal_status','QUALIFIED'),('boundary_witness_qualification','PASSED'),
    ('effect_attribution','VERIFIED'),('image_submission_allowed',True),('candidate_adoption_allowed',True),
    ('new_accepted_records',1),('new_permissions',True),('immutable_masks_modified',True),('generation_model_calls',1)])
def test_qualification_or_effects_cannot_be_relabelled_as_unqualified_feedback(field,new):
    value=native_value();value['report'][field]=new
    with pytest.raises(ValueError):summarize_boundary_probe(value)


def test_summary_refuses_changed_or_duplicate_instance_mapping_and_unbounded_free_text():
    value=native_value();value['report']['visual_review']['receipt']['output']['assessments'][0]['target_instance_id']='wrong'
    with pytest.raises(ValueError):summarize_boundary_probe(value)
    value=native_value();value['report']['mask_diagnostics']*=2
    with pytest.raises(ValueError):summarize_boundary_probe(value)
    value=native_value();value['report']['runtime_query']['question']='x'*65536
    with pytest.raises(ValueError,match='64KiB'):summarize_boundary_probe(value)


def fixture_authority(tmp_path,monkeypatch):
    cfg=tmp_path/'original';cfg.mkdir();evidence=cfg/'observed.json';evidence.write_text('native evidence')
    run=tmp_path/'run';run.mkdir();(run/'exit_code').write_text('0')
    metadata=tmp_path/'profile.json';metadata.write_text('{}');input_root=tmp_path/'original-media';input_root.mkdir()
    descriptor={'run':str(run),'failed_config':str(cfg),'failed_run':str(run),
        'profile_path':str(metadata),'worker_path':str(metadata),'bootstrap_path':str(metadata)}
    import sys
    authority=FrozenBoundaryFeedbackAuthority(config=cfg,descriptor=descriptor,python=sys.executable)
    authority._cached={**native_value(),'artifact_hashes':{str(evidence):file_sha256(evidence)},
        'store_roots':{'original':str(input_root)},'verification':{'frozen_contract_valid':True,'replay_matches_snapshot':True}}
    def forbidden(*a,**kw):raise AssertionError('cached authority must never dispatch a child/provider')
    monkeypatch.setattr('dataevolver.harness.boundary_feedback.subprocess.run',forbidden)
    return authority,evidence


def test_cached_registered_feedback_checks_original_bytes_and_does_not_dispatch(tmp_path,monkeypatch):
    authority,evidence=fixture_authority(tmp_path,monkeypatch)
    a,proof=authority.read();assert a==authority.read()[0] and proof==authority.read()[1]
    evidence.write_text('drift')
    with pytest.raises(ValueError,match='changed'):authority.read()


def test_registered_capture_recovery_is_cpu_only_and_refuses_output_inside_old_evidence(tmp_path,monkeypatch):
    authority,evidence=fixture_authority(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='overlap'):BoundaryProbeFeedbackOperator(authority=authority,output_root=authority.config/'new')
    op=BoundaryProbeFeedbackOperator(authority=authority,output_root=tmp_path/'output')
    call=OperatorCall('review','capture','key',op.spec.operator_id,'1',0,str(op.output_root/'capture'),{})
    receipt=op.execute(call)
    assert op.verify_effect(call,receipt).status is EvidenceStatus.VERIFIED
    assert op.recover(call).status is RecoveryStatus.RECOVERED
    assert not op._result()['production_qualification_passed']
    assert op._result()['feedback']['runtime_query']==native_value()['report']['runtime_query']
    # Changing an observed assessment invalidates the captured receipt.
    authority._cached['report']['visual_review']['receipt']['output']['assessments'][0]['visible_apertures']='supported'
    assert op.verify_effect(call,receipt).status is EvidenceStatus.CONTRADICTED


def test_same_domain_permits_only_one_new_prior_accounted_text_decision(tmp_path,monkeypatch):
    authority,_=fixture_authority(tmp_path,monkeypatch);value,_=authority.read()
    a=admission(value);domain=ProductionFeedbackReviewDomain(authority=authority,admission=a)
    from types import SimpleNamespace
    context=domain.context(SimpleNamespace(data={}))
    assert context['allowed_actions']==['REQUEST_CAPABILITY','ABSTAIN']
    assert not context['allowed_observations'] and not context['production_model_and_image_calls_allowed_by_domain']
    a['budget']['prior_usage']['model_calls']-=1
    with pytest.raises(ValueError):ProductionFeedbackReviewDomain(authority=authority,admission=a)
