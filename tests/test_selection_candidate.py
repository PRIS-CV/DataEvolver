import copy
import json

import pytest

from dataevolver.harness.selection_candidate import eligible_rows, validate_selection, baseline_selection
from dataevolver.harness.task_contracts import PROTECTED_GATES
from dataevolver.dataset.contracts import ArtifactRef, file_sha256


def pool():
    return {'catalog':{'rows':[{'id':str(i),'family_id':str(i),'source_id':'source','label_id':'label',
        'category':'objects','protected_gates':dict.fromkeys(PROTECTED_GATES,True),'review_status':'pass',
        'quality':{'clarity_proxy':.9-i*.1},'review_score':4-i} for i in range(4)]},
        'parameters':{'quotas':{'source':{'label':2}},'ranking':'hybrid','seed':0,'min_clarity':.4},
        'requirements':{'required_gates':sorted(PROTECTED_GATES),'review_required':True,'required_categories':['objects']}}


def test_selector_host_preserves_rows_and_checks_protected_gates():
    value=pool();before=copy.deepcopy(value)
    selected=validate_selection(value,baseline_selection(value))
    assert len(selected)==2 and value==before
    value['catalog']['rows'][0]['protected_gates']['privacy']=False
    with pytest.raises(ValueError,match='ineligible'):validate_selection(value,{'selected_ids':['0','1']})


@pytest.mark.parametrize('change', ['duplicate','invented','family','alias','quota','review','coverage','changed_gates','source_fields'])
def test_selector_rejects_permitted_looking_boundary_violations(change):
    value=pool();output={'selected_ids':['0','1']}
    if change=='duplicate':output['selected_ids']=['0','0']
    if change=='invented':output['selected_ids']=['missing','1']
    if change=='family':value['catalog']['rows'][1]['family_id']='0'
    if change=='alias':
        for row in value['catalog']['rows'][:2]:row['canonical_family_id']='merged'
    if change=='quota':output['selected_ids']=['0']
    if change=='review':value['catalog']['rows'][0]['review_status']='uncertain'
    if change=='coverage':value['requirements']['required_categories']=['missing']
    if change=='changed_gates':value['requirements']['required_gates']=['integrity']
    if change=='source_fields':output['rows']=[]
    with pytest.raises(ValueError):validate_selection(value,output)


def test_builtin_balanced_selector_respects_later_family_mergers():
    from dataevolver.dataset.quality import select_balanced
    value=pool();rows=value['catalog']['rows'][:2]
    for row in rows:row['canonical_family_id']='merged'
    with pytest.raises(ValueError,match='infeasible'):
        select_balanced(rows,value['parameters']['quotas'])


def test_benchmark_worker_rejects_cross_store_or_changed_media(tmp_path,monkeypatch):
    from dataevolver.adapters.benchmark_worker import bound_input_image
    from dataevolver.adapters.model_contracts import ModelRequest
    source=tmp_path/'source';source.mkdir();image=source/'image.png';image.write_bytes(b'actual fixture bytes')
    ref=ArtifactRef('source','image.png',file_sha256(image),'image/png')
    request=ModelRequest('request','benchmark.geneval2',({'role':'user','content':'{}'},),{'type':'object'},artifacts=(ref,))
    monkeypatch.setenv('BENCHMARK_SOURCE',str(source))
    payload={'inputs':[{'ref':ref.to_dict(),'path':str(image)}],'dataset_store':'output','dataset_root':str(tmp_path/'output'),
             'config':{'input_store_envs':{'source':'BENCHMARK_SOURCE'}}}
    assert bound_input_image(payload,request)==image
    bad=copy.deepcopy(payload);bad['inputs'][0]['path']=str(tmp_path/'elsewhere.png')
    with pytest.raises(ValueError,match='escaped'):bound_input_image(bad,request)
    image.write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):bound_input_image(payload,request)


def test_selection_scope_requires_real_build_and_outcome_authorities(tmp_path):
    from dataevolver.harness.selection_candidate import SelectionCandidateChecks
    with pytest.raises(ValueError,match='real build'):
        SelectionCandidateChecks(plan=None,package=None,store=None,request_authority=None,build_authority=None,
            runner=None,cases=[],phase='independent',outcome_authorities=[],output_root=tmp_path)


def test_candidate_slots_retain_closed_history_and_do_not_count_native_requests(tmp_path):
    from dataevolver.harness.candidate_limits import CandidateSlotLimits
    limits=CandidateSlotLimits(root=tmp_path,round_id='round2',quotas={'M':3,'R':1},
        closed_history={'native_requests':'7/8 closed','M_candidates':'1/1 closed'})
    first=limits.reserve(layer='M',candidate_id='candidate1',binding={'plan':'same'})
    assert limits.reserve(layer='M',candidate_id='candidate1',binding={'plan':'same'})==first
    with pytest.raises(ValueError,match='binding changed'):
        limits.reserve(layer='M',candidate_id='candidate1',binding={'plan':'different'})
    for i in [2,3]:limits.reserve(layer='M',candidate_id='candidate'+str(i),binding={})
    with pytest.raises(ValueError,match='exhausted'):limits.reserve(layer='M',candidate_id='candidate4',binding={})
    assert limits.usage()=={'M':3,'R':0}


def test_failed_host_workload_retains_failed_receipt_status(tmp_path):
    from dataevolver.harness.artifact_operator import ArtifactWorkloadOperator
    from dataevolver.harness.models import OperatorCall,EvidenceStatus
    class Failed(ArtifactWorkloadOperator):
        def workload(self,call):return {'failed':True}
        def result(self,call,report):
            result=super().result(call,report);result.update(status='FAILED',failure_code='NATIVE_FAILED')
            return result
    operator=Failed(operator_id='fixture.failed',output_root=tmp_path,configuration={})
    call=OperatorCall('fixture','call','call','fixture.failed','1',0,str(tmp_path/'call'))
    receipt=operator.execute(call)
    assert receipt.status.value=='FAILED' and receipt.return_code==1
    assert operator.verify_effect(call,receipt).status is EvidenceStatus.VERIFIED
    wrong=receipt.with_updates(status=type(receipt.status).SUCCESS,return_code=0)
    assert operator.verify_effect(call,wrong).status is EvidenceStatus.CONTRADICTED


def test_catalog_split_aliases_translate_at_supervision_boundary():
    from dataevolver.dataset.t2i_supervision import supervision_split
    assert [supervision_split(s) for s in ['train','dev','eval','validation','test']]==['train','validation','test','validation','test']
    with pytest.raises(ValueError):supervision_split('unregistered')


def test_answer_probability_uses_stable_full_vocabulary_normalization():
    import numpy as np
    from dataevolver.adapters.benchmark_worker import answer_probabilities
    # These float32 values came from a failing native response whose sum was >1.
    rounded=np.array([1.0120744263986126e-5,.0007095189648680389,.9990770816802979,.00020328059326857328],dtype=np.float32)
    assert float(rounded.astype(np.float64).sum())>1
    mass,probabilities,raw=answer_probabilities(np.log(rounded.astype(np.float64)),[0,1,2,3])
    assert mass<=1 and mass==pytest.approx(1.)
    assert abs(sum(probabilities)-mass)<1e-14
    with pytest.raises(ValueError,match='distinct'):answer_probabilities([0.,1.],[1,1])


@pytest.mark.parametrize('changed',['score','image','request','receipt'])
def test_benchmark_metrics_are_derived_from_bound_answer_evidence(tmp_path,changed):
    from dataevolver.dataset.benchmark_evaluation import GenerationBenchmarkEvaluation
    from dataevolver.dataset.contracts import ArtifactStore,payload_sha256
    from dataevolver.adapters.model_contracts import ModelRequest
    image=tmp_path/'image.png';image.write_bytes(b'fixture image')
    ref=ArtifactRef('source',image.name,file_sha256(image),'image/png')
    atom={'question':'Does the image contain the requested object?','answer':'yes'}
    manifest={'kind':'geneval2','split':'development','items':[{'id':'one','image':ref.to_dict(),'image_sha256':ref.sha256,'atoms':[atom]}]}
    evaluation=GenerationBenchmarkEvaluation(gateway=None,store=ArtifactStore({'source':tmp_path}),root=tmp_path/'journal',manifest=manifest)
    request=ModelRequest('benchmark-'+payload_sha256([evaluation.journal.binding,'one',0])[:28],
        'benchmark.geneval2',({'role':'user','content':json.dumps(atom)},),{'type':'object'},artifacts=(ref,),max_tokens=1)
    receipt={'output':{'atom_probability':.25,'answer_token_ids':[1],'answer_token_probabilities':[.25]}}
    value={'binding':evaluation.journal.binding,'id':'one','image_sha256':ref.sha256,
        'result':{'atom_probabilities':[.25],'soft_tifa_am':25.,'soft_tifa_gm':25.},
        'evidence':{'status':'VERIFIED','requests':[request.to_dict()],'receipts':[receipt],
            'model_receipt_sha256s':[payload_sha256(receipt)]}}
    evaluation.verify_item(value)
    if changed=='score':value['result']['soft_tifa_gm']=100.
    if changed=='image':value['image_sha256']='0'*64
    if changed=='request':value['evidence']['requests'][0]['request_id']='different'
    if changed=='receipt':value['evidence']['receipts'][0]['output']['atom_probability']=1.
    with pytest.raises(ValueError):evaluation.verify_item(value)


def test_capability_request_schema_requires_empty_observation_tool():
    from jsonschema import validate,ValidationError
    from dataevolver.harness.runtime_rsi import ACTION_SCHEMA
    action={'action':'REQUEST_CAPABILITY','layer':'M','hypothesis':'Missing interface','evidence_ids':['verified-evidence'],
        'observation':{'tool':'','parameters_json':'{}'},'patch':{'kind':'capability_gap','parameters_json':'{}'},
        'experience_id':'','reason':'Request inert implementation only.'}
    validate(action,ACTION_SCHEMA)
    action['observation']['tool']='none'
    with pytest.raises(ValidationError):validate(action,ACTION_SCHEMA)
