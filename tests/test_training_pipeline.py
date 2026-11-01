"""All media and backend responses here are explicitly synthetic fixtures."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from dataevolver.adapters.model_contracts import ModelPolicy,ModelProfile
from dataevolver.adapters.model_gateway import ModelGateway
from dataevolver.dataset.contracts import ArtifactStore
from dataevolver.dataset.supervision import write_image,load_dataset
from dataevolver.dataset.training_pipeline import TrainingOperations,QUALITY_CHECKS,validate_production_contract


@pytest.fixture
def pipeline(tmp_path):
    store=ArtifactStore({"data":tmp_path/"data","bundle":tmp_path/"bundle"})
    image=write_image(store,"data","fixture/target.png",Image.new("RGB",(32,24),"white"))
    masks=[];observed=[]
    for i,left in enumerate((4,20),1):
        mask=np.zeros((24,32),dtype="uint8");mask[4:12,left:left+8]=255
        ref=write_image(store,"data",f"fixture/mask{i}.png",Image.fromarray(mask))
        observed.append({"subject_id":f"s{i}","target_instance_id":f"i{i}","description":f"fixture object {i}",
                         "box_xyxy_normalized":[left/32,4/24,(left+8)/32,12/24]})
        masks.append({"subject_id":f"s{i}","target_instance_id":f"i{i}","description":f"fixture object {i}",
                      "box_xyxy":[left,4,left+8,12],"score":1.0,"mask":ref.to_dict()})
    sources=[("n1","scene.plan",{"training_goal":"goal:capability","task_config":"config:task_config"},["scene_plan"]),
      ("n2","image.generate",{"scene_plan":"node:n1.scene_plan"},["image"]),
      ("n3","subjects.resolve",{"image":"node:n2.image","subject_list":"node:n1.scene_plan"},["instance_bindings"]),
      ("n4","subjects.segment",{"image":"node:n2.image","instance_bindings":"node:n3.instance_bindings"},["mask_list"]),
      ("n5","subjects.extract",{"image":"node:n2.image","mask_list":"node:n4.mask_list"},["subject_references"]),
      ("n6","background.complete",{"image":"node:n2.image","mask_list":"node:n4.mask_list"},["background_visible","background_completed"]),
      ("n7","sample.assemble",{"background":"node:n6.background_completed","subject_references":"node:n5.subject_references",
        "target_image":"node:n2.image","instance_bindings":"node:n3.instance_bindings","scene_plan":"node:n1.scene_plan"},["training_record"]),
      ("n8","sample.review",{"training_record":"node:n7.training_record","mask_list":"node:n4.mask_list",
        "background_visible":"node:n6.background_visible"},["review_evidence"]),
      ("n9","dataset.export",{"training_records":"node:n7.training_record","review_evidence":"node:n8.review_evidence",
        "family_assignments":"node:n1.scene_plan","task_config":"config:task_config"},["dataset_bundle"])]
    nodes=[{"id":i,"operator":op,"input_bindings":[{"port":p,"source":s} for p,s in ins.items()],
            "outputs":[{"name":p} for p in outs],"depends_on":sorted({s[5:].split(".")[0] for s in ins.values() if s.startswith("node:")})}
           for i,op,ins,outs in sources]
    task={"slot_id":"fixture-slot","split":"pilot","seed":7,"selection_mode":"all","selected_subject_ids":["s1","s2"],
          "subjects":[{"subject_id":s["subject_id"],"description":s["description"]} for s in observed],
          "scene_prompt":"fixture natural scene","insertion_instruction":"fixture insertion"}
    config={"planning":{"goal":{"task_type":"multi_subject_insertion","construction_mode":"scene_decomposition","materials":[],
       "original_request":"fixture training goal","capability":"fixture capability","task_config":{"fixture_configuration":True}},
       "plan":{"dataset_plan":{"nodes":nodes}}},"production":{"evidence_kind":"fixture","count":1,"split_counts":{"pilot":1},
       "max_consecutive_failures":3,"tasks":[task],"checker_precheck":False,"recipe":{"width":32,"height":24,"steps":1,"true_cfg_scale":1.0,
          "review_max_tokens":512,"model_timeout_seconds":60,"mask_dilation_pixels":1,"background_instruction":"fixture remove",
          "sam_threshold":.3,"sam_mask_threshold":.5,"sam_min_box_iou":.1}},"bundle_store":"bundle"}
    outputs={"image":{"artifact":image.to_dict(),"actual_parameters":{"fixture_only":True},"image_size":[32,24]},
             "resolve":{"status":"pass","reason":"fixture only","coordinate_system":"normalized_0_1","subjects":observed},"segment":{"instances":masks},
             "review":{"status":"pass","summary":"fixture only","criteria":{k:{"status":"pass","reason":"fixture"} for k in QUALITY_CHECKS}}}
    profiles=tuple(ModelProfile(k,"mock","fixture-"+k,"fixture-1",("json","vision"),mock_output=v) for k,v in outputs.items())
    gateway=ModelGateway(profiles=profiles,policy=ModelPolicy({"training.generate":"image","training.background":"image",
        "training.resolve":"resolve","training.segment":"segment","training.review":"review"},None,usage_mode="accounting"),
        receipt_root=tmp_path/"gateway",artifact_store=store)
    parent=SimpleNamespace(call_id="fixture-call",campaign_id="fixture-campaign",to_dict=lambda:{"fixture_call":True})
    waits=[]
    operations=TrainingOperations(config,store,"data",gateway,tmp_path/"work",parent,lambda role:waits.append(role))
    return operations,task,waits,tmp_path


def test_all_nine_operations_preserve_records_and_load_independently(pipeline):
    operations,task,waits,tmp=pipeline
    record=operations.run_family(task,0)
    assert record["evidence_kind"]=="fixture" and record["status"]=="pass"
    assert len(record["input"]["subjects"])==2 and len(record["model_receipts"])==5
    assert operations.gateway.audit_usage()["model_calls"]==5
    outputs=copy.deepcopy(operations.last_outputs)
    operations.export([record],tmp/"bundle",[outputs])
    assert load_dataset(tmp/"bundle",allow_fixture=True)[0][0]==record
    # Replay all completed operations without a new model request or GPU wait.
    waits.clear()
    assert operations.run_family(task,0)==record
    operations.export([record],tmp/"bundle",[outputs])
    assert waits==[] and operations.gateway.audit_usage()["model_calls"]==5
    assert len(list((tmp/"work/steps").rglob("node_run.json")))==9


def test_background_cfg_activation_is_in_the_actual_request_only(pipeline):
    import json
    operations,task,_,_=pipeline
    operations.recipe['background_negative_prompt']=' '
    operations.run_family(task,0)
    background=next(r for r in operations.records if r['request']['role']=='training.background')
    generation=next(r for r in operations.records if r['request']['role']=='training.generate')
    assert json.loads(background['request']['messages'][0]['content'])['negative_prompt']==' '
    assert 'negative_prompt' not in json.loads(generation['request']['messages'][0]['content'])


def test_background_cfg_change_keeps_generation_and_training_goal_fixed(pipeline):
    import json
    operations,task,_,_=pipeline
    original=copy.deepcopy(operations.config['planning']['goal'])
    operations.recipe['true_cfg_scale']=4.0
    task['background_true_cfg_scale']=1.0
    row=operations.run_family(task,0)
    requests={r['request']['role']:json.loads(r['request']['messages'][0]['content']) for r in row['model_receipts']}
    assert requests['training.background']['true_cfg_scale']==1.0
    assert requests['training.generate']['true_cfg_scale']==4.0
    assert row['annotations']['original_goal']==original


def test_masked_background_changes_actual_conditioning_not_training_target(pipeline):
    import json
    operations,task,_,tmp=pipeline
    operations.recipe['background_conditioning']='masked_region'
    row=operations.run_family(task,0)
    background=next(r for r in row['model_receipts'] if r['request']['role']=='training.background')
    assert background['request']['artifacts'][0]!=row['target']['image']
    assert background['request']['artifacts'][0]==row['annotations']['background_completion']['conditioning_image']
    assert row['annotations']['background_completion']['conditioning_mode']=='masked_region'
    assert 'neutral gray missing regions' in json.loads(background['request']['messages'][0]['content'])['prompt']
    prompt=json.loads(background['request']['messages'][0]['content'])['prompt']
    assert not any(s['description'] in prompt for s in task['subjects'])
    assert row['annotations']['target_instances'] and row['annotations']['original_goal']==operations.config['planning']['goal']
    assert set(row['input'])=={'background','subjects','instruction'}
    outputs=copy.deepcopy(operations.last_outputs);operations.export([row],tmp/'bundle',[outputs])
    assert load_dataset(tmp/'bundle',allow_fixture=True)[0][0]==row


def test_rectangular_condition_is_traced_but_not_exposed_as_training_input(pipeline):
    operations,task,_,tmp=pipeline
    operations.recipe.update(background_conditioning='masked_region',
        background_conditioning_shape='bounding_rectangle')
    row=operations.run_family(task,0)
    completion=row['annotations']['background_completion']
    request=next(r['request'] for r in row['model_receipts'] if r['request']['role']=='training.background')
    assert request['artifacts']==[completion['conditioning_image']]
    assert completion['conditioning_region']!=completion['completion_region']
    assert completion['conditioning_shape']=='bounding_rectangle'
    assert set(row['input'])=={'background','subjects','instruction'}
    outputs=copy.deepcopy(operations.last_outputs)
    operations.export([row],tmp/'bundle',[outputs])
    assert load_dataset(tmp/'bundle',allow_fixture=True)[0][0]==row


def test_invalid_conditioning_shape_is_rejected_before_model_call(pipeline):
    from dataevolver.dataset.training_pipeline import validate_production_contract
    operations,_,_,_=pipeline
    operations.recipe['background_conditioning_shape']='ellipse'
    with pytest.raises(ValueError,match='unknown background conditioning shape'):
        validate_production_contract(operations.config)
    operations.recipe['background_conditioning_shape']='bounding_rectangle'
    with pytest.raises(ValueError,match='requires masked-region'):
        validate_production_contract(operations.config)


@pytest.mark.parametrize('selected_only',[False,True])
@pytest.mark.parametrize('visible_effects',[False,True])
def test_shadow_probe_is_a_costed_node_request_and_stays_out_of_training_inputs(pipeline,selected_only,visible_effects):
    import json
    from dataclasses import replace
    from dataevolver.adapters.model_gateway import MockModelProvider
    from dataevolver.dataset.training_pipeline import SHADOW_SCHEMA
    operations,task,waits,tmp=pipeline
    if selected_only:task.update(selected_subject_ids=['s1'],selection_mode='selected')
    operations.recipe.update(probe_shadows=True,background_conditioning='masked_region',background_feather_pixels=1,probe_visible_effects=visible_effects)
    observed={'status':'pass','reason':'fixture observation only','coordinate_system':'normalized_0_1000',
        'shadows':[{'subject_id':'s1','box_xyxy_normalized':[0,500,400,900],'reason':'fixture shadow'},
                   {'subject_id':'s2','box_xyxy_normalized':[600,700,1000,1000],'reason':'fixture shadow'}],
        'no_shadow_subject_ids':[]}
    profile=operations.gateway.profiles['resolve']
    normal=MockModelProvider(profile,tmp/'native-fixture')
    probe=MockModelProvider(replace(profile,mock_output=observed),tmp/'native-fixture')
    class ConditionalFixtureProvider:
        def health(self):return {'status':'fixture'}
        def invoke(self,request,digest):return (probe if request.response_schema==SHADOW_SCHEMA else normal).invoke(request,digest)
        def lookup(self,request,digest):return (probe if request.response_schema==SHADOW_SCHEMA else normal).lookup(request,digest)
    operations.gateway.factories['resolve']=ConditionalFixtureProvider
    row=operations.run_family(task,0)
    model=next(r for r in row['model_receipts'] if r['purpose']=='active_shadow_probe')
    prompt=json.loads(model['request']['messages'][0]['content'])['prompt']
    if visible_effects:
        assert 'above the objects' in prompt and 'steam' in prompt
        assert 'on the supporting surface' not in prompt
    else:assert 'on the supporting surface' in prompt
    assert model['operation']=='background.complete' and model['request']['role']=='training.resolve'
    assert len(model['request']['artifacts'])==1 and operations.gateway.audit_usage()['model_calls']==6
    assert row['annotations']['background_completion']['shadow_observation']
    shadow=row['annotations']['background_completion']['shadow_observation']
    import json
    from dataevolver.dataset.contracts import ArtifactRef
    bound=json.loads(operations.store.resolve(ArtifactRef.from_dict(shadow)).read_text())
    assert [s['subject_id'] for s in bound['shadows']]==task['selected_subject_ids']
    if selected_only:
        assert [s['subject_id'] for s in bound['preserved_shadow_observations']]==['s2']
        region=np.asarray(Image.open(operations.store.resolve(ArtifactRef.from_dict(row['annotations']['completion_region']))))
        assert region[20,30]==0  # Preserved shadow cannot widen the editable region.
    assert set(row['input'])=={'background','subjects','instruction'}
    outputs=copy.deepcopy(operations.last_outputs);operations.export([row],tmp/'bundle',[outputs])
    assert load_dataset(tmp/'bundle',allow_fixture=True)[0][0]==row
    waits.clear();assert operations.run_family(task,0)==row and waits==[]


def test_shadow_probe_requires_complete_attribution_and_declared_coordinates():
    from dataevolver.dataset.training_pipeline import normalize_shadow_observation,QualityRejected
    raw={'status':'pass','reason':'fixture','coordinate_system':'normalized_0_1000',
        'shadows':[{'subject_id':'s1','box_xyxy_normalized':[0,500,400,900],'reason':'fixture'}],
        'no_shadow_subject_ids':['s2']}
    assert normalize_shadow_observation(raw,['s1','s2'])['shadows'][0]['box_xyxy_normalized']==[0,.5,.4,.9]
    with pytest.raises(QualityRejected):normalize_shadow_observation(raw,['s1','s2','s3'])
    with pytest.raises(QualityRejected):normalize_shadow_observation({**raw,'status':'uncertain'},['s1','s2'])
    with pytest.raises(ValueError):normalize_shadow_observation({**raw,'coordinate_system':'pixels'},['s1','s2'])


def test_shadow_observation_routes_known_preserved_subjects_out_of_edit_regions():
    from dataevolver.dataset.training_pipeline import normalize_shadow_observation,QualityRejected
    # Mirrors a real selected probe that also described the preserved book.
    raw={'status':'pass','reason':'fixture reproducing observed subject routing',
        'coordinate_system':'normalized_0_1000','no_shadow_subject_ids':[],
        'shadows':[{'subject_id':key,'box_xyxy_normalized':box,'reason':'fixture'}
            for key,box in [('s1',[180,630,420,680]),('s2',[720,550,920,590]),('s3',[430,490,720,530])]]}
    with pytest.raises(QualityRejected):normalize_shadow_observation(raw,['s1','s2'])
    value=normalize_shadow_observation(raw,['s1','s2'],preserved_ids=['s3'])
    assert [s['subject_id'] for s in value['shadows']]==['s1','s2']
    assert [s['subject_id'] for s in value['preserved_shadow_observations']]==['s3']
    assert value['raw_observation']==raw
    with pytest.raises(QualityRejected):normalize_shadow_observation(raw,['s1','s2'],preserved_ids=['other'])
    with pytest.raises(QualityRejected):normalize_shadow_observation(raw,['s1','s2','missing'],preserved_ids=['s3'])


def test_preserved_shadow_absence_cannot_satisfy_missing_selected_evidence():
    from dataevolver.dataset.training_pipeline import normalize_shadow_observation,QualityRejected
    raw={'status':'pass','reason':'fixture','coordinate_system':'normalized_0_1',
        'shadows':[],'no_shadow_subject_ids':['s1','s2','s3']}
    value=normalize_shadow_observation(raw,['s1','s2'],preserved_ids=['s3'])
    assert value['no_shadow_subject_ids']==['s1','s2']
    assert value['preserved_no_shadow_subject_ids']==['s3']
    with pytest.raises(QualityRejected):normalize_shadow_observation(raw,['s1','s2','missing'],preserved_ids=['s3'])
    with pytest.raises(QualityRejected):normalize_shadow_observation(raw,['s1','s2'],preserved_ids=['s2','s3'])


def test_recorded_agent_shadow_correction_changes_conditioning_and_reloads(pipeline):
    from dataevolver.dataset.contracts import ArtifactRef,canonical_json
    from dataevolver.dataset.supervision import write_artifact
    operations,task,_,tmp=pipeline
    source=operations.gateway.profiles['image'].mock_output['artifact']
    candidate=write_image(operations.store,'data','fixture/failed.png',Image.new('RGB',(32,24),'gray'))
    evidence={'kind':'agent_visual_observation','status':'pass','reason':'fixture manual observation only',
        'source_image':source,'rejected_candidate':candidate.to_dict(),'coordinate_system':'normalized_0_1',
        'shadows':[{'subject_id':'s1','box_xyxy_normalized':[0,.7,.4,1],'reason':'fixture residual'}],
        'no_shadow_subject_ids':['s2']}
    ref=write_artifact(operations.store,'data','fixture/agent-observation.json',canonical_json(evidence).encode(),'application/json')
    operations.config['production']['shadow_region_corrections']={'fixture-slot-a0':ref.to_dict()}
    operations.recipe['background_conditioning']='masked_region'
    row=operations.run_family(task,0)
    completed=row['annotations']['background_completion']
    assert completed['shadow_region_correction']==ref.to_dict()
    region=np.asarray(Image.open(operations.store.resolve(ArtifactRef.from_dict(completed['completion_region']))))
    assert region[22,2]==255
    assert operations.gateway.audit_usage()['model_calls']==5
    assert set(row['input'])=={'background','subjects','instruction'}
    outputs=copy.deepcopy(operations.last_outputs);operations.export([row],tmp/'bundle',[outputs])
    assert load_dataset(tmp/'bundle',allow_fixture=True)[0][0]==row


def test_agent_shadow_correction_rejects_wrong_source_and_attribution(pipeline):
    from dataevolver.dataset.contracts import ArtifactRef
    from dataevolver.dataset.training_pipeline import validate_shadow_correction,QualityRejected
    operations,_,_,_=pipeline
    target=ArtifactRef.from_dict(operations.gateway.profiles['image'].mock_output['artifact'])
    other=write_image(operations.store,'data','fixture/failed.png',Image.new('RGB',(32,24),'gray'))
    raw={'kind':'agent_visual_observation','status':'pass','reason':'fixture only',
        'source_image':target.to_dict(),'rejected_candidate':other.to_dict(),
        'coordinate_system':'normalized_0_1','shadows':[], 'no_shadow_subject_ids':['s1','s2']}
    assert validate_shadow_correction(raw,target,['s1','s2'])['status']=='pass'
    with pytest.raises(ValueError):validate_shadow_correction({**raw,'source_image':other.to_dict()},target,['s1','s2'])
    with pytest.raises(ValueError):validate_shadow_correction({**raw,'kind':'model_observation'},target,['s1','s2'])
    with pytest.raises(QualityRejected):validate_shadow_correction({**raw,'no_shadow_subject_ids':['s1','s3']},target,['s1','s2'])


def test_negative_checker_cannot_accept_unchanged_background(pipeline):
    operations,task,waits,tmp=pipeline
    operations.config["production"]["checker_precheck"]=True
    with pytest.raises(ValueError,match="unchanged background"):
        operations.run_family(task,0)
    assert operations.gateway.audit_usage()["model_calls"]==5
    assert not (tmp/"bundle").exists()


def test_uncertain_or_unrelated_negative_rejection_is_not_checker_proof():
    from dataevolver.dataset.training_pipeline import negative_rejected
    result={"status":"fail","criteria":{k:{"status":"pass","reason":"fixture"} for k in QUALITY_CHECKS}}
    result["criteria"]["reference_identity"]["status"]="fail"
    assert not negative_rejected(result)
    result["criteria"]["selected_removal"]["status"]="uncertain"
    assert not negative_rejected(result)
    result["criteria"]["selected_removal"]["status"]="fail"
    assert negative_rejected(result)


def test_incompatible_media_ports_are_not_silently_compiled(pipeline):
    operations,_,_,_=pipeline
    config=copy.deepcopy(operations.config)
    config["planning"]["plan"]["dataset_plan"]["nodes"][4]["outputs"]=[{"name":"one_best_foreground"}]
    with pytest.raises(ValueError,match="compiler adapter"):
        validate_production_contract(config)


def _generation_handoff(pipeline):
    operations,task,waits,tmp=pipeline
    operations.task={**task,"attempt_index":0}
    operations.family_id=task["slot_id"]+"-a0"
    inputs={"scene_plan":operations._write("handoff-scene",{"task":operations.task})}
    output=operations.generate(inputs)
    previous=operations.records[0]
    config=copy.deepcopy(operations.config)
    config["production"]["generation_reuse"]={operations.family_id:{"request":previous["request"],
        "receipt_hash":previous["receipt_hash"],"source_config_sha256":"f"*64}}
    parent=SimpleNamespace(call_id="fixture-gpu-call",campaign_id="fixture-gpu-campaign",to_dict=lambda:{"fixture_gpu_call":True})
    next_ops=TrainingOperations(config,operations.store,"data",operations.gateway,tmp/"gpu-work",parent,lambda r:waits.append(r))
    next_ops.task=operations.task;next_ops.family_id=operations.family_id;waits.clear()
    return next_ops,inputs,output,waits


def test_deployment_handoff_recovers_generation_without_resource_wait_or_new_call(pipeline):
    operations,inputs,expected,waits=_generation_handoff(pipeline)
    assert operations.generate(inputs)==expected
    assert waits==[] and operations.gateway.audit_usage()["model_calls"]==1
    assert operations.records[0]["receipt"]["reused"] is True


def test_handoff_missing_receipt_never_regenerates_the_source(pipeline):
    from dataevolver.adapters.model_gateway import ModelError
    operations,inputs,_,waits=_generation_handoff(pipeline)
    operations.gateway.recover=lambda request:None
    with pytest.raises(ModelError) as error:operations.generate(inputs)
    assert error.value.code=="PRIOR_EFFECT_MISSING" and waits==[]


def test_general_model_reuse_remains_cache_only(pipeline):
    operations,inputs,expected,waits=_generation_handoff(pipeline)
    previous=operations.config['production'].pop('generation_reuse')
    operations.config['production']['model_reuse']={operations.family_id:{'generate':previous[operations.family_id]}}
    validate_production_contract(operations.config)
    assert operations.generate(inputs)==expected
    assert waits==[] and operations.gateway.audit_usage()['model_calls']==1
    operations.config['production']['model_reuse'][operations.family_id]['segment']=previous[operations.family_id]
    with pytest.raises(ValueError,match='phase role'):validate_production_contract(operations.config)


def test_observation_receipt_is_reused_without_reobserving_the_image(pipeline):
    operations,task,waits,tmp=pipeline
    operations.task={**task,'attempt_index':0};operations.family_id=task['slot_id']+'-a0'
    image=operations.generate({'scene_plan':operations._write('reuse-scene',{'task':operations.task})})['image']
    inputs={'image':image,'subject_list':operations._write('reuse-subjects',{'task':operations.task})}
    expected=operations.resolve(inputs)
    previous=operations.records[-1]
    config=copy.deepcopy(operations.config)
    config['production']['model_reuse']={operations.family_id:{'resolve':{'request':previous['request'],
        'receipt_hash':previous['receipt_hash'],'source_config_sha256':'f'*64}}}
    parent=SimpleNamespace(call_id='fixture-next-call',campaign_id='fixture-next',to_dict=lambda:{'fixture_next':True})
    following=TrainingOperations(config,operations.store,'data',operations.gateway,tmp/'next-work',parent,lambda r:waits.append(r))
    following.task=operations.task;following.family_id=operations.family_id;waits.clear()
    assert following.resolve(inputs)==expected
    assert waits==[] and operations.gateway.audit_usage()['model_calls']==2


def test_handoff_request_change_and_unbound_family_are_rejected(pipeline):
    from dataevolver.adapters.model_gateway import ModelError
    operations,inputs,_,waits=_generation_handoff(pipeline)
    operations.recipe["steps"]+=1
    with pytest.raises(ModelError) as error:operations.generate(inputs)
    assert error.value.code=="REQUEST_CONFLICT" and waits==[]
    config=copy.deepcopy(operations.config)
    value=next(iter(config["production"]["generation_reuse"].values()))
    config["production"]["generation_reuse"]={"unbound-a0":value}
    with pytest.raises(ValueError,match="initial task family"):validate_production_contract(config)


def test_sharded_admission_requires_two_stable_observations(tmp_path,monkeypatch):
    import json
    from dataevolver.harness.adapters import training_supervision as adapter
    op=object.__new__(adapter.TrainingSupervisionOperator)
    op.custom_wait=None;op.policy=SimpleNamespace(bindings={'training.generate':'image'})
    op.contract={'models':{'workers':{'image':{'shard_max_memory_mib':[48000,48000],
        'shard_min_free_mib':[64000,64000],'resource_profile':{'min_free_mib':64000,'max_utilization':100}}}},
        'production':{'resource_wait_seconds':60}}
    calls=[]
    def observe(*a,**kw):
        calls.append(1)
        return SimpleNamespace(stdout='0, gpu-a, 70000, 100\n1, gpu-b, 90000, 100\n')
    monkeypatch.setattr(adapter.subprocess,'run',observe)
    monkeypatch.setattr(adapter.time,'sleep',lambda seconds:None)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    op._wait(SimpleNamespace(work_dir=str(tmp_path)),'training.generate')
    assert len(calls)==2
    status=json.loads((tmp_path/'resource-status.json').read_text())
    assert status['state']=='READY_SHARDED' and len(status['selected_gpus'])==2
    assert adapter.os.environ['CUDA_VISIBLE_DEVICES']=='gpu-a,gpu-b'


@pytest.mark.parametrize('unit,box',[
    ('normalized_0_1',[.108,.368,.495,.750]),
    ('normalized_0_1000',[108,368,495,750])])
def test_declared_coordinate_units_convert_to_same_canonical_box(unit,box):
    from dataevolver.dataset.training_pipeline import normalize_instance_observation
    raw={'coordinate_system':unit,'subjects':[{'subject_id':'fixture-s1','box_xyxy_normalized':box}]}
    normalized=normalize_instance_observation(raw)
    assert normalized['subjects'][0]['box_xyxy_normalized']==[.108,.368,.495,.750]
    assert normalized['raw_observation']==raw


@pytest.mark.parametrize('unit,box',[
    (None,[108,368,495,750]),
    ('normalized_0_1',[108,368,495,750]),
    ('normalized_0_1000',[495,368,108,750])])
def test_ambiguous_or_invalid_boxes_are_not_silently_rescaled(unit,box):
    from dataevolver.dataset.training_pipeline import normalize_instance_observation
    with pytest.raises(ValueError,match='coordinate system'):
        normalize_instance_observation({'coordinate_system':unit,'subjects':[{'box_xyxy_normalized':box}]})


def _batch(pipeline, *, rejected=False):
    from dataevolver.harness.adapters.training_supervision import TrainingSupervisionOperator
    from dataevolver.harness.models import OperatorCall
    operations,_,waits,tmp=pipeline
    config=copy.deepcopy(operations.config)
    profiles=[p.to_dict() for p in operations.gateway.profiles.values()]
    if rejected:
        next(p for p in profiles if p["profile_id"]=="resolve")["mock_output"]={"status":"fail","reason":"fixture missing subject","coordinate_system":"normalized_0_1","subjects":[]}
    config.update(stores={"data":"FIXTURE_DATA","bundle":"FIXTURE_BUNDLE"},dataset_store="data",
                  models={"profiles":profiles,"policy":operations.gateway.policy.to_dict()})
    config["production"]["resource_wait_seconds"]=60
    op=TrainingSupervisionOperator(allowed_write_root=tmp/"campaign",contract=config,
        roots={k:str(v) for k,v in operations.store.roots.items()},wait_for_role=lambda role:waits.append(role))
    call=OperatorCall("fixture-campaign","fixture-batch","fixture-key",op.spec.operator_id,"1",0,str(tmp/"campaign/execute"),config)
    return op,call,waits,tmp


def test_batch_accounts_all_steps_and_reuses_without_gpu(pipeline):
    import json
    from dataevolver.harness.models import StageStatus,EvidenceStatus,RecoveryStatus
    op,call,waits,tmp=_batch(pipeline)
    receipt=op.execute(call)
    assert receipt.status is StageStatus.SUCCESS and receipt.stage_result["metrics"]["local_model_calls"]==5
    assert op.verify_effect(call,receipt).status is EvidenceStatus.VERIFIED
    waits.clear()
    assert op.recover(call).status is RecoveryStatus.RECOVERED
    assert op.execute(call).reused and waits==[]
    assert op.gateway(call).audit_usage()["model_calls"]==5
    manifest=json.loads((tmp/"campaign/execute/production.json").read_text())
    manifest["accepted"]=[]
    (tmp/"campaign/execute/production.json").write_text(json.dumps(manifest))
    assert op.verify_effect(call,receipt).status is EvidenceStatus.CONTRADICTED


def test_batch_quality_stop_keeps_failures_and_costs(pipeline):
    import json
    from dataevolver.harness.models import StageStatus
    op,call,waits,tmp=_batch(pipeline,rejected=True)
    receipt=op.execute(call)
    assert receipt.status is StageStatus.FAILED
    result=json.loads((tmp/"campaign/execute/production.json").read_text())
    assert result["stop_reason"]=="NO_PROGRESS" and len(result["failures"])==3
    assert result["usage"]["model_calls"]==6 and len(result["slots"])==1
    assert result["accepted"]==[] and not (tmp/"bundle").exists()
    waits.clear()
    assert op.execute(call).reused and waits==[]


def test_batch_refuses_roots_overlapping_read_only_sources(pipeline):
    from dataevolver.harness.adapters.training_supervision import TrainingSupervisionOperator
    op,call,_,tmp=_batch(pipeline)
    config=copy.deepcopy(op.contract);config["stores"]["source"]="FIXTURE_SOURCE"
    with pytest.raises(ValueError,match="read-only source"):
        TrainingSupervisionOperator(allowed_write_root=tmp/"campaign",contract=config,
          roots={"data":tmp/"data","bundle":tmp/"bundle","source":tmp/"campaign/sources"})


@pytest.mark.parametrize("rejected",[False,True])
def test_existing_controller_owns_media_batch_and_replays(pipeline,monkeypatch,rejected):
    import json
    from dataevolver.harness.cli import _controller
    from dataevolver.harness.models import GoalSpec
    from dataevolver.harness.training_run import trial_key
    from dataevolver.dataset.contracts import file_sha256
    op,_,_,tmp=_batch(pipeline,rejected=rejected)
    monkeypatch.setenv("FIXTURE_DATA",str(tmp/"data"));monkeypatch.setenv("FIXTURE_BUNDLE",str(tmp/"bundle"))
    # Fixtures bypass waiting only; model effects and costs still use ModelGateway.
    monkeypatch.setattr(type(op),"_wait",lambda *args:None)
    evaluator=tmp/"evaluator.json"
    evaluator.write_text(json.dumps({"schema_version":"dataevolver.harness.evaluator.v1","evaluator_id":"fixture-media","revision":"1",
        "hard_constraints":{"artifact_valid":{"required":True},"all_tasks_recorded":{"required":True},"all_qualified":{"required":True}},
        "target_metrics":{"qualified_ratio":{"required":True,"minimum":1.0}}}))
    goal=GoalSpec.from_dict({"schema_version":"dataevolver.harness.goal.v1","campaign_id":"fixture-harness",
        "objective":{"real_data_execution":True,"evidence_kind":"fixture","initial_trial_key":trial_key(op.contract,
         file_sha256(evaluator),{k:str(v) for k,v in op.store.roots.items()})},"initial_operator_id":"data.training_supervision",
        "route_candidates":["data.training_supervision"],"budgets":{"max_rounds":1,"max_operator_calls":1,"max_experiments":0,
        "max_retries":0,"max_wall_seconds":120,"max_planner_calls":0},"resource_policy":{"allow_gpu":True,"allow_network":False,
        "allowed_operators":["data.training_supervision"]},
        "frozen_evaluator":{"path":str(evaluator),"sha256":file_sha256(evaluator)},"planner":{"kind":"deterministic"},
        "initial_operator_parameters":op.contract,"interventions":[],"lease_seconds":180})
    controller=_controller(goal,tmp/"campaigns",create=True,idempotency_key="fixture-media-key")
    controller.run()
    assert (controller.status().state.value=="COMPLETE") is not rejected
    assert controller.replay().to_dict()==controller.status().to_dict()
    assert controller._frozen()==(True,())
    if rejected:
        assert not (tmp/"bundle").exists()
    else:
        assert len(load_dataset(tmp/"bundle",allow_fixture=True))==1
