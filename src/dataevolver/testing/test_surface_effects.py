"""Synthetic invariants, not real media quality or method improvement evidence."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dataevolver.dataset.contracts import ArtifactStore,canonical_json,file_sha256,payload_sha256
from dataevolver.dataset.supervision import write_image,artifact_closure,complete_background_composite
from dataevolver.dataset.surface_effects import (rasterize_surface_support,compile_surface_artifacts,
    audit_surface_composition,validate_surface_observation,UnsupportedSurfaceRepair)
from dataevolver.testing.test_training_rsi_domain import fixture_domain,observe,proposal
from dataevolver.harness.state_store import atomic_write_json


def observed():
    return {'status':'supported','reason':'synthetic attributed shadow',
        'surfaces':[{'surface_id':'table','description':'synthetic table plane',
            'polygon':[[0,0],[1,0],[1,1],[0,1]]}],
        'subjects':[{'subject_id':'s1','surface_ids':['table'],'effect_assessment':'attributed','reason':'fixture only'}],
        'effects':[{'subject_id':'s1','surface_id':'table','kind':'cast_shadow',
            'polygon':[[.2,.2],[.5,.2],[.5,.4],[.2,.4]],'reason':'synthetic support'}],
        'preservation_regions':[],'uncertain_regions':[]}


def masks():
    a=np.zeros((32,32),dtype='uint8');a[10:18,9:13]=255
    b=np.zeros_like(a);b[24:28,25:29]=255
    return {'s1':Image.fromarray(a),'s2':Image.fromarray(b)}


def raster(value):
    m=masks()
    return rasterize_surface_support(value,[{'subject_id':'s1'}],[{'subject_id':'s2'}],
        size=(32,32),load_mask=lambda x:m[x['subject_id']],edge_margin=1)


def test_condition_and_composition_share_exact_support_and_relocatable_closure(tmp_path):
    store=ArtifactStore({'data':str(tmp_path/'dataset')});m=masks()
    selected=[{'subject_id':'s1','mask':write_image(store,'data','s1.png',m['s1']).to_dict()}]
    kept=[{'subject_id':'s2','mask':write_image(store,'data','s2.png',m['s2']).to_dict()}]
    # Deliberately different raw and source, allowing exact pixel witnesses.
    source=write_image(store,'data','source.png',Image.new('RGB',(32,32),(15,25,35)))
    rejected=write_image(store,'data','rejected.png',Image.new('RGB',(32,32),(25,35,45)))
    contract={'source_image':source.to_dict(),'rejected_candidate':rejected.to_dict(),
        'observation':observed(),'proposal_actor':'runtime_agent'}
    visible,region,condition,evidence=compile_surface_artifacts(store,'data','fixture',source,selected,kept,contract,edge_margin=1)
    saved=json.loads(store.resolve(evidence).read_text())
    assert saved['support']==saved['conditioning_region']==saved['composition_alpha']
    a=np.asarray(Image.open(store.resolve(region)))>0
    c=np.asarray(Image.open(store.resolve(condition)))
    assert np.all(c[a]==127) and np.all(c[~a]==[15,25,35])
    raw=write_image(store,'data','raw.png',Image.new('RGB',(32,32),(120,130,140)))
    completed=complete_background_composite(store,'data','fixture',source,raw,region)
    from dataevolver.dataset.contracts import ArtifactRef
    result=audit_surface_composition(store,source,raw,completed,region,ArtifactRef.from_dict(saved['protected']))
    assert result['restored_supported_source_effect_pixels']==0 and not result['quality_or_attribution_established']
    # Idempotent exact recovery and a real copied store with recursive reference loading.
    assert compile_surface_artifacts(store,'data','fixture',source,selected,kept,contract,edge_margin=1)[-1]==evidence
    import shutil
    shutil.copytree(store.roots['data'],tmp_path/'relocated')
    copied=ArtifactStore({'data':str(tmp_path/'relocated')})
    for ref in artifact_closure({'evidence':evidence.to_dict()},copied):
        if ref.media_type=='image/png':
            with Image.open(copied.resolve(ref)) as image:image.load()
    broken=write_image(store,'data','broken.png',Image.new('RGB',(32,32),(1,2,3)))
    with pytest.raises(ValueError,match='violates'):audit_surface_composition(store,source,raw,broken,region,ArtifactRef.from_dict(saved['protected']))


@pytest.mark.parametrize('change', ['uncertain','unbound','cross_surface','protected','nonfinite','self_intersection'])
def test_unsafe_attribution_or_geometry_fails_before_generation(change):
    value=observed()
    if change=='uncertain':value['subjects'][0]['effect_assessment']='uncertain';value['effects']=[]
    if change=='unbound':value['effects'][0]['subject_id']='preserved'
    if change=='cross_surface':value['surfaces'][0]['polygon']=[[0,0],[.3,0],[.3,1],[0,1]]
    if change=='protected':value['uncertain_regions']=[{'polygon':value['effects'][0]['polygon'],'reason':'ambiguous light'}]
    if change=='nonfinite':value['effects'][0]['polygon'][0][0]=float('nan')
    if change=='self_intersection':value['effects'][0]['polygon']=[[0,0],[1,1],[1,0],[0,1]]
    with pytest.raises((ValueError, __import__('jsonschema').ValidationError)):raster(value)


def test_new_capability_does_not_accept_external_geometry_or_bypass_old_recipe_gate(tmp_path):
    domain,state,trial=fixture_domain(tmp_path);observe(domain,state,trial)
    domain.value['surface_capability']={'candidate_method':True}
    domain.task['selected_subject_ids']=['s1']
    path=Path(state.data['rsi_observations'][0]['stage_result']['evidence_refs'][0])
    row={'tool':'inspect_surfaces','source':trial['source'],'candidate':trial['candidate'],
        'output':observed(),'support_feasibility':{'supported':True}}
    atomic_write_json(path,row)
    p=proposal();p['patch']={'kind':'surface_effect_repair','parameters_json':json.dumps({'observation_id':'probe-fixture'})}
    compiled=domain.compile_intervention(p,state)
    assert compiled.candidate_delta['parameters']['surface_evidence']['sha256']==payload_sha256(row)
    p['patch']['parameters_json']=json.dumps({'observation_id':'probe-fixture','polygon':[[0,0],[1,0],[1,1]]})
    with pytest.raises(ValueError,match='only accepts'):domain.compile_intervention(p,state)
    with pytest.raises(ValueError,match='old recipe'):domain.compile_intervention(proposal(),state)
    row['support_feasibility']['supported']=False;atomic_write_json(path,row)
    p['patch']['parameters_json']=json.dumps({'observation_id':'probe-fixture'})
    with pytest.raises(ValueError,match='feasible'):domain.compile_intervention(p,state)


def test_mask_mismatch_and_preserved_instance_conflict_rejected():
    m=masks();m['s2']=m['s1']
    with pytest.raises(UnsupportedSurfaceRepair,match='overlap'):
        rasterize_surface_support(observed(),[{'subject_id':'s1'}],[{'subject_id':'s2'}],size=(32,32),load_mask=lambda x:m[x['subject_id']])


def test_trial_prepares_evidence_bound_contract_with_unchanged_models_and_early_cache(tmp_path,monkeypatch):
    from dataevolver.harness.adapters import training_rsi as module
    from dataevolver.dataset.contracts import ArtifactRef
    domain,state,trial=fixture_domain(tmp_path);domain.task['selected_subject_ids']=['s1']
    domain.value['surface_capability']={'candidate_method':True}
    domain.contract.update(models={'fixed':'fixture'},stores={'data':'DATA','bundle':'BUNDLE'})
    store=domain.scope('baseline')
    rejected=write_image(store,'data','reject.png',Image.new('RGB',(32,32)))
    source=write_image(store,'data','source.png',Image.new('RGB',(32,32)))
    trial.update(source=source.to_dict(),candidate=rejected.to_dict())
    trial_path=Path(state.data['last_stage_result']['evidence_refs'][0]);atomic_write_json(trial_path,trial)
    observe(domain,state,trial)
    path=Path(state.data['rsi_observations'][0]['stage_result']['evidence_refs'][0])
    row={'tool':'inspect_surfaces','source':trial['source'],'candidate':trial['candidate'],
        'output':observed(),'support_feasibility':{'supported':True}}
    atomic_write_json(path,row)
    p=proposal();p['patch']={'kind':'surface_effect_repair','parameters_json':json.dumps({'observation_id':'probe-fixture'})}
    compiled=domain.compile_intervention(p,state)
    experience={'capability_id':'fixture-k','version':'1','snapshot_sha256':'c'*64}
    compiled=compiled.with_updates(parameters={**compiled.parameters,'patch':{**compiled.candidate_delta,'experience':experience}})
    domain.value['seed'].update(config_sha256='a'*64,model_records=[{
        'request':{'request_id':'fixture-'+phase},'receipt_hash':'b'*64} for phase in module.EARLY_PHASES])
    monkeypatch.setattr(module,'TrainingSupervisionOperator',lambda **kw:kw)
    operator=module.TrainingRSITrialOperator(domain)
    prepared,store,inner=operator._prepare(type('Call',(),{'parameters':compiled.parameters})())
    assert prepared['models']==domain.contract['models']
    assert set(prepared['production']['model_reuse'][domain.family])==module.EARLY_PHASES
    surface=ArtifactRef.from_dict(prepared['production']['surface_effect_contracts'][domain.family])
    value=json.loads(store.resolve(surface).read_text())
    assert value['observation']==observed() and value['proposal_actor']=='runtime_agent'
    assert value['experience']==experience
    assert 'shadow_region_corrections' not in prepared['production']


@pytest.mark.parametrize('reconstruct',[False,True])
def test_background_node_uses_surface_contract_in_actual_media_path(tmp_path,reconstruct):
    from dataevolver.dataset.training_pipeline import TrainingOperations
    from dataevolver.dataset.supervision import write_artifact
    store=ArtifactStore({'data':str(tmp_path/'data')});m=masks()
    target=write_image(store,'data','target.png',Image.new('RGB',(32,32),(10,20,30)))
    rejected=write_image(store,'data','rejected.png',Image.new('RGB',(32,32),(11,21,31)))
    instances=[{'subject_id':name,'description':name,'box_xyxy':[0,0,10,10],
        'mask':write_image(store,'data',name+'.png',image).to_dict()} for name,image in m.items()]
    def document(path,value):return write_artifact(store,'data',path,canonical_json(value).encode(),'application/json')
    surface=document('observed-contract.json',{'source_image':target.to_dict(),'rejected_candidate':rejected.to_dict(),
        'observation':observed(),'proposal_actor':'runtime_agent',
        **({'reconstruction':{'context_padding_fraction':.1}} if reconstruct else {})})
    image=document('image.json',{'image':target.to_dict()});masklist=document('masks.json',{'instances':instances})
    op=TrainingOperations.__new__(TrainingOperations)
    op.store=store;op.dataset_store='data';op.family_id='fixture-a0';op.allow_fixture=True;op.records=[]
    op.config={'production':{'surface_effect_contracts':{op.family_id:surface.to_dict()}}}
    op.recipe={'probe_shadows':False,'mask_dilation_pixels':16,'background_instruction':'Original requirements',
        'width':32,'height':32,'steps':1,'true_cfg_scale':2,'background_conditioning':'masked_region',
        'background_conditioning_shape':'bounding_rectangle','background_feather_pixels':4}
    op.task={'selected_subject_ids':['s1'],'selected_edge_margin_pixels':1}
    requests=[]
    def model(role,parameters,schema,artifacts,**kw):
        requests.append((role,parameters,artifacts))
        op.records.append({'request':{'request_id':'fixture-background'}})
        raw=write_image(store,'data','model-output.png',Image.new('RGB',(32,32),(200,210,220)))
        return {'artifact':raw.to_dict(),'actual_parameters':parameters}
    op._model=model
    result=op.background({'image':image,'mask_list':masklist})
    row=op._read(result['background_completed'])
    assert row['completion_region']==row['conditioning_region']==row['completion_blend']
    assert row['conditioning_shape']==('surface_context_crop' if reconstruct else 'surface_effect_contract') and row['feather_pixels']==0
    assert bool(row.get('context_reconstruction'))==reconstruct
    if reconstruct:
        projection=op._read(__import__('dataevolver.dataset.contracts',fromlist=['ArtifactRef']).ArtifactRef.from_dict(row['context_reconstruction']))
        assert projection['raw_model_output']==row['raw_model_output'] and projection['projected_model_output']==row['raw_edit']
    assert requests[0][2][0].to_dict()==row['conditioning_image']
    assert row['surface_composition_audit']['protected_pixels_equal_source']


def test_partial_attribution_preserves_unknown_visible_surface_but_removes_selected_silhouettes():
    value=observed();value['subjects'].append({'subject_id':'s2','surface_ids':['table'],
        'effect_assessment':'uncertain','reason':'unresolved fixture illumination'})
    value['uncertain_regions']=[{'polygon':[[.7,.7],[1,.7],[1,1],[.7,1]],'reason':'unknown surface effect'}]
    m=masks()
    result=rasterize_surface_support(value,[{'subject_id':'s1'},{'subject_id':'s2'}],[],
        size=(32,32),load_mask=lambda x:m[x['subject_id']],edge_margin=1)
    support=np.asarray(result.support)>0;protected=np.asarray(result.protected)>0
    assert result.diagnostics['partial_attribution'] and result.diagnostics['unresolved_subject_ids']==['s2']
    assert not result.diagnostics['complete_goal_established']
    assert np.all(support[np.asarray(m['s2'])>0]) and not np.any(support&protected)
    value['uncertain_regions']=[{'polygon':value['effects'][0]['polygon'],'reason':'dependency unresolved'}]
    with pytest.raises(UnsupportedSurfaceRepair,match='uncertain'):
        rasterize_surface_support(value,[{'subject_id':'s1'},{'subject_id':'s2'}],[],size=(32,32),load_mask=lambda x:m[x['subject_id']])


def test_surface_recheck_compiles_exact_registered_output_without_inference(tmp_path):
    domain,state,trial=fixture_domain(tmp_path);observe(domain,state,trial)
    domain.value['surface_capability']={'candidate_method':True}
    path=Path(state.data['rsi_observations'][0]['stage_result']['evidence_refs'][0])
    row={'source':trial['source'],'candidate':trial['candidate'],'tool':'inspect_surfaces',
        'output':observed(),'support_feasibility':{'supported':False,'reason':'old version'}}
    atomic_write_json(path,row);before=path.read_bytes()
    decision={'observation':{'tool':'recheck_surface_support','parameters_json':json.dumps({'observation_id':'probe-fixture'})}}
    _,params=domain.compile_observation(decision,state)
    assert params['parameters']['path']==str(path) and params['parameters']['sha256']==file_sha256(path)
    assert path.read_bytes()==before
    row['candidate']={'sha256':'c'*64};atomic_write_json(path,row)
    with pytest.raises(ValueError,match='current source/candidate'):domain.compile_observation(decision,state)
