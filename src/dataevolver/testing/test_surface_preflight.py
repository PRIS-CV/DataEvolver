"""Matched CPU contracts over independent synthetic geometries; no quality claims."""
import json
import numpy as np
import pytest
from PIL import Image
from dataevolver.dataset.surface_preflight import (
    OBSERVATION_VERSION,typed_surface_support,UnsupportedReconstruction)
from dataevolver.dataset.contracts import payload_sha256
from dataevolver.testing.test_surface_effects import observed


def case(offset=0):
    value=observed(); value['schema_version']=OBSERVATION_VERSION
    value['effects']=[];value['subjects'][0]['effect_assessment']='none_observed'
    value['uncertain_regions']=[{'polygon':[[.1,.1],[.4,.1],[.4,.4],[.1,.4]],
        'kind':'reconstruction','subject_ids':['s1'],'reason':'unseen surface inside authorized hole'}]
    m=np.zeros((32,32),dtype='uint8');m[4+offset:10+offset,4:10]=255
    instance=Image.fromarray(m);old=Image.fromarray(m)
    return value,instance,old


def run(value,instance,old,**kw):
    return typed_surface_support(value,[{'subject_id':'s1'}],[],size=(32,32),
        load_mask=lambda x:instance,existing_support=old,bindings={'source':{'sha256':'a'*64}},**kw)


@pytest.mark.parametrize('offset',[0,7,15])
def test_reconstruction_uncertainty_never_expands_existing_authority(offset):
    value,instance,old=case(offset)
    out=run(value,instance,old,padding_fraction=.1,model_size=(64,64))
    assert np.array_equal(np.asarray(out.support),np.asarray(old))
    receipt=out.diagnostics;digest=receipt.pop('receipt_sha256')
    assert payload_sha256(receipt)==digest
    assert receipt['image_model_calls']==0 and not receipt['quality_verified']
    assert receipt['uncertainty_added_editable_pixels']==receipt['context_added_editable_pixels']==0
    crop=receipt['crop'];x,y=crop['source_origin_xy'];sx,sy=crop['source_to_model_scale']
    assert (12-x)*sx/sx+x==12 and (13-y)*sy/sy+y==13
    assert len(crop['source_crop_xyxy'])==4


@pytest.mark.parametrize('kind',['preservation','effect_attribution'])
def test_existing_support_cannot_override_visible_or_attribution_uncertainty(kind):
    value,instance,old=case();a=np.asarray(old).copy();a[10:14,4:10]=255;old=Image.fromarray(a)
    row={'polygon':[[.12,.3],[.35,.3],[.35,.46],[.12,.46]],'reason':'ambiguous visible surface'}
    if kind=='preservation':value['preservation_regions']=[row]
    else:value['uncertain_regions']=[{**row,'kind':kind,'subject_ids':['s1']}]
    with pytest.raises(UnsupportedReconstruction) as error:run(value,instance,old)
    receipt=error.value.receipt
    assert not receipt['supported'] and all(x['pixels']>0 and x['subject_ids']==['s1'] for x in receipt['conflicts'])
    assert any('existing_support' in x['category'] for x in receipt['conflicts'])


def test_new_attributed_effects_cannot_expand_into_reconstruction_or_protection():
    value,instance,old=case()
    value['subjects'][0]['effect_assessment']='attributed'
    value['effects']=observed()['effects']
    value['uncertain_regions'][0]['polygon']=value['effects'][0]['polygon']
    with pytest.raises(UnsupportedReconstruction) as error:run(value,instance,old)
    assert 'effect_expansion_into_reconstruction_uncertainty' in str(error.value)
    value['uncertain_regions']=[]
    out=run(value,instance,old)
    assert out.diagnostics['effect_expansion_pixels']>0
    value['preservation_regions']=[{'polygon':value['effects'][0]['polygon'],'reason':'visible furniture'}]
    with pytest.raises(UnsupportedReconstruction,match='protected_visible'):run(value,instance,old)


def test_legacy_evidence_not_silently_retyped_and_invalid_support_rejected():
    value,instance,old=case();value.pop('schema_version')
    with pytest.raises(__import__('jsonschema').ValidationError):run(value,instance,old)
    value['schema_version']=OBSERVATION_VERSION
    value['uncertain_regions'][0]['subject_ids']=['foreign']
    with pytest.raises(ValueError,match='unbound'):run(value,instance,old)
    value['uncertain_regions'][0]['subject_ids']=['s1']
    with pytest.raises(ValueError,match='binary'):run(value,instance,Image.new('L',(32,32),128))
    with pytest.raises(UnsupportedReconstruction,match='missing_selected'):run(value,instance,Image.new('L',(32,32)))
    value['preservation_regions']=[{'polygon':[[0,0],[1,0],[1,1],[0,1]],'reason':'visible protected content'}]
    with pytest.raises(UnsupportedReconstruction,match='protected_visible'):run(value,instance,old)


def test_full_frame_context_and_geometry_failure_have_receipts():
    value,instance,old=case();value['uncertain_regions']=[]
    with pytest.raises(UnsupportedReconstruction,match='unsupported_context_crop') as error:
        run(value,instance,Image.new('L',(32,32),255),padding_fraction=0)
    assert error.value.receipt['image_model_calls']==0
    assert error.value.receipt['conflicts'][-1]['pixels']==0


def test_typed_contract_relocates_prior_support_and_keeps_composition_exact(tmp_path):
    import shutil
    from dataevolver.dataset.contracts import ArtifactStore,ArtifactRef
    from dataevolver.dataset.supervision import write_image,artifact_closure,complete_background_composite
    from dataevolver.dataset.surface_effects import compile_surface_artifacts,audit_surface_composition
    from dataevolver.dataset.context_reconstruction import prepare_context_condition,project_context_output
    store=ArtifactStore({'data':str(tmp_path/'data'),'rsi_evidence':str(tmp_path/'evidence')})
    value,instance,old=case()
    source=write_image(store,'data','source.png',Image.new('RGB',(32,32),(10,20,30)))
    rejected=write_image(store,'rsi_evidence','unique/reject.png',Image.new('RGB',(32,32),(11,21,31)))
    prior=write_image(store,'rsi_evidence','unique/prior-support.png',old)
    mask=write_image(store,'data','mask.png',instance)
    contract={'source_image':source.to_dict(),'rejected_candidate':rejected.to_dict(),
        'existing_authorized_support':prior.to_dict(),'observation':value,'proposal_actor':'runtime_agent',
        'reconstruction':{'context_padding_fraction':.1}}
    args=(store,'data','fixture',source,[{'subject_id':'s1','mask':mask.to_dict()}],[],contract)
    visible,region,condition,evidence=compile_surface_artifacts(*args)
    assert compile_surface_artifacts(*args)[-1]==evidence
    # Prior support is kept in its own store/path, so compilation can write its
    # immutable output even when a fresh observation changes safe effect support.
    assert region.path!=prior.path and store.resolve(prior).exists()
    image,transform=prepare_context_condition(store,'data','fixture',source,condition,region,
        contract['reconstruction'],model_size=(64,64))
    raw=write_image(store,'data','raw.png',Image.new('RGB',(64,64),(200,210,220)))
    projected,projection=project_context_output(store,'data','fixture',source,raw,transform)
    completed=complete_background_composite(store,'data','fixture',source,projected,region)
    doc=json.loads(store.resolve(evidence).read_text())
    audit=audit_surface_composition(store,source,projected,completed,region,ArtifactRef.from_dict(doc['protected']))
    assert audit['outside_pixels_equal_source'] and audit['supported_pixels_equal_raw']
    assert doc['support']==doc['conditioning_region']==doc['composition_alpha']
    assert doc['diagnostics']['bindings']['instance_masks']==[mask.to_dict()]
    for name in ('data','rsi_evidence'):shutil.copytree(store.roots[name],tmp_path/'moved'/name)
    relocated=ArtifactStore({name:str(tmp_path/'moved'/name) for name in ('data','rsi_evidence')})
    refs=list(artifact_closure({'surface':evidence.to_dict(),'projection':projection.to_dict()},relocated))
    assert prior in refs and mask in refs and raw in refs
    for ref in refs:
        if ref.media_type=='image/png':
            with Image.open(relocated.resolve(ref)) as im:im.load()


def test_runtime_requires_fresh_typed_evidence_and_binds_actual_authorized_mask(tmp_path):
    from pathlib import Path
    from dataevolver.testing.test_training_rsi_domain import fixture_domain,observe
    from dataevolver.harness.state_store import atomic_write_json
    from dataevolver.dataset.supervision import write_image
    domain,state,trial=fixture_domain(tmp_path);observe(domain,state,trial)
    domain.value['surface_capability']={'typed_preflight':{'candidate_method':True}}
    domain.task['selected_subject_ids']=['s1']
    path=Path(state.data['rsi_observations'][0]['stage_result']['evidence_refs'][0])
    old={'tool':'inspect_surfaces','source':trial['source'],'candidate':trial['candidate'],'output':observed()}
    atomic_write_json(path,old)
    decision={'observation':{'tool':'recheck_surface_support','parameters_json':json.dumps({'observation_id':'probe-fixture'})}}
    with pytest.raises(ValueError,match='fresh typed'):domain.compile_observation(decision,state)
    value,instance,support=case()
    store=domain.scope(trial['scope'])
    source=write_image(store,'data','source.png',Image.new('RGB',(32,32)))
    prior=write_image(store,'data','prior.png',support)
    mask=write_image(store,'data','mask.png',instance)
    trial.update(source=source.to_dict(),media=[{}, {}, {}, {'artifact':prior.to_dict()}])
    root=store.roots['data']/'families'/domain.family
    atomic_write_json(root/'mask-list.json',{'instances':[{'subject_id':'s1','mask':mask.to_dict()}]})
    compiled=domain.compile_surface_support(trial,value)
    assert compiled.diagnostics['bindings']['existing_authorized_support']==prior.to_dict()
    assert compiled.diagnostics['bindings']['instance_masks']==[mask.to_dict()]
    assert np.array_equal(np.asarray(compiled.support),np.asarray(support))
    # Replacing the bytes behind an ArtifactRef fails before any editor call.
    store.resolve(prior).write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA|hash|checksum|changed|sha256'):
        domain.compile_surface_support(trial,value)


def test_registered_visual_operator_requests_typed_schema_and_runs_cpu_receipt(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from dataevolver.testing.test_training_rsi_domain import fixture_domain
    from dataevolver.harness.adapters.training_rsi import TrainingRSIObservationOperator
    from dataevolver.harness.models import RecoveryStatus
    from dataevolver.harness.state_store import atomic_write_json
    from dataevolver.dataset.supervision import write_image
    domain,state,trial=fixture_domain(tmp_path)
    domain.value['surface_capability']={'typed_preflight':{'candidate_method':True}}
    domain.task['selected_subject_ids']=['s1']
    value,instance,support=case();captured={}
    store=domain.scope(trial['scope'])
    source=write_image(store,'data','source.png',Image.new('RGB',(32,32),(10,20,30)))
    prior=write_image(store,'data','prior.png',support)
    mask=write_image(store,'data','mask.png',instance)
    trial.update(source=source.to_dict(),candidate=source.to_dict(),media=[
        {'label':label,'artifact':ref.to_dict()} for label,ref in [('source',source),('candidate',source),('raw',source),('editable region',prior)]])
    root=store.roots['data']/'families'/domain.family
    atomic_write_json(root/'mask-list.json',{'instances':[{'subject_id':'s1','mask':mask.to_dict()}]})
    path=domain.campaign/'trial.json';atomic_write_json(path,trial)
    def complete(messages,*,schema):
        captured.update(schema=schema,prompt=json.loads(messages[0]['content']))
        return json.dumps(value)
    domain.backend=SimpleNamespace(gateway=SimpleNamespace(store=store),set_artifacts=lambda refs:None,
        complete_json=complete,last_trace=lambda:{'fixture':True})
    op=TrainingRSIObservationOperator(domain)
    monkeypatch.setattr(op,'_check',lambda call:None)
    monkeypatch.setattr(op,'recover',lambda call:SimpleNamespace(status=RecoveryStatus.RETRY_SAFE))
    monkeypatch.setattr(op,'_finish',lambda call,result,path,started:json.loads(path.read_text()))
    call=SimpleNamespace(parameters={'trial_path':str(path),'trial_sha256':payload_sha256(trial),
        'tool':'inspect_surfaces','parameters':{'question':'Inspect fixture typed uncertainty'}},work_dir=str(tmp_path/'probe'))
    result=op.execute(call)
    assert captured['schema']['properties']['schema_version']['enum']==[OBSERVATION_VERSION]
    assert captured['prompt']['typed_contract']['existing_authorized_support']==prior.to_dict()
    assert result['support_feasibility']['supported'] and result['support_feasibility']['image_model_calls']==0
    assert result['existing_authorized_support']==prior.to_dict()


def test_legacy_preflight_is_a_safe_registered_diagnostic_not_an_edit(tmp_path):
    from pathlib import Path
    from dataevolver.testing.test_training_rsi_domain import fixture_domain,observe
    from dataevolver.harness.state_store import atomic_write_json
    domain,state,trial=fixture_domain(tmp_path);observe(domain,state,trial)
    domain.value['surface_capability']={'typed_preflight':{'candidate_method':True}}
    path=Path(state.data['rsi_observations'][0]['stage_result']['evidence_refs'][0])
    atomic_write_json(path,{'tool':'inspect_surfaces','source':trial['source'],'candidate':trial['candidate'],'output':observed()})
    old=path.read_bytes()
    _,parameters=domain.compile_observation({'observation':{'tool':'preflight_reconstruction',
        'parameters_json':json.dumps({'observation_id':'probe-fixture','context_padding_fraction':.1})}},state)
    assert parameters['parameters']['path']==str(path) and path.read_bytes()==old
    with pytest.raises(ValueError,match='fresh typed'):
        domain.compile_surface_support(trial,observed(),.1)


def test_runtime_schema_uses_provider_supported_keywords_and_cpu_keeps_uniqueness():
    from dataevolver.dataset.surface_preflight import TYPED_SURFACE_SCHEMA
    assert 'uniqueItems' not in json.dumps(TYPED_SURFACE_SCHEMA)
    value,instance,old=case()
    value['uncertain_regions'][0]['subject_ids']=['s1','s1']
    with pytest.raises(ValueError,match='repeats selected instance'):
        run(value,instance,old)
