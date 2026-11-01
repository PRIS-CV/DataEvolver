"""Authority algebra, not image quality, on multiple independent fixture masks."""
from copy import deepcopy
import json
import numpy as np
import pytest
from PIL import Image
from dataevolver.dataset.surface_authority import conservative_surface_support
from dataevolver.dataset.surface_preflight import UnsupportedReconstruction,OBSERVATION_VERSION
from dataevolver.dataset.contracts import payload_sha256
from dataevolver.testing.test_surface_preflight import case


def run(value,mask,old,**kw):
    return conservative_surface_support(value,[{'subject_id':'s1'}],[],size=mask.size,
        load_mask=lambda x:mask,existing_support=old,**kw)


@pytest.mark.parametrize('offset',[0,7,15])
def test_uncertain_prior_effects_are_preserved_without_dropping_task_silhouette(offset):
    value,mask,old=case(offset)
    a=np.asarray(old).copy();a[12:15,4:12]=255;old=Image.fromarray(a)
    value['uncertain_regions'].append({'kind':'effect_attribution','subject_ids':['s1'],
        'polygon':[[.1,.35],[.45,.35],[.45,.55],[.1,.55]],'reason':'unknown visible effect owner'})
    out=run(value,mask,old,padding_fraction=.1,model_size=(64,64))
    s=np.asarray(out.support)>0;m=np.asarray(mask)>0;prior=np.asarray(old)>0
    assert np.all(s[m]) and not np.any(s&~prior)
    assert out.diagnostics['uncertain_prior_effect_pixels_preserved']>0
    assert out.diagnostics['effect_expansion_pixels']==0 and out.diagnostics['support_subset_of_prior_authority']
    assert out.diagnostics['partial_execution'] and out.diagnostics['execution_status']=='partial_supported'
    assert out.diagnostics['coverage'][0]['completion_established'] is False
    assert not out.diagnostics['full_removal_verified'] and out.diagnostics['image_model_calls']==0
    assert not np.any((np.asarray(out.protected)>0)&s)
    d=dict(out.diagnostics);sha=d.pop('receipt_sha256');assert payload_sha256(d)==sha
    # Changes in uncertain reconstruction outside existing authority add no pixels.
    other=deepcopy(value);other['uncertain_regions'][0]['polygon']=[[.6,.6],[1,.6],[1,1],[.6,1]]
    assert np.array_equal(np.asarray(run(other,mask,old).support),np.asarray(out.support))


def test_protected_visible_overlapping_silhouette_is_unsupported():
    value,mask,old=case()
    value['preservation_regions']=[{'polygon':[[0,0],[.5,0],[.5,.5],[0,.5]],'reason':'visible protected fixture'}]
    with pytest.raises(UnsupportedReconstruction,match='silhouette_protected_visible'):run(value,mask,old)
    with pytest.raises(UnsupportedReconstruction,match='outside_prior_authority'):
        run(case()[0],mask,Image.new('L',mask.size))


def test_new_effects_and_uncertainty_never_expand_previous_authority():
    value,mask,old=case();value['subjects'][0]['effect_assessment']='attributed'
    value['effects']=[{'subject_id':'s1','surface_id':'table','kind':'cast_shadow',
        'polygon':[[.1,.1],[.7,.1],[.7,.7],[.1,.7]],'reason':'runtime fixture effect'}]
    out=run(value,mask,old)
    assert np.array_equal(np.asarray(out.support),np.asarray(old))
    assert out.diagnostics['new_effect_pixels_outside_prior_authority_preserved']>0
    assert out.diagnostics['coverage'][0]['observed_effect_pixels_outside_execution']>0
    assert out.diagnostics['unresolved_subject_ids']==['s1']


def test_receiving_surface_or_attribution_gaps_exclude_effect_pixels_but_not_selected_mask():
    value,mask,old=case();a=np.asarray(old).copy();a[15:20,4:12]=255;old=Image.fromarray(a)
    value['surfaces'][0]['polygon']=[[0,0],[1,0],[1,.4],[0,.4]]
    out=run(value,mask,old)
    assert np.all(np.asarray(out.support)[15:20,4:12]==0)
    assert out.diagnostics['excluded_prior_authority_pixels']==40
    assert np.all(np.asarray(out.support)[np.asarray(mask)>0]==255)


def test_runtime_preparation_uses_portable_references_and_original_preflight_hash(tmp_path,monkeypatch):
    from pathlib import Path
    from dataevolver.testing.test_training_rsi_domain import fixture_domain,observe,proposal
    from dataevolver.testing.test_surface_effects import observed,masks
    from dataevolver.harness.adapters import training_rsi as module
    from dataevolver.harness.state_store import atomic_write_json
    from dataevolver.dataset.supervision import write_image,artifact_closure
    from dataevolver.dataset.contracts import ArtifactRef
    from dataevolver.dataset.surface_effects import compile_surface_artifacts
    domain,state,trial=fixture_domain(tmp_path);domain.task['selected_subject_ids']=['s1']
    domain.value['surface_capability']={'typed_preflight':{},'reconstruction':{'fixture':True},
        'authority_separation':{'fixture':True},'candidate_method':True}
    # Nonempty identity enables typed routing, independently of fixture hashes.
    domain.value['surface_capability']['typed_preflight']={'fixture':True}
    domain.contract.update(models={'fixed':'fixture'},stores={'data':'DATA','bundle':'BUNDLE'})
    domain.contract['production']['recipe']={'width':32,'height':32}
    store=domain.scope('baseline');source=write_image(store,'data','source.png',Image.new('RGB',(32,32),(10,20,30)))
    rejected=write_image(store,'data',domain.family+'/background_completed.png',Image.new('RGB',(32,32),(11,21,31)))
    old=Image.new('L',(32,32));old.paste(255,(8,8,21,21))
    prior=write_image(store,'data',domain.family+'/surface-repair/support.png',old)
    instances=[{'subject_id':sid,'mask':write_image(store,'data',sid+'.png',im).to_dict()} for sid,im in masks().items()]
    atomic_write_json(store.roots['data']/'families'/domain.family/'mask-list.json',{'instances':instances})
    trial.update(source=source.to_dict(),candidate=rejected.to_dict(),media=[{},{},{},{'artifact':prior.to_dict()}])
    atomic_write_json(Path(state.data['last_stage_result']['evidence_refs'][0]),trial);observe(domain,state,trial)
    value=observed();value['schema_version']=OBSERVATION_VERSION
    observed_path=Path(state.data['rsi_observations'][0]['stage_result']['evidence_refs'][0])
    compiled=domain.compile_surface_support(trial,value,.1)
    atomic_write_json(observed_path,{'tool':'inspect_surfaces','source':trial['source'],'candidate':trial['candidate'],
        'existing_authorized_support':prior.to_dict(),'output':value,'support_feasibility':compiled.diagnostics})
    p=proposal();p['patch']={'kind':'surface_context_reconstruction','parameters_json':json.dumps({'observation_id':'probe-fixture','context_padding_fraction':.1})}
    intervention=domain.compile_intervention(p,state)
    original=intervention.candidate_delta['parameters']['preflight_receipt']
    domain.value['seed'].update(config_sha256='a'*64,model_records=[{'request':{'request_id':'fixture-'+phase},'receipt_hash':'b'*64} for phase in module.EARLY_PHASES])
    monkeypatch.setattr(module,'TrainingSupervisionOperator',lambda **kw:kw)
    prepared,candidate_store,inner=module.TrainingRSITrialOperator(domain)._prepare(type('Call',(),{'parameters':intervention.parameters})())
    contractref=ArtifactRef.from_dict(prepared['production']['surface_effect_contracts'][domain.family])
    document=json.loads(candidate_store.resolve(contractref).read_text())
    assert document['proposal_preflight_sha256']==payload_sha256(original) and 'proposal_preflight' not in document
    assert document['existing_authorized_support']['store']=='rsi_evidence'
    # Simulate strict cache's source/mask copies, followed by a new output with
    # the same dataset path and a different hash from the rejected candidate.
    for ref in [source,*[ArtifactRef.from_dict(x['mask']) for x in instances]]:
        module._copy_exact(store.resolve(ref),candidate_store.roots[ref.store]/ref.path)
    write_image(candidate_store,'data',rejected.path,Image.new('RGB',(32,32),(200,210,220)))
    result=compile_surface_artifacts(candidate_store,'data',domain.family,source,
        [instances[0]],[instances[1]],document,model_size=(32,32))
    refs=list(artifact_closure({'surface':result[-1].to_dict()},candidate_store))
    assert all(ref.sha256!=rejected.sha256 or ref.store=='rsi_evidence' for ref in refs)
    assert any(ref.path.endswith('prior-support.png') for ref in refs)


def test_registered_preflight_accepts_a_verified_cpu_recheck_as_input(tmp_path):
    from pathlib import Path
    from dataevolver.testing.test_training_rsi_domain import fixture_domain,observe
    from dataevolver.harness.state_store import atomic_write_json
    domain,state,trial=fixture_domain(tmp_path);observe(domain,state,trial)
    domain.value['surface_capability']={'typed_preflight':{'fixture':True},'authority_separation':{'fixture':True}}
    path=Path(state.data['rsi_observations'][0]['stage_result']['evidence_refs'][0])
    value=case()[0]
    atomic_write_json(path,{'tool':'recheck_surface_support','source':trial['source'],'candidate':trial['candidate'],
        'output':value,'support_feasibility':{'supported':True}})
    _,parameters=domain.compile_observation({'observation':{'tool':'preflight_reconstruction',
        'parameters_json':json.dumps({'observation_id':'probe-fixture','context_padding_fraction':.03})}},state)
    assert parameters['parameters']['observation_id']=='probe-fixture'
    assert parameters['parameters']['context_padding_fraction']==.03
    assert parameters['parameters']['path']==str(path)
