"""Independent CPU authority/solver fixtures; no generated-image quality claim."""
from copy import deepcopy
import json
import shutil

import numpy as np
import pytest
from PIL import Image

from dataevolver.dataset.surface_composition import (
    OBSERVATION_VERSION, coverage_surface_support, harmonize_arrays, harmonize_surface_edit)
from dataevolver.dataset.surface_preflight import UnsupportedReconstruction
from dataevolver.dataset.contracts import ArtifactStore, canonical_json
from dataevolver.dataset.supervision import write_image, write_artifact, artifact_closure
from dataevolver.dataset.context_reconstruction import prepare_context_condition, project_context_output


def case(offset=0):
    shape=(40,40);mask=np.zeros(shape,dtype='uint8');mask[8:13,7+offset:12+offset]=255
    value={'schema_version':OBSERVATION_VERSION,'status':'supported','reason':'independent fixture',
        'surfaces':[{'surface_id':'table','description':'fixture receiving surface','polygon':[[0,0],[1,0],[1,1],[0,1]]}],
        'subjects':[{'subject_id':'s1','surface_ids':['table'],'effect_assessment':'none_observed',
                     'boundary_assessment':'none_observed','reason':'fixture subject'}],
        'effects':[],'removal_boundaries':[],'preservation_regions':[],'uncertain_regions':[]}
    return value,Image.fromarray(mask),Image.fromarray(mask.copy())


def run(value,mask,old,**kwargs):
    return coverage_surface_support(value,[{'subject_id':'s1'}],[],size=mask.size,
        load_mask=lambda x:mask,existing_support=old,**kwargs)


@pytest.mark.parametrize('offset',[0,9,20])
def test_boundary_witnesses_never_silently_expand_authority(offset):
    value,mask,old=case(offset)
    value['subjects'][0]['boundary_assessment']='observed'
    value['removal_boundaries']=[{'subject_id':'s1','reason':'fine boundary outside mask',
        'polygon':[[.1,.1],[.9,.1],[.9,.5],[.1,.5]]}]
    with pytest.raises(UnsupportedReconstruction,match='uncovered_selected_boundary') as exc:
        run(value,mask,old)
    assert exc.value.receipt['coverage_added_authority_pixels']==0
    assert not exc.value.receipt['supported'] and exc.value.receipt['image_model_calls']==0


def test_only_fresh_attributable_effects_add_pixels_and_protection_still_wins():
    value,mask,old=case()
    value['subjects'][0].update(effect_assessment='attributed',boundary_assessment='observed')
    poly=[[.12,.12],[.5,.12],[.5,.5],[.12,.5]]
    value['effects']=[{'subject_id':'s1','surface_id':'table','kind':'cast_shadow','polygon':poly,'reason':'observed fixture effect'}]
    value['removal_boundaries']=[{'subject_id':'s1','polygon':poly,'reason':'covered fixture boundary'}]
    result=run(value,mask,old,padding_fraction=.1,model_size=(64,64))
    assert result.diagnostics['fresh_attributed_effect_expansion_pixels']>0
    assert result.diagnostics['coverage_supported'] and not result.diagnostics['complete_goal_established']
    assert result.diagnostics['coverage_added_authority_pixels']==0
    for kind in ['effect_attribution','reconstruction']:
        bad=deepcopy(value);bad['uncertain_regions']=[{'kind':kind,'subject_ids':['s1'],'polygon':poly,'reason':'uncertain fixture'}]
        with pytest.raises(UnsupportedReconstruction):run(bad,mask,old)
    bad=deepcopy(value);bad['preservation_regions']=[{'polygon':poly,'reason':'visible protected fixture'}]
    with pytest.raises(UnsupportedReconstruction,match='protected'):run(bad,mask,old)
    bad=deepcopy(value);bad['surfaces'][0]['polygon']=[[0,0],[.2,0],[.2,.2],[0,.2]]
    with pytest.raises(UnsupportedReconstruction,match='crosses_receiving_surface'):run(bad,mask,old)


def test_uncertainty_without_conflict_still_blocks_image_eligibility():
    value,mask,old=case();value['subjects'][0]['boundary_assessment']='uncertain'
    with pytest.raises(UnsupportedReconstruction,match='unresolved_removal_coverage') as exc:run(value,mask,old)
    assert exc.value.receipt['unresolved_subject_ids']==['s1']
    assert exc.value.receipt['editable_pixels']==25


@pytest.mark.parametrize('offset',[0,6])
def test_harmonic_offset_correction_preserves_generated_detail_and_source_outside(offset):
    yy,xx=np.indices((36,40));source=np.repeat((70+xx+yy)[:,:,None],3,axis=2).astype('uint8')
    generated=source+35
    generated[15:19,16+offset:20+offset]+=15
    support=np.zeros(source.shape[:2],dtype='uint8');support[8:28,8+offset:30+offset]=255
    protected=np.zeros_like(support);protected[:6]=255
    actual,proof=harmonize_arrays(source,generated,support,protected)
    assert np.array_equal(actual[support==0],source[support==0])
    assert np.array_equal(actual[protected>0],source[protected>0])
    assert np.all(actual[15:19,16+offset:20+offset]==source[15:19,16+offset:20+offset]+15)
    assert np.max(np.abs(actual[8:12,8+offset:30+offset].astype(int)-source[8:12,8+offset:30+offset]))<=1
    assert proof['unauthorized_changed_pixels']==0 and not proof['quality_verified']
    assert np.array_equal(harmonize_arrays(source,generated,support,protected)[0],actual)


def test_harmonic_solver_rejects_protected_support_and_unanchored_components():
    source=np.full((12,12,3),100,dtype='uint8');mask=np.full((12,12),255,dtype='uint8')
    with pytest.raises(ValueError,match='Dirichlet boundary'):harmonize_arrays(source,source,mask,np.zeros_like(mask))
    with pytest.raises(ValueError,match='protected'):harmonize_arrays(source,source,mask,mask)
    mask[0,0]=100
    with pytest.raises(ValueError,match='binary'):harmonize_arrays(source,source,mask,np.zeros_like(mask))


def test_distinct_surface_tones_do_not_mix_across_boundaries():
    source=np.full((24,24,3),100,dtype='uint8');generated=source.copy()
    generated[:12]+=30;generated[12:]-=30
    support=np.zeros((24,24),dtype='uint8');support[5:19,5:19]=255
    labels=np.ones((24,24),dtype='int32');labels[12:]=2
    result,proof=harmonize_arrays(source,generated,support,np.zeros_like(support),labels)
    assert np.array_equal(result,source) and proof['components']==2


def test_missing_or_ambiguous_surface_requires_new_registered_evidence():
    value,mask,old=case();value['surfaces'][0]['polygon']=[[0,0],[.1,0],[.1,.1],[0,.1]]
    with pytest.raises(UnsupportedReconstruction,match='receiving_surface_coverage'):run(value,mask,old)
    value,mask,old=case();value['surfaces'].append({**value['surfaces'][0],'surface_id':'other'})
    with pytest.raises(UnsupportedReconstruction,match='receiving_surface_coverage'):run(value,mask,old)


def artifacts(tmp_path):
    store=ArtifactStore({'data':str(tmp_path/'data')});value,mask,old=case()
    source=write_image(store,'data','source.png',Image.new('RGB',mask.size,(100,110,120)))
    rejected=write_image(store,'data','rejected.png',Image.new('RGB',mask.size,(101,111,121)))
    support=write_image(store,'data','support.png',mask)
    protected=write_image(store,'data','protected.png',Image.new('L',mask.size))
    condition=write_image(store,'data','condition.png',Image.composite(Image.new('RGB',mask.size,(127,127,127)),Image.new('RGB',mask.size,(100,110,120)),mask))
    bindings={'source':source.to_dict(),'candidate':rejected.to_dict()}
    coverage=run(value,mask,old,bindings=bindings,padding_fraction=.1,model_size=(64,64)).diagnostics
    _,transform=prepare_context_condition(store,'data','fixture',source,condition,support,{'context_padding_fraction':.1},model_size=(64,64))
    raw=write_image(store,'data','raw.png',Image.new('RGB',(64,64),(140,150,160)))
    projected,projection=project_context_output(store,'data','fixture',source,raw,transform)
    contract={'source_image':source.to_dict(),'rejected_candidate':rejected.to_dict(),
        'observation':value,'proposal_actor':'runtime_agent','support':support.to_dict(),
        'conditioning_region':support.to_dict(),'composition_alpha':support.to_dict(),
        'protected':protected.to_dict(),'diagnostics':coverage}
    ref=write_artifact(store,'data','contract.json',canonical_json(contract).encode(),'application/json')
    return store,source,projected,projection,ref,contract


def test_projection_bound_cpu_artifacts_are_portable_and_idempotent(tmp_path):
    store,source,projected,projection,ref,contract=artifacts(tmp_path)
    output,audit=harmonize_surface_edit(store,'data','fixture',source,projected,ref,projection)
    assert harmonize_surface_edit(store,'data','fixture',source,projected,ref,projection)==(output,audit)
    proof=json.loads(store.resolve(audit).read_text())
    assert proof['inverse_projection_verified'] and proof['unauthorized_changed_pixels']==0
    shutil.copytree(store.roots['data'],tmp_path/'relocated')
    relocated=ArtifactStore({'data':str(tmp_path/'relocated')})
    closure=list(artifact_closure({'audit':audit.to_dict()},relocated))
    assert source in closure and projection in closure and ref in closure
    for entry in closure:
        if entry.media_type=='image/png':Image.open(relocated.resolve(entry)).load()


@pytest.mark.parametrize('mutation',['source','candidate','observation','alpha','projection'])
def test_stale_or_inconsistent_contracts_are_rejected_before_composition(tmp_path,mutation):
    store,source,projected,projection,ref,contract=artifacts(tmp_path)
    if mutation=='source':contract['source_image']['sha256']='0'*64
    elif mutation=='candidate':contract['rejected_candidate']['sha256']='0'*64
    elif mutation=='observation':contract['observation']['reason']='different evidence'
    elif mutation=='alpha':contract['composition_alpha']=contract['protected']
    else:
        doc=json.loads(store.resolve(projection).read_text());doc['source_crop_xyxy']=[0,0,1,1]
        projection=write_artifact(store,'data','bad-projection.json',canonical_json(doc).encode(),'application/json')
    ref=write_artifact(store,'data','bad-contract.json',canonical_json(contract).encode(),'application/json')
    with pytest.raises(ValueError):harmonize_surface_edit(store,'data','bad',source,projected,ref,projection)


def test_registered_domain_requires_fresh_current_coverage_before_compiling_trial(tmp_path):
    from pathlib import Path
    from dataevolver.testing.test_training_rsi_domain import fixture_domain, observe, proposal
    from dataevolver.harness.state_store import atomic_write_json
    domain,state,trial=fixture_domain(tmp_path);domain.task['selected_subject_ids']=['s1']
    domain.value['surface_capability']={'typed_preflight':{'fixture':True},
        'reconstruction':{'fixture':True},'composition':{'fixture':True}}
    domain.contract['production']['recipe']={'width':64,'height':64}
    store=domain.scope('baseline');value,mask,old=case()
    source=write_image(store,'data','source-coverage.png',Image.new('RGB',mask.size,(100,110,120)))
    rejected=write_image(store,'data','candidate-coverage.png',Image.new('RGB',mask.size,(110,120,130)))
    prior=write_image(store,'data','support-coverage.png',old)
    instance={'subject_id':'s1','mask':write_image(store,'data','s1-mask.png',mask).to_dict()}
    atomic_write_json(store.roots['data']/'families'/domain.family/'mask-list.json',{'instances':[instance]})
    trial.update(source=source.to_dict(),candidate=rejected.to_dict(),media=[{},{},{},{'artifact':prior.to_dict()}])
    atomic_write_json(Path(state.data['last_stage_result']['evidence_refs'][0]),trial)
    observe(domain,state,trial)
    compiled=domain.compile_surface_support(trial,value,.1)
    observation=Path(state.data['rsi_observations'][0]['stage_result']['evidence_refs'][0])
    doc={'tool':'inspect_surfaces','source':trial['source'],'candidate':trial['candidate'],
        'existing_authorized_support':prior.to_dict(),'output':value,'support_feasibility':compiled.diagnostics}
    atomic_write_json(observation,doc)
    p=proposal();p['patch']={'kind':'surface_context_reconstruction','parameters_json':json.dumps({
        'observation_id':'probe-fixture','context_padding_fraction':.1})}
    result=domain.compile_intervention(p,state)
    assert result.candidate_delta['parameters']['preflight_receipt']['coverage_supported']
    assert result.candidate_delta['parameters']['preflight_receipt']['quality_verified'] is False
    stale=deepcopy(doc);stale['candidate']['sha256']='f'*64;atomic_write_json(observation,stale)
    with pytest.raises(ValueError,match='current-source'):domain.compile_intervention(p,state)
