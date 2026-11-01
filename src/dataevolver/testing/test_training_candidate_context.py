"""Production evidence mapping fixtures; no task quality labels or models."""
import copy
import hashlib
import json
import os
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dataevolver.dataset.contracts import ArtifactRef,ArtifactStore,file_sha256,payload_sha256
from dataevolver.dataset.surface_composition import validate_coverage_observation
from dataevolver.dataset.surface_effects import UnsupportedSurfaceRepair
from dataevolver.harness.authority_qualification import authenticated
from dataevolver.harness.candidate_sandbox import ProductionContextRunner,TrustedContextRunner,SandboxLimits,ProductionSandboxLimits
from dataevolver.harness.candidate_trust import CandidateTrustContext,canonical_digest
from dataevolver.harness.training_candidate_context import compile_training_evidence,sign_production_context,verify_requested_observation
from dataevolver.harness.state_store import atomic_write_json


def raster_fixture(tmp_path):
    root=tmp_path/'media';root.mkdir()
    def save(name,im):
        p=root/(name+'.png');im.save(p)
        return ArtifactRef('media',p.name,file_sha256(p),'image/png').to_dict()
    size=(9,8);white=Image.new('RGB',size,'white')
    silhouette=np.zeros((8,9),dtype=np.uint8);silhouette[2:5,2:5]=255;silhouette[3,3]=0
    prior=silhouette.copy();prior[5,4]=255
    base=[save('source',white),save('candidate',Image.new('RGB',size,'blue')),save('raw',white),
          save('support',Image.fromarray(prior)),save('blend',Image.fromarray(prior)),save('condition',white),
          save('reference',white),save('mask',Image.fromarray(silhouette))]
    labels=['original target scene','candidate background','projected edit','editable region','composition blend alpha',
            'actual editing condition','reference subject-a','instance mask subject-a']
    trial={'source':base[0],'candidate':base[1],'source_family':'fixture-a0','media':[
        {'label':label,'artifact':ref} for label,ref in zip(labels,base)],
        'patch':{'parameters':{'preflight_receipt':{'crop':{'source_crop_xyxy':[1,1,8,7]}}}}}
    polygon=[[2/8,2/7],[4/8,2/7],[4/8,4/7],[2/8,4/7]]
    observed={'schema_version':'dataevolver.removal_coverage_observation.v1','status':'unsupported',
        'reason':'Fixture outer envelope is not aperture authority.',
        'surfaces':[{'surface_id':'plane','description':'fixture plane','polygon':[[0,0],[1,0],[1,1],[0,1]]}],
        'subjects':[{'subject_id':'subject-a','surface_ids':['plane'],'effect_assessment':'none_observed',
                     'boundary_assessment':'observed','reason':'fixture aperture'}],
        'effects':[],'preservation_regions':[{'polygon':[[0,0],[1/8,0],[1/8,1/7],[0,1/7]],'reason':'outside support'}],
        'uncertain_regions':[],'removal_boundaries':[{'subject_id':'subject-a','polygon':polygon,'reason':'outer envelope only'}]}
    origin={'trial':trial,'family_records':[],'contract':{'models':{'workers':{'fixture':{'sha256':'a'*64}}},
        'production':{'recipe':{'steps':3},'tasks':[{'slot_id':'fixture','seed':7,'selected_subject_ids':['subject-a']}]}}}
    registered={'observation':{'source':base[0],'candidate':base[1],
        'existing_authorized_support':base[3],'output':observed}}
    return origin,registered,ArtifactStore({'media':root})


def test_registered_unsupported_is_structurally_readable_without_becoming_supported(tmp_path):
    _,registered,_=raster_fixture(tmp_path);observed=registered['observation']['output'];before=copy.deepcopy(observed)
    validate_coverage_observation(observed,['subject-a'],structural_only=True)
    with pytest.raises(UnsupportedSurfaceRepair):validate_coverage_observation(observed,['subject-a'])
    assert observed==before and observed['status']=='unsupported'


def test_aperture_envelope_never_grants_boundary_authority(tmp_path):
    origin,registered,store=raster_fixture(tmp_path);evidence,manifest,projection=compile_training_evidence(origin,registered,store)
    required=evidence['instances'][0]['required_boundary'];witness=evidence['observations'][0]['pixels']
    assert [3,3] in required and [3,3] not in witness
    assert [3,3] not in evidence['instances'][0]['mask']
    assert projection['boundary_envelope_added_authority_pixels']==0
    assert projection['full_protected_pixels']>0 and projection['relevant_protected_pixels']==0
    assert manifest['transform']['crop']==[1,1,7,6]
    assert manifest['frozen']['seed']==7
    assert projection['registered_status']=='unsupported' and not projection['image_submission_allowed']


@pytest.mark.parametrize('change',['candidate','source','support','dimensions','selected'])
def test_stale_real_input_bindings_are_rejected(tmp_path,change):
    origin,registered,store=raster_fixture(tmp_path)
    if change in {'candidate','source'}:registered['observation'][change]['sha256']='f'*64
    elif change=='support':registered['observation']['existing_authorized_support']={**registered['observation']['existing_authorized_support'],'sha256':'f'*64}
    elif change=='dimensions':
        p=store.roots['media']/'mask.png';Image.new('L',(8,8)).save(p)
        origin['trial']['media'][-1]['artifact']['sha256']=file_sha256(p)
    else:origin['contract']['production']['tasks'][0]['selected_subject_ids']=[]
    with pytest.raises(ValueError):compile_training_evidence(origin,registered,store)


def test_protected_pixels_and_uncertainty_cannot_become_permission(tmp_path):
    origin,registered,store=raster_fixture(tmp_path);obs=registered['observation']['output']
    obs['subjects'][0]['boundary_assessment']='uncertain';obs['removal_boundaries']=[]
    obs['preservation_regions'][0]['polygon']=[[2/8,2/7],[4/8,2/7],[4/8,4/7],[2/8,4/7]]
    obs['uncertain_regions']=[{'kind':'effect_attribution','subject_ids':['subject-a'],
        'polygon':[[2/8,2/7],[4/8,2/7],[4/8,4/7],[2/8,4/7]],'reason':'fixture ambiguity'}]
    evidence,_,_=compile_training_evidence(origin,registered,store)
    assert evidence['observations'][0]['pixels']==[]
    assert [2,2] in evidence['protected']['pixels']
    assert any(row['kind']=='boundary' and row['required'] for row in evidence['uncertainty'])
    assert any(row['kind']=='attribution' and row['required'] for row in evidence['uncertainty'])


def test_uncertain_boundary_envelope_remains_inert_diagnostic_evidence(tmp_path):
    origin,registered,store=raster_fixture(tmp_path);obs=registered['observation']['output']
    obs['subjects'][0]['boundary_assessment']='uncertain';before=copy.deepcopy(obs)
    evidence,_,projection=compile_training_evidence(origin,registered,store)
    assert evidence['observations'][0]['pixels']==[]
    assert [3,3] in evidence['instances'][0]['required_boundary']
    assert any(row['kind']=='boundary' and row['required'] for row in evidence['uncertainty'])
    assert obs==before and projection['registered_status']=='unsupported'
    with pytest.raises(ValueError):validate_coverage_observation(obs,['subject-a'])


def signed_fixture(tmp_path,source='TRUSTED_REGISTRARS={}\ndef resolve(p):return {"keys":sorted(TRUSTED_REGISTRARS)}'):
    from types import SimpleNamespace
    origin,registered,store=raster_fixture(tmp_path);evidence,manifest,_=compile_training_evidence(origin,registered,store)
    package=SimpleNamespace(compute_hash=lambda:'a'*64,implementation={'sha256':hashlib.sha256(source.encode()).hexdigest()})
    refs=[ArtifactRef('host','origin-'+str(i)+'.json',str(i)*64).to_dict() for i in range(3)]
    bindings={k:'b'*64 for k in ['trial_authority_sha256','observation_authority_sha256','request_authority_sha256',
        'trial_sha256','observation_sha256','compiler_sha256']}
    payload,context=sign_production_context(evidence,manifest,context_id='registered-context',package=package,
        request_sha256='c'*64,input_artifact=ArtifactRef('host','input.json','0'*64).to_dict(),
        family_id='regression',origin_bindings=bindings,evidence_artifacts=refs)
    return source,payload,context


def test_signature_is_host_owned_and_input_file_hash_matches_serialization(tmp_path):
    _,payload,context=signed_fixture(tmp_path)
    assert authenticated(payload,context)
    p=tmp_path/'input.json';atomic_write_json(p,payload)
    assert context.input_artifact['sha256']==file_sha256(p)
    assert context.scope=='registered_production_runtime'
    payload['evidence']['instances'][0]['mask'].append([0,0])
    assert not authenticated(payload,context)
    assert 'private' not in json.dumps(context.to_dict()).lower()


@pytest.mark.skipif(platform.system()!='Linux' or os.geteuid()!=0,reason='real Linux containment required')
def test_actual_production_context_is_distinct_from_synthetic_and_binds_exact_input(tmp_path):
    source,payload,context=signed_fixture(tmp_path);runner=ProductionContextRunner()
    result=runner.run_with_context(source,payload,context)
    assert result['status']=='COMPLETED' and result['isolation_verified']
    assert result['runtime']['context_scope']=='registered_production_runtime'
    assert result['output']=={'keys':['registered-training-evidence']}
    assert result['binding']['trusted_context_sha256']==context.compute_hash()
    with pytest.raises(ValueError):TrustedContextRunner().run_with_context(source,payload,context)
    synthetic=CandidateTrustContext(context.context_id,context.package_sha256,context.implementation_sha256,
        context.qualification_request_sha256,context.input_artifact,context.payload_sha256,context.fixture_family_id,context.registrars)
    with pytest.raises(ValueError):runner.run_with_context(source,payload,synthetic)
    with pytest.raises(ValueError):runner.run_with_context(source+'\n',payload,context)
    with pytest.raises(ValueError):runner.run_with_context(source,{**payload,'parameters':{'grant':True}},context)


@pytest.mark.parametrize('field,value',[('scope','synthetic_authority_contract'),('namespace','__builtins__'),
    ('origin_bindings',{}),('evidence_artifacts',[])])
def test_production_context_cannot_drop_origins_or_change_globals(tmp_path,field,value):
    _,_,context=signed_fixture(tmp_path)
    with pytest.raises(ValueError):replace(context,**{field:value})


def test_large_production_output_is_explicit_bounded_and_leaves_legacy_policy_unchanged():
    assert ProductionSandboxLimits(output_bytes=16777216).output_bytes==16777216
    with pytest.raises(ValueError):SandboxLimits(output_bytes=16777216)
    with pytest.raises(ValueError):ProductionSandboxLimits(output_bytes=16777217)
    with pytest.raises(ValueError):ProductionSandboxLimits(memory_bytes=2147483648)


def test_nonbinary_alpha_cannot_be_reinterpreted_as_authority(tmp_path):
    origin,registered,store=raster_fixture(tmp_path)
    p=store.roots['media']/'support.png';im=Image.open(p).copy();im.putpixel((3,3),128);im.save(p)
    origin['trial']['media'][3]['artifact']['sha256']=file_sha256(p)
    with pytest.raises(ValueError,match='binary'):compile_training_evidence(origin,registered,store)


@pytest.mark.parametrize('crop',[[0,0,10,8],[0,0,0,8],[0.,0,8,8]])
def test_original_crop_must_remain_an_exact_bounded_transform(tmp_path,crop):
    origin,registered,store=raster_fixture(tmp_path)
    origin['trial']['patch']['parameters']['preflight_receipt']['crop']['source_crop_xyxy']=crop
    with pytest.raises(ValueError,match='pixel transform'):compile_training_evidence(origin,registered,store)


def test_original_request_payload_citation_and_committed_file_digest_remain_distinct(tmp_path):
    origin,registered,_=raster_fixture(tmp_path)
    p=tmp_path/'observation.json';atomic_write_json(p,registered['observation'])
    registered.update(observation_path=str(p),observation_sha256=file_sha256(p),
                      observation_payload_sha256=payload_sha256(registered['observation']))
    assert registered['observation_sha256']!=registered['observation_payload_sha256']
    request={'source':origin['trial']['source'],'rejected_candidate':origin['trial']['candidate'],
        'evidence':[{'path':str(p),'sha256':registered['observation_payload_sha256']}]}
    verify_requested_observation(request,origin,registered)
    request['evidence'][0]['sha256']=registered['observation_sha256']
    with pytest.raises(ValueError,match='cited'):verify_requested_observation(request,origin,registered)
    request['evidence'][0]['sha256']=registered['observation_payload_sha256']
    registered['observation']['output']['reason']='tampered after original citation'
    with pytest.raises(ValueError,match='cited'):verify_requested_observation(request,origin,registered)
