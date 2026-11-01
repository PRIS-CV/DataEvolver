"""Pinned Turbo8 geometry is rejected before any model/backend construction."""
import builtins
from pathlib import Path

import pytest

from dataevolver.adapters import local_visual_worker,visual_models,training_media_worker
from dataevolver.adapters.model_contracts import ModelProfile,ModelRequest
from dataevolver.adapters.visual_models import (
    H3_LORA,
    H3_LORA_SHA256,
    LocalVisualProvider,
    SCHEMA,
    validate_h3_parameters,
    validate_visual_worker,
)
from dataevolver.dataset.contracts import ArtifactRef,ArtifactStore,canonical_json,file_sha256,payload_sha256

PARAMETERS={'prompt':'A fixed development scene','width':768,'height':768,'num_frames':22,'steps':8,'fps':24}

def config():
    return {'schema_version':SCHEMA,'kind':'h3_turbo','precision':'lightx2v-fp8-torchao',
        'model_path_env':'H3_MODEL','python_env':'H3_PYTHON','file_hashes':{'model_index.json':'0'*64},
        'runtime_versions':{'torch':'fixture','transformers':'fixture','av':'fixture'},
        'implementation_sha256':file_sha256(Path(local_visual_worker.__file__)),
        'binding_sha256':file_sha256(Path(visual_models.__file__)),
        'transport_sha256':file_sha256(Path(training_media_worker.__file__)),
        'threads':1,'max_images':1,'image_side':1024,'max_input_tokens':4096,'max_frames':256,
        'execution_device':'cuda','gpu_uuid':'GPU-fixture',
        'resource_profile':{'min_free_mib':32768,'max_utilization':100,'cpu_offload':True,'estimate_basis':'CPU fixture'},
        'dependencies':{'lora':{'path_env':'LORA','file_hashes':{H3_LORA:H3_LORA_SHA256}},
            **{name:{'path_env':name.upper(),'file_hashes':{'fixture':'0'*64}} for name in ['fp8','backend','adaln']}}}

@pytest.mark.parametrize('frames',[22,39,56,124,192,243])
def test_exact_pinned_backend_frame_counts_are_accepted(frames):
    cfg=config();validate_visual_worker(cfg)
    validate_h3_parameters({**PARAMETERS,'num_frames':frames},cfg,1)

@pytest.mark.parametrize('frames',[4,5,8,16,20,21,23,38,55,57,68,128,256,True,22.0])
def test_incompatible_frame_counts_are_rejected_instead_of_aligned(frames):
    with pytest.raises(ValueError,match=r'exact 17\*n\+5'):
        validate_h3_parameters({**PARAMETERS,'num_frames':frames},config(),1)

def test_worker_ceiling_cannot_exclude_every_valid_request():
    with pytest.raises(ValueError,match='22-frame'):validate_visual_worker({**config(),'max_frames':16})

def test_exact_frame_count_still_respects_original_ceiling_steps_fps_and_dimensions():
    for parameters,cfg,count in [({**PARAMETERS,'num_frames':124},{**config(),'max_frames':56},1),
        ({**PARAMETERS,'steps':4},config(),1),({**PARAMETERS,'fps':25},config(),1),
        ({**PARAMETERS,'width':767},config(),1),(PARAMETERS,config(),0)]:
        with pytest.raises(ValueError):validate_h3_parameters(parameters,cfg,count)

def test_provider_rejects_sixteen_frames_before_native_worker(tmp_path,monkeypatch):
    cfg=config();profile=ModelProfile('h3','local_python','lightx2v/Minimax-h3-Turbo',
        'sha256:'+payload_sha256(cfg),('json','vision'))
    provider=LocalVisualProvider(profile,cfg,ArtifactStore({'data':tmp_path/'data'}),'data',tmp_path/'native')
    request=ModelRequest('invalid-sixteen','visual.video.generate',
        ({'role':'user','content':canonical_json({**PARAMETERS,'num_frames':16})},),{'type':'object'},
        (ArtifactRef('data','image.png','0'*64,'image/png'),))
    monkeypatch.setattr(training_media_worker.TrainingMediaProvider,'invoke',lambda *_:pytest.fail('invalid geometry reached model worker'))
    with pytest.raises(ValueError,match='exact 17'):provider.invoke(request,'0'*64)
    assert not provider.root.exists()

def test_worker_rejects_sixteen_frames_before_torch_or_backend_import(monkeypatch):
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name in {'torch','lightx2v'}:pytest.fail('invalid geometry imported model/backend runtime')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(ValueError,match='exact 17'):
        local_visual_worker._h3({'config':config(),'inputs':[{}]},
            {**PARAMETERS,'num_frames':16},{},{})
