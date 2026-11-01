"""Native transport fixtures; these do not execute image or vision models."""
import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

from dataevolver.adapters.model_contracts import ModelProfile, ModelRequest
from dataevolver.adapters.model_gateway import ModelError
from dataevolver.adapters import training_media_worker as worker
from dataevolver.dataset.contracts import ArtifactStore, payload_sha256, file_sha256
from dataevolver.dataset.supervision import write_image


@pytest.fixture
def provider(tmp_path,monkeypatch):
    model=tmp_path/"model";model.mkdir();(model/"config.json").write_text("{}");assets={"config.json":file_sha256(model/"config.json")}
    config={"kind":"qwen21","model_path_env":"FIXTURE_MODEL","python_env":"FIXTURE_PYTHON","file_hashes":assets,
      "runtime_versions":{"python":"fixture"},"implementation_sha256":file_sha256(Path(worker.__file__)),
      "threads":1,"max_images":4,"image_side":1024,"max_input_tokens":8192,
      "resource_profile":{"min_free_mib":64000,"max_utilization":25,"cpu_offload":True,"estimate_basis":"fixture only"}}
    profile=ModelProfile("image","local_python","Qwen/Qwen-Image-2.1","sha256:"+payload_sha256(assets),("json","vision"))
    store=ArtifactStore({"data":tmp_path/"data"})
    monkeypatch.setenv("FIXTURE_MODEL",str(model));monkeypatch.setenv("FIXTURE_PYTHON",sys.executable)
    return worker.TrainingMediaProvider(profile,config,store,"data",tmp_path/"native")


def req():
    return ModelRequest("fixture-request","training.scene",({"role":"user","content":"{}"},),
       {"type":"object","properties":{"artifact":{"type":"object"}},"required":["artifact"],"additionalProperties":False})


@pytest.mark.parametrize('unit_rule', ['image_subject', 'video_frame_subject'])
def test_native_usage_keeps_sam_statistics_and_rejects_changed_binding(provider, unit_rule):
    record = {'request_hash': 'e'*64, 'native_units_per_attempt': 2 if unit_rule == 'image_subject' else 6}
    provider.root.mkdir()
    details, totals = provider.audit_native_usage('audit', record, unit_rule, None)
    assert details['sam_native_usage'] is None and totals['sam_requests_without_native_accounting'] == 1
    value = {'binding': provider.binding(record['request_hash']), 'gateway_requests': 1}
    value.update({'input_images': 1, 'forward_attempts': 2} if unit_rule == 'image_subject' else
                 {'input_videos': 1, 'decoded_frames': 3, 'subjects': 2,
                  'frame_forward_attempts': 2, 'frame_subject_units_attempted': 4})
    native = provider.root/'audit.native.sam-usage.json'
    native.write_text(json.dumps(value))
    details, totals = provider.audit_native_usage('audit', record, unit_rule, None)
    assert details['sam_native_usage']['journal_sha256'] == file_sha256(native)
    assert totals == {'sam_image_subject_forwards_observed': 2 if unit_rule == 'image_subject' else 0,
        'sam_video_frame_forwards_observed': 2 if unit_rule == 'video_frame_subject' else 0,
        'sam_video_frame_subject_units_observed': 4 if unit_rule == 'video_frame_subject' else 0,
        'sam_requests_without_native_accounting': 0}
    value['binding'] = 'changed'
    native.write_text(json.dumps(value))
    with pytest.raises(ModelError, match='accounting changed'):
        provider.audit_native_usage('audit', record, unit_rule, None)


def test_committed_cache_never_checks_gpu_or_submits(provider,monkeypatch):
    digest="a"*64;request=req()
    ref=write_image(provider.store,"data","fixture.png",Image.new("RGB",(4,4),"white"))
    response={"model":provider.profile.model,"output":{"artifact":ref.to_dict()},
              "usage":{"images":1,"cost":None},"finish_reason":"stop"}
    provider.root.mkdir()
    provider.path(request).write_text(json.dumps({"binding":provider.binding(digest),"sha256":payload_sha256(response),"response":response}))
    monkeypatch.setattr(worker,"gpu_observation",lambda c:pytest.fail("cache checked GPU"))
    assert provider.invoke(request,digest)["output"]["artifact"]==ref.to_dict()
    provider.store.roots["data"].joinpath("fixture.png").write_text("corrupted")
    with pytest.raises(ModelError) as error:provider.lookup(request,digest)
    assert error.value.code=="INVALID_RECEIPT"


def test_orphaned_image_cannot_become_a_cache_hit(provider,monkeypatch):
    digest="b"*64
    monkeypatch.setattr(worker,"gpu_observation",lambda c:{"uuid":"fixture"})
    write_image(provider.store,"data","models/"+digest+"/image.png",Image.new("RGB",(4,4),"red"))
    with pytest.raises(ModelError) as error:provider.invoke(req(),digest)
    assert error.value.code=="AMBIGUOUS_EFFECT"
    assert not provider.path(req()).exists()


def test_failure_receipt_is_retained_without_resubmission(provider):
    digest="c"*64;request=req();provider.root.mkdir()
    with pytest.raises(ModelError) as error:provider.lookup(request,digest)
    assert error.value.code=="AMBIGUOUS_EFFECT"
    provider.path(request).with_suffix(".failure.json").write_text(json.dumps({"binding":provider.binding(digest),
       "code":"ValueError","message":"fixture missing subject"}))
    with pytest.raises(ModelError) as error:provider.lookup(request,digest)
    assert error.value.code=="RECORDED_FAILURE"


def test_resource_guard_uses_worker_profile(provider,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","fixture-gpu")
    monkeypatch.setattr(worker.subprocess,"run",lambda *a,**k:type("R",(),{"stdout":"0, fixture-uuid, 70000, 100"})())
    with pytest.raises(ModelError) as error:worker.gpu_observation(provider.config)
    assert error.value.code=="RESOURCE_BUSY"
    monkeypatch.setattr(worker.subprocess,"run",lambda *a,**k:type("R",(),{"stdout":"0, fixture-uuid, 70000, 0"})())
    assert worker.gpu_observation(provider.config)["uuid"]=="fixture-uuid"


def test_cpu_admission_requires_hidden_cuda_and_host_memory(provider,monkeypatch):
    monkeypatch.setattr(worker,"_available_cpu_mib",lambda:100000)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","fixture-gpu")
    with pytest.raises(ModelError,match="hidden CUDA"):worker.cpu_observation(provider.config)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    observed=worker.cpu_observation(provider.config)
    assert observed["device"]=="cpu" and observed["gpu_access"] is False
    monkeypatch.setattr(worker,"_available_cpu_mib",lambda:100)
    with pytest.raises(ModelError,match="host memory"):worker.cpu_observation(provider.config)


def test_cpu_native_payload_cannot_query_or_expose_gpu(provider,monkeypatch):
    provider.config["execution_device"]="cpu"
    provider.config["resource_profile"]["cpu_offload"]=False
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setattr(worker,"_available_cpu_mib",lambda:100000)
    monkeypatch.setattr(worker,"gpu_observation",lambda *a:pytest.fail("CPU deployment queried GPU"))
    observed={}
    def execute(*args,**kwargs):
        observed["env"]=kwargs["env"]
        return type("R",(),{"returncode":1})()
    monkeypatch.setattr(worker.subprocess,"run",execute)
    with pytest.raises(ModelError) as error:provider.invoke(req(),"d"*64)
    assert error.value.code=="LOCAL_WORKER_FAILED"
    payload=json.loads((provider.root/(req().request_id+".job.json")).read_text())
    assert observed["env"]["CUDA_VISIBLE_DEVICES"]==""
    assert payload["gpu_launch"] is None and payload["deployment_launch"]["gpu_access"] is False


def test_cpu_configuration_cannot_enable_gpu_offload(provider):
    provider.config["execution_device"]="cpu"
    with pytest.raises(ValueError,match="offloading"):worker.validate_worker(provider.config)
    provider.config["resource_profile"]["cpu_offload"]=False
    worker.validate_worker(provider.config)
    provider.config["execution_device"]="automatic"
    with pytest.raises(ValueError,match="execution device"):worker.validate_worker(provider.config)


def test_two_card_placement_checks_both_reserves_and_restores_mapping(provider,monkeypatch):
    provider.config.update(shard_max_memory_mib=[48000,48000],shard_min_free_mib=[64000,64000])
    provider.config["resource_profile"].update(cpu_offload=False,max_utilization=100)
    worker.validate_worker(provider.config)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","gpu-a,gpu-b")
    responses={"gpu-a":"0, gpu-a, 70000, 100","gpu-b":"1, gpu-b, 90000, 100"}
    monkeypatch.setattr(worker.subprocess,"run",lambda *a,**kw:type("R",(),{"stdout":responses[os.environ['CUDA_VISIBLE_DEVICES']]})())
    placement=worker.gpu_observation(provider.config)
    assert [d['uuid'] for d in placement['devices']]==['gpu-a','gpu-b']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='gpu-a,gpu-b'
    assert worker.sharded_loading(provider.config)=={'device_map':'balanced','max_memory':{0:'48000MiB',1:'48000MiB'}}
    responses['gpu-b']='1, gpu-b, 1000, 100'
    with pytest.raises(ModelError,match='resource profile'):worker.gpu_observation(provider.config)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='gpu-a,gpu-b'


def test_sharding_requires_two_mappings_and_real_activation_reserves(provider,monkeypatch):
    provider.config.update(shard_max_memory_mib=[48000,48000],shard_min_free_mib=[64000,64000])
    provider.config['resource_profile']['cpu_offload']=False
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','gpu-a')
    with pytest.raises(ModelError,match='two explicit'):worker.gpu_observation(provider.config)
    provider.config['shard_min_free_mib']=[40000,64000]
    with pytest.raises(ValueError,match='activation reserves'):worker.validate_worker(provider.config)


def test_transformers_set_loading_metadata_is_json_serializable():
    info={'missing_keys':set(),'unexpected_keys':{'fixture-b','fixture-a'},'mismatched_keys':set()}
    result=worker.loading_info_summary(info)
    assert json.loads(json.dumps(result))['unexpected_keys']==['fixture-a','fixture-b']


def test_negative_prompt_is_forwarded_and_effective_guidance_is_audited(provider,monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    class FixtureGenerator:
        def manual_seed(self,seed):return self
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(bfloat16='fixture-bf16',
        Generator=lambda **kw:FixtureGenerator(),inference_mode=nullcontext))
    captured={}
    class FakePipeline:
        @classmethod
        def from_pretrained(cls,*a,**kw):return cls()
        def to(self,device):assert device=='cpu';return self
        def __call__(self,**kw):
            captured.update(kw)
            return SimpleNamespace(images=[Image.new('RGB',(4,4),'white')])
    monkeypatch.setitem(sys.modules,'diffusers',SimpleNamespace(QwenImage21Pipeline=FakePipeline))
    (Path(os.environ['FIXTURE_MODEL'])/'model_index.json').write_text(json.dumps({'_class_name':'QwenImage21Pipeline'}))
    provider.config.update(execution_device='cpu');provider.config['resource_profile']['cpu_offload']=False
    payload={'model_path':os.environ['FIXTURE_MODEL'],'inputs':[],'config':provider.config,
             'request':{'seed':7},'dataset_store':'data','dataset_root':str(provider.store.roots['data']),'output_prefix':'fixture-guidance'}
    parameters={'prompt':'fixture-only background','width':4,'height':4,'steps':1,'true_cfg_scale':4.0,'negative_prompt':' '}
    runtime={}
    worker._qwen_image(payload,parameters,runtime)
    assert captured['negative_prompt']==' '
    assert runtime['guidance']['classifier_free_guidance_enabled'] is True


@pytest.mark.parametrize('count', [1, 2, 10])
def test_qwen21_preserves_reference_image_order(provider, monkeypatch, tmp_path, count):
    from contextlib import nullcontext
    from types import SimpleNamespace
    captured = {}
    class Generator:
        def manual_seed(self, seed):
            return self
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(bfloat16='fixture',
        Generator=lambda **kw: Generator(), inference_mode=nullcontext))
    class Pipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def to(self, device):
            return self
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(images=[Image.new('RGB', (4, 4))])
    monkeypatch.setitem(sys.modules, 'diffusers', SimpleNamespace(QwenImage21Pipeline=Pipeline))
    (Path(os.environ['FIXTURE_MODEL']) / 'model_index.json').write_text('{"_class_name":"QwenImage21Pipeline"}')
    provider.config.update(execution_device='cpu')
    provider.config['resource_profile']['cpu_offload'] = False
    inputs = []
    for i in range(count):
        path = tmp_path / f'reference-{i}.png'
        Image.new('RGB', (4, 4), (i, 0, 0)).save(path)
        inputs.append({'path': str(path)})
    payload = {'model_path': os.environ['FIXTURE_MODEL'], 'inputs': inputs, 'config': provider.config,
               'request': {'seed': 7}, 'dataset_store': 'data',
               'dataset_root': str(provider.store.roots['data']), 'output_prefix': 'fixture-ordered'}
    worker._qwen_image(payload, {'prompt': 'fixture', 'width': 4, 'height': 4,
                               'steps': 1, 'true_cfg_scale': 1.0}, {})
    images = captured['image'] if isinstance(captured['image'], list) else [captured['image']]
    assert [image.getpixel((0, 0)) for image in images] == [(i, 0, 0) for i in range(count)]
