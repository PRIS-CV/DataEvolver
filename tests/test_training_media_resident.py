"""Resident transport engineering fixtures; no model inference or quality claims."""
import json
import os
import socket
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_training_media_worker import provider as provider, req
from dataevolver.adapters import training_media_worker as worker
from dataevolver.adapters import training_media_resident as resident
from dataevolver.adapters.model_gateway import ModelError
from dataevolver.dataset.contracts import file_sha256,payload_sha256


@pytest.fixture
def deployment(provider,tmp_path,monkeypatch):
    provider.config['resource_profile'].update(cpu_offload=False,max_utilization=100)
    handler=tmp_path/'handler.py';handler.write_text('# fixture only\n')
    document={'deployment_id':'fixture-resident','pid':os.getpid(),
        'process_start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split(') ',1)[1].split()[19],
        'gpu_uuids':['gpu-a','gpu-b'],'socket_path':str(tmp_path/'fixture.sock'),
        'handler_path':str(handler),'handler_sha256':file_sha256(handler),
        'resident_implementation_sha256':file_sha256(Path(resident.__file__)),
        'models':{'qwen21':{'model':provider.profile.model,'path':os.environ['FIXTURE_MODEL'],
            'assets_sha256':payload_sha256(provider.config['file_hashes']),'device_map':{'transformer':1,'text_encoder':0}}},
        'authorized_job_roots':[str(provider.root)],'authorized_dataset_roots':[str(provider.store.roots['data'])]}
    path=tmp_path/'identity.json';path.write_text(json.dumps(document))
    provider.config.update(execution_device='cuda',resident={'socket_env':'FIXTURE_SOCKET','identity_env':'FIXTURE_IDENTITY',
        'identity_sha256':file_sha256(path),'implementation_sha256':file_sha256(Path(resident.__file__)),
        'activation_reserve_mib':[16000,16000]})
    monkeypatch.setenv('FIXTURE_SOCKET',document['socket_path']);monkeypatch.setenv('FIXTURE_IDENTITY',str(path))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','gpu-a,gpu-b')
    status={'state':'READY','pid':os.getpid(),'cuda_visible_devices':'gpu-a,gpu-b',
        'models':{'qwen21':{'ready':True,'device_map':document['models']['qwen21']['device_map']}},
        'memory':[{'device':0,'allocated_mib':10000,'reserved_mib':14000},
                  {'device':1,'allocated_mib':10000,'reserved_mib':14000}]}
    return provider,path,document,status


def test_resident_identity_pins_assets_handler_and_implementation(deployment):
    p,path,doc,status=deployment
    worker.validate_worker(p.config);assert resident.identity(p.config)['pid']==os.getpid()
    Path(doc['handler_path']).write_text('changed')
    with pytest.raises(ModelError,match='handler changed'):resident.identity(p.config)


def test_resident_configuration_cannot_reload_or_enable_offloading(deployment):
    p,*_=deployment
    p.config['shard_max_memory_mib']=[48000,48000]
    with pytest.raises(ValueError,match='actual frozen placement'):worker.validate_worker(p.config)
    del p.config['shard_max_memory_mib'];p.config['resource_profile']['cpu_offload']=True
    with pytest.raises(ValueError,match='without offloading'):worker.validate_worker(p.config)


def test_resident_admission_counts_only_own_reusable_cache(deployment,monkeypatch):
    p,path,doc,status=deployment
    monkeypatch.setattr(resident.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='0, gpu-a, 12000, 100\n1, gpu-b, 18000, 100'))
    observed=resident.resident_observation(p.config,status=status)
    assert observed['devices'][0]['usable_activation_mib']==16000
    status['memory'][0]['allocated_mib']=14000
    with pytest.raises(ModelError) as error:resident.resident_observation(p.config,status=status)
    assert error.value.code=='RESOURCE_BUSY'


def test_restart_or_placement_change_is_not_a_compatible_deployment(deployment,monkeypatch):
    p,path,doc,status=deployment
    status['pid']+=1
    with pytest.raises(ModelError,match='process, placement'):resident.resident_observation(p.config,status=status)


def test_single_gpu_admission_never_uses_unlisted_gpu1_capacity(deployment,monkeypatch):
    p,path,doc,status=deployment
    doc['gpu_uuids']=['gpu-a'];path.write_text(json.dumps(doc))
    p.config['resident'].update(identity_sha256=file_sha256(path),activation_reserve_mib=[16000])
    status['cuda_visible_devices']='gpu-a';monkeypatch.setenv('CUDA_VISIBLE_DEVICES','gpu-a')
    monkeypatch.setattr(resident.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='0, gpu-a, 11000, 100\n1, gpu-b, 200000, 0'))
    worker.validate_worker(p.config)
    with pytest.raises(ModelError) as error:resident.resident_observation(p.config,status=status)
    assert error.value.code=='RESOURCE_BUSY'
    status['memory'][0]['allocated_mib']=9000
    assert len(resident.resident_observation(p.config,status=status)['devices'])==1


def test_cpu_resident_uses_host_memory_without_hiding_shared_host_cuda(deployment,monkeypatch):
    p,path,doc,status=deployment
    p.config.update(execution_device='cpu')
    p.config['resident']['activation_reserve_mib']=[]
    doc['models']['qwen21']['device_map']={'':'cpu'};path.write_text(json.dumps(doc))
    p.config['resident']['identity_sha256']=file_sha256(path)
    status['models']['qwen21']['device_map']={'':'cpu'}
    monkeypatch.setattr(worker,'_available_cpu_mib',lambda:100000)
    monkeypatch.setattr(resident.subprocess,'run',lambda *a,**k:pytest.fail('CPU admission touched another GPU'))
    worker.validate_worker(p.config)
    assert resident.resident_observation(p.config,status=status)['device']=='resident_cpu'
    status['models']['qwen21']['device_map']={'':0}
    with pytest.raises(ModelError,match='process, placement'):resident.resident_observation(p.config,status=status)


def test_cpu_deployment_observation_still_uses_resident_identity(deployment,monkeypatch):
    p,*_=deployment;p.config['execution_device']='cpu'
    monkeypatch.setattr(resident,'resident_observation',lambda c:{'device':'resident_cpu','pid':123})
    monkeypatch.setattr(worker,'cpu_observation',lambda *a:pytest.fail('lost resident identity'))
    assert worker.deployment_observation(p.config)['pid']==123


def test_hardware_binding_preserves_assets_and_refuses_gpu1_tensor_placement(deployment,tmp_path):
    import copy
    from dataevolver.harness.training_rsi_run import bind_resident_deployment
    p,path,doc,status=deployment
    doc['gpu_uuids']=['gpu-a'];doc['models']['qwen21']['device_map']={'transformer':'cuda:0'}
    path.write_text(json.dumps(doc))
    prompt=tmp_path/'prompt.json';prompt.write_text(json.dumps({'pid':doc['pid'],'models':{}}))
    target={'models':{'workers':{'image':copy.deepcopy(p.config)}}};env={}
    new={'media_identity':str(path),'prompt_identity':str(prompt),'gpu_uuids':['gpu-a'],
         'activation_reserve_mib':{'qwen21':[16000]}}
    old_assets=copy.deepcopy(target['models']['workers']['image']['file_hashes'])
    bind_resident_deployment(target,env,new)
    assert target['models']['workers']['image']['file_hashes']==old_assets
    assert target['models']['workers']['image']['resident']['activation_reserve_mib']==[16000]
    assert env['CUDA_VISIBLE_DEVICES']=='gpu-a'
    doc['models']['qwen21']['device_map']={'transformer':'cuda:1'};path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='only physical GPU0'):bind_resident_deployment(target,env,new)


def test_tensor_placement_without_hf_metadata_is_derived_not_assumed():
    model=SimpleNamespace(parameters=lambda:iter([SimpleNamespace(device='cpu')]))
    assert worker.actual_device_map(model)=={'':'cpu'}
    model=SimpleNamespace(parameters=lambda:iter([SimpleNamespace(device='cpu'),SimpleNamespace(device='cuda:0')]))
    with pytest.raises(ValueError,match='single verified'):worker.actual_device_map(model)


def test_sam_cpu_tracker_does_not_disqualify_gpu0_detector_binding(deployment,tmp_path):
    import copy
    from dataevolver.harness.training_rsi_run import bind_resident_deployment
    p,path,doc,status=deployment
    doc['gpu_uuids']=['gpu-a']
    model=doc['models'].pop('qwen21');model['device_map']={'detector':'cuda:0','unused_video_tracker':'cpu'}
    doc['models']['sam3_instances']=model;path.write_text(json.dumps(doc))
    prompt=tmp_path/'prompt.json';prompt.write_text(json.dumps({'pid':doc['pid'],'models':{}}))
    config=copy.deepcopy(p.config);config['kind']='sam3_instances'
    target={'models':{'workers':{'sam':config}}}
    bind_resident_deployment(target,{}, {'media_identity':str(path),'prompt_identity':str(prompt),
        'gpu_uuids':['gpu-a'],'activation_reserve_mib':{'sam3_instances':[8192]}})
    assert target['models']['workers']['sam']['execution_device']=='cuda'


def test_completed_native_bypasses_resident_health_and_loading(deployment,monkeypatch):
    p,*_=deployment;request=req();digest='a'*64;p.root.mkdir()
    from dataevolver.dataset.supervision import write_image
    from PIL import Image
    ref=write_image(p.store,'data','fixture.png',Image.new('RGB',(4,4),'white'))
    response={'model':p.profile.model,'output':{'artifact':ref.to_dict()},'usage':{},'finish_reason':'stop'}
    p.path(request).write_text(json.dumps({'binding':p.binding(digest),'sha256':payload_sha256(response),'response':response}))
    monkeypatch.setattr(resident,'resident_rpc',lambda *a,**k:pytest.fail('cache contacted resident socket'))
    assert p.invoke(request,digest)==response


def test_provider_routes_to_resident_without_starting_a_subprocess(deployment,monkeypatch):
    p,path,doc,status=deployment
    monkeypatch.setattr(worker,'deployment_observation',lambda c:{'device':'resident_cuda'})
    monkeypatch.setattr(worker.subprocess,'run',lambda *a,**k:pytest.fail('resident call started subprocess'))
    seen=[]
    def rpc(config,message,**kw):
        seen.append(message)
        job=json.loads(Path(message['job_path']).read_text())
        response={'model':p.profile.model,'output':{},'usage':{},'finish_reason':'stop'}
        Path(job['native_output']).write_text(json.dumps({'binding':job['binding'],'sha256':payload_sha256(response),'response':response}))
        return {'ok':True}
    monkeypatch.setattr(resident,'resident_rpc',rpc)
    p.invoke(req(),'b'*64)
    assert len(seen)==1 and seen[0]['operation']=='execute_payload'


def test_socket_timeout_after_submission_is_ambiguous_and_never_retried(deployment):
    p,path,doc,status=deployment;server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.bind(doc['socket_path']);server.listen(1)
    received=[];release=threading.Event()
    def accept():
        with server:
            conn,_=server.accept()
            with conn:received.append(conn.recv(65536));release.wait(2)
    thread=threading.Thread(target=accept);thread.start()
    try:
        with pytest.raises(ModelError) as error:resident.resident_rpc(p.config,{'operation':'execute_payload'},timeout=.03)
        assert error.value.code=='AMBIGUOUS_EFFECT' and not error.value.retry_safe
    finally:release.set();thread.join()
    assert len(received)==1


@pytest.mark.parametrize('kind',['qwen21','qwen38_review','sam3_instances'])
def test_native_execution_reuses_actual_object_and_does_not_reset_threads(tmp_path,monkeypatch,kind):
    component=object();captured=[]
    payload={'config':{'resident':{'fixture':True},'kind':kind,'execution_device':'cuda'},
        'request':{'seed':1,'request_id':'fixture','messages':[{'content':'{}'}]},
        'profile':{'model':'fixture','revision':'fixture'},'native_output':str(tmp_path/'native.json'),'binding':'fixture'}
    p=tmp_path/'job.json';p.write_text(json.dumps(payload))
    monkeypatch.setattr(worker,'_verified_runtime',lambda *a:{})
    fake_cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:2,
        max_memory_allocated=lambda i:100,get_device_name=lambda i:'fixture')
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=fake_cuda,manual_seed=lambda s:None))
    def run(payload,parameters,runtime,*,components):captured.append(components);return {},{},'stop'
    monkeypatch.setattr(worker,{'qwen21':'_qwen_image','qwen38_review':'_qwen_review','sam3_instances':'_sam_instances'}[kind],run)
    worker.run_payload(p,resident_models={kind:component},resident_status={'fixture':True})
    assert captured==[component]
    assert json.loads(Path(payload['native_output']).read_text())['response']['runtime']['weights_reloaded'] is False


def test_native_intent_blocks_repeated_submission_before_inference(deployment,monkeypatch):
    p,path,doc,status=deployment;p.root.mkdir();request=req();digest='c'*64
    native=p.path(request);intent=native.with_suffix('.resident.intent.json');intent.write_text('{}')
    payload={'config':p.config,'request':request.to_dict(),'request_hash':digest,'profile':p.profile.to_dict(),
        'dataset_store':'data','dataset_root':str(p.store.roots['data']),'output_prefix':'models/'+digest,
        'native_output':str(native),'binding':p.binding(digest)}
    job=p.root/(request.request_id+'.job.json');job.write_text(json.dumps(payload))
    monkeypatch.setattr(worker,'run_payload',lambda *a,**k:pytest.fail('duplicate inference'))
    import time
    with pytest.raises(ModelError) as error:resident.execute_resident_job({'job_path':str(job),'identity_sha256':file_sha256(path),'deadline_epoch':time.time()+10},{},status,p.root,path)
    assert error.value.code=='AMBIGUOUS_EFFECT'


def test_resident_loop_survives_malformed_client_and_failed_request(deployment,monkeypatch,tmp_path):
    p,path,doc,status=deployment;status['inference_requests']=0;models={'fixture':object()}
    fake_cuda=SimpleNamespace(device_count=lambda:2,memory_allocated=lambda i:100,memory_reserved=lambda i:200)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=fake_cuda))
    seen=[]
    def fail(message,actual,*args):
        seen.append(actual['fixture']);raise ValueError('fixture model failure')
    monkeypatch.setattr(resident,'execute_resident_job',fail)
    server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.bind(doc['socket_path']);server.listen(4)
    def run():
        try:resident.serve_resident(server,models,status,tmp_path,path)
        except OSError:pass  # Only the fixture closes its listener.
    thread=threading.Thread(target=run);thread.start()
    def call(raw):
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
            client.settimeout(2);client.connect(doc['socket_path']);client.sendall(raw);return json.loads(client.recv(65536))
    try:
        assert not call(b'not-json\n')['ok']
        server.settimeout(.05)
        assert not call(b'{"operation":"execute_payload"}')['ok']
        health=call(b'{"operation":"health"}');assert health['ok'] and health['status']['state']=='READY'
        assert seen==[models['fixture']]
    finally:server.close();thread.join(timeout=2)
    assert not thread.is_alive()


def test_controller_waits_for_activation_capacity_without_weight_loading(deployment,monkeypatch,tmp_path):
    from dataevolver.harness.adapters import training_supervision as adapter
    import time
    p,path,doc,status=deployment;op=object.__new__(adapter.TrainingSupervisionOperator)
    op.custom_wait=None;op.policy=SimpleNamespace(bindings={'training.generate':'image'})
    op.contract={'models':{'workers':{'image':p.config}},'production':{'resource_deadline_epoch':time.time()+30}}
    observations=[]
    monkeypatch.setattr(resident,'resident_observation',lambda c:observations.append(True) or {'fixture':True})
    monkeypatch.setattr(adapter.time,'sleep',lambda s:None)
    op._wait(SimpleNamespace(work_dir=str(tmp_path)),'training.generate')
    assert len(observations)==2
    assert json.loads((tmp_path/'resource-status.json').read_text())['state']=='READY_RESIDENT'


def test_server_uses_bound_model_env_and_recovers_native_without_another_execution(deployment,monkeypatch):
    import time
    p,path,doc,status=deployment;p.root.mkdir();request=req();digest='e'*64
    native=p.path(request)
    payload={'config':p.config,'request':request.to_dict(),'request_hash':digest,'profile':p.profile.to_dict(),
        'dataset_store':'data','dataset_root':str(p.store.roots['data']),'output_prefix':'models/'+digest,
        'native_output':str(native),'model_path':doc['models']['qwen21']['path'],'binding':p.binding(digest)}
    job=p.root/(request.request_id+'.job.json');job.write_text(json.dumps(payload));seen=[]
    monkeypatch.delenv(p.config['model_path_env'])
    def run(job,*,resident_models,resident_status):
        assert os.environ['FIXTURE_MODEL']==doc['models']['qwen21']['path']
        seen.append(resident_models);native.write_text('{}')
    monkeypatch.setattr(worker,'run_payload',run)
    message={'job_path':str(job),'identity_sha256':file_sha256(path),'deadline_epoch':time.time()+10}
    models={'fixture':object()}
    first=resident.execute_resident_job(message,models,status,p.root,path)
    second=resident.execute_resident_job(message,models,status,p.root,path)
    assert first['weights_reloaded'] is False and second['reused_native'] is True
    assert seen==[models]
