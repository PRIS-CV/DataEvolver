"""Costed prompt refinement using two already resident Qwen PE components.

This is a model transport beneath ModelGateway, not an execution controller.
The image, review and SAM workers and their resident objects are unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from dataevolver.dataset.contracts import file_sha256, payload_sha256, identifier
from dataevolver.harness.state_store import atomic_write_json
from .model_contracts import ModelProfile, ModelRequest
from .model_gateway import ModelError
from .training_media_worker import TrainingMediaProvider

PE_MODELS = {"pe_t2i":"Qwen/Qwen-Image-2.1-PE-T2I", "pe_i2i":"Qwen/Qwen-Image-2.1-PE-I2I"}
PE_ROLES = {"training.refine_t2i":"pe_t2i", "training.refine_i2i":"pe_i2i"}
PE_T2I_SCHEMA = {"type":"object", "properties":{
    "rewritten_prompt":{"type":"string","minLength":1}, "wh_ratio":{"type":"string"}},
    "required":["rewritten_prompt","wh_ratio"], "additionalProperties":False}
PE_I2I_SCHEMA = {"type":"object", "properties":{
    "rewritten_prompt":{"type":"string","minLength":1}, "wh_ratio":{"type":"string"},
    "ratio_follow":{"type":"string"}},
    "required":["rewritten_prompt","wh_ratio","ratio_follow"], "additionalProperties":False}
for _schema in (PE_T2I_SCHEMA,PE_I2I_SCHEMA):
    _schema['properties']['raw_text_artifact']={'type':'object'}


def parse_refinement(text, kind):
    # Keep raw thinking in the native log, never substitute it for the answer.
    answer=text.rsplit("</think>",1)[-1].strip()
    if answer.startswith("```json\n") and answer.endswith("```"):answer=answer[8:-3].strip()
    value=json.loads(answer)
    if 'raw_text_artifact' in value:raise ValueError('raw evidence references are owned by the transport, not the model')
    from .model_gateway import validate_response_schema
    validate_response_schema(PE_T2I_SCHEMA if kind=="pe_t2i" else PE_I2I_SCHEMA,value)
    if not value["rewritten_prompt"].strip():raise ValueError("empty PE execution prompt")
    if kind=="pe_i2i" and bool(value["wh_ratio"])==bool(value["ratio_follow"]):
        raise ValueError("PE editing aspect ratio choices must be exclusive")
    return value


def execute_resident_pe_job(message,components,identity_path):
    """Execute one PE effect in the serial combined host, without loading weights."""
    path=Path(identity_path).resolve()
    if file_sha256(path)!=message.get('deployment_sha256'):
        raise ValueError('PE deployment binding changed')
    deployment=json.loads(path.read_text())
    if deployment['pid']!=os.getpid():raise ValueError('PE resident process changed')
    if file_sha256(Path(deployment['handler_path']))!=deployment['implementation_sha256']:
        raise ValueError('PE registered handler changed')
    job=Path(message['job_path']).resolve()
    if not job.is_relative_to(Path(deployment['authorized_root']).resolve()):
        raise ValueError('PE job escaped authorized root')
    payload=json.loads(job.read_text());request=ModelRequest.from_dict(payload['request'])
    if payload['config']['deployment_sha256']!=message['deployment_sha256']:
        raise ValueError('PE request identity changed')
    native=Path(payload['native_output']).resolve()
    if native.parent!=job.parent or native.name!=request.request_id+'.native.json':
        raise ValueError('PE native path changed')
    if native.exists():return {'ok':True,'reused_native':True}
    intent=native.with_suffix('.resident.intent.json');failure=native.with_suffix('.failure.json')
    orphan=Path(payload['dataset_root'])/payload['output_prefix']
    if intent.exists() or failure.exists() or (orphan.exists() and any(orphan.iterdir())):
        raise ModelError('AMBIGUOUS_EFFECT','PE effect already started; reconcile native journal')
    if time.time()>message['deadline_epoch']:raise ValueError('PE request expired')
    native.parent.mkdir(parents=True,exist_ok=True)
    with intent.open('x') as handle:json.dump({'binding':payload['binding'],'started_at':time.time(),'pid':os.getpid()},handle)
    try:
        _execute_pe(payload,components[payload['config']['kind']],deployment)
        atomic_write_json(intent,{'binding':payload['binding'],'completed_at':time.time(),'pid':os.getpid()})
        return {'ok':True,'resident_pid':os.getpid(),'weights_reloaded':False}
    except Exception as exc:
        if not native.exists():atomic_write_json(failure,{'binding':payload['binding'],'code':getattr(exc,'code',type(exc).__name__),
            'message':str(exc)[:2000]})
        raise


def validate_pe_worker(config):
    required={"kind","deployment_env","deployment_sha256","file_hashes","implementation_sha256",
              "activation_reserve_mib","max_input_tokens"}
    if set(config)!=required or config["kind"] not in PE_MODELS:raise ValueError("unregistered prompt refinement worker")
    identifier(config["deployment_env"])
    if config["implementation_sha256"]!=file_sha256(Path(__file__)):raise ValueError("PE implementation changed")
    if not config["file_hashes"] or any(type(config[k]) is not int or config[k]<1
            for k in ["activation_reserve_mib","max_input_tokens"]):raise ValueError("PE worker lacks frozen assets or limits")


def pe_identity(config):
    validate_pe_worker(config)
    path=Path(os.environ[config["deployment_env"]])
    if file_sha256(path)!=config["deployment_sha256"]:raise ModelError("CONFIG_MISSING","PE deployment changed")
    value=json.loads(path.read_text());model=value["models"][config["kind"]]
    if model["model"]!=PE_MODELS[config["kind"]] or model["assets_sha256"]!=payload_sha256(config["file_hashes"]):
        raise ModelError("CONFIG_MISSING","PE model identity differs from frozen assets")
    if value["implementation_sha256"]!=config["implementation_sha256"]:
        raise ModelError("CONFIG_MISSING","resident PE source differs from caller")
    if file_sha256(Path(value["handler_path"]))!=config["implementation_sha256"]:
        raise ModelError("CONFIG_MISSING","registered PE handler source changed")
    ticks=(Path("/proc")/str(value["pid"])/"stat").read_text().rsplit(")",1)[1].split()[19]
    if ticks!=value["process_start_ticks"]:raise ModelError("RESOURCE_UNAVAILABLE","PE resident restarted")
    return value


def pe_observation(config):
    expected=pe_identity(config);status=json.loads(Path(expected["status_path"]).read_text())
    if status["pid"]!=expected["pid"] or status["state"]!="READY":
        raise ModelError("RESOURCE_BUSY","PE resident is not ready",retry_safe=True)
    model=expected['models'][config['kind']]
    if set(map(str,model['device_map'].values()))=={'cpu'}:
        available=next(int(line.split()[1])//1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        if available<config['activation_reserve_mib']:raise ModelError('RESOURCE_BUSY','PE CPU activation reserve unavailable',retry_safe=True)
        return {'resident_pid':expected['pid'],'device':'resident_cpu','available_mib':available,
            'required_activation_mib':config['activation_reserve_mib'],'weights_already_loaded':True}
    result=subprocess.run(["nvidia-smi","-i",expected["gpu_uuid"],"--query-gpu=memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=20)
    free,util=[int(v.strip()) for v in result.stdout.strip().split(",")]
    own=max(0,int(status["reserved_mib"]-status["allocated_mib"]))
    if free+own<config["activation_reserve_mib"] or util>100:
        raise ModelError("RESOURCE_BUSY","PE activation reserve unavailable",retry_safe=True)
    return {"resident_pid":expected["pid"],"gpu_uuid":expected["gpu_uuid"],"free_mib":free,
        "owned_reusable_cache_mib":own,"required_activation_mib":config["activation_reserve_mib"],"weights_already_loaded":True}


class PromptEnhancementProvider(TrainingMediaProvider):
    """Reuse the media worker's durable lookup and binding, with a PE transport."""
    def __init__(self,profile,config,store,dataset_store,receipt_root):
        validate_pe_worker(config)
        if profile.model!=PE_MODELS[config["kind"]] or profile.revision!="sha256:"+payload_sha256(config["file_hashes"]):
            raise ValueError("PE profile differs from selected component")
        self.profile,self.config,self.store=profile,config,store
        self.dataset_store,self.root=dataset_store,Path(receipt_root)

    def invoke(self,request,request_hash):
        if self.path(request).exists():return self.lookup(request,request_hash)
        expected=pe_identity(self.config);placement=pe_observation(self.config)
        if (request.role not in PE_ROLES or PE_ROLES[request.role]!=self.config["kind"] or
                (self.config["kind"]=="pe_t2i" and request.artifacts) or
                (self.config["kind"]=="pe_i2i" and len(request.artifacts)!=1)):
            raise ModelError("INVALID_INPUT","PE role or image count differs from frozen mode")
        self.root.mkdir(parents=True,exist_ok=True)
        job=self.root/(request.request_id+".job.json")
        payload={"binding":self.binding(request_hash),"request":request.to_dict(),"request_hash":request_hash,
            "profile":self.profile.to_dict(),"config":self.config,"native_output":str(self.path(request)),
            "dataset_store":self.dataset_store,"dataset_root":str(self.store.roots[self.dataset_store]),
            "output_prefix":"models/"+request_hash,
            "inputs":[{"ref":ref.to_dict(),"path":str(self.store.resolve(ref))} for ref in request.artifacts],
            "deployment_launch":placement}
        atomic_write_json(job,payload)
        sent=False
        try:
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
                client.settimeout(request.timeout_seconds);client.connect(expected["socket_path"])
                message={"job_path":str(job.resolve()),"deployment_sha256":self.config["deployment_sha256"],
                    "deadline_epoch":time.time()+request.timeout_seconds}
                sent=True;client.sendall(json.dumps(message).encode());chunks=[];size=0
                while True:
                    data=client.recv(65536)
                    if not data:break
                    chunks.append(data);size+=len(data)
                    if size>1024*1024:raise ValueError("PE response exceeded bound")
                json.loads(b"".join(chunks))
        except (OSError,ValueError) as exc:
            if self.path(request).exists():return self.lookup(request,request_hash)
            raise ModelError("AMBIGUOUS_EFFECT" if sent else "RESOURCE_UNAVAILABLE",
                "PE transport incomplete; native journal must be reconciled before another request") from exc
        return self.lookup(request,request_hash)


def _execute_pe(payload,components,deployment):
    import torch
    from PIL import Image,ImageOps
    request=ModelRequest.from_dict(payload["request"]);config=payload["config"];kind=config["kind"]
    validate_pe_worker(config);profile=ModelProfile.from_dict(payload["profile"])
    if payload["binding"]!=payload_sha256([payload["request_hash"],profile.to_dict(),config,payload["dataset_store"]]):
        raise ValueError("PE request binding changed")
    if profile.model!=PE_MODELS[kind] or profile.revision!="sha256:"+deployment["models"][kind]["assets_sha256"]:
        raise ValueError("PE loaded model differs from request")
    if PE_ROLES.get(request.role)!=kind or (kind=="pe_t2i" and request.artifacts) or (kind=="pe_i2i" and len(request.artifacts)!=1):
        raise ValueError("PE task mode or image count changed")
    if len(payload["inputs"])!=len(request.artifacts):raise ValueError("PE input binding changed")
    root=Path(payload["dataset_root"]).resolve()
    if not root.is_relative_to(Path(deployment["authorized_dataset_root"]).resolve()) or payload["output_prefix"]!="models/"+payload["request_hash"]:
        raise ValueError("PE raw evidence escaped its authorized dataset")
    parameters=json.loads(request.messages[0]["content"])
    if (set(parameters)-{"prompt","enable_thinking"} or "prompt" not in parameters or len(request.messages)!=1
            or type(parameters.get("enable_thinking",True)) is not bool):
        raise ValueError("PE accepts the bound refinement instruction and an explicit thinking mode only")
    enable_thinking=parameters.get("enable_thinking",True)
    model,processor=components["model"],components["processor"]
    content=[]
    for value,ref in zip(payload["inputs"],request.artifacts):
        if value["ref"]!=ref.to_dict() or file_sha256(Path(value["path"]))!=ref.sha256:
            raise ValueError("PE input bytes changed")
        with Image.open(value["path"]) as image:content.append({"type":"image","image":ImageOps.exif_transpose(image).convert("RGB")})
    content.append({"type":"text","text":parameters["prompt"]})
    messages=[{"role":"system","content":[{"type":"text","text":components["system_prompt"]}]},
              {"role":"user","content":content}]
    inputs=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,
        return_dict=True,return_tensors="pt",enable_thinking=enable_thinking).to(model.device)
    if inputs["input_ids"].shape[-1]>config["max_input_tokens"]:raise ValueError("PE context exceeds frozen limit")
    torch.manual_seed(request.seed);torch.cuda.manual_seed_all(request.seed)
    started=time.monotonic()
    from transformers.generation.streamers import BaseStreamer
    class Progress(BaseStreamer):
        def __init__(self):self.started=False;self.count=0
        def put(self,value):
            if not self.started:self.started=True;return
            self.count+=value.numel()
            if self.count==1 or self.count%16==0:
                atomic_write_json(Path(payload['native_output']).with_suffix('.progress.json'),
                    {'pid':os.getpid(),'request_id':request.request_id,'output_tokens':self.count,
                     'generation_seconds':time.monotonic()-started,'at':time.time()})
        def end(self):
            atomic_write_json(Path(payload['native_output']).with_suffix('.progress.json'),
                {'pid':os.getpid(),'request_id':request.request_id,'output_tokens':self.count,
                 'generation_seconds':time.monotonic()-started,'at':time.time(),'generation_ended':True})
    with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=request.max_tokens,
        do_sample=True,temperature=1.0,top_p=.95,top_k=20,streamer=Progress())
    tokens=generated[0,inputs["input_ids"].shape[-1]:]
    raw=processor.tokenizer.decode(tokens,skip_special_tokens=True)
    native=Path(payload["native_output"]);native.with_suffix(".raw.txt").write_text(raw)
    from dataevolver.dataset.contracts import ArtifactStore
    from dataevolver.dataset.supervision import write_artifact
    raw_ref=write_artifact(ArtifactStore({payload["dataset_store"]:root}),payload["dataset_store"],
        payload["output_prefix"]+"/pe-raw.txt",raw.encode(),"text/plain")
    value=parse_refinement(raw,kind)
    value['raw_text_artifact']=raw_ref.to_dict()
    eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
    finish="stop" if len(tokens) and int(tokens[-1]) in eos else "length"
    response={"model":profile.model,"output":value,"usage":{"input_tokens":int(inputs["attention_mask"].sum()),
        "output_tokens":len(tokens),"images":len(request.artifacts),"wall_seconds":time.monotonic()-started,"cost":None},
        "finish_reason":finish,"provider_request_id":request.request_id,"runtime":{"resident_pid":os.getpid(),
        "weights_reloaded":False,"implementation_sha256":file_sha256(Path(__file__)),
        "device_map":loaded_device_map(model),"gpu_uuid":deployment["gpu_uuid"],
        "execution_device":str(model.device),
        "runtime_versions":deployment.get('runtime_versions'),
        "raw_text_sha256":file_sha256(native.with_suffix(".raw.txt")),"enable_thinking":enable_thinking,
        "raw_text_artifact":raw_ref.to_dict(),
        "sampling":{"temperature":1.0,"top_p":.95,"top_k":20,"seed":request.seed}}}
    atomic_write_json(native,{"binding":payload["binding"],"response":response,"sha256":payload_sha256(response)})


def remaining_load_mib(config,loaded):
    import math
    remaining=sum(value['total_weight_bytes'] for kind,value in config['models'].items()
        if kind not in loaded and config.get('placement',{}).get(kind,'cuda:0')=='cuda:0')
    return math.ceil(remaining/2**20)+8192+2048


def loaded_device_map(model):
    devices={str(parameter.device) for parameter in model.parameters()}
    if not devices or 'meta' in devices:raise ValueError('PE parameters were not materialized')
    declared=getattr(model,'hf_device_map',None)
    if declared is not None:return declared
    if len(devices)==1:return {'':next(iter(devices))}
    return {name:str(parameter.device) for name,parameter in model.named_parameters()}


def serve_pe(config_path,state_dir,socket_path):
    import torch
    from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
    config=json.loads(Path(config_path).read_text());state_dir=Path(state_dir);state_dir.mkdir(parents=True,exist_ok=True)
    import importlib.metadata as package_metadata
    import platform
    for name,version in config['runtime_versions'].items():
        actual=platform.python_version() if name=='python' else package_metadata.version(name)
        if actual!=version:raise ValueError('PE runtime version changed: '+name)
    identity_path=state_dir/"identity.json";status_path=state_dir/"status.json"
    if identity_path.exists():raise ValueError("existing PE deployment must be reconciled; no duplicate loading")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=config["gpu_uuid"]:raise ValueError("PE host GPU mapping changed")
    stable=0
    while stable<2:
        result=subprocess.run(["nvidia-smi","-i",config["gpu_uuid"],"--query-gpu=memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=20)
        free,util=[int(v.strip()) for v in result.stdout.strip().split(",")]
        base=json.loads(Path(config["base_status_path"]).read_text())
        ready=free>=config["minimum_start_free_mib"] and util<=100 and base["pid"]==config["base_pid"] and base["state"]=="READY"
        stable=stable+1 if ready else 0
        atomic_write_json(state_dir/"resource-status.json",{"free_mib":free,"utilization":util,"stable_observations":stable,
            "minimum_start_free_mib":config["minimum_start_free_mib"],"model_request_submitted":False,"at":time.time()})
        if stable<2:
            if time.time()>=config["resource_deadline_epoch"]:raise TimeoutError("PE resident resource admission deadline expired")
            time.sleep(30)
    models={};metadata={}
    atomic_write_json(status_path,{'pid':os.getpid(),'state':'LOADING','models':metadata,'inference_requests':0})
    for kind,value in config["models"].items():
        if kind not in PE_MODELS or value["model"]!=PE_MODELS[kind]:raise ValueError("PE component role changed")
        root=Path(value["path"])
        for name,digest in value["file_hashes"].items():
            if file_sha256(root/name)!=digest:raise ValueError("PE model asset changed: "+name)
        if json.loads((root/"config.json").read_text())["architectures"]!=["Qwen3_5ForConditionalGeneration"]:
            raise ValueError("PE root architecture is not the prompt model")
        processor=AutoProcessor.from_pretrained(root,local_files_only=True)
        placement=config.get('placement',{}).get(kind,'cuda:0')
        if placement not in {'cpu','cuda:0'}:raise ValueError('unregistered PE placement')
        if placement=='cuda:0':
            load_required=remaining_load_mib(config,models)
            stable=0
            while stable<2:
                result=subprocess.run(['nvidia-smi','-i',config['gpu_uuid'],'--query-gpu=memory.free,utilization.gpu',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)
                free,util=[int(v.strip()) for v in result.stdout.strip().split(',')]
                stable=stable+1 if free>=load_required and util<=100 else 0
                atomic_write_json(state_dir/'resource-status.json',{'free_mib':free,'utilization':util,
                    'stable_observations':stable,'models_already_loaded':list(models),'at':time.time(),
                    'minimum_start_free_mib':load_required,'model_request_submitted':False})
                if stable<2:
                    if time.time()>=config['resource_deadline_epoch']:raise TimeoutError('PE GPU loading admission expired')
                    time.sleep(30)
        elif next(int(line.split()[1])//1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))<65536:raise ValueError('PE CPU loading requires 64GiB available memory')
        model=Qwen3_5ForConditionalGeneration.from_pretrained(root,dtype=torch.bfloat16,
            local_files_only=True,attn_implementation="sdpa",device_map={"":'cpu' if placement=='cpu' else 0}).eval()
        if {str(parameter.device) for parameter in model.parameters()}!={placement}:
            raise ValueError("PE parameters differ from frozen execution placement")
        models[kind]={"processor":processor,"model":model,"system_prompt":(root/"system_prompt.txt").read_text().strip()}
        metadata[kind]={"model":value["model"],"path":str(root),"assets_sha256":payload_sha256(value["file_hashes"]),
            "device_map":loaded_device_map(model),"loaded_at":time.time(),
            "parameter_devices":sorted({str(parameter.device) for parameter in model.parameters()}),
            "parameter_bytes":sum(parameter.numel()*parameter.element_size() for parameter in model.parameters())}
        atomic_write_json(status_path,{'pid':os.getpid(),'state':'LOADING','models':metadata,'inference_requests':0})
    if set(models)!=set(PE_MODELS):raise ValueError("both PE components must remain loaded")
    deployment={"pid":os.getpid(),"process_start_ticks":(Path("/proc")/str(os.getpid())/"stat").read_text().rsplit(")",1)[1].split()[19],
        "models":metadata,"gpu_uuid":config["gpu_uuid"],"socket_path":str(socket_path),"status_path":str(status_path),
        "implementation_sha256":file_sha256(Path(__file__)),"authorized_root":config["authorized_root"]}
    deployment["authorized_dataset_root"]=config["authorized_dataset_root"]
    deployment["handler_path"]=str(Path(__file__).resolve())
    deployment['runtime_versions']=config['runtime_versions']
    atomic_write_json(identity_path,deployment);identity_path.chmod(0o600)
    status={"pid":os.getpid(),"state":"READY","models":metadata,"inference_requests":0}
    handlers={deployment['implementation_sha256']:_execute_pe}
    def update():
        status.update(updated_at=time.time(),allocated_mib=torch.cuda.memory_allocated(0)/2**20,
            reserved_mib=torch.cuda.memory_reserved(0)/2**20)
        atomic_write_json(status_path,status)
    if Path(socket_path).exists():raise ValueError("PE socket already exists; refuse to replace another service")
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        server.bind(str(socket_path));Path(socket_path).chmod(0o600);server.listen(4);server.settimeout(30);update()
        while True:
            try:conn,_=server.accept()
            except socket.timeout:update();continue
            with conn:
                native=None;payload=None
                try:
                    conn.settimeout(10);raw=b""
                    while True:
                        data=conn.recv(65536)
                        if not data:raise ValueError("incomplete PE transport envelope")
                        raw+=data
                        if len(raw)>60000:raise ValueError("invalid PE transport envelope")
                        try:message=json.loads(raw);break
                        except json.JSONDecodeError:continue
                    if message["deployment_sha256"]!=file_sha256(identity_path):raise ValueError("PE deployment binding changed")
                    current=json.loads(identity_path.read_text())
                    for field in ('pid','process_start_ticks','gpu_uuid','models','authorized_root','authorized_dataset_root'):
                        if current[field]!=deployment[field]:raise ValueError('loaded PE identity changed: '+field)
                    handler_path=Path(current['handler_path']);handler_hash=file_sha256(handler_path)
                    if handler_hash!=current['implementation_sha256']:raise ValueError('PE handler source differs from registered revision')
                    if handler_hash not in handlers:
                        import importlib.util
                        spec=importlib.util.spec_from_file_location('dataevolver.adapters._pe_handler_'+handler_hash,handler_path)
                        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                        handlers[handler_hash]=module._execute_pe
                    job=Path(message["job_path"]).resolve()
                    if not job.is_relative_to(Path(config["authorized_root"]).resolve()):raise ValueError("PE job escaped authorized root")
                    payload=json.loads(job.read_text());request=ModelRequest.from_dict(payload["request"])
                    native=Path(payload["native_output"])
                    if native.parent!=job.parent or native.name!=request.request_id+".native.json":raise ValueError("PE native path changed")
                    if native.exists():result={"ok":True,"reused_native":True}
                    else:
                        intent=native.with_suffix(".resident.intent.json")
                        if intent.exists() or native.with_suffix(".failure.json").exists():raise ModelError("AMBIGUOUS_EFFECT","PE effect already started; reconcile native journal")
                        if time.time()>message["deadline_epoch"]:raise ValueError("PE request expired before inference")
                        with intent.open("x") as handle:json.dump({"binding":payload["binding"],"started_at":time.time(),"pid":os.getpid()},handle)
                        status.update(state="EXECUTING",active_job=str(job));update()
                        status['handler_sha256']=handler_hash
                        handlers[handler_hash](payload,models[payload["config"]["kind"]],current)
                        atomic_write_json(intent,{"binding":payload["binding"],"completed_at":time.time(),"pid":os.getpid()})
                        status["inference_requests"]+=1;result={"ok":True}
                except Exception as exc:
                    if (getattr(exc,'code',None)!='AMBIGUOUS_EFFECT' and native is not None and payload is not None
                            and not native.exists() and not native.with_suffix(".failure.json").exists()):
                        atomic_write_json(native.with_suffix(".failure.json"),{"binding":payload["binding"],
                            "code":getattr(exc,"code",type(exc).__name__),"message":str(exc)[:2000]})
                    result={"ok":False,"error":str(exc)[:2000]}
                finally:
                    # Only this PE process's unused allocator blocks are freed;
                    # all PE model weights remain resident for the next request.
                    torch.cuda.empty_cache();status.update(state="READY",active_job=None);update()
                try:conn.sendall(json.dumps(result).encode())
                except OSError:pass


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--serve",required=True);parser.add_argument("--state-dir",required=True);parser.add_argument("--socket",required=True)
    args=parser.parse_args();serve_pe(args.serve,args.state_dir,args.socket)
