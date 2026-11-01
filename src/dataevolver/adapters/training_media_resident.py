"""Unix-socket execution transport for already loaded media model objects.

The existing Controller and ModelGateway retain all planning, request ownership,
accounting and adoption. This module never loads weights or retries a request.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, identifier, payload_sha256
from dataevolver.harness.state_store import atomic_write_json
from .model_contracts import ModelRequest
from .model_gateway import ModelError


def validate_resident_config(config):
    value=config["resident"]
    required={"socket_env","identity_env","identity_sha256","implementation_sha256","activation_reserve_mib"}
    if set(value)!=required or config.get("execution_device") not in {"cuda","cpu"} or config["resource_profile"]["cpu_offload"]:
        raise ValueError("resident transport requires an explicit CUDA deployment without offloading")
    if "shard_max_memory_mib" in config or "shard_min_free_mib" in config:
        raise ValueError("resident weights use their actual frozen placement, not a new loading budget")
    for key in ("socket_env","identity_env"):identifier(value[key])
    if value["implementation_sha256"]!=file_sha256(Path(__file__)):
        raise ValueError("resident transport implementation changed")
    reserve=value["activation_reserve_mib"]
    count = {0} if config['execution_device']=='cpu' else {1,2}
    if not isinstance(reserve,list) or len(reserve) not in count or any(type(x) is not int or x<=0 for x in reserve):
        raise ValueError("resident transport requires one reserve per CUDA device, or none for a CPU model")
    if not isinstance(value["identity_sha256"],str) or len(value["identity_sha256"])!=64:
        raise ValueError("resident deployment identity must be frozen")


def identity(config):
    if config.get('schema_version') == 'dataevolver.generation_benchmark_worker.v1':
        from .benchmark_worker import benchmark_model
        selected_model = benchmark_model(config)
    elif config.get('schema_version') == 'dataevolver.local_visual_worker.v1':
        from .visual_models import MODELS
        selected_model = MODELS[config["kind"]]
    else:
        from .training_media_worker import MODELS
        selected_model = MODELS[config["kind"]]
    value=config["resident"]
    path=Path(os.environ.get(value["identity_env"],""))
    if not path.is_file() or file_sha256(path)!=value["identity_sha256"]:
        raise ModelError("CONFIG_MISSING","resident deployment identity is missing or changed")
    result=json.loads(path.read_text());model=result["models"][config["kind"]]
    if (model["model"]!=selected_model or model["assets_sha256"]!=payload_sha256(config["file_hashes"]) or
            Path(os.environ[config["model_path_env"]]).resolve()!=Path(model["path"]).resolve()):
        raise ModelError("INVALID_RECEIPT","resident loaded model differs from frozen assets")
    if os.environ.get(value["socket_env"])!=result["socket_path"]:
        raise ModelError("CONFIG_MISSING","resident endpoint differs from its frozen identity")
    if file_sha256(Path(result["handler_path"]))!=result["handler_sha256"]:
        raise ModelError("CONFIG_MISSING","resident request handler changed")
    if result["resident_implementation_sha256"]!=value["implementation_sha256"]:
        raise ModelError("CONFIG_MISSING","resident source differs from the deployment identity")
    return result


def resident_rpc(config,message,*,timeout=10):
    target=identity(config)["socket_path"]
    sent=False
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
            client.settimeout(timeout);client.connect(target)
            raw=json.dumps(message,separators=(",",":")).encode()
            if len(raw)>60000:raise ValueError("resident request exceeded its bounded envelope")
            # Mark before sendall: a partial send may still cause an effect.
            sent=True;client.sendall(raw);chunks=[];length=0
            while True:
                data=client.recv(65536)
                if not data:break
                chunks.append(data);length+=len(data)
                if length>2*1024*1024:raise ValueError("resident response exceeded its bound")
            result=json.loads(b"".join(chunks))
            if not isinstance(result,dict):raise ValueError("resident response must be an object")
            return result
    except (OSError,ValueError) as exc:
        uncertain=sent and message.get("operation")=="execute_payload"
        raise ModelError("AMBIGUOUS_EFFECT" if uncertain else "RESOURCE_UNAVAILABLE",
            "resident socket did not return a complete response; no automatic resend",retry_safe=not uncertain) from exc


def resident_observation(config,*,status=None):
    expected=identity(config)
    if status is None:
        response=resident_rpc(config,{"operation":"health"})
        if not response.get("ok"):raise ModelError("RESOURCE_UNAVAILABLE","resident host health failed")
        status=response["status"]
    try:
        start=Path("/proc")/str(expected["pid"])/"stat"
        ticks=start.read_text().split(") ",1)[1].split()[19]
        model=status["models"][config["kind"]]
        if (status["pid"]!=expected["pid"] or ticks!=expected["process_start_ticks"] or
                status["state"] not in {"READY","EXECUTING"} or not model["ready"] or
                model["device_map"]!=expected["models"][config["kind"]]["device_map"] or
                status["cuda_visible_devices"].split(",")!=expected["gpu_uuids"]):
            raise ValueError("resident host or placement changed")
        if os.environ.get("CUDA_VISIBLE_DEVICES","").split(",")!=expected["gpu_uuids"]:
            raise ValueError("caller GPU mapping differs from its resident deployment")
        if config['execution_device']=='cpu':
            if set(map(str,model['device_map'].values()))!={'cpu'}:
                raise ValueError('CPU worker is not backed by CPU resident tensors')
            from .training_media_worker import _available_cpu_mib
            available=_available_cpu_mib()
            required=config['resource_profile']['min_free_mib']
            if available<required:
                raise ModelError('RESOURCE_BUSY','resident CPU activation reserve unavailable',retry_safe=True)
            return {'device':'resident_cpu','resident_pid':expected['pid'],'deployment_id':expected['deployment_id'],
                'device_map':model['device_map'],'weights_already_loaded':True,'available_mib':available,
                'required_activation_mib':required,'devices':[],'observed_at':time.time()}
        if len(expected['gpu_uuids'])!=len(config['resident']['activation_reserve_mib']):
            raise ValueError('activation reserves differ from frozen CUDA mapping')
        result=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"],check=True,text=True,capture_output=True,timeout=20)
        rows={}
        for line in result.stdout.splitlines():
            index,uuid,free,util=[x.strip() for x in line.split(",")]
            rows[uuid]={"physical_index":int(index),"uuid":uuid,"free_mib":int(free),"utilization":int(util)}
        memory={v["device"]:v for v in status["memory"]};devices=[]
        for i,(uuid,reserve) in enumerate(zip(expected["gpu_uuids"],config["resident"]["activation_reserve_mib"])):
            row=dict(rows[uuid]);own=memory[i]
            cache=max(0,int(own["reserved_mib"]-own["allocated_mib"]))
            row.update(process_device=f"cuda:{i}",owned_reusable_cache_mib=cache,
                usable_activation_mib=row["free_mib"]+cache,required_activation_mib=reserve)
            if row["usable_activation_mib"]<reserve or row["utilization"]>config["resource_profile"]["max_utilization"]:
                raise ModelError("RESOURCE_BUSY","resident activation reserve is unavailable",retry_safe=True)
            devices.append(row)
    except ModelError:raise
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
        raise ModelError("RESOURCE_UNAVAILABLE","cannot verify resident process, placement and GPU resources",retry_safe=True) from exc
    return {"device":"resident_cuda","resident_pid":expected["pid"],"deployment_id":expected["deployment_id"],
        "device_map":model["device_map"],"weights_already_loaded":True,"devices":devices,"observed_at":time.time()}


def execute_resident_job(message,models,status,state_dir,identity_path):
    """Called by the serial resident host, using actual model object references."""
    from .training_media_worker import validate_worker,run_payload
    expected_path=Path(identity_path).resolve()
    if file_sha256(expected_path)!=message.get("identity_sha256"):
        raise ValueError("request does not bind the installed resident identity")
    expected=json.loads(expected_path.read_text())
    job=Path(message["job_path"]).resolve()
    if not any(job.is_relative_to(Path(p).resolve()) for p in expected["authorized_job_roots"]):
        raise ValueError("request job escaped its registered campaign")
    payload=json.loads(job.read_text());config=payload["config"]
    if config.get('schema_version') == 'dataevolver.local_visual_worker.v1':
        from .visual_models import validate_visual_worker as validate_worker
        from .local_visual_worker import run_payload
    elif config.get('schema_version') == 'dataevolver.generation_benchmark_worker.v1':
        from .benchmark_worker import validate_worker, run_payload
    validate_worker(config)
    digest=payload["request_hash"]
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("native request hash is not canonical")
    if not config.get("resident"):raise ValueError("a resident host cannot execute an ephemeral model payload")
    os.environ[config["resident"]["identity_env"]]=str(expected_path)
    os.environ[config["resident"]["socket_env"]]=expected["socket_path"]
    model_path=expected["models"][config["kind"]]["path"]
    if Path(payload.get("model_path",model_path)).resolve()!=Path(model_path).resolve():
        raise ValueError("request model path differs from the loaded model")
    os.environ[config["model_path_env"]]=model_path
    identity(config)
    if config["resident"]["identity_sha256"]!=message["identity_sha256"]:
        raise ValueError("native request identity changed")
    request=ModelRequest.from_dict(payload["request"])
    native=Path(payload["native_output"]).resolve()
    if native.parent!=job.parent or native.name!=request.request_id+".native.json":
        raise ValueError("native output escaped its request journal")
    if payload["binding"]!=payload_sha256([payload["request_hash"],payload["profile"],config,payload["dataset_store"]]):
        raise ValueError("native request binding changed")
    dataset=Path(payload["dataset_root"]).resolve()
    if str(dataset) not in expected["authorized_dataset_roots"]:
        raise ValueError("dataset output root is not authorized")
    if payload["output_prefix"]!="models/"+payload["request_hash"]:
        raise ValueError("model output prefix changed")
    # A completed native effect is a recovery, never another inference.
    if native.exists():return {"ok":True,"reused_native":True,"native_output":str(native)}
    failure=native.with_suffix(".failure.json")
    if failure.exists():raise ModelError("RECORDED_FAILURE","native failure already recorded")
    intent=native.with_suffix(".resident.intent.json")
    orphan=dataset/payload["output_prefix"]
    if intent.exists() or (orphan.exists() and any(orphan.iterdir())):
        raise ModelError("AMBIGUOUS_EFFECT","resident request already started or has uncommitted artifacts")
    if time.time()>message["deadline_epoch"]:raise ModelError("RESOURCE_UNAVAILABLE","request expired before execution")
    intent.parent.mkdir(parents=True,exist_ok=True)
    with intent.open("x") as handle:
        json.dump({"state":"STARTED","binding":payload["binding"],"started_at":time.time(),"resident_pid":os.getpid()},handle)
    try:
        if os.getpid()!=expected["pid"]:raise ValueError("only the actual resident process can execute this request")
        run_payload(job,resident_models=models,resident_status=status)
        atomic_write_json(intent,{"state":"COMPLETED","binding":payload["binding"],"finished_at":time.time(),"resident_pid":os.getpid()})
        return {"ok":True,"native_output":str(native),"resident_pid":os.getpid(),"weights_reloaded":False}
    except Exception as exc:
        if not native.exists():
            atomic_write_json(failure,{"binding":payload["binding"],"code":getattr(exc,"code",type(exc).__name__),
                "message":str(exc)[:2000],"failed_at":time.time(),"resident_pid":os.getpid()})
        raise


def serve_resident(server,models,status,state_dir,identity_path,*,prompt_identity_path=None):
    """Keep loaded objects alive across bad clients, timeouts and model failures.

    The private bootstrap calls this loop without replacing its process or model
    objects. No fork, model reload, planning or retry happens here.
    """
    import torch
    state_dir=Path(state_dir)
    def update():
        status["updated_at"]=time.time()
        status["memory"]=[{"device":i,"allocated_mib":torch.cuda.memory_allocated(i)/2**20,
            "reserved_mib":torch.cuda.memory_reserved(i)/2**20} for i in range(torch.cuda.device_count())]
        atomic_write_json(state_dir/"status.json",status)
    status.update(state="READY",inference_handler="attached",handler_identity_sha256=file_sha256(Path(identity_path)),
        resident_implementation_sha256=file_sha256(Path(__file__)))
    update();server.settimeout(30)
    while True:
        try:conn,_=server.accept()
        except socket.timeout:update();continue
        with conn:
            try:
                conn.settimeout(10);raw=b""
                while True:
                    data=conn.recv(65536)
                    if not data:raise ValueError("incomplete resident request")
                    raw+=data
                    if len(raw)>60000:raise ValueError("resident request exceeded its bound")
                    try:message=json.loads(raw);break
                    except json.JSONDecodeError:
                        if raw.endswith(b"\n"):raise ValueError("malformed resident request")
                if not isinstance(message,dict):raise ValueError("resident envelope must be an object")
                if message.get("operation")=="health":
                    update();result={"ok":True,"status":status}
                elif message.get("operation")=="execute_payload":
                    status.update(state="EXECUTING",active_job=message.get("job_path"));update()
                    try:
                        result=execute_resident_job(message,models,status,state_dir,identity_path)
                        if not result.get("reused_native"):status["inference_requests"]+=1
                    finally:
                        status.update(state="READY",active_job=None);update()
                elif prompt_identity_path is not None and 'deployment_sha256' in message:
                    from .training_prompt_enhancement import execute_resident_pe_job
                    status.update(state='EXECUTING',active_job=message.get('job_path'));update()
                    try:
                        result=execute_resident_pe_job(message,models,prompt_identity_path)
                        if not result.get('reused_native'):status['inference_requests']+=1
                    finally:
                        status.update(state='READY',active_job=None);update()
                else:raise ValueError("unsupported resident operation")
            except Exception as exc:
                status.update(state="READY",active_job=None,last_error={"code":getattr(exc,"code",type(exc).__name__),
                    "message":str(exc)[:1000],"at":time.time()});update()
                result={"ok":False,"error":str(exc)[:2000],"error_code":getattr(exc,"code",type(exc).__name__)}
            try:conn.sendall(json.dumps(result).encode())
            except OSError:
                # A lost caller does not release the requested resident models.
                pass
