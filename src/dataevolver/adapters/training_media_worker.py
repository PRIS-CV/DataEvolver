"""Offline, explicitly registered media workers with durable native receipts.

ModelGateway owns reservation, identity and replay. This transport executes one
bound request and never selects another model, retries, or adopts an output.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, file_sha256, identifier, payload_sha256
from dataevolver.dataset.supervision import refs, write_image
from dataevolver.harness.state_store import atomic_write_json
from .model_gateway import ModelError
from .model_contracts import ModelRequest, ModelProfile


KINDS = {"qwen21", "qwen38_review", "sam3_instances"}
MODELS = {"qwen21": "Qwen/Qwen-Image-2.1", "qwen38_review": "Qwen/Qwen3.8-27B", "sam3_instances": "SAM3"}


def validate_worker(config):
    required = {"kind", "model_path_env", "python_env", "file_hashes", "runtime_versions", "implementation_sha256",
                "threads", "max_images", "image_side", "max_input_tokens", "resource_profile"}
    if set(config)-{"execution_device","shard_max_memory_mib","shard_min_free_mib","resident"} != required or config["kind"] not in KINDS:
        raise ValueError("unregistered training media worker")
    if config.get("execution_device","cuda") not in {"cpu","cuda"}:
        raise ValueError("unregistered execution device")
    if config.get("execution_device")=="cpu" and config["resource_profile"]["cpu_offload"]:
        raise ValueError("pure CPU execution cannot enable GPU offloading")
    if "resident" in config:
        from .training_media_resident import validate_resident_config
        validate_resident_config(config)
    if "shard_max_memory_mib" in config or "shard_min_free_mib" in config:
        maximum,minimum=config.get("shard_max_memory_mib"),config.get("shard_min_free_mib")
        if (config.get("execution_device")=="cpu" or config["kind"]=="sam3_instances" or
                not isinstance(maximum,list) or not isinstance(minimum,list) or len(maximum)!=2 or len(minimum)!=2 or
                any(type(x) is not int or x<=0 for x in maximum+minimum) or
                any(low<=high for high,low in zip(maximum,minimum)) or config["resource_profile"]["cpu_offload"]):
            raise ValueError("two-card sharding requires explicit weight limits and activation reserves")
    for key in ("model_path_env", "python_env"):
        identifier(config[key])
    if config["implementation_sha256"] != file_sha256(Path(__file__)):
        raise ValueError("training worker implementation changed")
    for key in ("threads", "max_images", "image_side", "max_input_tokens"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("invalid worker limit: " + key)
    if not config["file_hashes"] or not config["runtime_versions"]:
        raise ValueError("model assets and runtime must be frozen")
    for name, digest in config["file_hashes"].items():
        ArtifactRef("model", name, digest)
    resource = config["resource_profile"]
    if (set(resource) != {"min_free_mib", "max_utilization", "cpu_offload", "estimate_basis"} or
            type(resource["min_free_mib"]) is not int or resource["min_free_mib"] <= 0 or
            type(resource["max_utilization"]) is not int or not 0 <= resource["max_utilization"] <= 100 or
            type(resource["cpu_offload"]) is not bool or not resource["estimate_basis"]):
        raise ValueError("resource profile must describe this deployment")


def gpu_observation(config):
    if config.get("resident"):
        from .training_media_resident import resident_observation
        return resident_observation(config)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if config.get("shard_max_memory_mib"):
        mappings=visible.split(",")
        if len(mappings)!=2 or len(set(mappings))!=2 or not all(mappings):
            raise ModelError("RESOURCE_UNAVAILABLE","sharded worker requires two explicit GPU mappings",retry_safe=True)
        devices=[]
        for mapping,minimum in zip(mappings,config["shard_min_free_mib"]):
            single={k:v for k,v in config.items() if not k.startswith("shard_")}
            single["resource_profile"]={**single["resource_profile"],"min_free_mib":minimum}
            os.environ["CUDA_VISIBLE_DEVICES"]=mapping
            try:devices.append(gpu_observation(single))
            finally:os.environ["CUDA_VISIBLE_DEVICES"]=visible
        return {"device":"cuda_sharded","devices":devices,"process_devices":["cuda:0","cuda:1"],
                "max_memory_mib":config["shard_max_memory_mib"],"observed_at":time.time()}
    if not visible or "," in visible:
        raise ModelError("RESOURCE_UNAVAILABLE", "one explicit physical GPU mapping is required", retry_safe=True)
    try:
        result = subprocess.run(["nvidia-smi", "-i", visible, "--query-gpu=index,uuid,memory.free,utilization.gpu",
                                 "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True, timeout=20)
        index, uuid, free, utilization = [x.strip() for x in result.stdout.strip().split(",")]
    except (OSError,subprocess.SubprocessError,ValueError) as exc:
        raise ModelError("RESOURCE_UNAVAILABLE","cannot verify the selected physical GPU",retry_safe=True) from exc
    value = {"physical_index": int(index), "uuid": uuid, "free_mib": int(free),
             "utilization": int(utilization), "process_device": "cuda:0", "observed_at": time.time()}
    profile = config["resource_profile"]
    if value["free_mib"] < profile["min_free_mib"] or value["utilization"] > profile["max_utilization"]:
        raise ModelError("RESOURCE_BUSY", "selected GPU no longer matches this worker's resource profile", retry_safe=True)
    return value


def _available_cpu_mib():
    values={line.split(":",1)[0]:int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")}
    return values["MemAvailable"]//1024


def cpu_observation(config):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise ModelError("RESOURCE_UNAVAILABLE","pure CPU worker requires explicitly hidden CUDA devices",retry_safe=True)
    available=_available_cpu_mib()
    if available<config["resource_profile"]["min_free_mib"]:
        raise ModelError("RESOURCE_BUSY","host memory does not meet the CPU worker reserve",retry_safe=True)
    return {"device":"cpu","available_mib":available,"threads":config["threads"],"gpu_access":False,"observed_at":time.time()}


def deployment_observation(config, *, resident_status=None):
    if config.get('resident'):
        from .training_media_resident import resident_observation
        if resident_status is None:
            return resident_observation(config)
        return resident_observation(config, status=resident_status)
    return cpu_observation(config) if config.get("execution_device")=="cpu" else gpu_observation(config)


def torch_device(config):
    return "cpu" if config.get("execution_device")=="cpu" else "cuda:0"


def sharded_loading(config):
    if not config.get("shard_max_memory_mib"):return {}
    return {"device_map":"balanced","max_memory":{i:f"{value}MiB" for i,value in enumerate(config["shard_max_memory_mib"])}}


def loading_info_summary(info):
    # Transformers may return sets. Preserve the audit fields as JSON arrays.
    return {key:sorted(info.get(key,[]),key=str) for key in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")}


def actual_device_map(model):
    mapping=getattr(model,'hf_device_map',None)
    if mapping:return mapping
    # CPU and explicitly .to()-placed Transformers/Diffusers objects may have
    # no hf_device_map. Derive placement from the actual already loaded tensors.
    if hasattr(model,'components'):
        mapping={}
        for name,component in model.components.items():
            if not hasattr(component,'parameters'):continue
            devices={str(p.device) for p in component.parameters()}
            if len(devices)!=1:raise ValueError('component has no single verified tensor device')
            mapping[name]=next(iter(devices))
        if not mapping:raise ValueError('pipeline has no verified component tensors')
        return mapping
    devices={str(p.device) for p in model.parameters()}
    if len(devices)!=1:raise ValueError('model has no single verified tensor device')
    return {'':next(iter(devices))}


class TrainingMediaProvider:
    worker_module = "dataevolver.adapters.training_media_worker"
    def __init__(self, profile, config, store, dataset_store, receipt_root):
        validate_worker(config)
        if profile.model != MODELS[config["kind"]] or profile.revision != "sha256:"+payload_sha256(config["file_hashes"]):
            raise ValueError("worker model identity differs from frozen selected assets")
        self.profile, self.config, self.store = profile, config, store
        self.dataset_store, self.root = dataset_store, Path(receipt_root)

    def health(self):
        return {"status": "CONFIGURED", "kind": self.config["kind"], "inference_performed": False}

    def path(self, request):
        return self.root / (request.request_id+".native.json")

    def binding(self, digest):
        return payload_sha256([digest, self.profile.to_dict(), self.config, self.dataset_store])

    def audit_native_usage(self, request_id: str, record: dict, unit_rule: str,
                          artifact_store: ArtifactStore | None) -> tuple[dict, dict[str, int]]:
        """Read SAM accounting without launching a worker or loading a model."""
        native = self.root / (request_id + '.native.sam-usage.json')
        subject_usage = None
        if native.exists():
            value = json.loads(native.read_text())
            count = value.get('forward_attempts') if unit_rule == 'image_subject' else value.get('frame_subject_units_attempted')
            valid = (value.get('binding') == self.binding(record['request_hash'])
                and value.get('gateway_requests') == 1
                and type(count) is int and 0 <= count <= record['native_units_per_attempt'])
            if unit_rule == 'image_subject':
                valid = valid and value.get('input_images') == 1
            else:
                frames, subjects, forwards = (value.get(k) for k in ('decoded_frames', 'subjects', 'frame_forward_attempts'))
                valid = (valid and value.get('input_videos') == 1
                    and all(type(n) is int for n in (frames, subjects, forwards))
                    and 1 <= frames <= 256 and 1 <= subjects <= 16 and 0 <= forwards <= frames
                    and frames * subjects == record['native_units_per_attempt'] and count == forwards * subjects)
            if not valid:
                raise ModelError('INVALID_RECEIPT', 'SAM native subject accounting changed')
            subject_usage = {**value, 'journal_sha256': file_sha256(native)}
            if artifact_store is not None:
                for key, root in artifact_store.roots.items():
                    if native.resolve().is_relative_to(root):
                        subject_usage['journal'] = ArtifactRef(key, native.resolve().relative_to(root).as_posix(),
                            subject_usage['journal_sha256']).to_dict()
                        break
        observed = subject_usage or {}
        details = {'sam_native_usage': subject_usage, 'sam_unit_rule': unit_rule}
        totals = {
            'sam_image_subject_forwards_observed': observed.get('forward_attempts', 0) if unit_rule == 'image_subject' else 0,
            'sam_video_frame_forwards_observed': observed.get('frame_forward_attempts', 0) if unit_rule == 'video_frame_subject' else 0,
            'sam_video_frame_subject_units_observed': observed.get('frame_subject_units_attempted', 0) if unit_rule == 'video_frame_subject' else 0,
            'sam_requests_without_native_accounting': int(subject_usage is None),
        }
        return details, totals

    def lookup(self, request, request_hash):
        path = self.path(request)
        if not path.exists():
            failure = path.with_suffix(".failure.json")
            if failure.exists():
                value = json.loads(failure.read_text())
                if value.get("binding") != self.binding(request_hash):
                    raise ModelError("INVALID_RECEIPT","failed native request binding changed")
                if value.get("code")=="QUALITY_REJECTED":
                    raise ModelError("QUALITY_REJECTED","recorded rejected instance: "+str(value.get("message")))
                raise ModelError("RECORDED_FAILURE",str(value.get("code"))+": "+str(value.get("message")))
            raise ModelError("AMBIGUOUS_EFFECT", "no committed native receipt; automatic resubmission is forbidden")
        try:
            value = json.loads(path.read_text()); response = value["response"]
            if value["binding"] != self.binding(request_hash) or value["sha256"] != payload_sha256(response):
                raise ValueError("native request binding changed")
            for ref in refs(response["output"]):
                if ref.store != self.dataset_store:
                    raise ValueError("model output escaped its writable dataset store")
                asset = self.store.resolve(ref)
                if ref.media_type.startswith("image/"):
                    from PIL import Image
                    with Image.open(asset) as image:
                        image.load()
            return response
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ModelError("INVALID_RECEIPT", "native effect failed independent readback") from exc

    def invoke(self, request, request_hash):
        if self.path(request).exists():
            return self.lookup(request, request_hash)
        if len(request.artifacts) > self.config["max_images"]:
            raise ModelError("INVALID_INPUT", "media count exceeds configured worker capacity", retry_safe=True)
        model_path = os.environ.get(self.config["model_path_env"], "")
        interpreter = os.environ.get(self.config["python_env"], "")
        if not Path(model_path).is_dir() or not Path(interpreter).is_file():
            raise ModelError("CONFIG_MISSING", "selected model assets or interpreter are unavailable", retry_safe=True)
        placement = deployment_observation(self.config)
        output_prefix = "models/"+request_hash
        output_root = self.store.roots[self.dataset_store]/output_prefix
        if output_root.exists() and any(output_root.iterdir()):
            raise ModelError("AMBIGUOUS_EFFECT", "orphaned model output exists without a native receipt")
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"binding": self.binding(request_hash), "request": request.to_dict(), "request_hash": request_hash,
            "profile": self.profile.to_dict(), "config": self.config, "model_path": model_path,
            "inputs": [{"path": str(self.store.resolve(ref)), "ref":ref.to_dict()} for ref in request.artifacts],
            "dataset_store": self.dataset_store, "dataset_root": str(self.store.roots[self.dataset_store]),
            "output_prefix": output_prefix, "native_output": str(self.path(request)), "deployment_launch":placement,
            "gpu_launch":placement if torch_device(self.config)!="cpu" else None}
        job = self.root/(request.request_id+".job.json"); atomic_write_json(job, payload)
        if self.config.get("resident"):
            from .training_media_resident import resident_rpc
            result = resident_rpc(self.config, {"operation":"execute_payload", "job_path":str(job.resolve()),
                "identity_sha256":self.config["resident"]["identity_sha256"],
                "deadline_epoch":time.time()+request.timeout_seconds}, timeout=request.timeout_seconds)
            if self.path(request).exists():return self.lookup(request, request_hash)
            failure=self.path(request).with_suffix(".failure.json")
            if failure.exists():return self.lookup(request, request_hash)
            raise ModelError("AMBIGUOUS_EFFECT", "resident call returned without a committed native receipt: "+str(result.get("error","")))
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   OMP_NUM_THREADS=str(self.config["threads"]), MKL_NUM_THREADS=str(self.config["threads"]),
                   PYTHONFAULTHANDLER="1", PYTHONDONTWRITEBYTECODE="1")
        if torch_device(self.config)=="cpu":env["CUDA_VISIBLE_DEVICES"]=""
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            env.pop(key, None)
        with (self.root/(request.request_id+".log")).open("a", buffering=1) as log:
            log.write(json.dumps({"event":"started", "request_hash":request_hash, "deployment":placement})+"\n")
            try:
                result = subprocess.run([interpreter,"-m",self.worker_module,"--payload",str(job)],
                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=request.timeout_seconds, check=False)
            except subprocess.TimeoutExpired:
                if self.path(request).exists():
                    return self.lookup(request, request_hash)
                raise ModelError("AMBIGUOUS_EFFECT", "worker timed out without a committed native receipt") from None
            log.write(json.dumps({"event":"exited", "return_code":result.returncode})+"\n")
        if result.returncode:
            failure = self.root/(request.request_id+".native.failure.json")
            detail = json.loads(failure.read_text()) if failure.exists() else {}
            code = detail.get("code") if detail.get("code") in {"RESOURCE_BUSY", "RESOURCE_UNAVAILABLE", "QUALITY_REJECTED"} else "LOCAL_WORKER_FAILED"
            raise ModelError(code, f"worker exit {result.returncode}: {detail.get('message','inspect worker log')}")
        return self.lookup(request, request_hash)


def _verified_runtime(payload, resident_status=None):
    import importlib.metadata as metadata
    import platform
    config = payload["config"]; validate_worker(config)
    profile = ModelProfile.from_dict(payload["profile"])
    request = ModelRequest.from_dict(payload["request"])
    if profile.model != MODELS[config["kind"]] or profile.revision != "sha256:"+payload_sha256(config["file_hashes"]):
        raise ValueError("worker payload changed selected model identity")
    if len(payload["inputs"]) != len(request.artifacts) or any(
            value["ref"] != ref.to_dict() for value,ref in zip(payload["inputs"],request.artifacts)):
        raise ValueError("worker input order or artifact binding changed")
    runtime = {"python":platform.python_version(), "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
               "gpu_launch":payload.get("gpu_launch"), "deployment_launch":payload.get("deployment_launch"),
               "execution_device":torch_device(config), "implementation_sha256":config["implementation_sha256"]}
    for name, expected in config["runtime_versions"].items():
        value = platform.python_version() if name == "python" else metadata.version(name)
        if value != expected:
            raise ValueError("frozen runtime changed: "+name)
        runtime[name] = value
    root = Path(payload["model_path"]).resolve()
    for name, digest in config["file_hashes"].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != digest:
            raise ValueError("frozen model file changed: "+name)
    for value in payload["inputs"]:
        if file_sha256(Path(value["path"])) != value["ref"]["sha256"]:
            raise ValueError("media input bytes changed")
    if config.get("resident"):
        from .training_media_resident import resident_observation
        if resident_status is None:raise ValueError("resident inference requires the actual loaded model host")
        runtime["deployment_before_load"] = resident_observation(config,status=resident_status)
    else:runtime["deployment_before_load"] = deployment_observation(config)
    return runtime


def _qwen_image(payload, parameters, runtime, *, components=None):
    import torch
    from PIL import Image, ImageOps
    from diffusers import QwenImage21Pipeline
    index = json.loads((Path(payload["model_path"])/"model_index.json").read_text())
    if index.get("_class_name") != "QwenImage21Pipeline":
        raise ValueError("selected image model architecture changed")
    if len(payload["inputs"]) > 10 or set(parameters)-{"negative_prompt"} != {"prompt", "width", "height", "steps", "true_cfg_scale"}:
        raise ValueError("unexpected first-production image parameters or input count")
    if "negative_prompt" in parameters and not isinstance(parameters["negative_prompt"],str):
        raise ValueError("negative_prompt must be explicit text")
    pipe = components if components is not None else QwenImage21Pipeline.from_pretrained(payload["model_path"], torch_dtype=torch.bfloat16, local_files_only=True,
                                             **sharded_loading(payload["config"]))
    if payload["config"].get("shard_max_memory_mib") or payload["config"].get("resident"):
        runtime["device_map"]=actual_device_map(pipe)
        if any(str(v) in {"cpu","disk"} for v in runtime['device_map'].values()):
            raise ValueError("sharded image model did not remain on the selected GPUs")
    elif payload["config"]["resource_profile"]["cpu_offload"]:
        pipe.enable_model_cpu_offload(gpu_id=0)
    else:
        pipe.to(torch_device(payload["config"]))
    kwargs = dict(prompt=parameters["prompt"], width=parameters["width"], height=parameters["height"],
        num_inference_steps=parameters["steps"], true_cfg_scale=parameters["true_cfg_scale"],
        generator=torch.Generator(device="cpu" if payload["config"].get("shard_max_memory_mib") or payload["config"].get("resident") else torch_device(payload["config"])).manual_seed(payload["request"]["seed"]))
    if "negative_prompt" in parameters:kwargs["negative_prompt"]=parameters["negative_prompt"]
    runtime["guidance"]={"requested_true_cfg_scale":parameters["true_cfg_scale"],
        "negative_prompt_provided":"negative_prompt" in parameters,
        "classifier_free_guidance_enabled":parameters["true_cfg_scale"]>1 and "negative_prompt" in parameters}
    if payload["inputs"]:
        images = []
        for item in payload["inputs"]:
            with Image.open(item["path"]) as opened:
                images.append(ImageOps.exif_transpose(opened).convert("RGB"))
        kwargs["image"] = images[0] if len(images) == 1 else images
    with torch.inference_mode():
        image = pipe(**kwargs).images[0]
    if image.size != (parameters["width"],parameters["height"]):
        raise ValueError("image backend changed requested output dimensions")
    store = ArtifactStore({payload["dataset_store"]:payload["dataset_root"]})
    ref = write_image(store,payload["dataset_store"],payload["output_prefix"]+"/image.png",image)
    return {"artifact":ref.to_dict(),"actual_parameters":parameters,"image_size":list(image.size)}, {
        "images":1,"input_tokens":None,"output_tokens":None,"cost":None}, "stop"


def _qwen_review(payload, parameters, runtime, *, components=None):
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
    if set(parameters) != {"prompt"}:
        raise ValueError("review parameters must contain only the bound inspection prompt")
    processor = components["processor"] if components is not None else AutoProcessor.from_pretrained(payload["model_path"],local_files_only=True)
    model = components["model"] if components is not None else Qwen3_5ForConditionalGeneration.from_pretrained(payload["model_path"],dtype=torch.bfloat16,
        local_files_only=True,attn_implementation="sdpa",**sharded_loading(payload["config"]))
    if payload["config"].get("shard_max_memory_mib") or payload["config"].get("resident"):
        runtime["device_map"]=actual_device_map(model)
        allowed={'cpu'} if torch_device(payload['config'])=='cpu' else {'0','1','cuda:0','cuda:1'}
        if not set(map(str,runtime['device_map'].values()))<=allowed:
            raise ValueError("sharded reviewer did not remain on the selected GPUs")
    else:model.to(torch_device(payload["config"]))
    model.eval()
    content = []
    for value in payload["inputs"]:
        with Image.open(value["path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGBA")
        canvas = Image.new("RGBA",image.size,(127,127,127,255));canvas.alpha_composite(image)
        image = canvas.convert("RGB");image.thumbnail((payload["config"]["image_side"],)*2)
        content.append({"type":"image","image":image})
    content.append({"type":"text","text":parameters["prompt"]})
    messages = [{"role":"system","content":"Return only JSON satisfying this schema. Use uncertain when observations are insufficient. "+
        json.dumps(payload["request"]["response_schema"])},{"role":"user","content":content}]
    inputs = processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors="pt",
                                           return_dict=True,enable_thinking=False).to(torch_device(payload["config"]))
    if inputs["input_ids"].shape[-1] > payload["config"]["max_input_tokens"]:
        raise ValueError("review input exceeds its frozen context configuration")
    with torch.inference_mode():
        generated = model.generate(**inputs,max_new_tokens=payload["request"]["max_tokens"],do_sample=False)
    tokens = generated[0,inputs["input_ids"].shape[-1]:]
    text = processor.decode(tokens,skip_special_tokens=True).strip()
    Path(payload["native_output"]).with_suffix(".raw.txt").write_text(text)
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    output = json.loads(text)
    eos = model.generation_config.eos_token_id;eos = eos if isinstance(eos,list) else [eos]
    finish = "stop" if len(tokens) and int(tokens[-1]) in eos else "length"
    return output,{"images":len(payload["inputs"]),"input_tokens":int(inputs["attention_mask"].sum()),
        "output_tokens":len(tokens),"cost":None},finish


def _iou(first, second):
    left,top=max(first[0],second[0]),max(first[1],second[1])
    right,bottom=min(first[2],second[2]),min(first[3],second[3])
    intersection=max(0,right-left)*max(0,bottom-top)
    union=max(0,first[2]-first[0])*max(0,first[3]-first[1])+max(0,second[2]-second[0])*max(0,second[3]-second[1])-intersection
    return intersection/union if union else 0


def _sam_instances(payload, parameters, runtime, *, components=None):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import Sam3Processor,Sam3VideoModel
    if len(payload["inputs"]) != 1 or set(parameters) != {"subjects","threshold","mask_threshold","min_box_iou"}:
        raise ValueError("SAM3 requires one target and bound instance prompts")
    # Reject a broken observation before loading either detector or tracker.
    import math
    for subject in parameters["subjects"]:
        normalized=subject["box_xyxy_normalized"]
        if (len(normalized)!=4 or any(not math.isfinite(x) or not 0<=x<=1 for x in normalized) or
                normalized[0]>=normalized[2] or normalized[1]>=normalized[3]):
            raise ValueError("invalid observed subject bounding box")
    if components is None:
        processor = Sam3Processor.from_pretrained(payload["model_path"],local_files_only=True)
        video_model, info = Sam3VideoModel.from_pretrained(payload["model_path"],dtype=torch.float32,
            local_files_only=True,output_loading_info=True)
        model = video_model.detector_model.to(torch_device(payload["config"])).eval()
    else:
        processor,model,info=components["processor"],components["model"],components["loading_info"]
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise ValueError("SAM3 video checkpoint did not load all detector/tracker weights")
    runtime["loading_info"] = loading_info_summary(info)
    with Image.open(payload["inputs"][0]["path"]) as opened:image=opened.convert("RGB")
    store = ArtifactStore({payload["dataset_store"]:payload["dataset_root"]})
    instances, masks = [], []
    runtime['sam_counts'] = {'gateway_requests': 1, 'input_images': 1, 'forward_attempts': 0}
    for subject in parameters["subjects"]:
        identity=subject["subject_id"];identifier(identity)
        normalized=subject["box_xyxy_normalized"]
        box=[normalized[0]*image.width,normalized[1]*image.height,normalized[2]*image.width,normalized[3]*image.height]
        inputs=processor(images=image,text=subject["description"],input_boxes=[[box]],input_boxes_labels=[[1]],return_tensors="pt").to(torch_device(payload["config"]))
        runtime['sam_counts']['forward_attempts'] += 1
        if payload.get('native_output'):
            atomic_write_json(Path(payload['native_output']).with_suffix('.sam-usage.json'),
                {'binding': payload['binding'], **runtime['sam_counts']})
        with torch.inference_mode():outputs=model(**inputs)
        result=processor.post_process_instance_segmentation(outputs,threshold=parameters["threshold"],
            mask_threshold=parameters["mask_threshold"],target_sizes=inputs["original_sizes"].tolist())[0]
        candidates=[(_iou(box,b.tolist()),float(score),i) for i,(b,score) in enumerate(zip(result["boxes"],result["scores"]))]
        if not candidates:raise ModelError("QUALITY_REJECTED","SAM3 found no instance for "+identity)
        overlap,score,index=max(candidates)
        if overlap<parameters["min_box_iou"]:raise ModelError("QUALITY_REJECTED","SAM3 mask does not match observed instance "+identity)
        mask=result["masks"][index].detach().cpu().numpy().astype(bool).squeeze()
        if mask.shape!=(image.height,image.width) or not mask.any():raise ValueError("invalid SAM3 instance mask")
        for previous in masks:
            if np.logical_and(mask,previous).sum()/max(1,np.logical_or(mask,previous).sum())>.8:
                raise ModelError("QUALITY_REJECTED","different subject IDs map to the same segmented instance")
        masks.append(mask)
        ref=write_image(store,payload["dataset_store"],payload["output_prefix"]+"/"+identity+".png",Image.fromarray(mask.astype("uint8")*255))
        instances.append({"subject_id":identity,"target_instance_id":subject["target_instance_id"],"description":subject["description"],
            "mask":ref.to_dict(),"box_xyxy":result["boxes"][index].tolist(),"score":score,"observed_box_iou":overlap})
    return {"instances":instances},{"images":1,"input_tokens":None,"output_tokens":None,"cost":None},"stop"


def run_payload(path, *, resident_models=None, resident_status=None):
    payload=json.loads(Path(path).read_text());runtime=_verified_runtime(payload,resident_status)
    import torch
    devices=(len(resident_status['cuda_visible_devices'].split(','))
        if resident_status and 'cuda_visible_devices' in resident_status else
        (2 if payload['config'].get('shard_max_memory_mib') or payload['config'].get('resident') else 1))
    if torch_device(payload["config"])!="cpu" and (not torch.cuda.is_available() or torch.cuda.device_count()!=devices):
        raise ValueError("the explicitly configured CUDA device mapping is required")
    if resident_models is None:
        if payload["config"].get("resident"):raise ValueError("resident models must never be loaded by an ephemeral worker")
        torch.set_num_threads(payload["config"]["threads"])
        if not payload['config'].get('resident'):
            torch.set_num_interop_threads(1)
    torch.manual_seed(payload["request"]["seed"])
    parameters=json.loads(payload["request"]["messages"][0]["content"])
    started=time.monotonic()
    function={"qwen21":_qwen_image,"qwen38_review":_qwen_review,"sam3_instances":_sam_instances}[payload["config"]["kind"]]
    if resident_models is not None:
        components=resident_models[payload["config"]["kind"]]
        runtime["resident_pid"]=os.getpid();runtime["weights_reloaded"]=False
        output,usage,finish=function(payload,parameters,runtime,components=components)
    else:output,usage,finish=function(payload,parameters,runtime)
    if torch_device(payload["config"])=="cpu":
        import resource
        runtime.update(device="cpu",peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_gpu_memory_bytes=0)
    else:runtime.update(peak_gpu_memory_bytes=sum(torch.cuda.max_memory_allocated(i) for i in range(devices)),
        peak_gpu_memory_by_device=[torch.cuda.max_memory_allocated(i) for i in range(devices)],
        device="resident_cuda" if payload["config"].get("resident") else ("cuda_sharded" if devices==2 else "cuda:0"),
        gpu_names=[torch.cuda.get_device_name(i) for i in range(devices)])
    response={"model":payload["profile"]["model"],"revision":payload["profile"]["revision"],"output":output,
        "usage":{**usage,"wall_seconds":time.monotonic()-started},"finish_reason":finish,
        "provider_request_id":payload["request"]["request_id"],"runtime":runtime}
    atomic_write_json(Path(payload["native_output"]),{"binding":payload["binding"],"sha256":payload_sha256(response),"response":response})


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--payload",required=True)
    args=parser.parse_args()
    try:run_payload(args.payload)
    except Exception as exc:
        payload=json.loads(Path(args.payload).read_text())
        atomic_write_json(Path(payload["native_output"]).with_suffix(".failure.json"),{
            "binding":payload["binding"],"code":getattr(exc,"code",type(exc).__name__),"message":str(exc)[:2000],"failed_at":time.time()})
        raise
