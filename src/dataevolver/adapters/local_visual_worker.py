"""Offline visual execution under the existing ModelGateway native receipt ABI."""
from __future__ import annotations

import json
import os
import platform
import time
from collections import Counter
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactStore, file_sha256, payload_sha256
from dataevolver.dataset.supervision import write_artifact, write_image
from dataevolver.harness.state_store import atomic_write_json
from .model_contracts import ModelProfile, ModelRequest
from .model_gateway import ModelError
from .training_media_worker import _qwen_image, _qwen_review, _sam_instances, deployment_observation
from .visual_models import (H3_REVISION, LocalVisualProvider, MODELS, ROLES, TEXT_ROLES,
                            validate_h3_parameters, validate_visual_request, validate_visual_worker, verify_assets)


def _storage_dtypes(model):
    """Inspect tensor subclass storage, not its outward BF16 logical dtype."""
    counts = Counter()
    def visit(tensor):
        flatten = getattr(tensor, "__tensor_flatten__", None)
        if flatten:
            names, _ = flatten()
            for name in names:
                value = getattr(tensor, name)
                if hasattr(value, "dtype"): visit(value)
        else:
            counts[str(tensor.dtype)] += tensor.numel()
    for parameter in model.parameters(): visit(parameter)
    return dict(counts)


def _precision(model, runtime, component):
    dtypes = _storage_dtypes(model)
    if not any("float8" in name and count for name, count in dtypes.items()):
        raise ValueError("requested FP8 did not produce actual FP8 parameter storage")
    runtime["precision"] = {"backend": "torchao", "mode": "dynamic_activation_float8_weight",
        "component": component, "storage_elements_by_dtype": dtypes,
        "other_components": "released BF16/FP32 precision; no whole-model FP8 claim"}


def _load_qwen(payload, runtime):
    import torch
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, TorchAoConfig
    kind = payload["config"]["kind"]
    quant = Float8DynamicActivationFloat8WeightConfig()
    root = payload["model_path"]
    if kind == "qwen21":
        from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
        from diffusers import TorchAoConfig as DiffusersAoConfig
        transformer = QwenImage21Transformer2DModel.from_pretrained(root, subfolder="transformer",
            torch_dtype=torch.bfloat16, local_files_only=True, device_map={"": 0},
            quantization_config=DiffusersAoConfig(quant))
        _precision(transformer, runtime, "image transformer")
        return QwenImage21Pipeline.from_pretrained(root, transformer=transformer,
            torch_dtype=torch.bfloat16, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(root, dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa", device_map={"": 0},
        quantization_config=TorchAoConfig(quant, modules_to_not_convert=["lm_head", "visual"]))
    _precision(model, runtime, "language model linear layers; vision encoder excluded")
    return {"model": model.eval(), "processor": AutoProcessor.from_pretrained(root, local_files_only=True)}


def _pe(payload, parameters, runtime, components):
    import torch
    from PIL import Image, ImageOps
    from .training_prompt_enhancement import parse_refinement
    kind = payload["config"]["kind"]
    count = len(payload["inputs"])
    if set(parameters) != {"prompt"} or (count != 0 if kind == "pe_t2i" else not 1 <= count <= 10):
        raise ValueError("PE requires its exact text/image prompt role")
    content = []
    for item in payload["inputs"]:
        with Image.open(item["path"]) as image:
            content.append({"type": "image", "image": ImageOps.exif_transpose(image).convert("RGB")})
    content.append({"type": "text", "text": parameters["prompt"]})
    messages = [{"role": "system", "content": (Path(payload["model_path"])/"system_prompt.txt").read_text()},
                {"role": "user", "content": content}]
    processor, model = components["processor"], components["model"]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True, enable_thinking=True).to("cuda:0")
    if inputs["input_ids"].shape[-1] > payload["config"]["max_input_tokens"]:
        raise ValueError("PE input exceeds its frozen context bound")
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=True, temperature=1., top_p=.95, top_k=20,
                                   max_new_tokens=payload["request"]["max_tokens"])
    tokens = generated[0, inputs["input_ids"].shape[-1]:]
    raw = processor.decode(tokens, skip_special_tokens=True).strip()
    Path(payload["native_output"]).with_suffix(".raw.txt").write_text(raw)
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    return parse_refinement(raw, kind), {"images": len(payload["inputs"]),
        "input_tokens": int(inputs["attention_mask"].sum()), "output_tokens": len(tokens), "cost": None}, (
        "stop" if len(tokens) and int(tokens[-1]) in eos else "length")


def _video_frames(path, *, max_frames, image_side):
    import av
    frames = []
    with av.open(str(path)) as video:
        for frame in video.decode(video=0):
            if len(frames) >= max_frames or max(frame.width, frame.height) > image_side:
                raise ValueError("video exceeds its frozen frame/dimension bound")
            frames.append(frame.to_image().convert("RGB"))
    if not frames: raise ValueError("video has no decodable frames")
    return frames


def _review(payload, parameters, runtime, components):
    if payload["request"]["role"] not in {"visual.inspect.video", "visual.review.video"}:
        return _qwen_review(payload, parameters, runtime, components=components)
    frames = _video_frames(payload["inputs"][0]["path"],
        max_frames=payload["config"]["max_frames"], image_side=payload["config"]["image_side"])
    indices = sorted({0, len(frames)//2, len(frames)-1})
    directory = Path(payload["dataset_root"])/payload["output_prefix"]
    directory.mkdir(parents=True, exist_ok=True)
    inputs = []
    for index in indices:
        path = directory/f"review-frame-{index:05d}.png"; frames[index].save(path)
        inputs.append({"path": str(path)})
    runtime["video_observation"] = {"source": payload["inputs"][0]["ref"], "frame_indices": indices,
                                    "decoded_frame_count": len(frames), "scope": "sampled frames only"}
    return _qwen_review({**payload, "inputs": inputs}, parameters, runtime, components=components)


def _text(payload, runtime, components):
    """The same loaded selected model, using the original bounded text messages."""
    import torch
    request = ModelRequest.from_dict(payload['request'])
    messages = [dict(m) for m in request.messages]
    instruction = 'Return only JSON satisfying this exact schema: ' + json.dumps(request.response_schema)
    if messages[0]['role'] == 'system':
        messages[0]['content'] += '\n' + instruction
    else:
        messages.insert(0, {'role': 'system', 'content': instruction})
    processor, model = components['processor'], components['model']
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pt', enable_thinking=False).to('cuda:0')
    tokens_in = inputs['input_ids'].shape[-1]
    if tokens_in + request.max_tokens > payload['config']['max_input_tokens']:
        raise ValueError('text request and output exceed the frozen context capacity')
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=request.max_tokens, do_sample=False)
    tokens = generated[0, tokens_in:]
    raw = processor.decode(tokens, skip_special_tokens=True).strip()
    Path(payload['native_output']).with_suffix('.raw.txt').write_text(raw)
    if raw.startswith('```json\n') and raw.endswith('```'):
        raw = raw[8:-3].strip()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    runtime['text_request'] = {'role': request.role, 'thinking': False,
        'rendered_messages_sha256': payload_sha256(messages)}
    return json.loads(raw), {'images': 0, 'input_tokens': tokens_in,
        'output_tokens': len(tokens), 'cost': None}, 'stop' if len(tokens) and int(tokens[-1]) in eos else 'length'


def _sam_video(payload, parameters, runtime):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    if len(payload["inputs"]) != 1 or set(parameters) != {"subjects"} or not parameters["subjects"]:
        raise ValueError("video SAM requires one video and observed text subjects")
    prompts = [subject["description"] for subject in parameters["subjects"]]
    if len(set(prompts)) != len(prompts):
        raise ValueError("ambiguous identical text prompts cannot identify distinct subjects")
    frames = _video_frames(payload["inputs"][0]["path"],
        max_frames=payload["config"]["max_frames"], image_side=payload["config"]["image_side"])
    processor = Sam3VideoProcessor.from_pretrained(payload["model_path"], local_files_only=True)
    model, info = Sam3VideoModel.from_pretrained(payload["model_path"], local_files_only=True,
        dtype=torch.float32, output_loading_info=True)
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise ValueError("SAM3 video checkpoint is incomplete")
    model = model.to("cuda:0").eval()
    parameter_dtypes = _storage_dtypes(model)
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('SAM3 video floating parameters must remain FP32')
    session = processor.init_video_session(video=frames, inference_device="cuda:0",
        inference_state_device="cpu", video_storage_device="cpu", dtype=torch.float32)
    session = processor.add_text_prompt(session, text=prompts)
    store = ArtifactStore({payload["dataset_store"]: payload["dataset_root"]})
    records, identities = [], {}
    subjects = parameters['subjects']
    counts = {'gateway_requests':1, 'input_videos':1, 'decoded_frames':len(frames),
        'subjects':len(subjects), 'frame_forward_attempts':0, 'frame_subject_units_attempted':0}
    usage_path = Path(payload['native_output']).with_suffix('.sam-usage.json')
    def write_counts():atomic_write_json(usage_path, {'binding':payload['binding'], **counts})
    original_forward = model.forward
    def counted_forward(*args, **kwargs):
        frame = kwargs.get('frame_idx')
        if type(frame) is not int or frame != counts['frame_forward_attempts'] or frame >= len(frames):
            raise ModelError('ACCOUNTING_MISMATCH', 'SAM3 video attempted an unreserved or repeated frame')
        counts['frame_forward_attempts'] += 1
        counts['frame_subject_units_attempted'] += len(subjects)
        write_counts()
        return original_forward(*args, **kwargs)
    write_counts(); model.forward = counted_forward
    try:
        with torch.inference_mode():
            for result in model.propagate_in_video_iterator(session, max_frame_num_to_track=len(frames)):
                if result.frame_idx != len(records):
                    raise ValueError('SAM3 returned frames in an unexpected order')
                processed = processor.postprocess_outputs(session, result)
                objects = processed["object_ids"].tolist()
                current = []
                for subject in parameters["subjects"]:
                    found = processed["prompt_to_obj_ids"].get(subject["description"], [])
                    if len(found) != 1 or found[0] not in objects:
                        raise ModelError("QUALITY_REJECTED", "SAM3 tracking has missing or ambiguous subject")
                    identity = int(found[0])
                    if subject["subject_id"] in identities and identities[subject["subject_id"]] != identity:
                        raise ModelError("QUALITY_REJECTED", "SAM3 changed a tracked subject identity")
                    identities[subject["subject_id"]] = identity
                    mask = processed["masks"][objects.index(identity)].detach().cpu().numpy()
                    if mask.ndim == 3 and mask.shape[0] == 1:mask = mask[0]
                    if not np.isfinite(mask).all():raise ModelError('QUALITY_REJECTED', 'SAM3 video returned nonfinite mask values')
                    mask = mask.astype(np.uint8)*255
                    if mask.shape != (frames[result.frame_idx].height,frames[result.frame_idx].width) or not mask.any():
                        raise ModelError('QUALITY_REJECTED', 'SAM3 video returned an empty or incorrectly sized instance mask')
                    ref = write_image(store, payload["dataset_store"], payload["output_prefix"]+
                        f"/frame-{len(records):05d}-{subject['subject_id']}.png", Image.fromarray(mask))
                    current.append({"subject_id": subject["subject_id"], "object_id": identity, "mask": ref.to_dict()})
                if len(set(identities.values())) != len(identities):
                    raise ModelError("QUALITY_REJECTED", "distinct subjects map to the same tracked object")
                records.append({"frame_index": len(records), "instances": current})
    finally:
        model.forward = original_forward
    if len(records) != len(frames): raise ValueError("SAM3 did not cover every input frame")
    if counts['frame_forward_attempts'] != len(frames):raise ValueError('SAM3 did not account for every frame forward')
    runtime["precision"] = {"component": "SAM3 detector and tracker", "mode": "fp32", 'storage_elements_by_dtype':parameter_dtypes}
    runtime['sam_video_counts'] = counts
    return {"frames": records, "frame_count": len(frames)}, {
        "images": 1, "input_tokens": None, "output_tokens": None, "cost": None}, "stop"


def _h3(payload, parameters, runtime, roots):
    import subprocess
    import sys
    validate_h3_parameters(parameters, payload['config'], len(payload['inputs']))
    import torch
    source = Path(roots["backend"])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != H3_REVISION:
        raise ValueError("LightX2V differs from the verified Turbo8 backend revision")
    sys.path.insert(0, str(source))
    os.environ.update(DTYPE="BF16", SENSITIVE_LAYER_DTYPE="BF16")
    from lightx2v import LightX2VPipeline
    from lightx2v.utils.set_config import build_startup_config
    from lightx2v.models.networks.minimax_h3.adaln_cache import _build_spec, _cache_path, _validate_cache
    dependencies = payload["config"]["dependencies"]
    def one(name, suffix):
        candidates = [Path(roots[name])/p for p in dependencies[name]["file_hashes"] if p.endswith(suffix)]
        if len(candidates) != 1: raise ValueError("dependency must name one exact "+name+" asset")
        return str(candidates[0])
    config = {"infer_steps": 8, "fps": 24, "enable_cfg": False, "cpu_offload": True,
        "offload_granularity": "block", "use_adaln_cache": True, "adaln_cache_dir": roots["adaln"],
        "text_encoder_cpu_offload": True, "text_encoder_offload_granularity": "block",
        "text_encoder_release_block_offload_buffers": True, "vae_cpu_offload": True,
        "video_vae_offload_granularity": "block", "vae_decode_parallel": False,
        "attn_type": "torch_sdpa", "qwen3vl_attn_type": "torch_sdpa", "rms_type": "torch",
        "rope_type": "minimax_h3_triton_rope", "feature_caching": "NoCaching",
        "use_compile": False, "vae_use_compile": False, "video_flow_shift": 6., "audio_flow_shift": 3.,
        "h3_step_update": "training_euler", "vae_spatial_scale_factor": 16,
        "dit_quantized": True, "dit_quant_scheme": "fp8-torchao", "dit_quantized_ckpt": one("fp8", ".safetensors"),
        "lora_dynamic_apply": True, "lora_configs": [{"path": one("lora", ".safetensors"), "strength": 1., "alpha": 8}],
        "num_frames": parameters["num_frames"], "size": [parameters["height"], parameters["width"]]}
    directory = Path(payload["dataset_root"])/payload["output_prefix"]
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory/"backend-config.json"; atomic_write_json(config_path, config)
    startup = build_startup_config({"model_cls": "minimax_h3", "model_variant": "fl2av", "task": "i2av",
                                    "model_path": payload["model_path"], "config_json": str(config_path)})
    cache = _cache_path(startup)
    for name in ('manifest.json', 'adaln_cache.safetensors'):
        path = cache/name
        key = path.resolve().relative_to(Path(roots['adaln']).resolve()).as_posix()
        if dependencies['adaln']['file_hashes'].get(key) != file_sha256(path):
            raise ValueError('AdaLN manifest and table bytes must both match the frozen dependency')
    if not _validate_cache(cache, _build_spec(startup)):
        raise ValueError("frozen AdaLN cache does not match the Turbo8 schedule")
    pipe = LightX2VPipeline(task="i2av", model_path=payload["model_path"], model_cls="minimax_h3", model_variant="fl2av")
    pipe.create_generator(config_json=str(config_path))
    model = pipe.runner.model
    objects = list(model._iter_weight_objects(model.pre_weight, model.transformer_weights, model.post_weight))
    weights = [getattr(obj, "weight", None) if getattr(obj, "weight", None) is not None
               else getattr(obj, "pin_weight", None) for obj in objects]
    dtypes = Counter(str(weight.dtype) for weight in weights if isinstance(weight, torch.Tensor))
    branches = sum(bool(getattr(obj, "has_lora_branch", False)) for obj in objects)
    if not dtypes.get("torch.float8_e4m3fn") or not branches:
        raise ValueError("H3 did not load actual FP8 weights and Turbo LoRA branches")
    original, evaluations = model.infer, []
    def infer(inputs):
        evaluations.append(1)
        return original(inputs)
    model.infer = infer
    result = pipe.generate(task="i2av", prompt=parameters["prompt"], image_path=payload["inputs"][0]["path"],
        seed=payload["request"]["seed"], num_frames=parameters["num_frames"],
        size=(parameters["height"], parameters["width"]), return_result_tensor=True)
    video, audio = result["video"], result["audio"]
    if (len(evaluations) != 8 or tuple(video.shape) != (1, parameters["num_frames"], 3, parameters["height"], parameters["width"]) or
            not torch.isfinite(video).all() or not torch.isfinite(audio).all() or not audio.numel() or result["sampling_rate"] != 32000):
        raise ValueError('H3 actual NFE or decoded tensors differ from the requested profile: '
            + str({'actual_nfe': len(evaluations), 'actual_video_shape': list(video.shape),
                   'expected_video_shape': [1, parameters['num_frames'], 3, parameters['height'], parameters['width']],
                   'actual_audio_shape': list(audio.shape), 'actual_sampling_rate': result['sampling_rate']}))
    from lightx2v.utils.ltx2_media_io import encode_video
    from lightx2v.models.video_encoders.hf.ltx2.audio_vae.ops import Audio
    target = directory/"video.mp4"
    frames = (video[0].permute(0, 2, 3, 1).float()*255).round().clamp(0, 255).to(torch.uint8).contiguous()
    encode_video(video=frames, fps=24, audio=Audio(waveform=audio[0].float(), sampling_rate=32000),
                 output_path=str(target), video_chunks_number=1)
    import av
    with av.open(str(target)) as container:
        stream = container.streams.video[0]
        if (float(stream.average_rate) != 24 or [stream.width, stream.height] != [parameters["width"], parameters["height"]] or
                sum(1 for _ in container.decode(video=0)) != parameters["num_frames"]):
            raise ValueError("encoded H3 video failed independent full decode")
    with av.open(str(target)) as container:
        if not container.streams.audio or not sum(f.samples for f in container.decode(audio=0)):
            raise ValueError("H3 audio failed independent decode")
    store = ArtifactStore({payload["dataset_store"]: payload["dataset_root"]})
    ref = write_artifact(store, payload["dataset_store"], payload["output_prefix"]+"/verified.mp4", target.read_bytes(), "video/mp4")
    runtime.update(precision={"backend": "LightX2V", "dit": "fp8-torchao", "lora": "bf16",
        "sensitive_layers": "bf16/fp32", "loaded_weight_dtypes": dict(dtypes), "lora_branches": branches},
        backend_revision=H3_REVISION, actual_nfe=len(evaluations), backend_config_sha256=file_sha256(config_path))
    return {"artifact": ref.to_dict(), "actual_parameters": parameters,
            "frame_count": parameters["num_frames"], "fps": 24}, {
        "images": 1, "input_tokens": None, "output_tokens": None, "cost": None}, "stop"


def run_payload(path, *, resident_models=None, resident_status=None):
    import importlib.metadata as metadata
    payload = json.loads(Path(path).read_text())
    config = payload["config"]; validate_visual_worker(config)
    request = ModelRequest.from_dict(payload["request"])
    validate_visual_request(request, config)
    profile = ModelProfile.from_dict(payload["profile"])
    store = ArtifactStore({payload["dataset_store"]: payload["dataset_root"]})
    provider = LocalVisualProvider(profile, config, store, payload["dataset_store"], Path(payload["native_output"]).parent)
    if (provider.binding(payload["request_hash"]) != payload["binding"] or ROLES.get(request.role) != config["kind"] or
            len(payload["inputs"]) != len(request.artifacts) or len(request.artifacts) > config["max_images"]):
        raise ValueError("visual worker request binding changed")
    for value, ref in zip(payload["inputs"], request.artifacts):
        if value["ref"] != ref.to_dict() or file_sha256(Path(value["path"])) != ref.sha256:
            raise ValueError("visual input artifact binding changed")
    for name, expected in config["runtime_versions"].items():
        actual = platform.python_version() if name == "python" else metadata.version(name)
        if actual != expected: raise ValueError("frozen visual runtime changed: "+name)
    entry = resident_models[config['kind']] if resident_models is not None else None
    if bool(config.get('resident')) != (entry is not None and resident_status is not None):
        raise ValueError('resident visual execution requires the actual loaded host and its status')
    roots = {'model': entry['model_path']} if entry is not None else verify_assets(config)
    if str(Path(payload["model_path"]).resolve()) != roots["model"]:
        raise ValueError("visual model root changed")
    runtime = {"runtime_versions": config["runtime_versions"],
               "deployment": deployment_observation(config, resident_status=resident_status),
               "implementation_sha256": config["implementation_sha256"]}
    placement = runtime['deployment']
    uuids = [d['uuid'] for d in placement['devices']] if 'devices' in placement else [placement['uuid']]
    if uuids != [config['gpu_uuid']]:
        raise ValueError('the physical GPU differs from the frozen deployment')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("one explicitly mapped CUDA device is required")
    torch.set_num_threads(config["threads"])
    if entry is None:
        torch.set_num_interop_threads(1)
    torch.manual_seed(request.seed)
    parameters = None if request.role in TEXT_ROLES else json.loads(request.messages[0]["content"])
    started = time.monotonic(); kind = config["kind"]
    if kind == "h3_turbo": output, usage, finish = _h3(payload, parameters, runtime, roots)
    elif kind == "sam3":
        if request.role == "visual.segment.video": output, usage, finish = _sam_video(payload, parameters, runtime)
        else:
            output, usage, finish = _sam_instances(payload, parameters, runtime,
                components=entry['components'] if entry is not None else None)
            runtime["precision"] = {"component": "SAM3 image detector", "mode": "fp32"}
    else:
        if entry is None:
            components = _load_qwen(payload, runtime)
        else:
            components = entry['components']
            runtime.update(entry['loading_runtime'])
        if request.role in TEXT_ROLES: output, usage, finish = _text(payload, runtime, components)
        elif kind == "qwen21": output, usage, finish = _qwen_image(payload, parameters, runtime, components=components)
        elif kind.startswith("pe_"): output, usage, finish = _pe(payload, parameters, runtime, components)
        else: output, usage, finish = _review(payload, parameters, runtime, components)
    runtime.update(peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(), gpu_name=torch.cuda.get_device_name(0),
        resident=entry is not None, weights_reloaded=entry is None)
    response = {"model": MODELS[kind], "revision": profile.revision, "output": output,
        "usage": {**usage, "wall_seconds": time.monotonic()-started}, "finish_reason": finish,
        "provider_request_id": request.request_id, "runtime": runtime}
    atomic_write_json(Path(payload["native_output"]), {"binding": payload["binding"], "sha256": payload_sha256(response), "response": response})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--payload", required=True)
    args = parser.parse_args()
    try: run_payload(args.payload)
    except Exception as exc:
        payload = json.loads(Path(args.payload).read_text())
        atomic_write_json(Path(payload["native_output"]).with_suffix(".failure.json"), {
            "binding": payload["binding"], "code": getattr(exc, "code", type(exc).__name__),
            "message": str(exc)[:2000], "failed_at": time.time()})
        raise
