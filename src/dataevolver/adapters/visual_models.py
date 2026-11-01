"""Exact local visual roles and offline asset inspection, without loading weights.

This module supplies providers to ModelGateway. It does not own a request budget,
choose replacement models, or qualify a model from an inventory result.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactRef, file_sha256, identifier, payload_sha256
from .training_media_worker import TrainingMediaProvider, validate_worker


SCHEMA = "dataevolver.local_visual_worker.v1"
TEXT_ROLES = frozenset({'operator.build', 'agent.runtime_rsi', 'knowledge.distill'})
MODELS = {
    "qwen21": "Qwen/Qwen-Image-2.1",
    "pe_t2i": "Qwen/Qwen-Image-2.1-PE-T2I",
    "pe_i2i": "Qwen/Qwen-Image-2.1-PE-I2I",
    "qwen38_review": "Qwen/Qwen3.8-27B",
    "sam3": "SAM/sam3",
    "h3_turbo": "lightx2v/Minimax-h3-Turbo",
}
ROLES = {
    "visual.image.generate": "qwen21", "visual.image.edit": "qwen21",
    "visual.prompt.t2i": "pe_t2i", "visual.prompt.i2i": "pe_i2i",
    "visual.inspect": "qwen38_review", "visual.inspect.video": "qwen38_review", "visual.segment.image": "sam3",
    "visual.review.image": "qwen38_review", "visual.review.video": "qwen38_review",
    "visual.caption.refine": "qwen38_review",
    "visual.segment.video": "sam3", "visual.video.generate": "h3_turbo",
}
ROLES.update({role: 'qwen38_review' for role in TEXT_ROLES})
ARCHITECTURES = {
    "qwen21": ("model_index.json", "_class_name", "QwenImage21Pipeline"),
    "pe_t2i": ("config.json", "architectures", ["Qwen3_5ForConditionalGeneration"]),
    "pe_i2i": ("config.json", "architectures", ["Qwen3_5ForConditionalGeneration"]),
    "qwen38_review": ("config.json", "architectures", ["Qwen3_5ForConditionalGeneration"]),
    "sam3": ("config.json", "architectures", ["Sam3VideoModel"]),
    "h3_turbo": ("model_index.json", "_class_name", "MiniMaxH3ModularPipeline"),
}
H3_REVISION = "cc9429192971815afff8368f74bd10adcc6fed48"
H3_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors"
H3_LORA_SHA256 = "9b0efe3613b43a84e30febaa43af27432ea9d0711eac7bba904b2556b175f6d4"
PRECISION = {
    "qwen21": "torchao-fp8-dynamic", "pe_t2i": "torchao-fp8-dynamic",
    "pe_i2i": "torchao-fp8-dynamic", "qwen38_review": "torchao-fp8-dynamic",
    "sam3": "fp32", "h3_turbo": "lightx2v-fp8-torchao",
}


def inspect_assets(kind, root, *, hash_weights=False):
    """Inspect only required components; PE roots may contain unrelated pipelines."""
    if kind not in MODELS:
        raise ValueError("unknown visual model kind")
    root = Path(root).resolve()
    name, field, expected = ARCHITECTURES[kind]
    if json.loads((root/name).read_text()).get(field) != expected:
        raise ValueError("selected model architecture differs from its role")
    components = [root]
    if kind == "qwen21":
        components = [root/part for part in ("transformer", "text_encoder", "vae")]
    elif kind == "h3_turbo":
        # transformer_ref is for reference conditioning; this adapter is I2AV.
        components = [root/part for part in ("transformer", "text_encoder", "vae", "audio_vae")]
    files = {name}
    weights = set()
    for component in components:
        if not component.is_dir():
            raise ValueError("required model component is absent: "+component.name)
        indexes = sorted(component.glob("*.safetensors.index.json"))
        for index in indexes:
            files.add(index.relative_to(root).as_posix())
            mapping = json.loads(index.read_text()).get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("weight index has no tensor mapping")
            for value in set(mapping.values()):
                path = component/value
                if not path.resolve().is_relative_to(root) or not path.is_file() or not path.stat().st_size:
                    raise ValueError("weight index references a missing or escaped shard")
                weights.add(path.relative_to(root).as_posix())
        if not indexes:
            candidates = list(component.glob("*.safetensors"))
            if not candidates:
                raise ValueError("required component has no safetensors weights")
            weights.update(p.relative_to(root).as_posix() for p in candidates)
        files.update(p.relative_to(root).as_posix() for p in component.glob("*.json"))
    if kind.startswith("pe_"):
        if not (root/"system_prompt.txt").is_file():
            raise ValueError("prompt enhancement requires its released system prompt")
        files.add("system_prompt.txt")
    # Tokenizers/processors are execution assets too, even though not weights.
    for directory in [root, root/"tokenizer", root/"processor", root/"scheduler", root/"audio_scheduler"]:
        if directory.is_dir():
            files.update(p.relative_to(root).as_posix() for p in directory.iterdir()
                         if p.is_file() and p.suffix in {".json", ".txt", ".model", ".jinja"})
    for name in files | weights:
        if not (root/name).resolve().is_relative_to(root) or not (root/name).stat().st_size:
            raise ValueError("empty or escaped model asset")
    hashes = {name: file_sha256(root/name) for name in sorted(files | (weights if hash_weights else set()))}
    return {"kind": kind, "model": MODELS[kind], "architecture_checked": True,
            "weight_files": sorted(weights), "weight_bytes": sum((root/n).stat().st_size for n in weights),
            "file_hashes": hashes, "weights_hashed": hash_weights,
            "inference_performed": False, "execution_ready": False,
            "precision": PRECISION[kind]}


def validate_visual_worker(config):
    if config.get("schema_version") != SCHEMA or config.get("kind") not in MODELS:
        raise ValueError("unregistered local visual worker")
    kind = config["kind"]
    if config.get("precision") != PRECISION[kind]:
        raise ValueError("precision must match the explicit component implementation")
    if config.get("execution_device") != "cuda" or config.get("shard_max_memory_mib"):
        raise ValueError("local visual workers require a single-GPU deployment")
    if config.get('resident') and kind == 'h3_turbo':
        raise ValueError('H3 has no registered resident visual implementation')
    if config["implementation_sha256"] != file_sha256(Path(__file__).with_name("local_visual_worker.py")):
        raise ValueError("local visual worker implementation changed")
    if config.get("binding_sha256") != file_sha256(Path(__file__)):
        raise ValueError("visual role binding implementation changed")
    if config.get("transport_sha256") != file_sha256(Path(__file__).with_name("training_media_worker.py")):
        raise ValueError("visual transport implementation changed")
    if type(config.get("max_frames")) is not int or not 1 <= config["max_frames"] <= 256:
        raise ValueError("visual video frame ceiling must be explicit")
    if kind == 'h3_turbo' and config['max_frames'] < 22:
        raise ValueError('Turbo8 frame ceiling must permit at least one exact 22-frame request')
    if not isinstance(config.get('gpu_uuid'), str) or not config['gpu_uuid'].startswith('GPU-'):
        raise ValueError('visual deployment must freeze its physical GPU UUID')
    common = {k: v for k, v in config.items() if k not in
              {"schema_version", "precision", "binding_sha256", "transport_sha256", "dependencies", "max_frames", "gpu_uuid"}}
    common["kind"] = "qwen21"
    common["implementation_sha256"] = file_sha256(Path(__file__).with_name("training_media_worker.py"))
    validate_worker(common)
    offload = config['resource_profile']['cpu_offload']
    if kind == 'h3_turbo' and not offload:
        raise ValueError('Turbo8 requires the explicit single-GPU CPU-offload profile')
    if kind not in {'h3_turbo', 'qwen21'} and offload:
        raise ValueError('this component has no registered CPU-offload implementation')
    deps = config.get("dependencies", {})
    if set(deps) != ({"lora", "fp8", "backend", "adaln"} if kind == "h3_turbo" else set()):
        raise ValueError("model dependencies must be explicitly frozen")
    for value in deps.values():
        if set(value) != {"path_env", "file_hashes"} or not value["file_hashes"]:
            raise ValueError("dependency requires an environment reference and file hashes")
        identifier(value["path_env"])
        for path, digest in value["file_hashes"].items():
            ArtifactRef("asset", path, digest)
    if kind == 'h3_turbo' and deps['lora']['file_hashes'] != {H3_LORA: H3_LORA_SHA256}:
        raise ValueError('Turbo8 must bind the exact released 8-step LoRA')
    required_packages = {"torch", "transformers"}
    if kind in {'h3_turbo', 'qwen38_review', 'sam3'}: required_packages.add('av')
    if kind == "qwen21": required_packages |= {"diffusers", "torchao"}
    elif kind != "h3_turbo" and kind != "sam3": required_packages.add("torchao")
    if not required_packages <= set(config["runtime_versions"]):
        raise ValueError("precision backend runtime versions must be frozen")


def verify_assets(config):
    root = Path(os.environ[config["model_path_env"]]).resolve()
    inventory = inspect_assets(config["kind"], root)
    required = set(inventory["file_hashes"]) | set(inventory["weight_files"])
    if not required <= set(config["file_hashes"]):
        raise ValueError("worker must freeze all required weights and processor assets")
    roots = {"model": (root, config["file_hashes"])}
    for name, dep in config["dependencies"].items():
        roots[name] = (Path(os.environ[dep["path_env"]]).resolve(), dep["file_hashes"])
    if 'backend' in roots:
        source, hashes = roots['backend']
        required_python = {p.relative_to(source).as_posix() for p in (source/'lightx2v').rglob('*.py')}
        if not required_python or not required_python <= set(hashes):
            raise ValueError('all imported LightX2V Python sources must be frozen')
    for name, (directory, hashes) in roots.items():
        for relative, digest in hashes.items():
            path = (directory/relative).resolve()
            if not path.is_relative_to(directory) or file_sha256(path) != digest:
                raise ValueError("frozen visual asset changed: "+name+"/"+relative)
    return {name: str(path) for name, (path, _) in roots.items()}


class LocalVisualProvider(TrainingMediaProvider):
    worker_module = "dataevolver.adapters.local_visual_worker"

    def __init__(self, profile, config, store, dataset_store, receipt_root):
        validate_visual_worker(config)
        if (profile.backend != "local_python" or profile.model != MODELS[config["kind"]] or
                profile.revision != "sha256:"+payload_sha256(config)):
            raise ValueError("visual profile differs from the frozen selected model")
        self.profile, self.config, self.store = profile, config, store
        self.dataset_store, self.root = dataset_store, Path(receipt_root)

    def invoke(self, request, request_hash):
        validate_visual_request(request, self.config)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != self.config['gpu_uuid']:
            raise ValueError('visual deployment GPU mapping differs from its frozen UUID')
        return super().invoke(request, request_hash)


def validate_h3_parameters(parameters, config, input_count):
    required = {'prompt', 'width', 'height', 'num_frames', 'steps', 'fps'}
    if (set(parameters) != required or input_count != 1 or parameters['steps'] != 8 or parameters['fps'] != 24
            or type(parameters['num_frames']) is not int or not 22 <= parameters['num_frames'] <= config['max_frames']
            or (parameters['num_frames'] - 5) % 17
            or any(type(parameters[n]) is not int or parameters[n] % 32
                   or not 32 <= parameters[n] <= config['image_side'] for n in ('width', 'height'))):
        raise ValueError('Turbo8 requires exact 17*n+5 frames starting at 22, 8 steps, 24 FPS and bounded dimensions')


def validate_visual_request(request, config):
    if ROLES.get(request.role) != config["kind"]:
        raise ValueError("visual role bound to the wrong model")
    if request.role in TEXT_ROLES:
        if request.artifacts:
            raise ValueError('bounded text reasoning roles cannot consume undeclared media')
        return
    text_only = request.role in {"visual.image.generate", "visual.prompt.t2i"}
    image_count = len(request.artifacts)
    multi_image = request.role in {"visual.image.edit", "visual.prompt.i2i"}
    if (len(request.messages) != 1 or request.messages[0]["role"] != "user" or
            (text_only and image_count) or
            (multi_image and not 1 <= image_count <= 10) or
            (not text_only and not multi_image and image_count != 1)):
        raise ValueError("visual request has the wrong message or media count")
    if request.artifacts:
        media = "video/" if request.role in {"visual.segment.video", "visual.inspect.video", "visual.review.video"} else "image/"
        if not all(ref.media_type.startswith(media) for ref in request.artifacts):
            raise ValueError("visual role received an unsupported media type")
    parameters = json.loads(request.messages[0]["content"])
    if "prompt" in parameters and (not isinstance(parameters["prompt"], str) or not parameters["prompt"].strip()):
        raise ValueError("visual prompt must be nonempty text")
    if config['kind'] == 'h3_turbo':
        validate_h3_parameters(parameters, config, len(request.artifacts))
    if config["kind"] == "qwen21":
        import math
        if (set(parameters)-{"negative_prompt"} != {"prompt", "width", "height", "steps", "true_cfg_scale"} or
                any(type(parameters[k]) is not int or not 16 <= parameters[k] <= config["image_side"] or parameters[k] % 16
                    for k in ("width", "height")) or
                type(parameters["steps"]) is not int or not 1 <= parameters["steps"] <= 100 or
                type(parameters["true_cfg_scale"]) not in {int, float} or
                not math.isfinite(parameters["true_cfg_scale"]) or not 1 <= parameters["true_cfg_scale"] <= 20):
            raise ValueError("image parameters exceed the frozen worker limits")
    if config["kind"] == "sam3":
        subjects = parameters.get("subjects", [])
        if not subjects or len(subjects) > 16:
            raise ValueError("SAM3 requires a bounded nonempty subject list")
        for subject in subjects:
            identifier(subject["subject_id"]); identifier(subject["target_instance_id"])
            if not isinstance(subject["description"], str) or not subject["description"].strip():
                raise ValueError("SAM3 requires an observed text description")
        if len({s["subject_id"] for s in subjects}) != len(subjects):
            raise ValueError("SAM3 subject IDs must be unique")
