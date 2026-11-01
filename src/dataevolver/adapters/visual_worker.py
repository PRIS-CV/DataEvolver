"""Registered offline G2 workers with durable effects and one subprocess per call.

ModelGateway reserves every invocation, including failures. This module owns
only the media effect; it cannot plan, retry, alter budgets or select a model.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path, PurePosixPath

from dataevolver.dataset.contracts import ArtifactRef, canonical_json, file_sha256, identifier, payload_sha256
from dataevolver.harness.state_store import atomic_write_json
from .model_gateway import ModelError


def validate_visual_worker(config):
    required = {"kind", "model_path_env", "python_env", "file_hashes", "runtime_versions", "threads"}
    if set(config) != required or config["kind"] not in {"qwen_edit_gpu", "qwen_vl_cpu"}:
        raise ValueError("unregistered visual worker")
    for name in ("model_path_env", "python_env"):
        identifier(config[name])
    if type(config["threads"]) is not int or not 1 <= config["threads"] <= 16:
        raise ValueError("invalid visual worker CPU budget")
    if not config["file_hashes"] or not config["runtime_versions"]:
        raise ValueError("visual worker requires frozen assets and runtime versions")
    for name, digest in config["file_hashes"].items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
            raise ValueError("invalid model inventory path")
        if len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest):
            raise ValueError("invalid model file hash")


class VisualLocalProvider:
    def __init__(self, profile, config, store, dataset_store, receipt_root):
        validate_visual_worker(config)
        self.profile, self.config, self.store = profile, config, store
        self.dataset_store, self.root = dataset_store, Path(receipt_root)

    def health(self):
        return {"status": "CONFIGURED", "inference_performed": False, "kind": self.config["kind"]}

    def _path(self, request):
        return self.root / f"{request.request_id}.native.json"

    def _binding(self, request_hash):
        return payload_sha256([request_hash, self.profile.to_dict(), self.config, self.dataset_store])

    def lookup(self, request, request_hash):
        path = self._path(request)
        if not path.exists():
            raise ModelError("AMBIGUOUS_EFFECT", "no committed visual effect; automatic resubmission is forbidden")
        try:
            record = json.loads(path.read_text())
            response = record["response"]
            if record["binding"] != self._binding(request_hash) or record["sha256"] != payload_sha256(response):
                raise ValueError("visual native effect binding changed")
            if self.config["kind"] == "qwen_edit_gpu":
                ref = ArtifactRef.from_dict(response["output"]["artifact"])
                if ref.store != self.dataset_store:
                    raise ValueError("editing output store changed")
                image = self.store.resolve(ref)
                from PIL import Image
                with Image.open(image) as opened:
                    opened.load()
                    if list(opened.size) != response["output"]["image_size"]:
                        raise ValueError("edited image dimensions changed")
            return response
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ModelError("INVALID_RECEIPT", "visual native effect or image is invalid") from exc

    def invoke(self, request, request_hash):
        if self._path(request).exists():
            return self.lookup(request, request_hash)
        model = os.environ.get(self.config["model_path_env"], "")
        python = os.environ.get(self.config["python_env"], "")
        if not model or not Path(python).is_file():
            raise ModelError("CONFIG_MISSING", "visual model or interpreter is missing", retry_safe=True)
        if len(request.artifacts) > 2 or (self.config["kind"] == "qwen_edit_gpu" and len(request.artifacts) != 1):
            raise ModelError("INVALID_INPUT", "visual input count changed", retry_safe=True)
        if self.config["kind"] == "qwen_edit_gpu" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise ModelError("CONFIG_MISSING", "editing requires an explicitly selected GPU", retry_safe=True)
        gpu_launch = None
        if self.config["kind"] == "qwen_edit_gpu":
            # Recheck the selected physical GPU for every new real invocation.
            # It is shared; this worker never terminates another user's process.
            try:
                gpu = subprocess.run(["nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
                         "--query-gpu=uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20, check=True)
                uuid, free, utilization = [x.strip() for x in gpu.stdout.strip().split(",")]
                gpu_launch = {"uuid": uuid, "free_mib": int(free), "utilization": int(utilization)}
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                raise ModelError("RESOURCE_UNAVAILABLE", "cannot verify selected GPU before invocation", retry_safe=True) from exc
            if int(free) < 100000 or int(utilization) > 30:
                raise ModelError("RESOURCE_BUSY", "selected GPU no longer meets the frozen 100000 MiB/30 percent launch policy", retry_safe=True)
        self.root.mkdir(parents=True, exist_ok=True)
        output_relative = f"images/{request_hash}.png"
        image_output = self.store.roots[self.dataset_store] / output_relative
        payload = {"binding": self._binding(request_hash), "request": request.to_dict(),
                   "config": self.config, "profile": self.profile.to_dict(), "model_path": model,
                   "images": [{"path": str(self.store.resolve(ref)), "sha256": ref.sha256} for ref in request.artifacts],
                   "image_output": str(image_output), "output_relative": output_relative,
                   "dataset_store": self.dataset_store, "native_output": str(self._path(request)), "gpu_launch": gpu_launch}
        job = self.root / f"{request.request_id}.job.json"
        atomic_write_json(job, payload)
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   OMP_NUM_THREADS=str(self.config["threads"]), MKL_NUM_THREADS=str(self.config["threads"]),
                   PYTHONFAULTHANDLER="1", PYTHONDONTWRITEBYTECODE="1")
        if self.config["kind"] == "qwen_vl_cpu":
            env.update(CUDA_VISIBLE_DEVICES="", ATEN_CPU_CAPABILITY="avx2")
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            env.pop(name, None)
        started = time.monotonic()
        log = self.root / f"{request.request_id}.log"
        with log.open("a", buffering=1) as handle:
            handle.write(canonical_json({"event": "started", "request_hash": request_hash}) + "\n")
            try:
                result = subprocess.run([python, "-m", "dataevolver.adapters.visual_worker", "--payload", str(job)],
                                        env=env, stdout=handle, stderr=subprocess.STDOUT,
                                        timeout=request.timeout_seconds, check=False, shell=False)
                handle.write(canonical_json({"return_code": result.returncode, "wall_seconds": time.monotonic()-started}) + "\n")
            except subprocess.TimeoutExpired:
                handle.write(canonical_json({"event": "timeout", "wall_seconds": time.monotonic()-started}) + "\n")
                if self._path(request).exists():
                    return self.lookup(request, request_hash)
                raise ModelError("AMBIGUOUS_EFFECT", "visual worker timed out; its image is not a committed candidate") from None
        if result.returncode != 0:
            raise ModelError("LOCAL_WORKER_FAILED", f"visual worker failed with exit {result.returncode}; no successful receipt")
        return self.lookup(request, request_hash)


def _verify_assets(payload):
    import importlib.metadata as metadata
    import platform
    config = payload["config"]
    validate_visual_worker(config)
    runtime = {"python": platform.python_version()}
    for name, version in config["runtime_versions"].items():
        runtime[name] = platform.python_version() if name == "python" else metadata.version(name)
        if runtime[name] != version:
            raise ValueError(f"frozen runtime changed: {name}")
    root = Path(payload["model_path"]).resolve()
    for name, digest in config["file_hashes"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != digest:
            raise ValueError(f"frozen model asset changed: {name}")
    for row in payload["images"]:
        if file_sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError("visual input image changed")
    runtime["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    runtime["gpu_launch"] = payload.get("gpu_launch")
    return runtime


def run_payload(path):
    payload = json.loads(Path(path).read_text())
    runtime = _verify_assets(payload)
    import torch
    from PIL import Image, ImageOps
    torch.set_num_threads(payload["config"]["threads"])
    torch.set_num_interop_threads(1)
    request, profile = payload["request"], payload["profile"]
    torch.manual_seed(request["seed"])
    started = time.monotonic()
    kind = payload["config"]["kind"]
    if kind == "qwen_edit_gpu":
        from diffusers import QwenImage21Pipeline
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("editing requires exactly one explicitly mapped GPU")
        runtime.update(device="cuda:0", gpu_name=torch.cuda.get_device_name(0))
        parameters = json.loads(request["messages"][0]["content"])
        if set(parameters) != {"prompt", "seed", "width", "height", "steps", "true_cfg_scale", "guidance_scale", "negative_prompt"}:
            raise ValueError("editing parameters changed")
        if parameters["seed"] != request["seed"]:
            raise ValueError("request and generator seeds differ")
        output = Path(payload["image_output"])
        # An orphaned file never becomes a successful receipt or a cache hit.
        if output.exists():
            raise ValueError("uncommitted output already exists; reconcile before another invocation")
        pipe = QwenImage21Pipeline.from_pretrained(payload["model_path"], torch_dtype=torch.bfloat16,
                                                       local_files_only=True).to("cuda:0")
        with Image.open(payload["images"][0]["path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        generator = torch.Generator(device="cuda:0").manual_seed(parameters["seed"])
        with torch.inference_mode():
            result = pipe(image=image, prompt=parameters["prompt"], generator=generator,
                          height=parameters["height"], width=parameters["width"],
                          num_inference_steps=parameters["steps"], true_cfg_scale=parameters["true_cfg_scale"],
                          negative_prompt=parameters["negative_prompt"],
                          num_images_per_prompt=1)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.png")
        result.images[0].save(temporary)
        with Image.open(temporary) as opened:
            opened.load()
            if opened.size != (parameters["width"], parameters["height"]):
                raise ValueError("editing backend returned unexpected dimensions")
        temporary.replace(output)
        ref = ArtifactRef(payload["dataset_store"], payload["output_relative"], file_sha256(output), "image/png")
        runtime["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
        value = {"artifact": ref.to_dict(), "actual_parameters": parameters,
                 "image_size": [parameters["width"], parameters["height"]], "runtime": runtime}
        usage = {"images": 1, "input_tokens": None, "output_tokens": None, "cost": None}
        finish = "stop"
    else:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        from types import MethodType
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionRotaryEmbedding, Qwen3_5TextRotaryEmbedding
        from .chat_worker import numpy_vision_rotary, numpy_text_rotary
        # Keep the already validated CPU dispatch/RoPE path explicit.
        torch.backends.mkldnn.enabled = False
        processor = AutoProcessor.from_pretrained(payload["model_path"], local_files_only=True)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(payload["model_path"], local_files_only=True,
                    dtype=torch.bfloat16, attn_implementation="eager").to("cpu").eval()
        for module in model.modules():
            if isinstance(module, Qwen3_5VisionRotaryEmbedding):
                module.forward = MethodType(numpy_vision_rotary, module)
            if isinstance(module, Qwen3_5TextRotaryEmbedding):
                if module.rope_type != "default":
                    raise ValueError("unsupported CPU text RoPE")
                module.forward = MethodType(numpy_text_rotary, module)
        messages = [{"role": "system", "content": "Return only JSON. No markdown. Follow this schema exactly: " +
                     canonical_json(request["response_schema"])}] + list(request["messages"])
        if payload["images"]:
            content = []
            for row in payload["images"]:
                with Image.open(row["path"]) as opened:
                    image = ImageOps.pad(ImageOps.exif_transpose(opened).convert("RGB"), (448, 448))
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": "Images are in the specified order. Use uncertain when evidence is insufficient."})
            messages.append({"role": "user", "content": content})
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                         return_tensors="pt", return_dict=True, enable_thinking=False)
        if inputs["input_ids"].shape[-1] > 8192:
            raise ValueError("visual review input budget exceeded")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=request["max_tokens"], do_sample=False)
        tokens = generated[0, inputs["input_ids"].shape[-1]:]
        raw = processor.decode(tokens, skip_special_tokens=True).strip()
        (Path(payload["native_output"]).with_suffix(".raw.txt")).write_text(raw)
        if raw.startswith("```json\n") and raw.endswith("```"):
            raw = raw[8:-3].strip()
        value = json.loads(raw)
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        finish = "stop" if len(tokens) and int(tokens[-1]) in eos else "length"
        usage = {"images": len(payload["images"]), "input_tokens": int(inputs["attention_mask"].sum()),
                 "output_tokens": len(tokens), "cost": None}
    response = {"model": profile["model"], "revision": profile["revision"], "output": value,
                "usage": {**usage, "wall_seconds": time.monotonic()-started}, "runtime": runtime,
                "finish_reason": finish, "provider_request_id": request["request_id"]}
    atomic_write_json(Path(payload["native_output"]), {"binding": payload["binding"],
                      "sha256": payload_sha256(response), "response": response})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    run_payload(parser.parse_args().payload)
