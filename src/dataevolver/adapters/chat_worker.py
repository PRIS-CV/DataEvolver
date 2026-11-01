"""Offline Qwen vision/text worker, sharing CLIP's durable native effect transport.

CPU execution is explicit. No service, download, arbitrary model code, fallback
provider or implicit GPU allocation is started by a request.
"""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath

from .clip_worker import ClipCPUProvider
from dataevolver.dataset.contracts import file_sha256, identifier, payload_sha256
from dataevolver.harness.state_store import atomic_write_json


def validate_chat_config(config):
    fields = {"kind", "model_path_env", "python_env", "threads", "max_images", "file_hashes", "runtime_versions", "implementation_sha256",
              "precision", "image_side", "disable_mkldnn", "cpu_capability", "rotary_backend"}
    if set(config) != fields or config["kind"] != "qwen_vl_cpu":
        raise ValueError("unregistered local chat worker")
    if config["implementation_sha256"] != file_sha256(Path(__file__)):
        raise ValueError("chat worker implementation differs from frozen version")
    if (config["precision"] not in {"float32", "bfloat16"} or config["image_side"] not in {224, 256} or
            type(config["disable_mkldnn"]) is not bool or config["cpu_capability"] not in {"default", "avx2", "avx512"}
            or config["rotary_backend"] not in {"native", "numpy"}):
        raise ValueError("unsupported CPU precision or image preparation")
    for name in ("model_path_env", "python_env"):
        identifier(config[name], name)
    if type(config["threads"]) is not int or not 1 <= config["threads"] <= 16:
        raise ValueError("chat worker thread budget must be between one and sixteen")
    if type(config["max_images"]) is not int or not 1 <= config["max_images"] <= 8:
        raise ValueError("chat image batch must be between one and eight")
    hashes = config["file_hashes"]
    required = {"config.json", "model.safetensors.index.json", "preprocessor_config.json",
                "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "merges.txt", "vocab.json"}
    if not isinstance(hashes, dict) or not required.issubset(hashes):
        raise ValueError("chat model and processor inventory is incomplete")
    for name, digest in hashes.items():
        if (PurePosixPath(name).name != name or name in {".", ".."} or "\\" in name or
                not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid chat model file identity")
    versions = config["runtime_versions"]
    if (not isinstance(versions, dict) or not {"python", "torch", "transformers"}.issubset(versions) or
            set(versions) - {"python", "torch", "transformers", "tokenizers", "safetensors", "numpy", "pillow"} or
            any(not isinstance(v, str) or not v for v in versions.values())):
        raise ValueError("chat runtime versions must be explicit")


class ChatCPUProvider(ClipCPUProvider):
    worker_module = "dataevolver.adapters.chat_worker"
    require_images = False

    def __init__(self, profile, config, store, receipt_root):
        validate_chat_config(config)
        self.profile, self.config, self.store, self.root = profile, config, store, Path(receipt_root)


def run_payload(path):
    import importlib.metadata
    import platform
    import time

    payload = json.loads(Path(path).read_text())
    config, request, profile = payload["config"], payload["request"], payload["profile"]
    validate_chat_config(config)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or not 1 <= request["max_tokens"] <= 512:
        raise ValueError("CPU chat visibility or token budget changed")
    runtime = {"python": platform.python_version(), "device": "cpu", "threads": config["threads"],
               "cuda_visible_devices": "", "precision": config["precision"], "image_side": config["image_side"],
               "disable_mkldnn": config["disable_mkldnn"], "rotary_backend": config["rotary_backend"]}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "numpy", "pillow"):
        runtime[name] = importlib.metadata.version(name)
    if any(runtime[name] != version for name, version in config["runtime_versions"].items()):
        raise ValueError("chat runtime changed from frozen versions")
    model_path = Path(payload["model_path"]).resolve()
    for name, digest in config["file_hashes"].items():
        asset = (model_path / name).resolve()
        if not asset.is_relative_to(model_path) or not asset.is_file() or file_sha256(asset) != digest:
            raise ValueError("chat model inventory changed")
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    if not set(index["weight_map"].values()).issubset(config["file_hashes"]):
        raise ValueError("chat checkpoint shard is not hash-bound")
    if json.loads((model_path / "config.json").read_text()).get("model_type") != "qwen3_5":
        raise ValueError("worker supports the registered Qwen3.5 architecture only")
    if len(payload["images"]) > config["max_images"]:
        raise ValueError("chat image budget exceeded")
    for image in payload["images"]:
        if file_sha256(Path(image["path"])) != image["sha256"]:
            raise ValueError("chat image input changed")
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    started = time.monotonic()
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = not config["disable_mkldnn"]
    torch.manual_seed(request["seed"])
    runtime["cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    if os.environ.get("ATEN_CPU_CAPABILITY") != config["cpu_capability"]:
        raise ValueError("CPU kernel dispatch differs from frozen configuration")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, dtype=getattr(torch, config["precision"]), attn_implementation="eager").to("cpu").eval()
    if config["rotary_backend"] == "numpy":
        from types import MethodType
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionRotaryEmbedding, Qwen3_5TextRotaryEmbedding
        patched, text_patched = 0, 0
        for module in model.modules():
            if isinstance(module, Qwen3_5VisionRotaryEmbedding):
                module.forward = MethodType(numpy_vision_rotary, module)
                patched += 1
            elif isinstance(module, Qwen3_5TextRotaryEmbedding):
                if module.rope_type != "default":
                    raise ValueError("NumPy text rotary supports default bounded-context RoPE only")
                module.forward = MethodType(numpy_text_rotary, module)
                text_patched += 1
        if patched != 1 or text_patched != 1:
            raise ValueError("registered CPU rotary modules were not found exactly once")
        runtime["rotary_modules_patched"] = patched
        runtime["text_rotary_modules_patched"] = text_patched
    messages = [dict(m) for m in request["messages"]]
    messages.insert(0, {"role": "system", "content": "Return only JSON satisfying this schema. No markdown or explanation: " +
                     json.dumps(request["response_schema"], separators=(",", ":"))})
    if payload["images"]:
        content = []
        for value in payload["images"]:
            with Image.open(value["path"]) as opened:
                image = ImageOps_for_chat(opened, config["image_side"])
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": "Use the images in their exact order. Answer the requested JSON schema."})
        messages.append({"role": "user", "content": content})
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True, enable_thinking=False)
    if inputs["input_ids"].shape[-1] > 8192:
        raise ValueError("chat input token limit exceeded")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=request["max_tokens"], do_sample=False)
    tokens = output[0, inputs["input_ids"].shape[-1]:]
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    text = processor.decode(tokens, skip_special_tokens=True).strip()
    # A fenced JSON object is accepted as syntax only, never executable content.
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    value = json.loads(text)
    response = {"model": profile["model"], "revision": profile["revision"], "output": value,
                "finish_reason": "stop" if int(tokens[-1]) in eos else "length",
                "provider_request_id": request["request_id"], "runtime": runtime,
                "usage": {"input_tokens": int(inputs["attention_mask"].sum()), "output_tokens": len(tokens),
                          "images": len(payload["images"]), "cost": None, "wall_seconds": time.monotonic()-started}}
    atomic_write_json(Path(payload["output"]), {"binding": payload["binding"],
                      "sha256": payload_sha256(response), "response": response})


def ImageOps_for_chat(opened, image_side):
    from PIL import ImageOps
    from PIL import Image
    image = ImageOps.exif_transpose(opened).convert("RGB")
    return ImageOps.pad(image, (image_side, image_side), method=Image.Resampling.BICUBIC, color=(127, 127, 127))


def numpy_vision_rotary(module, x, position_ids):
    """Equivalent CPU vision RoPE, avoiding an unstable native trig kernel.

    This is explicitly selected in a frozen worker config, never a fallback
    chosen after a failed model request. Learned parameters remain unchanged.
    """
    import numpy as np
    import torch
    if x.device.type != "cpu":
        raise ValueError("NumPy rotary implementation is CPU-only")
    frequencies = position_ids[..., None].float() * module.inv_freq.to(device=x.device, dtype=torch.float32)
    values = frequencies.detach().numpy()
    cosine = torch.from_numpy(np.cos(values)) * module.attention_scaling
    sine = torch.from_numpy(np.sin(values)) * module.attention_scaling
    return module.recomposition_frequencies(cosine), module.recomposition_frequencies(sine)


def numpy_text_rotary(module, x, position_ids):
    """Default text RoPE with the same explicit CPU trig implementation."""
    import numpy as np
    import torch
    if x.device.type != "cpu" or module.rope_type != "default":
        raise ValueError("NumPy text rotary requires CPU and default RoPE")
    positions = position_ids.expand(3, -1, -1)
    frequencies = positions[..., None].float() * module.inv_freq.to(device=x.device, dtype=torch.float32)
    values = frequencies.detach().numpy()
    cosine = torch.from_numpy(np.cos(values)) * module.attention_scaling
    sine = torch.from_numpy(np.sin(values)) * module.attention_scaling
    return module.recomposition_frequencies(cosine).to(x.dtype), module.recomposition_frequencies(sine).to(x.dtype)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    run_payload(parser.parse_args().payload)
