"""Offline CPU CLIP provider with a subprocess deadline and native receipts.

Heavy imports occur only in the child, after gateway cache/recovery checks.
Weights stay in their existing read-only model cache; no automatic downloads.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path, PurePosixPath

from dataevolver.dataset.contracts import file_sha256, identifier, payload_sha256, positive_int
from dataevolver.harness.state_store import atomic_write_json
from .model_gateway import ModelError


def validate_worker_config(config):
    required = {"kind", "model_path_env", "python_env", "threads", "max_images", "file_hashes"}
    if not required.issubset(config) or set(config) - required - {"runtime_versions"} or config["kind"] != "clip_cpu":
        raise ValueError("only the registered offline CPU CLIP worker is supported")
    for name in ("model_path_env", "python_env"):
        identifier(config[name], name)
    for name in ("threads", "max_images"):
        positive_int(config[name], name)
    if config["threads"] > 16 or config["max_images"] > 64:
        raise ValueError("CPU worker exceeds its bounded thread/image contract")
    hashes = config["file_hashes"]
    required_files = {"config.json", "preprocessor_config.json", "model.safetensors", "tokenizer.json",
                      "tokenizer_config.json", "special_tokens_map.json", "merges.txt", "vocab.json"}
    if not isinstance(hashes, dict) or not required_files.issubset(hashes):
        raise ValueError("CLIP requires a bound model and processor file inventory")
    for name, digest in hashes.items():
        if PurePosixPath(name).name != name or name in {".", ".."} or "\\" in name:
            raise ValueError("model inventory contains an invalid filename")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("model inventory requires SHA-256 hashes")
    versions = config.get("runtime_versions", {})
    if (not isinstance(versions, dict) or set(versions) - {"python", "torch", "transformers", "tokenizers", "safetensors", "numpy", "pillow"}
            or any(not isinstance(v, str) or not v for v in versions.values())):
        raise ValueError("invalid CPU worker runtime version contract")


class ClipCPUProvider:
    worker_module = "dataevolver.adapters.clip_worker"
    require_images = True
    def __init__(self, profile, config, store, receipt_root):
        validate_worker_config(config)
        self.profile, self.config, self.store, self.root = profile, config, store, Path(receipt_root)

    def health(self):
        return {"status": "CONFIGURED", "model": self.profile.model, "device": "cpu", "inference_performed": False}

    def _binding(self, request_hash):
        return payload_sha256([self.profile.to_dict(), self.config, request_hash])

    def _path(self, request):
        return self.root / f"{request.request_id}.native.json"

    def lookup(self, request, request_hash):
        path = self._path(request)
        if not path.exists():
            raise ModelError("AMBIGUOUS_EFFECT", "CPU worker has no committed native result; automatic resend is forbidden")
        try:
            value = json.loads(path.read_text())
            if value["binding"] != self._binding(request_hash) or value["sha256"] != payload_sha256(value["response"]):
                raise ValueError("native CPU model receipt binding changed")
            return value["response"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ModelError("INVALID_RECEIPT", "native CPU model receipt is invalid") from exc

    def invoke(self, request, request_hash):
        if self._path(request).exists():
            return self.lookup(request, request_hash)
        if (self.require_images and not request.artifacts) or len(request.artifacts) > self.config["max_images"]:
            raise ModelError("INVALID_INPUT", "CPU worker image batch is empty or exceeds its bound")
        model_path = os.environ.get(self.config["model_path_env"], "")
        python = os.environ.get(self.config["python_env"], "")
        if not model_path or not python or not Path(python).is_file():
            raise ModelError("CONFIG_MISSING", "CPU worker interpreter/model environment is missing", retry_safe=True)
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"binding": self._binding(request_hash), "profile": self.profile.to_dict(), "config": self.config,
                   "model_path": model_path, "request": request.to_dict(), "output": str(self._path(request)),
                   "images": [{"path": str(self.store.resolve(ref)), "sha256": ref.sha256} for ref in request.artifacts]}
        job = self.root / f"{request.request_id}.job.json"
        atomic_write_json(job, payload)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   OMP_NUM_THREADS=str(self.config["threads"]), MKL_NUM_THREADS=str(self.config["threads"]),
                   TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
        env["PYTHONFAULTHANDLER"] = "1"
        if "cpu_capability" in self.config:
            env["ATEN_CPU_CAPABILITY"] = self.config["cpu_capability"]
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            env.pop(name, None)
        try:
            result = subprocess.run([python, "-m", self.worker_module, "--payload", str(job)],
                                    env=env, capture_output=True, text=True, timeout=request.timeout_seconds,
                                    check=False, shell=False)
        except subprocess.TimeoutExpired:
            if self._path(request).exists():
                return self.lookup(request, request_hash)
            raise ModelError("AMBIGUOUS_EFFECT", "CPU model worker exceeded its deadline and was stopped") from None
        (self.root / f"{request.request_id}.log").write_text(f"return_code={result.returncode}\n" + (result.stdout + result.stderr)[-32000:])
        if result.returncode != 0:
            raise ModelError("LOCAL_WORKER_FAILED", f"CPU model worker failed (exit {result.returncode}); inspect its private runtime log")
        return self.lookup(request, request_hash)


def run_payload(path):
    import importlib.metadata
    import platform
    payload = json.loads(Path(path).read_text())
    config, request, profile = payload["config"], payload["request"], payload["profile"]
    validate_worker_config(config)
    runtime = {"python": platform.python_version(), "device": "cpu", "threads": config["threads"],
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "numpy", "pillow"):
        runtime[name] = importlib.metadata.version(name)
    if any(runtime[name] != expected for name, expected in config.get("runtime_versions", {}).items()):
        raise ValueError("CPU worker runtime changed from the frozen version contract")
    model_path = Path(payload["model_path"]).resolve()
    for name, digest in config["file_hashes"].items():
        asset = (model_path / name).resolve()
        if not asset.is_relative_to(model_path) or not asset.is_file() or file_sha256(asset) != digest:
            raise ValueError("cached model/processor inventory changed")
    if json.loads((model_path / "config.json").read_text()).get("model_type") != "clip":
        raise ValueError("cached model is not CLIP")
    texts = json.loads(request["messages"][0]["content"])["texts"]
    if not 2 <= len(texts) <= 16 or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("unsupported image-text scoring task")
    if len(payload["images"]) > config["max_images"]:
        raise ValueError("CPU image budget exceeded")
    for image in payload["images"]:
        if file_sha256(Path(image["path"])) != image["sha256"]:
            raise ValueError("CPU model image changed")
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
    model = CLIPModel.from_pretrained(model_path, local_files_only=True).to("cpu").eval()
    scores, tokens = [], 0
    with torch.inference_mode():
        for offset in range(0, len(payload["images"]), 4):
            values = payload["images"][offset:offset + 4]
            images = []
            for value in values:
                with Image.open(value["path"]) as opened:
                    images.append(opened.convert("RGB"))
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
            tokens += int(inputs["attention_mask"].sum().item())
            output = model(**inputs)
            similarities = (output.logits_per_image / model.logit_scale.exp()).float().clamp(-1, 1).tolist()
            scores.extend({"artifact_sha256": value["sha256"], "similarities": similarity}
                          for value, similarity in zip(values, similarities))
    response = {"model": profile["model"], "revision": profile["revision"], "output": {"scores": scores},
                "finish_reason": "stop", "provider_request_id": request["request_id"],
                "runtime": runtime,
                "usage": {"input_tokens": tokens, "output_tokens": 0, "images": len(scores), "cost": None}}
    atomic_write_json(Path(payload["output"]), {"binding": payload["binding"], "sha256": payload_sha256(response), "response": response})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    run_payload(parser.parse_args().payload)
