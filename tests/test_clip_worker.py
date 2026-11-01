import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataevolver.adapters import clip_worker
from dataevolver.adapters.clip_worker import ClipCPUProvider
from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile, ModelRequest
from dataevolver.adapters.model_gateway import ModelError, ModelGateway
from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, file_sha256, payload_sha256
from dataevolver.harness.state_store import atomic_write_json


def settings(tmp_path, monkeypatch):
    config = {"kind": "clip_cpu", "model_path_env": "TEST_CLIP_MODEL", "python_env": "TEST_CLIP_PYTHON",
              "threads": 2, "max_images": 8,
              "file_hashes": {n: "0" * 64 for n in ("config.json", "preprocessor_config.json", "tokenizer.json", "model.safetensors",
                                                     "tokenizer_config.json", "special_tokens_map.json", "merges.txt", "vocab.json")}}
    monkeypatch.setenv("TEST_CLIP_MODEL", str(tmp_path / "weights"))
    monkeypatch.setenv("TEST_CLIP_PYTHON", sys.executable)
    image = tmp_path / "image.png"
    image.write_bytes(b"image bytes handled by a fixture provider")
    store = ArtifactStore({"images": tmp_path})
    ref = ArtifactRef("images", image.name, file_sha256(image), "image/png")
    profile = ModelProfile("clip", "local_python", "fixture-clip", "fixture-revision", ("json", "vision"))
    request = ModelRequest("score", "embedder", ({"role": "user", "content": '{"texts":["leaf","other"]}'},),
                           {"type": "object", "properties": {"scores": {"type": "array"}}, "required": ["scores"]},
                           (ref,), max_tokens=1)
    return config, store, profile, request


def test_native_receipt_recovers_crash_without_repeated_inference(tmp_path, monkeypatch):
    config, store, profile, request = settings(tmp_path, monkeypatch)
    child_calls = []

    def child(argv, **kwargs):
        child_calls.append(argv)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "" and kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        job = json.loads(Path(argv[-1]).read_text())
        response = {"model": profile.model, "output": {"scores": [{"artifact_sha256": request.artifacts[0].sha256,
                    "similarities": [.8, .1]}]}, "finish_reason": "stop", "usage": {"images": 1, "output_tokens": 0}}
        atomic_write_json(Path(job["output"]), {"binding": job["binding"], "response": response, "sha256": payload_sha256(response)})
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(clip_worker.subprocess, "run", child)
    provider = lambda: ClipCPUProvider(profile, config, store, tmp_path / "native")
    gateway = ModelGateway(profiles=(profile,), policy=ModelPolicy({"embedder": "clip"}, 1),
                           receipt_root=tmp_path / "gateway", artifact_store=store, providers={"clip": provider})
    receipt_method = gateway._receipt
    monkeypatch.setattr(gateway, "_receipt", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("crash before gateway receipt")))
    with pytest.raises(RuntimeError, match="crash"):
        gateway.invoke(request)
    monkeypatch.setattr(gateway, "_receipt", receipt_method)
    recovered = gateway.invoke(request)
    assert recovered.reused and len(child_calls) == 1
    assert json.loads((tmp_path / "gateway/score.intent.json").read_text())["attempts"] == 1
    monkeypatch.delenv("TEST_CLIP_MODEL")
    monkeypatch.delenv("TEST_CLIP_PYTHON")
    gateway.factories["clip"] = lambda: (_ for _ in ()).throw(AssertionError("complete cache created a worker"))
    assert gateway.invoke(request).reused


def test_unconfirmed_worker_timeout_cannot_be_automatically_reissued(tmp_path, monkeypatch):
    config, store, profile, request = settings(tmp_path, monkeypatch)
    calls = []

    def timeout(argv, **kwargs):
        calls.append(argv)
        raise clip_worker.subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(clip_worker.subprocess, "run", timeout)
    gateway = ModelGateway(profiles=(profile,), policy=ModelPolicy({"embedder": "clip"}, 1),
                           receipt_root=tmp_path / "gateway", artifact_store=store,
                           providers={"clip": lambda: ClipCPUProvider(profile, config, store, tmp_path / "native")})
    for _ in range(2):
        with pytest.raises(ModelError) as error:
            gateway.invoke(request)
        assert error.value.code == "AMBIGUOUS_EFFECT"
    assert len(calls) == 1


def test_changed_model_inventory_and_native_receipt_fail_closed(tmp_path, monkeypatch):
    config, store, profile, request = settings(tmp_path, monkeypatch)
    provider = ClipCPUProvider(profile, config, store, tmp_path / "native")
    provider.root.mkdir()
    atomic_write_json(provider._path(request), {"binding": "changed", "response": {}, "sha256": payload_sha256({})})
    with pytest.raises(ModelError) as error:
        provider.lookup(request, "f" * 64)
    assert error.value.code == "INVALID_RECEIPT"
    bad = dict(config, threads=128)
    with pytest.raises(ValueError, match="bounded"):
        ClipCPUProvider(profile, bad, store, tmp_path)
