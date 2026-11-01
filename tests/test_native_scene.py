"""Real CPU subprocess fixtures; no Blender, UE5 or model inference."""
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from dataevolver.harness.adapters.native_scene import NativeSceneOperator
from dataevolver.harness.models import EvidenceStatus, OperatorCall


def fixture_scene(tmp_path, *, fail=False, delay=0):
    source = tmp_path / "input.json"
    source.write_text('{"fixture":true}')
    worker = tmp_path / "worker.py"
    worker.write_text('''import hashlib,json,sys,time
from pathlib import Path
request=json.loads(Path(sys.argv[1]).read_text());root=Path(sys.argv[2])
counter=root/'calls.txt';counter.write_text(str(int(counter.read_text())+1 if counter.exists() else 1))
''' + f'time.sleep({delay})\n' + ('sys.exit(7)\n' if fail else '''
(root/'summary.json').write_text(json.dumps({'fixture':True}))
digest=hashlib.sha256(json.dumps(request,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
(root/'result.json').write_text(json.dumps({'status':'completed','request_sha256':digest}))
'''))
    config = {"schema_version": "dataevolver.native_scene.v2", "stage": "custom_worker",
        "argv": [str(Path(sys.executable).resolve()), str(worker), "{request}", "{output}"],
        "files": {str(Path(sys.executable).resolve()): file_sha256(Path(sys.executable)), str(worker): file_sha256(worker)},
        "inputs": {"scene_spec": {"path": str(source), "sha256": file_sha256(source)}},
        "outputs": {"summary": "summary.json"}, "timeout_seconds": 1 if delay else 10,
        "allowed_env_keys": [], "environment_sha256": payload_sha256({})}
    operator = NativeSceneOperator(output_root=tmp_path / "campaigns/scene", configuration=config)
    call = OperatorCall("scene", "one", "one", operator.spec.operator_id, "1", 0,
                        str(operator.output_root / "calls/one"))
    return operator, call, source, worker


@pytest.mark.parametrize("value", ["other-model", None, ""])
def test_worker_environment_drift_blocks_dispatch(tmp_path, monkeypatch, value):
    operator, call, _, _ = fixture_scene(tmp_path)
    key = "DATAEVOLVER_FIXTURE_MODEL"
    config = {**operator.configuration, "allowed_env_keys": [key],
              "environment_sha256": payload_sha256({key: "selected-model"})}
    operator = NativeSceneOperator(output_root=operator.output_root, configuration=config)
    if value is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("changed environment dispatched a worker"))
    with pytest.raises(ValueError, match="environment changed"):
        operator.execute(call)
    assert not (Path(call.work_dir) / "native/intent.json").exists()


def test_worker_compiler_binds_environment_values_without_storing_them(tmp_path, monkeypatch):
    from dataevolver.workflows.scene.prepare import prepare
    operator, _, source, worker = fixture_scene(tmp_path)
    key = "DATAEVOLVER_FIXTURE_MODEL"
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"stage": "custom_worker", "argv": operator.configuration["argv"],
        "files": [str(worker)], "inputs": {"source": str(source)}, "outputs": {"summary": "summary.json"},
        "timeout_seconds": 10, "allowed_env_keys": [key], "allow_gpu": False, "allow_network": False}))
    hashes = []
    for value, directory in (("selected-model", "first"), ("other-model", "second")):
        monkeypatch.setenv(key, value)
        prepare(profile, tmp_path / directory, "scene", tmp_path / "campaigns")
        text = (tmp_path / directory / "goal.json").read_text()
        goal = json.loads(text)
        config = goal["objective"]["host_workload"]["configuration"]["configuration"]
        assert config["schema_version"] == "dataevolver.native_scene.v2"
        assert config["environment_sha256"] == payload_sha256({key: value})
        assert goal["objective"]["initial_trial_key"]["environment_contract_hash"] == config["environment_sha256"]
        assert value not in text
        hashes.append(config["environment_sha256"])
    assert hashes[0] != hashes[1]


def test_scene_completion_recovery_and_native_receipt_reuse(tmp_path):
    operator, call, _, _ = fixture_scene(tmp_path)
    receipt = operator.execute(call)
    assert operator.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    assert operator.execute(call).reused
    native = Path(call.work_dir) / "native"
    assert json.loads((native / "native.json").read_text())["quality_verified"] is False
    # Crash after native completion, before the outer workload receipt commits.
    operator.path(call).unlink()
    operator.execute(call)
    assert (native / "calls.txt").read_text() == "1"


@pytest.mark.parametrize("changed", ["input", "worker", "output", "exit", "result"])
def test_scene_rejects_drift_without_launching_again(tmp_path, changed):
    operator, call, source, worker = fixture_scene(tmp_path)
    receipt = operator.execute(call)
    native = Path(call.work_dir) / "native"
    target = {"input": source, "worker": worker, "output": native / "summary.json",
              "exit": native / "exit.json", "result": native / "result.json"}[changed]
    target.write_text('{}')
    assert operator.verify_effect(call, receipt).status is EvidenceStatus.CONTRADICTED
    with pytest.raises(ValueError):
        operator.execute(call)
    assert (native / "calls.txt").read_text() == "1"


@pytest.mark.parametrize("delay", [0, 20])
def test_scene_failure_and_timeout_retain_evidence_without_resubmission(tmp_path, delay):
    operator, call, _, _ = fixture_scene(tmp_path, fail=not delay, delay=delay)
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
        operator.execute(call)
    native = Path(call.work_dir) / "native"
    exit_record = json.loads((native / "exit.json").read_text())
    assert exit_record["timed_out"] is bool(delay)
    if not delay:
        assert exit_record["returncode"] == 7
    with pytest.raises(RuntimeError, match="reconcile"):
        operator.execute(call)
    assert (native / "calls.txt").read_text() == "1"


def test_scene_rejects_escaping_outputs_and_call_paths(tmp_path):
    operator, call, _, _ = fixture_scene(tmp_path)
    with pytest.raises(ValueError):
        NativeSceneOperator(output_root=operator.output_root,
                            configuration={**operator.configuration, "stage": "../escape"})
    config = {**operator.configuration, "outputs": {"summary": "../outside.json"}}
    with pytest.raises(ValueError, match="relative"):
        NativeSceneOperator(output_root=operator.output_root, configuration=config)
    with pytest.raises(ValueError, match="root"):
        operator.execute(replace(call, work_dir=str(tmp_path / "outside")))


def test_native_blender_export_checks_input_before_using_scene_api(tmp_path, monkeypatch):
    import runpy
    from types import SimpleNamespace
    from dataevolver.paths import SRC_ROOT
    source = tmp_path / 'scene.blend'
    source.write_bytes(b'fixture-input')
    request = tmp_path / 'request.json'
    request.write_text(json.dumps({'stage': 'blender', 'inputs': {
        'scene': {'path': str(source), 'sha256': '0'*64}}}))
    monkeypatch.setitem(sys.modules, 'bpy', SimpleNamespace())
    monkeypatch.setattr(sys, 'argv', ['blender', '--', '--request', str(request), '--output', str(tmp_path / 'native')])
    with pytest.raises(ValueError, match='input changed'):
        runpy.run_path(str(SRC_ROOT / 'dataevolver/workflows/scene/ue5/export_frozen_scene.py'))


@pytest.mark.parametrize("command", ["worker-compile", "scene-compile"])
def test_compiled_scene_uses_existing_controller_and_cli(tmp_path, capsys, command):
    from dataevolver.harness.cli import main
    operator, _, source, worker = fixture_scene(tmp_path)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"stage": "custom_worker", "argv": operator.configuration["argv"],
        "files": [str(worker)], "inputs": {"scene_spec": str(source)},
        "outputs": {"summary": "summary.json"}, "timeout_seconds": 10, "allowed_env_keys": [],
        "allow_gpu": False, "allow_network": False}))
    campaigns = tmp_path / "campaigns"
    compiled = tmp_path / "compiled"
    assert main([command, "--profile", str(profile), "--output", str(compiled),
                 "--campaign", "scene", "--campaign-root", str(campaigns)]) == 0
    assert not campaigns.exists()
    assert main(["create", "--goal", str(compiled / "goal.json"), "--campaign-root", str(campaigns),
                 "--idempotency-key", "create-scene"]) == 0
    assert main(["step", "--campaign", "scene", "--campaign-root", str(campaigns), "--max-actions", "8"]) == 0
    assert main(["status", "--campaign", "scene", "--campaign-root", str(campaigns)]) == 0
    last = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert last["state"] == "COMPLETE"
    counters = list(campaigns.rglob("calls.txt"))
    assert len(counters) == 1 and counters[0].read_text() == "1"
