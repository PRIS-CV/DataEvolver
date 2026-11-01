"""Local subprocess fixtures test routing/recovery, never real Planner quality."""
import json
import os
import sys

import pytest

from dataevolver.adapters.codex_planner import CodexCLIPlannerProvider, GatewayPlannerBackend
from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile, ModelRequest
from dataevolver.adapters.model_gateway import ModelError, ModelGateway
from dataevolver.dataset.contracts import ArtifactStore, file_sha256


def setup(tmp_path, monkeypatch, *, delay=0, output=None, model='gpt-6.1-sol', effort=None):
    binary = tmp_path / 'fixture-cli'
    binary.write_text('#!' + sys.executable + '\n' + f'''
import json,sys,time
from pathlib import Path
counter=Path({str(tmp_path / 'calls.json')!r})
counter.write_text(json.dumps(json.loads(counter.read_text())+1 if counter.exists() else 1))
time.sleep({delay})
output=Path(sys.argv[sys.argv.index('--output-last-message')+1])
output.write_text({json.dumps(output if output is not None else {'ok': True})!r})
print(json.dumps({{'type':'turn.completed','usage':{{'input_tokens':7,'output_tokens':4}}}}))
''')
    binary.chmod(0o700)
    instance = tmp_path / 'private-instance'
    instance.mkdir()
    (instance / 'auth.json').write_text('{}')
    monkeypatch.setenv('TEST_RSI_CLI', str(binary))
    monkeypatch.setenv('TEST_RSI_INSTANCE', str(instance))
    config = {'binary_env': 'TEST_RSI_CLI', 'instance_env': 'TEST_RSI_INSTANCE',
              'binary_sha256': file_sha256(binary)}
    if effort is not None:
        config['effort'] = effort
    profile = ModelProfile('fixture-cli', 'local_python', model, 'fixture-cli-revision')
    store = ArtifactStore({'dataset': tmp_path / 'data'})
    provider = CodexCLIPlannerProvider(profile, config, store, tmp_path / 'native')
    gateway = ModelGateway(profiles=(profile,),
        policy=ModelPolicy({'agent.runtime_rsi': profile.profile_id}, None, usage_mode='accounting'),
        receipt_root=tmp_path / 'ledger', artifact_store=store, providers={profile.profile_id: lambda: provider})
    return provider, gateway, config


SCHEMA = {'type': 'object', 'required': ['ok'], 'additionalProperties': False,
          'properties': {'ok': {'type': 'boolean'}}}


def test_explicit_model_and_effort_reach_native_cli(tmp_path, monkeypatch):
    _, gateway, _ = setup(tmp_path, monkeypatch, model='fixture-future-model', effort='max')
    gateway.invoke(request())
    invocation = json.loads((tmp_path / 'native/fixture-request/invocation.json').read_text())
    argv = invocation['argv']
    assert argv[argv.index('--model') + 1] == 'fixture-future-model'
    assert 'model_reasoning_effort="max"' in argv
    assert invocation['model'] == 'fixture-future-model'


def request(timeout=5):
    return ModelRequest('fixture-request', 'agent.runtime_rsi', ({'role': 'user', 'content': 'fixture'},),
                        SCHEMA, timeout_seconds=timeout)


def test_completed_cli_effect_recovers_without_second_subprocess(tmp_path, monkeypatch):
    provider, gateway, _ = setup(tmp_path, monkeypatch)
    assert provider.health()['authenticated_request_verified'] is False
    first = gateway.invoke(request())
    assert first.output == {'ok': True}
    assert first.usage['input_tokens'] == 7 and first.usage['output_tokens'] == 4
    assert gateway.recover(request()).without_reuse_hash() == first.without_reuse_hash()
    assert json.loads((tmp_path / 'calls.json').read_text()) == 1
    args = json.loads((tmp_path / 'native/fixture-request/invocation.json').read_text())['argv']
    assert args[args.index('--sandbox')+1] == 'read-only' and 'multi_agent' in args


def test_cli_timeout_is_not_resubmitted_and_has_failure_evidence(tmp_path, monkeypatch):
    provider, gateway, _ = setup(tmp_path, monkeypatch, delay=20)
    with pytest.raises(ModelError):
        gateway.invoke(request(timeout=1))
    with pytest.raises(ModelError):
        gateway.invoke(request(timeout=1))
    assert json.loads((tmp_path / 'calls.json').read_text()) == 1
    failed = json.loads((tmp_path / 'native/fixture-request/native.failure.json').read_text())
    assert failed['automatic_retry_allowed'] is False
    assert failed['wall_seconds'] >= 1


def test_tampered_cli_event_stream_rejects_native_recovery(tmp_path, monkeypatch):
    provider, gateway, _ = setup(tmp_path, monkeypatch)
    gateway.invoke(request())
    (tmp_path / 'native/fixture-request/events.jsonl').write_text('{}\n')
    with pytest.raises(ModelError, match='binding changed'):
        provider.lookup(request(), request().compute_hash())


def test_invalid_structured_output_is_not_a_verified_planner_completion(tmp_path, monkeypatch):
    provider, gateway, _ = setup(tmp_path, monkeypatch, output={'unexpected': 'not a decision'})
    with pytest.raises(ModelError):
        gateway.invoke(request())
    assert not (tmp_path / 'native/fixture-request/native.json').exists()


def test_gateway_backend_binds_context_to_a_stable_request_id(tmp_path, monkeypatch):
    _, gateway, _ = setup(tmp_path, monkeypatch)
    backend = GatewayPlannerBackend(gateway, model_id='gpt-6.1-sol', timeout_seconds=5)
    messages = [{'role': 'user', 'content': 'fixture context'}]
    assert backend.complete_json(messages, schema=SCHEMA) == '{"ok": true}'
    assert backend.complete_json(messages, schema=SCHEMA) == '{"ok": true}'
    assert json.loads((tmp_path / 'calls.json').read_text()) == 1
