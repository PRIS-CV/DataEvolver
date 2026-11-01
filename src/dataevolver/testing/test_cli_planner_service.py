"""Loopback native-lookup transport fixtures; no real model is used here."""
import json
import multiprocessing
import socket
import time

import pytest

from dataevolver.adapters.cli_planner_service import CLIPlannerEndpointProvider, serve
from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile
from dataevolver.adapters.model_gateway import ModelGateway, ModelError
from dataevolver.dataset.contracts import ArtifactStore
from dataevolver.testing.test_codex_planner_provider import setup, request


@pytest.fixture
def remote(tmp_path, monkeypatch):
    _, _, worker = setup(tmp_path, monkeypatch)
    profile = ModelProfile('fixture-cli', 'local_python', 'gpt-6.1-sol', 'fixture-cli-revision')
    token = tmp_path / 'secret'; token.write_text('fixture-private-token')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    config = {'port': port, 'token_file': str(token), 'root': str(tmp_path / 'service'),
              'profile': profile.to_dict(), 'worker': worker}
    process = multiprocessing.get_context('fork').Process(target=serve, args=(config,), daemon=True)
    process.start()
    for _ in range(100):
        if (tmp_path / 'service/service.json').exists(): break
        time.sleep(.01)
    monkeypatch.setenv('FIXTURE_ENDPOINT', f'http://127.0.0.1:{port}')
    monkeypatch.setenv('FIXTURE_TOKEN', str(token))
    store = ArtifactStore({'dataset': tmp_path / 'data'})
    provider = CLIPlannerEndpointProvider(profile, {'endpoint_env': 'FIXTURE_ENDPOINT', 'token_file_env': 'FIXTURE_TOKEN'},
                                           store, tmp_path / 'client-native')
    gateway = ModelGateway(profiles=(profile,), policy=ModelPolicy({'agent.runtime_rsi': profile.profile_id}, None, usage_mode='accounting'),
        artifact_store=store, receipt_root=tmp_path / 'client-ledger', providers={profile.profile_id: lambda: provider})
    yield gateway, tmp_path, provider
    process.terminate(); process.join(3)


def test_inference_bridge_preserves_gateway_request_binding_and_native_audit(remote):
    gateway, tmp, provider = remote
    from dataclasses import replace
    req = replace(request(), role='agent.runtime_rsi')
    result = gateway.invoke(req)
    assert result.output == {'ok': True}
    assert gateway.recover(req).without_reuse_hash() == result.without_reuse_hash()
    assert gateway.audit_usage()['model_calls'] == 1
    assert json.loads((tmp / 'calls.json').read_text()) == 1
    assert (tmp / 'client-native/fixture-request/events.jsonl').exists()
    assert (tmp / 'client-native/fixture-request/native.json').exists()


def test_bridge_rejects_profile_bound_to_a_different_request_without_model_call(remote):
    _, tmp, provider = remote
    from dataclasses import replace
    with pytest.raises(ModelError):
        provider.invoke(replace(request(), role='agent.runtime_rsi'), 'a' * 64)
    assert not (tmp / 'calls.json').exists()
