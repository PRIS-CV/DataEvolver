"""Authenticated CLI inference provider; Harness remains the only controller.

This opt-in planner backend can contact its configured model service. It never
receives a shell tool or permission to execute data-production operations.
Credentials are read through a private instance environment variable only.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from dataevolver.harness.state_store import atomic_write_json
from .model_contracts import ModelRequest
from .model_gateway import ModelError, validate_response_schema


class CodexCLIPlannerProvider:
    def __init__(self, profile, config, store, native_root):
        self.profile, self.config, self.store = profile, dict(config), store
        self.root = Path(native_root).resolve()
        self.binary = Path(os.environ[config['binary_env']]).resolve()
        self.instance = Path(os.environ[config['instance_env']]).resolve()
        if profile.backend != 'local_python':
            raise ValueError('the registered CLI planner model identity changed')
        if config.get('effort') not in (None, 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'):
            raise ValueError('unsupported explicit Codex effort')
        if (not self.binary.is_file() or file_sha256(self.binary) != config['binary_sha256']
                or not (self.instance / 'auth.json').is_file()):
            raise ValueError('configured planner executable or private login is missing')

    def health(self):
        return {'status': 'CONFIGURED', 'model': self.profile.model,
                'authenticated_request_verified': False, 'real_model_call': False}

    def _paths(self, request):
        root = self.root / request.request_id
        return root, root / 'native.json', root / 'native.intent.json'

    def lookup(self, request, request_hash):
        root, native, intent = self._paths(request)
        if native.exists():
            value = json.loads(native.read_text())
            if (value['request_hash'] != request_hash or value['profile_hash'] != self.profile.compute_hash()
                    or file_sha256(root / 'final-response.json') != value['output_sha256']
                    or file_sha256(root / 'events.jsonl') != value['events_sha256']):
                raise ModelError('INVALID_RECEIPT', 'CLI planner native binding changed')
            validate_response_schema(request.response_schema, value['response']['output'])
            return value['response']
        if intent.exists():
            # A text request may have reached the service even if the client died.
            raise ModelError('UNKNOWN_EFFECT', 'CLI planner attempt has no verified native completion; no automatic resend')
        return None

    def invoke(self, request, request_hash):
        recovered = self.lookup(request, request_hash)
        if recovered is not None:
            return recovered
        root, native, intent = self._paths(request)
        root.mkdir(parents=True, exist_ok=True)
        images = []
        for artifact in request.artifacts:
            if artifact.media_type not in {'image/png', 'image/jpeg', 'image/webp'}:
                raise ModelError('INVALID_INPUT', 'runtime planner only accepts bound image artifacts')
            images.append(self.store.resolve(artifact))
        output, schema = root / 'final-response.json', root / 'output-schema.json'
        atomic_write_json(schema, request.response_schema)
        prompt = '\n\n'.join(message['role'].upper() + ':\n' + message['content'] for message in request.messages)
        (root / 'prompt.txt').write_text(prompt)
        command = [str(self.binary), '--disable', 'shell_tool', '--disable', 'unified_exec',
                   '--disable', 'multi_agent', '-c', 'model_provider="dataevolver_runtime"',
                   '-c', 'model_providers.dataevolver_runtime={name="OpenAI",wire_api="responses",requires_openai_auth=true,request_max_retries=0,stream_max_retries=0,supports_websockets=true}',
                   'exec', '--ignore-user-config', '--model', self.profile.model,
                   '--sandbox', 'read-only', '--skip-git-repo-check', '--json',
                   '--cd', str(root), '--output-schema', str(schema), '--output-last-message', str(output)]
        if self.config.get('effort'):
            command[1:1] = ['-c', 'model_reasoning_effort=' + json.dumps(self.config['effort'])]
        for image in images:
            command.extend(['--image', str(image)])
        command.append('-')
        env = dict(os.environ, CODEX_HOME=str(self.instance))
        atomic_write_json(root / 'invocation.json', {'argv': command,
            'model': self.profile.model, 'binary_sha256': self.config['binary_sha256'],
            'instance_env': self.config['instance_env'], 'credential_values_recorded': False,
            'model_compute': 'configured_remote_service', 'source_modification_tools_enabled': False})
        started = time.time()
        atomic_write_json(intent, {'request_hash': request_hash, 'profile_hash': self.profile.compute_hash(),
                                   'state': 'STARTED', 'started_at': started})
        process = None
        try:
            with (root / 'events.jsonl').open('w') as stdout, (root / 'stderr.log').open('w') as stderr:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                           text=True, env=env, start_new_session=True)
                atomic_write_json(intent, {'request_hash': request_hash, 'profile_hash': self.profile.compute_hash(),
                    'state': 'STARTED', 'started_at': started, 'pid': process.pid})
                process.stdin.write(prompt)
                process.stdin.close()
                try:
                    code = process.wait(timeout=request.timeout_seconds)
                except subprocess.TimeoutExpired as exc:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                    raise ModelError('UNKNOWN_EFFECT', 'planner client timed out; inspect its stored events before recovery') from exc
            if code != 0:
                raise ModelError('PLANNER_FAILED', f'CLI planner exited {code}; inspect stored stderr and events')
            events = []
            for line in (root / 'events.jsonl').read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            turns = [event for event in events if event.get('type') == 'turn.completed']
            if any(event.get('type') == 'item.completed' and event.get('item', {}).get('type') in
                   {'command_execution', 'file_change', 'mcp_tool_call', 'web_search'} for event in events):
                raise ModelError('UNAUTHORIZED_TOOL', 'inference backend attempted a tool outside Harness ownership')
            if not turns or not output.is_file():
                raise ModelError('UNKNOWN_EFFECT', 'planner lacks a completed turn and final structured output')
            if any(event.get('type') in {'turn.failed', 'error'} for event in events):
                raise ModelError('PLANNER_FAILED', 'planner reported a failed execution event')
            value = json.loads(output.read_text())
            validate_response_schema(request.response_schema, value)
            usage = turns[-1].get('usage', {})
            response = {'model': self.profile.model, 'output': value, 'finish_reason': 'stop',
                        'provider_request_id': request.request_id,
                        'usage': {'input_tokens': usage.get('input_tokens'),
                                  'output_tokens': usage.get('output_tokens'), 'images': len(images),
                                  'cost': None, 'wall_seconds': time.time() - started}}
            atomic_write_json(native, {'request_hash': request_hash, 'profile_hash': self.profile.compute_hash(),
                'response': response, 'output_sha256': file_sha256(output),
                'events_sha256': file_sha256(root / 'events.jsonl'),
                'model_weight_revision_available': False, 'planner_actor': 'DataEvolver_runtime'})
            atomic_write_json(intent, {'request_hash': request_hash, 'profile_hash': self.profile.compute_hash(),
                'state': 'COMPLETED', 'started_at': started, 'finished_at': time.time(), 'pid': process.pid})
            return response
        except BaseException as exc:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            atomic_write_json(root / 'native.failure.json', {'request_hash': request_hash,
                'error_code': getattr(exc, 'code', type(exc).__name__), 'error': str(exc)[:1500],
                'started_at': started, 'finished_at': time.time(), 'wall_seconds': time.time() - started,
                'automatic_retry_allowed': False})
            raise


class GatewayPlannerBackend:
    """PlannerBackend protocol using the same cache/effect ledger as other models."""
    backend_id = 'model-gateway-authenticated-cli'

    def __init__(self, gateway, *, model_id, artifacts=(), timeout_seconds=900):
        self.gateway, self.model_id = gateway, model_id
        self.artifacts, self.timeout_seconds = tuple(artifacts), timeout_seconds
        self._last_trace = {}

    def last_trace(self):
        return self._last_trace

    def set_artifacts(self, artifacts):
        self.artifacts = tuple(artifacts)

    def complete_json(self, messages, *, schema):
        binding = payload_sha256({'messages': messages, 'schema': schema,
                                  'artifacts': [ref.to_dict() for ref in self.artifacts]})
        request = ModelRequest(request_id='rsi-' + binding[:28], role='agent.runtime_rsi',
            messages=tuple(messages), response_schema=schema, artifacts=self.artifacts,
            max_tokens=4096, timeout_seconds=self.timeout_seconds, max_attempts=1, seed=0)
        receipt = self.gateway.recover(request)
        if receipt is None:
            receipt = self.gateway.invoke(request)
        self._last_trace = {'request': request.to_dict(), 'receipt': receipt.to_dict(),
                            'receipt_hash': receipt.without_reuse_hash()}
        return json.dumps(receipt.output, ensure_ascii=False)
