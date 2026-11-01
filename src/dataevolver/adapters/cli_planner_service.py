"""Stateless inference transport for a CLI instance on a different machine.

Only the Harness owns decisions, tools, campaigns and adoption. This service
accepts a bound ModelRequest and returns the CLI provider's native effect.
Deploy behind a loopback SSH tunnel; credentials stay at the configured instance.
"""
from __future__ import annotations

import argparse
import base64
import hmac
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .codex_planner import CodexCLIPlannerProvider
from .model_contracts import ModelProfile, ModelRequest
from .model_gateway import ModelError
from dataevolver.dataset.contracts import ArtifactStore, ArtifactRef, file_sha256, identifier, payload_sha256
from dataevolver.harness.state_store import atomic_write_json


def registered_roles(config):
    roles = config.get('allowed_roles', ['agent.runtime_rsi'])
    if (not isinstance(roles, list) or not roles or
            any(not isinstance(role, str) or not role.strip() for role in roles) or
            len(roles) != len(set(roles))):
        raise ValueError('inference roles must be an explicit nonempty unique list')
    return frozenset(roles)


def bound_request(message, profile, roles):
    request = ModelRequest.from_dict(message['request'])
    identifier(request.request_id)
    if (message['profile_hash'] != profile.compute_hash() or
            request.role not in roles or request.max_attempts != 1 or
            payload_sha256({'request': request.to_dict(), 'profile_hash': payload_sha256({
                'profile': profile.to_dict(), 'model': profile.model, 'endpoint': ''})}) != message['request_hash']):
        raise ValueError('unbound planner request')
    return request


class CLIPlannerEndpointProvider:
    """No request retry; lookup asks for an existing native completion only."""
    def __init__(self, profile, config, store, native_root):
        self.profile, self.config, self.store = profile, config, store
        self.root = Path(native_root)
        self.url = os.environ[config['endpoint_env']].rstrip('/')
        if not self.url.startswith('http://127.0.0.1:'):
            raise ValueError('CLI bridge must use the registered loopback SSH tunnel')
        self.token = Path(os.environ[config['token_file_env']]).read_text().strip()
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def health(self):
        return {'status': 'CONFIGURED', 'real_model_call': False}

    def _send(self, operation, request, request_hash):
        payload = {'operation': operation, 'request': request.to_dict(), 'request_hash': request_hash,
                   'profile_hash': self.profile.compute_hash()}
        if self.config.get('worker_identity_sha256'):
            payload['worker_identity_sha256'] = self.config['worker_identity_sha256']
        if operation == 'invoke':
            payload['artifacts'] = [{'ref': ref.to_dict(),
                'base64': base64.b64encode(self.store.resolve(ref).read_bytes()).decode()} for ref in request.artifacts]
        call = urllib.request.Request(self.url + '/inference',
            data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer ' + self.token,
                                                       'Content-Type': 'application/json'})
        try:
            with self.opener.open(call, timeout=request.timeout_seconds + 30) as response:
                data = response.read(8 * 1024 * 1024 + 1)
            if len(data) > 8 * 1024 * 1024:
                raise ValueError('oversized CLI native response')
            value = json.loads(data)
        except Exception as exc:
            raise ModelError('UNKNOWN_EFFECT' if operation == 'invoke' else 'UNAVAILABLE',
                             'CLI tunnel did not return a bound completion; inspect native records before retry') from exc
        if value.get('error_code'):
            raise ModelError(value['error_code'], value.get('error', 'CLI service rejected request'))
        if value.get('request_hash') != request_hash or value.get('profile_hash') != self.profile.compute_hash():
            raise ModelError('INVALID_RECEIPT', 'CLI service request/profile binding changed')
        if self.config.get('worker_identity_sha256') and payload_sha256(value.get('worker_identity')) != self.config['worker_identity_sha256']:
            raise ModelError('INVALID_RECEIPT', 'configured CLI worker implementation changed')
        target = self.root / request.request_id
        for name, content in value.get('audit', {}).items():
            if name not in {'native.json', 'native.intent.json', 'native.failure.json', 'invocation.json',
                            'events.jsonl', 'final-response.json', 'output-schema.json', 'prompt.txt', 'stderr.log'}:
                raise ModelError('INVALID_RECEIPT', 'CLI service audit name is unregistered')
            target.mkdir(parents=True, exist_ok=True)
            path = target / name
            raw = base64.b64decode(content, validate=True)
            if path.exists() and path.read_bytes() != raw:
                # An incomplete native intent may progress to completion.
                if name != 'native.intent.json':
                    raise ModelError('INVALID_RECEIPT', 'CLI native audit changed')
            path.write_bytes(raw)
        if value.get('response') is not None:
            native = json.loads((target / 'native.json').read_text())
            if (native['response'] != value['response'] or native['request_hash'] != request_hash or
                    native['profile_hash'] != self.profile.compute_hash() or
                    native['events_sha256'] != file_sha256(target / 'events.jsonl') or
                    native['output_sha256'] != file_sha256(target / 'final-response.json')):
                raise ModelError('INVALID_RECEIPT', 'CLI native completion failed independent binding checks')
        return value.get('response')

    def lookup(self, request, request_hash):
        return self._send('lookup', request, request_hash)

    def invoke(self, request, request_hash):
        return self._send('invoke', request, request_hash)


def serve(config):
    root = Path(config['root']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    token = Path(config['token_file']).read_text().strip()
    profile = ModelProfile.from_dict(config['profile'])
    roles = registered_roles(config)
    lock = threading.Lock()
    from . import codex_planner
    identity = {'profile_sha256': profile.compute_hash(), 'binary_sha256': config['worker']['binary_sha256'],
                'provider_implementation_sha256': file_sha256(Path(codex_planner.__file__)),
                'transport_implementation_sha256': file_sha256(Path(__file__))}
    if 'allowed_roles' in config:
        identity['allowed_roles'] = sorted(roles)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                if self.path != '/inference' or not hmac.compare_digest(
                        self.headers.get('Authorization', ''), 'Bearer ' + token):
                    self.send_error(403); return
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 48 * 1024 * 1024:
                    raise ValueError('invalid envelope size')
                message = json.loads(self.rfile.read(size))
                if message.get('worker_identity_sha256', payload_sha256(identity)) != payload_sha256(identity):
                    raise ValueError('configured inference worker identity changed')
                request = bound_request(message, profile, roles)
                stores = {ref.store: root / 'artifacts' / request.request_id / ref.store for ref in request.artifacts}
                store = ArtifactStore(stores)
                if message['operation'] == 'invoke':
                    if not lock.acquire(blocking=False):
                        raise ModelError('RESOURCE_BUSY', 'one configured CLI inference is already active')
                    try:
                        if len(message.get('artifacts', [])) != len(request.artifacts):
                            raise ValueError('unbound planner images')
                        for supplied, expected in zip(message['artifacts'], request.artifacts):
                            if ArtifactRef.from_dict(supplied['ref']) != expected:
                                raise ValueError('planner image binding changed')
                            path = stores[expected.store] / expected.path
                            path.parent.mkdir(parents=True, exist_ok=True)
                            raw = base64.b64decode(supplied['base64'], validate=True)
                            if path.exists() and path.read_bytes() != raw:
                                raise ValueError('planner image changed')
                            path.write_bytes(raw)
                            store.resolve(expected)
                        provider = CodexCLIPlannerProvider(profile, config['worker'], store, root / 'native')
                        response = provider.invoke(request, message['request_hash'])
                    finally:
                        lock.release()
                elif message['operation'] == 'lookup':
                    provider = CodexCLIPlannerProvider(profile, config['worker'], store, root / 'native')
                    response = provider.lookup(request, message['request_hash'])
                else:
                    raise ValueError('only inference and native lookup are registered')
                audit_root = root / 'native' / request.request_id
                audit = {p.name: base64.b64encode(p.read_bytes()).decode()
                         for p in audit_root.glob('*') if p.is_file() and p.name in {
                             'native.json', 'native.intent.json', 'native.failure.json', 'invocation.json',
                             'events.jsonl', 'final-response.json', 'output-schema.json', 'prompt.txt', 'stderr.log'}}
                result = {'request_hash': message['request_hash'], 'profile_hash': profile.compute_hash(),
                          'response': response, 'audit': audit, 'worker_identity': identity}
            except Exception as exc:
                result = {'error_code': getattr(exc, 'code', type(exc).__name__), 'error': str(exc)[:1500]}
            raw = json.dumps(result).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw))); self.end_headers()
            try:
                self.wfile.write(raw)
            except OSError:
                pass  # Native completion remains available after a lost HTTP caller.

    server = ThreadingHTTPServer(('127.0.0.1', config['port']), Handler)
    atomic_write_json(root / 'service.json', {'pid': os.getpid(), 'state': 'READY',
        'control_plane': False, 'models_loaded_here': False, 'model': profile.model,
        'worker_identity': identity, 'worker_identity_sha256': payload_sha256(identity),
        'authenticated_request_verified': False})
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    serve(json.loads(Path(args.config).read_text()))
