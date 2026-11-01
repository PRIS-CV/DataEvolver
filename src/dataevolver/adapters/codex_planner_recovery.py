"""Reconcile a completed CLI turn after local validation failed; no inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataevolver.adapters.model_contracts import ModelProfile, ModelRequest
from dataevolver.adapters.model_gateway import validate_response_schema
from dataevolver.dataset.contracts import ArtifactRef as ArtifactRef, ArtifactStore, file_sha256, payload_sha256
from dataevolver.harness.state_store import atomic_write_json


def reconcile(config, request):
    profile = ModelProfile.from_dict(config['profile'])
    root = Path(config['root']) / 'native' / request.request_id
    expected_hash = payload_sha256({'request': request.to_dict(), 'profile_hash': payload_sha256({
        'profile': profile.to_dict(), 'model': profile.model, 'endpoint': ''})})
    intent = json.loads((root / 'native.intent.json').read_text())
    failure = json.loads((root / 'native.failure.json').read_text())
    if failure['error_code'] != 'DEPENDENCY_MISSING':
        raise ValueError('this reconciler only handles completed-turn local schema dependency failures')
    if (intent['request_hash'] != expected_hash or failure['request_hash'] != expected_hash or
            intent['profile_hash'] != profile.compute_hash()):
        raise ValueError('native request/profile binding changed')
    invocation = json.loads((root / 'invocation.json').read_text())
    if invocation['model'] != profile.model or invocation['binary_sha256'] != config['worker']['binary_sha256']:
        raise ValueError('native CLI identity changed')
    prompt = '\n\n'.join(m['role'].upper() + ':\n' + m['content'] for m in request.messages)
    if (root / 'prompt.txt').read_text() != prompt or json.loads((root / 'output-schema.json').read_text()) != request.response_schema:
        raise ValueError('native prompt or output schema differs from the original request')
    store = ArtifactStore({r.store: Path(config['root']) / 'artifacts' / request.request_id / r.store for r in request.artifacts})
    images = [str(store.resolve(ref)) for ref in request.artifacts]
    args = invocation['argv']
    supplied = [args[i+1] for i, x in enumerate(args) if x == '--image']
    if supplied != images:
        raise ValueError('native images differ from the original request')
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    completed = [e for e in events if e.get('type') == 'turn.completed']
    if (not completed or any(e.get('type') in {'turn.failed', 'error'} for e in events) or
            any(e.get('type') == 'item.completed' and e.get('item', {}).get('type') != 'agent_message' for e in events)):
        raise ValueError('native turn is incomplete or contains unauthorized tool effects')
    value = json.loads((root / 'final-response.json').read_text())
    validate_response_schema(request.response_schema, value)
    usage = completed[-1].get('usage', {})
    response = {'model': profile.model, 'output': value, 'finish_reason': 'stop',
                'provider_request_id': request.request_id,
                'usage': {'input_tokens': usage.get('input_tokens'), 'output_tokens': usage.get('output_tokens'),
                          'images': len(images), 'cost': None, 'wall_seconds': failure['wall_seconds']}}
    native = {'request_hash': expected_hash, 'profile_hash': profile.compute_hash(), 'response': response,
              'output_sha256': file_sha256(root / 'final-response.json'),
              'events_sha256': file_sha256(root / 'events.jsonl'),
              'model_weight_revision_available': False, 'planner_actor': 'DataEvolver_runtime',
              'recovered_completed_turn': True}
    path = root / 'native.json'
    if path.exists() and json.loads(path.read_text()) != native:
        raise ValueError('refusing to overwrite a different native completion')
    atomic_write_json(path, native)
    proof = {'request_hash': expected_hash, 'native_sha256': file_sha256(path),
             'original_failure_sha256': file_sha256(root / 'native.failure.json'),
             'original_intent_sha256': file_sha256(root / 'native.intent.json'),
             'events_sha256': file_sha256(root / 'events.jsonl'),
             'reconciliation_source_sha256': file_sha256(Path(__file__)),
             'new_model_calls': 0, 'decision_edited': False}
    atomic_write_json(root / 'native.recovery.json', proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--request', required=True)
    args = parser.parse_args()
    print(json.dumps(reconcile(json.loads(Path(args.config).read_text()), ModelRequest.from_dict(json.loads(Path(args.request).read_text())))))
