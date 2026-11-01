"""Resume a failed Planner phase only from its verified existing model effect.

Run with the campaign's frozen source on PYTHONPATH. This additive maintenance
entrypoint records its own source hash; it never changes the frozen evaluator,
source, old events, failed trace, counters or model output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256
from dataevolver.adapters.model_contracts import ModelRequest
from dataevolver.harness.models import ControllerState, canonical_json
from dataevolver.harness.runtime_rsi import SYSTEM_PROMPT, ACTION_SCHEMA, validate_decision
from dataevolver.harness.state_store import CampaignLease, atomic_write_json


def recover_planner(controller):
    c = controller
    with CampaignLease(c.layout.lease_path, ttl_seconds=c.goal.lease_seconds):
        state = c.status()
        if state.data.get('rsi_recoveries'):
            last = state.data['rsi_recoveries'][-1]
            if state.state is not ControllerState.ABSTAIN:
                return {'status': 'ALREADY_RECOVERED', 'proof': last, 'new_model_calls': 0}
        if state.state is not ControllerState.ABSTAIN or state.terminal_reason != 'RSI_PLANNER_FAILED':
            raise ValueError('only a failed Planner phase can be reconciled')
        valid, drift = c._frozen()
        if not valid:
            raise ValueError('frozen contract drift before recovery: ' + str(drift))
        path = c.planner.trace_root / f'p{state.planner_calls-1:03d}' / 'trace.json'
        trace = json.loads(path.read_text())
        if trace['outcome'] != 'FAILED':
            raise ValueError('Planner trace is not a failed attempt')
        context = c.planner.context(state)
        if payload_sha256({'config': c.planner.config(), 'context': context}) != trace['context_hash']:
            raise ValueError('Planner recovery context changed')
        artifacts = (c.planner.domain.planner_artifacts(state) if hasattr(c.planner.domain, 'planner_artifacts')
                     else c.planner.backend.artifacts)
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': canonical_json(context)}]
        binding = payload_sha256({'messages': messages, 'schema': ACTION_SCHEMA, 'artifacts': [r.to_dict() for r in artifacts]})
        request = ModelRequest('rsi-' + binding[:28], 'agent.runtime_rsi', tuple(messages), ACTION_SCHEMA,
            artifacts=tuple(artifacts), max_tokens=4096, timeout_seconds=c.planner.backend.timeout_seconds,
            max_attempts=1, seed=0)
        # Recovery only: this command has no branch that invokes the provider.
        receipt = c.planner.backend.gateway.recover(request)
        if receipt is None:
            raise ValueError('no verified native completion exists; automatic resubmission is forbidden')
        actual = receipt.output
        # Older frozen code rejected a probe merely for naming its R/M target,
        # although that value is allowed by ACTION_SCHEMA. Validate the empty
        # effect shape while retaining the original model decision byte-for-byte.
        validation_view = actual
        compatibility = None
        if actual['action'] == 'OBSERVE' and actual['layer'] in {'R', 'M'}:
            validation_view = {**actual, 'layer': 'none'}
            compatibility = 'observation-target-layer-v1'
        validate_decision(validation_view, context)
        if actual['action'] == 'OBSERVE' and json.loads(actual['patch']['parameters_json']):
            raise ValueError('recovered observation cannot carry a hidden patch')
        decision = actual
        proof = {'failed_trace_sha256': file_sha256(path), 'request': request.to_dict(),
                 'receipt': receipt.to_dict(), 'receipt_hash': receipt.without_reuse_hash(),
                 'decision': decision, 'proposal_actor': 'runtime_agent', 'new_model_calls': 0,
                 'validation_compatibility': compatibility, 'decision_edited': False,
                 'recovery_source_sha256': file_sha256(Path(__file__))}
        recovery_path = path.with_name('recovery.json')
        if recovery_path.exists() and json.loads(recovery_path.read_text()) != proof:
            raise ValueError('Planner recovery proof changed')
        atomic_write_json(recovery_path, proof)
        data = {**state.data, 'rsi_pending_decision': decision,
                'rsi_recoveries': [*state.data.get('rsi_recoveries', []), {'path': str(recovery_path), 'sha256': file_sha256(recovery_path)}]}
        restored = c._commit(state, ControllerState.PROPOSE, 'RSI_PLANNER_COMPLETION_RECOVERED', 'RECOVERED',
                            data=data, terminal_reason='', idempotency_key='rsi:recovered:' + request.request_id)
        return {'status': restored.state.value, 'request_id': request.request_id, 'new_model_calls': 0,
                'decision': decision, 'proof': str(recovery_path)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir', required=True)
    args = parser.parse_args()
    from dataevolver.paths import REPO_ROOT
    from dataevolver.harness.cli import _existing_controller
    directory = Path(args.config_dir).resolve()
    if REPO_ROOT != directory / 'source':
        raise ValueError('use the campaign frozen source on PYTHONPATH for recovery')
    goal = json.loads((directory / 'goal.json').read_text())
    print(json.dumps(recover_planner(_existing_controller(directory / 'campaigns', goal['campaign_id'])), ensure_ascii=False))
