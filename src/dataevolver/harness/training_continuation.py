"""Evidence-preserving continuation after a completed, rejected experiment.

This allocates an explicitly recorded new finite tranche. It never resets a
prior counter, replays a failed experiment, or changes a prior frozen contract.
"""
import json
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256,payload_sha256
from .state_store import atomic_write_json


def verified_rejected_continuation(root,state,goal,slot):
    root=Path(root).resolve()
    if (state['state']!='ABSTAIN' or state.get('terminal_reason') not in
            {'BUDGET_EXHAUSTED','RSI_AGENT_ABSTAINED','CAPABILITY_REQUESTED'} or state['data'].get('pending_call') or
            goal.objective['source_family']!=slot):
        raise ValueError('continuation needs a terminal, reconciled same-source predecessor')
    if state.get('terminal_reason')=='CAPABILITY_REQUESTED':
        request=state['data'].get('rsi_capability_request')
        if not request or file_sha256(Path(request['path']))!=request['sha256']:
            raise ValueError('terminal capability request must be hash-verified')
    history=state['data'].get('rsi_history',[])
    if not history or history[-1]['decision']['decision']!='ROLLBACK':
        raise ValueError('continuation requires a completed rejected comparison')
    path=Path(state['data']['experiment_candidate_stage_result']['evidence_refs'][0]).resolve()
    if not path.is_relative_to(root):raise ValueError('rejected trial escaped predecessor')
    trial=json.loads(path.read_text())
    if trial.get('failure_kind') or trial.get('qualified') is not False:
        raise ValueError('unknown/incomplete model effects cannot be continued as quality failures')
    checked={str(path):file_sha256(path)}
    for effect in root.rglob('runtime-effect.json'):
        value=json.loads(effect.read_text())
        for name,sha in value['files'].items():
            if file_sha256(Path(name))!=sha:raise ValueError('predecessor effect evidence changed')
            checked[name]=sha
        checked[str(effect)]=file_sha256(effect)
    if not checked or not (path.parent/'runtime-effect.json').exists():
        raise ValueError('rejected trial has no committed registered effect')
    journals={name:{} for name in ('model_receipts','pe_model_receipts')}
    # A terminal Controller can coexist with an orphan model reservation.
    # A valid completion receipt, or an explicit known terminal failure with
    # duration, is required before allocating a different experiment.
    for intent in root.rglob('*.intent.json'):
        if intent.parent.name not in journals:continue
        record=json.loads(intent.read_text())
        receipt=intent.with_name(intent.name.replace('.intent.json','.receipt.json'))
        if receipt.exists():
            wrapper=json.loads(receipt.read_text());value=wrapper['receipt']
            if (wrapper['sha256']!=payload_sha256(value) or value['request_hash']!=record['request_hash']):
                raise ValueError('predecessor model receipt binding changed')
        elif record.get('state') not in {'INVALID_OUTPUT','RETRY_SAFE','FAILED'} or record.get('wall_seconds') is None:
            raise ValueError('unreconciled model reservation prevents continuation')
    for name in journals:
        for folder in root.rglob(name):
            if not folder.is_dir() or folder==root/'planner/model_receipts':continue
            for p in folder.glob('*.json'):
                if p.name.endswith(('.intent.json','.receipt.json')):
                    journals[name][str(p)]=file_sha256(p)
    prior=goal.planner['runtime']['domain'].get('prior_runtime',{})
    used=state['planner_calls']+prior.get('planner_calls',0)
    return {'trial_path':str(path),'trial':trial,'file_hashes':checked,'journals':journals,
        'planner_calls':used,'experiments_used':state['experiments_used']+prior.get('experiments_used',0),
        'comparisons':[*prior.get('comparisons',[]),*history],
        'previous_budget':goal.budgets.to_dict(),'source_specific_answer_supplied':False}


def bind_rejected_view(proof,out,scopes):
    trial=proof['trial'];old_scope=trial['scope']
    if old_scope not in scopes:raise ValueError('predecessor rejected scope is not frozen')
    view={**trial,'scope':'prior-rejected'}
    path=Path(out)/'continuation/rejected-view.json';atomic_write_json(path,view)
    return {'path':str(path),'sha256':file_sha256(path),
        'original_path':proof['trial_path'],'original_sha256':proof['file_hashes'][proof['trial_path']]},scopes[old_scope]
