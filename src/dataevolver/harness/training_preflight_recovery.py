"""Verify an unchanged runtime proposal stopped before any new model reservation."""
import json
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256,payload_sha256


def verified_preflight_proposal(prior_campaign,state):
    root=Path(prior_campaign)
    if state['state']!='ABSTAIN' or state['terminal_reason']!='RSI_AGENT_ABSTAINED':
        raise ValueError('preflight recovery needs a terminal, fully observed predecessor')
    history=state['data'].get('rsi_history',[])
    if len(history)!=1 or history[0]['decision']['decision']!='ROLLBACK':
        raise ValueError('only one failed preflight experiment can be recovered')
    candidate_path=Path(state['data']['experiment_candidate_stage_result']['evidence_refs'][0])
    if not candidate_path.resolve().is_relative_to(root.resolve()):raise ValueError('candidate escaped predecessor')
    trial=json.loads(candidate_path.read_text());work=Path(trial['inner_work_dir'])
    production=json.loads((work/'production.json').read_text())
    failures=production['failures']
    if (trial.get('failure_kind')!='incomplete_media_pipeline' or len(failures)!=1 or
            failures[0].get('error_code')!='KeyError' or failures[0].get('error')!="'status_path'"):
        raise ValueError('failure is not the verified PE deployment preflight defect')
    baseline=root/'rounds/r000/execute/trial.json';initial=json.loads(baseline.read_text())
    base_work=Path(initial['inner_work_dir'])
    def intents(directory):
        return {f'{name}/{p.name}':file_sha256(p) for name in ('model_receipts','pe_model_receipts')
                for p in (directory/name).glob('*.intent.json')}
    if intents(work)!=intents(base_work):raise ValueError('a new model reservation exists; automatic trial replay forbidden')
    if any((work/'native').rglob('*.job.json')) or any((work/'native').rglob('*.resident.intent.json')):
        raise ValueError('a native model job may have been submitted')
    decision=history[0]['proposal'];matches=[]
    for path in (root/'planner/decisions').glob('p*/trace.json'):
        trace=json.loads(path.read_text())
        receipt=trace.get('model_call',{}).get('receipt',{})
        if (trace.get('outcome')=='VALID_DECISION' and trace.get('decision')==decision and
                receipt.get('output')==decision and trace.get('proposal_actor')=='runtime_agent'):
            matches.append((path,trace))
    if len(matches)!=1:raise ValueError('proposal lacks one exact real runtime Planner completion')
    path,trace=matches[0]
    from dataevolver.adapters.model_contracts import ModelReceipt
    receipt=ModelReceipt.from_dict(trace['model_call']['receipt'])
    saved=root/'planner/model_receipts'/(receipt.request_id+'.receipt.json')
    wrapper=json.loads(saved.read_text())
    if (receipt.model!='gpt-6.1-sol' or receipt.backend!='local_python' or
            receipt.without_reuse_hash()!=trace['model_call']['receipt_hash'] or
            wrapper['sha256']!=payload_sha256(wrapper['receipt']) or
            ModelReceipt.from_dict(wrapper['receipt']).without_reuse_hash()!=receipt.without_reuse_hash()):
        raise ValueError('proposal is not bound to its original verified runtime Gateway receipt')
    if decision['action']!='RUN_EXPERIMENT' or decision['layer']!='R':raise ValueError('not an R experiment')
    if (trace['context']['source_sha256']!=initial['source']['sha256'] or
            trace['context']['current_trial']['candidate']!=initial['candidate']):
        raise ValueError('proposal was made for a different source/incumbent')
    evidence={str(p):file_sha256(p) for p in [path,saved,candidate_path,baseline,work/'production.json']}
    return {'decision':decision,'trace_path':str(path),'trace_sha256':file_sha256(path),
        'source':initial['source'],'candidate':initial['candidate'],'file_hashes':evidence,
        'new_model_reservations_before_failure':0,'decision_edited':False,
        'proposal_actor':'runtime_agent','prior_experiments_used':state['experiments_used']}
