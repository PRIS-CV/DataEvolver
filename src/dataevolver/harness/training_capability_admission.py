"""Verify a real runtime implementation request before enabling its operator.

The human supplies execution authority; generated requests supply evidence only.
The old image trial and capability diagnostic remain immutable and accounted.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

from dataevolver.dataset.contracts import file_sha256,payload_sha256
from .models import GoalSpec


def admit_surface_capability(directory, prior_runtime, source, goal_payload, *, reconstruction=False, typed_preflight=False, authority_separation=False, composition=False):
    from dataevolver.dataset import surface_effects
    if authority_separation and not typed_preflight:
        raise ValueError('conservative authority requires a typed reconstruction contract')
    if typed_preflight and not reconstruction:
        raise ValueError('typed reconstruction contract requires its registered context operator')
    if composition and (not typed_preflight or authority_separation):
        raise ValueError('surface composition needs fresh coverage authority, not the legacy partial-support contract')
    directory=Path(directory).resolve();prior_runtime=Path(prior_runtime).resolve()
    diagnostic=GoalSpec.from_json((directory/'goal.json').read_text())
    domain=diagnostic.planner['runtime']['domain']
    same_run=directory==prior_runtime
    if not same_run and Path(domain['prior_runtime']['config_dir']).resolve()!=prior_runtime:
        raise ValueError('surface capability must follow the actual same-source diagnostic and image predecessor')
    frozen=json.loads((directory/'frozen.json').read_text())
    if file_sha256(directory/'goal.json')!=frozen['goal_sha256']:
        raise ValueError('capability diagnostic frozen goal changed')
    checked={str(directory/'goal.json'):file_sha256(directory/'goal.json')}
    for name,sha in frozen['source_hashes'].items():
        path=directory/'source'/name
        if file_sha256(path)!=sha:raise ValueError('capability diagnostic source changed')
        checked[str(path)]=sha
    process=subprocess.run([sys.executable,'-m','dataevolver.harness.training_rsi_run','validate',
        '--config-dir',str(directory)],cwd=directory/'source',
        env={**os.environ,'PYTHONPATH':str(directory/'source/src')},capture_output=True,text=True)
    proof=json.loads(process.stdout)
    if (process.returncode not in (0,1) or not all(proof.get(k) for k in
            ('capability_request_recorded','frozen_contract_valid','replay_matches_snapshot')) or
            not proof.get('runtime_planner_decisions') or
            not all(t['real_model_receipt'] for t in proof['runtime_planner_decisions'])):
        raise ValueError('capability diagnostic lacks real model request, frozen/replay or terminal proof')
    campaign=directory/'campaigns'/diagnostic.campaign_id
    state=json.loads((campaign/'state.json').read_text());entry=state['data']['rsi_capability_request']
    if state['state']!='ABSTAIN' or state['terminal_reason']!='CAPABILITY_REQUESTED' or (state['experiments_used']!=0 and not same_run) or state['data'].get('pending_call'):
        raise ValueError('capability diagnostic must have zero new image experiments and no pending effect')
    path=Path(entry['path']).resolve()
    if not path.is_relative_to(campaign) or file_sha256(path)!=entry['sha256']:
        raise ValueError('capability request identity changed')
    request=json.loads(path.read_text())
    if (request['status']!='IMPLEMENTATION_REQUIRED' or request['proposal_actor']!='runtime_agent' or
            request['request']['implementation_kind']!=('artifact_contract' if typed_preflight and not authority_separation and not composition else 'registered_operator') or request['source']!=source or
            request['protected_goal_sha256']!=payload_sha256(goal_payload)):
        raise ValueError('capability request does not bind the frozen original task')
    for evidence in request['evidence']:
        if payload_sha256(json.loads(Path(evidence['path']).read_text()))!=evidence['sha256']:
            raise ValueError('capability request evidence changed')
        checked[evidence['path']]=file_sha256(Path(evidence['path']))
    checked.update({str(path):entry['sha256'],str(campaign/'state.json'):file_sha256(campaign/'state.json')})
    journals={}
    for folder in campaign.rglob('model_receipts'):
        for item in folder.glob('*.intent.json'):
            intent=json.loads(item.read_text());receipt=item.with_name(item.name.replace('.intent.json','.receipt.json'))
            if receipt.exists():
                wrapper=json.loads(receipt.read_text())
                if wrapper['sha256']!=payload_sha256(wrapper['receipt']) or wrapper['receipt']['request_hash']!=intent['request_hash']:
                    raise ValueError('capability diagnostic receipt binding changed')
            elif intent.get('state') not in {'FAILED','INVALID_OUTPUT','RETRY_SAFE'} or intent.get('wall_seconds') is None:
                raise ValueError('capability diagnostic has an unreconciled reservation')
            journals[str(item)]=file_sha256(item)
            if receipt.exists():journals[str(receipt)]=file_sha256(receipt)
    observations=[]
    for item in domain.get('prior_observations',[]):
        observations.append(item)
    for item in state['data'].get('rsi_observations',[]):
        if item['verification']['status']!='VERIFIED':raise ValueError('diagnostic has an unverified observation')
        observed=Path(item['stage_result']['evidence_refs'][0])
        observations.append({'path':str(observed),'sha256':file_sha256(observed),'item':item})
    for item in observations:
        if file_sha256(Path(item['path']))!=item['sha256']:raise ValueError('diagnostic observation changed')
        checked[item['path']]=item['sha256']
    preceding=GoalSpec.from_json((prior_runtime/'goal.json').read_text())
    preceding_state=json.loads((prior_runtime/'campaigns'/preceding.campaign_id/'state.json').read_text())
    prior_count=preceding_state['planner_calls']+preceding.planner['runtime']['domain'].get('prior_runtime',{}).get('planner_calls',0)
    diagnostic_count=state['planner_calls']+domain.get('prior_runtime',{}).get('planner_calls',0)
    if diagnostic_count<prior_count or (diagnostic_count==prior_count and not same_run):raise ValueError('capability diagnostic decisions were not accounted')
    value={'operator_id':surface_effects.OPERATOR_ID,'version':surface_effects.OPERATOR_VERSION,
        'implementation_sha256':file_sha256(Path(surface_effects.__file__)),
        'request_path':str(path),'request_sha256':entry['sha256'],
        'diagnostic_config':str(directory),'diagnostic_planner_calls':diagnostic_count-prior_count,
        'diagnostic_local_planner_calls':state['planner_calls'],'cumulative_planner_calls':diagnostic_count,
        'registered_observations':observations,
        'file_hashes':{**checked,**journals},
        'candidate_method':True,'independently_verified_method':False,
        'execution_authority':'standing human authorization; runtime request is evidence, not authorization'}
    if reconstruction:
        from dataevolver.dataset import context_reconstruction
        value['reconstruction']={'operator_id':context_reconstruction.OPERATOR_ID,
            'version':context_reconstruction.OPERATOR_VERSION,
            'implementation_sha256':file_sha256(Path(context_reconstruction.__file__)),
            'parameters':{'context_padding_fraction':{'minimum':0,'maximum':.25}},
            'max_model_edits_per_trial':1,'method_improvement_verified':False}
    if typed_preflight:
        from dataevolver.dataset import surface_preflight
        value['typed_preflight']={'contract_id':surface_preflight.CONTRACT_ID,
            'version':surface_preflight.CONTRACT_VERSION,
            'implementation_sha256':file_sha256(Path(surface_preflight.__file__)),
            'request_sha256':entry['sha256'],'requires_fresh_typed_observation':True,
            'uncertainty_grants_edit_authority':False,'preflight_image_calls':0}
    if authority_separation:
        from dataevolver.dataset import surface_authority
        value['authority_separation']={'operator_id':surface_authority.OPERATOR_ID,
            'version':surface_authority.OPERATOR_VERSION,
            'implementation_sha256':file_sha256(Path(surface_authority.__file__)),
            'request_sha256':entry['sha256'],'support_expansion_allowed':False,
            'partial_execution_is_completion':False,'method_improvement_verified':False}
    if composition:
        from dataevolver.dataset import surface_composition
        value['composition']={'operator_id':surface_composition.OPERATOR_ID,
            'version':surface_composition.OPERATOR_VERSION,
            'implementation_sha256':file_sha256(Path(surface_composition.__file__)),
            'request_sha256':entry['sha256'],'requires_fresh_coverage_observation':True,
            'coverage_grants_authority':False,'unresolved_coverage_blocks_image_request':True,
            'harmonization':'dirichlet_harmonic_correction','method_improvement_verified':False}
    return value
