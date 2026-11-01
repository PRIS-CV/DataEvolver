"""Transfer an exact completed observation decision rejected before dispatch.

This is not provider retry and cannot recover unknown or submitted effects.
The decision occupies its original local budget slot in the new snapshot.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from dataevolver.dataset.contracts import file_sha256,payload_sha256
from dataevolver.adapters.model_contracts import ModelReceipt
from .models import GoalSpec


def verified_observation_handoff(directory, *, prior_config, capability_request_sha256):
    directory=Path(directory).resolve()
    goal=GoalSpec.from_json((directory/'goal.json').read_text())
    domain=goal.planner['runtime']['domain']
    if Path(domain['prior_runtime']['config_dir']).resolve()!=Path(prior_config).resolve():
        raise ValueError('protocol recovery changed the actual image predecessor')
    if domain['surface_capability']['request_sha256']!=capability_request_sha256:
        raise ValueError('protocol recovery changed its capability request')
    process=subprocess.run([sys.executable,'-m','dataevolver.harness.training_rsi_run','validate','--config-dir',str(directory)],
        cwd=directory/'source',env={**os.environ,'PYTHONPATH':str(directory/'source/src')},capture_output=True,text=True)
    proof=json.loads(process.stdout)
    if process.returncode not in (0,1) or not all(proof.get(k) for k in ('frozen_contract_valid','replay_matches_snapshot')):
        raise ValueError('protocol predecessor frozen/replay validation failed')
    campaign=directory/'campaigns'/goal.campaign_id
    state=json.loads((campaign/'state.json').read_text())
    if state['state']=='PROPOSE' and state['data'].get('pending_call'):
        return verified_schema_observation_handoff(directory,prior_config=prior_config,capability_request_sha256=capability_request_sha256)
    local_observations=state['data'].get('rsi_observations',[])
    if (state['state']!='ABSTAIN' or state['terminal_reason']!='INVALID_RSI_OBSERVATION' or
            not 1<=state['planner_calls']<=goal.budgets.max_planner_calls or state['experiments_used']!=0 or
            state['operator_calls']!=1+len(local_observations) or state['data'].get('pending_call')):
        raise ValueError('observation recovery requires a rejected pre-dispatch decision with only verified preceding observations')
    decision=state['data']['rsi_pending_decision']
    if decision['action']!='OBSERVE' or decision['observation']['tool']!='preflight_reconstruction':
        raise ValueError('only a deterministic reconstruction preflight decision can transfer')
    params=json.loads(decision['observation']['parameters_json'])
    if set(params)!={'observation_id','context_padding_fraction'}:
        raise ValueError('observation recovery has hidden effect parameters')
    registered=[];verified_files={}
    for item in local_observations:
        path=Path(item['stage_result']['evidence_refs'][0]).resolve();effect=path.parent/'runtime-effect.json'
        if (item['verification']['status']!='VERIFIED' or not path.is_relative_to(campaign.resolve()) or not effect.exists()):
            raise ValueError('preceding observation is not a verified registered effect')
        manifest=json.loads(effect.read_text())
        for name,sha in manifest['files'].items():
            if file_sha256(Path(name))!=sha:raise ValueError('preceding observation effect changed')
            verified_files[name]=sha
        for p in path.parent.rglob('*.json'):verified_files[str(p)]=file_sha256(p)
        registered.append({'path':str(path),'sha256':file_sha256(path),'item':item})
    expected={Path(x['path']).parent.resolve() for x in registered}
    actual={p.resolve() for p in (campaign/'observations').iterdir()} if (campaign/'observations').exists() else set()
    if actual!=expected or any((campaign/'experiments').rglob('*')):
        raise ValueError('observation or experiment may already have been dispatched')
    baseline_path=campaign/'rounds/r000/execute/trial.json'
    baseline=json.loads(baseline_path.read_text());baseline_work=Path(baseline['inner_work_dir'])
    known=set(domain['seed']['file_hashes'].values()) | set(domain['seed'].get('additional_journals',{}).values()) | set(domain['seed'].get('additional_pe_journals',{}).values())
    for ledger in ('model_receipts','pe_model_receipts'):
        for item in (baseline_work/ledger).glob('*.intent.json'):
            if file_sha256(item) not in known:
                raise ValueError('bootstrap contains a new or unknown production reservation')
    if any((baseline_work/'native').rglob('*.job.json')) or any((baseline_work/'native').rglob('*.resident.intent.json')):
        raise ValueError('bootstrap may have submitted a new native production request')
    tracepath=campaign/'planner/decisions'/f"p{state['planner_calls']-1:03d}"/'trace.json';trace=json.loads(tracepath.read_text())
    receipt=ModelReceipt.from_dict(trace['model_call']['receipt'])
    if (trace['outcome']!='VALID_DECISION' or trace['decision']!=decision or receipt.output!=decision or
            trace['proposal_actor']!='runtime_agent' or receipt.model!='gpt-6.1-sol' or receipt.backend!='local_python' or
            receipt.without_reuse_hash()!=trace['model_call']['receipt_hash']):
        raise ValueError('protocol recovery lacks an exact real runtime completion')
    saved=campaign/'planner/model_receipts'/(receipt.request_id+'.receipt.json')
    intent=saved.with_name(receipt.request_id+'.intent.json')
    wrapper=json.loads(saved.read_text());reservation=json.loads(intent.read_text())
    if (wrapper['sha256']!=payload_sha256(wrapper['receipt']) or
            ModelReceipt.from_dict(wrapper['receipt']).without_reuse_hash()!=receipt.without_reuse_hash() or
            reservation['request_hash']!=receipt.request_hash):
        raise ValueError('protocol recovery receipt/reservation binding changed')
    trial=trace['context']['current_trial']
    matches=[x for x in domain['prior_observations']+registered if x['item']['call_id']==params['observation_id']]
    if len(matches)!=1 or file_sha256(Path(matches[0]['path']))!=matches[0]['sha256']:
        raise ValueError('protocol recovery observation provenance changed')
    observed=json.loads(Path(matches[0]['path']).read_text())
    if observed['source']!=trial['source'] or observed['candidate']!=trial['candidate']:
        raise ValueError('protocol recovery source or current candidate changed')
    files={str(p):file_sha256(p) for p in (directory/'goal.json',directory/'frozen.json',campaign/'state.json',baseline_path,baseline_work/'production.json',tracepath,saved,intent,Path(matches[0]['path']))}
    files.update(verified_files)
    local_traces=[];local_journals={}
    for p in sorted((campaign/'planner/decisions').glob('p*/trace.json')):
        t=json.loads(p.read_text());r=ModelReceipt.from_dict(t['model_call']['receipt'])
        journal=campaign/'planner/model_receipts'/(r.request_id+'.receipt.json')
        wrapper=json.loads(journal.read_text())
        if (t['outcome']!='VALID_DECISION' or r.output!=t['decision'] or
                t['model_call']['receipt_hash']!=r.without_reuse_hash() or
                wrapper['sha256']!=payload_sha256(wrapper['receipt']) or
                ModelReceipt.from_dict(wrapper['receipt']).without_reuse_hash()!=r.without_reuse_hash()):
            raise ValueError('preceding decision receipt changed')
        local_traces.append(str(p));files[str(p)]=file_sha256(p)
        for j in (journal,journal.with_name(r.request_id+'.intent.json')):
            local_journals[str(j)]=file_sha256(j);files[str(j)]=file_sha256(j)
    inherited=set(domain.get('continuation',{}).get('file_hashes',{}).values()) | set(domain['surface_capability'].get('file_hashes',{}).values()) | known | set(local_journals.values())
    for p in (campaign/'planner/model_receipts').glob('*.json'):
        if not p.name.endswith(('.intent.json','.receipt.json')):continue
        if p not in {intent,saved} and file_sha256(p) not in inherited:
            raise ValueError('unexpected local model reservation in protocol recovery')
    return {'decision':decision,'trace_path':str(tracepath),'source':trial['source'],'candidate':trial['candidate'],
        'file_hashes':files,'journal_files':local_journals,'registered_observations':registered,
        'recovered_decision_trace_paths':domain.get('protocol_observation_recovery',{}).get('recovered_decision_trace_paths',[])+local_traces,
        'prior_runtime_planner_calls':domain['prior_runtime']['planner_calls'],
        'decision_slots_consumed':state['planner_calls'],'local_decision_limit':goal.budgets.max_planner_calls,
        'remaining_decisions':goal.budgets.max_planner_calls-state['planner_calls'],
        'diagnostic_config':str(directory),'recovered_decision_edited':False,
        'observation_submitted_before_failure':False,'new_model_calls_for_recovery':0}


def verified_schema_observation_handoff(directory, *, prior_config, capability_request_sha256):
    """Authorize one changed-schema submission only after an exact HTTP 400.

    Successful completions, transport ambiguity, partial output and any image
    experiment are excluded. The Planner decision/question and budget slots
    remain unchanged; this function itself invokes no inference provider.
    """
    from copy import deepcopy
    from dataevolver.adapters.model_contracts import ModelProfile,ModelRequest
    from dataevolver.dataset.contracts import ArtifactRef
    from dataevolver.dataset.surface_preflight import TYPED_SURFACE_SCHEMA
    directory=Path(directory).resolve()
    goal=GoalSpec.from_json((directory/'goal.json').read_text())
    runtime=goal.planner['runtime'];domain=runtime['domain']
    if (Path(domain['prior_runtime']['config_dir']).resolve()!=Path(prior_config).resolve() or
            domain['surface_capability']['request_sha256']!=capability_request_sha256):
        raise ValueError('schema handoff changed image predecessor or capability request')
    checked=subprocess.run([sys.executable,'-m','dataevolver.harness.training_rsi_run','validate','--config-dir',str(directory)],
        cwd=directory/'source',env={**os.environ,'PYTHONPATH':str(directory/'source/src')},capture_output=True,text=True)
    validation=json.loads(checked.stdout)
    if checked.returncode not in (0,1) or not all(validation.get(k) for k in ('frozen_contract_valid','replay_matches_snapshot')):
        raise ValueError('schema predecessor frozen/replay validation failed')
    campaign=directory/'campaigns'/goal.campaign_id
    state=json.loads((campaign/'state.json').read_text())
    if (state['state']!='PROPOSE' or state['experiments_used']!=0 or state['planner_calls']!=2 or
            state['operator_calls']!=2 or state['data'].get('rsi_history')):
        raise ValueError('schema handoff requires an unchanged zero-experiment observation phase')
    run=directory.parent/directory.name.replace('-config-','-')
    if not (run/'exit_code').exists() or (run/'exit_code').read_text().strip()!='1':
        raise ValueError('schema predecessor process lacks a failure exit receipt')
    if (run/'worker_pid').exists():
        proc=Path('/proc')/(run/'worker_pid').read_text().strip()/'cmdline'
        if proc.exists() and str(run/'run.sh').encode() in proc.read_bytes():
            raise ValueError('schema predecessor process is still active')
    decision=state['data']['rsi_pending_decision'];call=state['data']['pending_call']
    parameters=json.loads(decision['observation']['parameters_json'])
    if (decision['action']!='OBSERVE' or decision['observation']['tool']!='inspect_surfaces' or
            set(parameters)!={'question'} or call['operator_id']!='data.training_rsi_observe' or
            call['call_id']!=goal.campaign_id+':p001:observe' or
            call['parameters']['tool']!='inspect_surfaces' or call['parameters']['parameters']!=parameters):
        raise ValueError('schema handoff does not bind the exact pending runtime inspection')
    tracepath=campaign/'planner/decisions/p001/trace.json';trace=json.loads(tracepath.read_text())
    receipt=ModelReceipt.from_dict(trace['model_call']['receipt'])
    if (trace['outcome']!='VALID_DECISION' or trace['decision']!=decision or receipt.output!=decision or
            trace['proposal_actor']!='runtime_agent' or receipt.model!='gpt-6.1-sol' or receipt.backend!='local_python' or
            receipt.without_reuse_hash()!=trace['model_call']['receipt_hash']):
        raise ValueError('schema handoff lacks its unchanged real Planner decision')
    trialpath=Path(call['parameters']['trial_path'])
    trial=json.loads(trialpath.read_text())
    if (payload_sha256(trial)!=call['parameters']['trial_sha256'] or
            trial['source']!=trace['context']['current_trial']['source'] or
            trial['candidate']!=trace['context']['current_trial']['candidate']):
        raise ValueError('schema observation source or current candidate changed')
    inherited=set(domain.get('continuation',{}).get('file_hashes',{}).values())
    inherited|=set(domain['surface_capability'].get('file_hashes',{}).values())
    inherited|=set(domain['seed'].get('additional_journals',{}).values())|set(domain['seed']['file_hashes'].values())
    failed=[];journal_files={}
    for intentpath in (campaign/'planner/model_receipts').glob('*.intent.json'):
        intent=json.loads(intentpath.read_text());rid=intentpath.name[:-len('.intent.json')]
        saved=intentpath.with_name(rid+'.receipt.json')
        journal_files[str(intentpath)]=file_sha256(intentpath)
        if saved.exists():
            wrapped=json.loads(saved.read_text());r=ModelReceipt.from_dict(wrapped['receipt'])
            if wrapped['sha256']!=payload_sha256(wrapped['receipt']) or r.request_hash!=intent['request_hash']:
                raise ValueError('schema handoff has an unbound existing receipt')
            journal_files[str(saved)]=file_sha256(saved)
        elif file_sha256(intentpath) not in inherited:
            if intent.get('state')!='FAILED' or intent.get('error_code')!='PLANNER_FAILED' or intent.get('wall_seconds') is None:
                raise ValueError('schema handoff has an unknown or incomplete reservation')
            failed.append((rid,intentpath,intent))
    saved_decision=campaign/'planner/model_receipts'/(receipt.request_id+'.receipt.json')
    if not saved_decision.exists() or ModelReceipt.from_dict(json.loads(saved_decision.read_text())['receipt']).without_reuse_hash()!=receipt.without_reuse_hash():
        raise ValueError('schema decision receipt differs from its Gateway journal')
    if len(failed)!=1:raise ValueError('schema handoff needs one exact failed observation reservation')
    rid,intentpath,intent=failed[0];native=campaign/'planner/native'/rid
    if any((native/n).exists() for n in ('native.json','final-response.json')):
        raise ValueError('schema failure contains a completed or partial response')
    events=[json.loads(line) for line in (native/'events.jsonl').read_text().splitlines() if line.strip()]
    allowed={'thread.started','turn.started','error','turn.failed'}
    if not events or any(e.get('type') not in allowed for e in events) or not any(e.get('type')=='turn.failed' for e in events):
        raise ValueError('schema recovery excludes inference output or ambiguous events')
    errors=[json.loads(e['message']) for e in events if e.get('type')=='error']
    if len(errors)!=1 or errors[0].get('status')!=400 or errors[0].get('error',{}).get('code')!='invalid_json_schema':
        raise ValueError('native failure is not an explicit provider schema rejection')
    message=errors[0]['error'].get('message','')
    if 'uncertain_regions.items.properties.subject_ids.uniqueItems' not in message:
        raise ValueError('schema rejection is outside the implemented compatibility repair')
    old_schema=json.loads((native/'output-schema.json').read_text())
    normalized=deepcopy(old_schema)
    if normalized['properties']['uncertain_regions']['items']['properties']['subject_ids'].pop('uniqueItems',None) is not True:
        raise ValueError('failed schema did not contain the rejected uniqueness keyword')
    version=normalized['properties']['schema_version'].get('const')
    normalized['properties']['schema_version']={'type':'string','enum':[version]}
    if normalized!=TYPED_SURFACE_SCHEMA:
        raise ValueError('schema repair changed semantic fields beyond the verified transport changes')
    prompt=(native/'prompt.txt').read_text()
    if not prompt.startswith('USER:\n'):raise ValueError('unexpected observation prompt envelope')
    content=prompt[len('USER:\n'):];document=json.loads(content)
    if (document['question']!=parameters['question'] or document['task']!=domain['production_contract']['production']['tasks'][0] or
            document['media_order']!=[x['label'] for x in trial['media']] or
            document['typed_contract']['existing_authorized_support']!=trial['media'][3]['artifact']):
        raise ValueError('failed native prompt differs from the actual runtime observation')
    messages=[{'role':'user','content':content}]
    artifacts=tuple(ArtifactRef.from_dict(x['artifact']) for x in trial['media'])
    binding=payload_sha256({'messages':messages,'schema':old_schema,'artifacts':[x.to_dict() for x in artifacts]})
    request=ModelRequest('rsi-'+binding[:28],'agent.runtime_rsi',tuple(messages),old_schema,artifacts=artifacts,
        max_tokens=4096,timeout_seconds=runtime['planner_timeout_seconds'],max_attempts=1,seed=0)
    profile=ModelProfile.from_dict(runtime['planner_profile'])
    expected=payload_sha256({'request':request.to_dict(),'profile_hash':payload_sha256({'profile':profile.to_dict(),'model':profile.model,'endpoint':''})})
    native_intent=json.loads((native/'native.intent.json').read_text());failure=json.loads((native/'native.failure.json').read_text())
    if (rid!=request.request_id or intent['request_hash']!=expected or native_intent['request_hash']!=expected or
            failure['request_hash']!=expected or native_intent['profile_hash']!=profile.compute_hash() or
            failure['error_code']!='PLANNER_FAILED' or failure.get('wall_seconds') is None):
        raise ValueError('schema failure request/profile/usage binding changed')
    baselinepath=campaign/'rounds/r000/execute/trial.json'
    baseline=json.loads(baselinepath.read_text());work=Path(baseline['inner_work_dir'])
    known=inherited|set(domain['seed'].get('additional_pe_journals',{}).values())
    for ledger in ('model_receipts','pe_model_receipts'):
        for p in (work/ledger).glob('*.intent.json'):
            if file_sha256(p) not in known:raise ValueError('schema handoff contains a new image reservation')
    if any((work/'native').rglob('*.job.json')) or any((campaign/'experiments').rglob('*')):
        raise ValueError('schema handoff contains an image execution')
    observations=[]
    for item in state['data'].get('rsi_observations',[]):
        p=Path(item['stage_result']['evidence_refs'][0])
        effect=p.parent/'runtime-effect.json'
        if item['verification']['status']!='VERIFIED' or not effect.exists():
            raise ValueError('schema handoff has an unverified preceding observation')
        value=json.loads(effect.read_text())
        if any(file_sha256(Path(name))!=sha for name,sha in value['files'].items()):
            raise ValueError('preceding registered observation changed')
        observations.append({'path':str(p),'sha256':file_sha256(p),'item':item})
    files={str(p):file_sha256(p) for p in [directory/'goal.json',directory/'frozen.json',campaign/'state.json',
        run/'exit_code',tracepath,trialpath,baselinepath,work/'production.json']}
    files.update(journal_files)
    files.update({str(p):file_sha256(p) for p in native.iterdir() if p.is_file()})
    for item in observations:
        for p in Path(item['path']).parent.rglob('*.json'):files[str(p)]=file_sha256(p)
    old_recovery=domain.get('protocol_observation_recovery',{})
    files.update(old_recovery.get('file_hashes',{}))
    if any(file_sha256(Path(name))!=sha for name,sha in files.items()):raise ValueError('schema ancestry changed')
    return {'decision':decision,'trace_path':str(tracepath),'source':trial['source'],'candidate':trial['candidate'],
        'file_hashes':files,'journal_files':journal_files,'registered_observations':observations,
        'prior_runtime_planner_calls':domain['prior_runtime']['planner_calls'],'decision_slots_consumed':state['planner_calls'],
        'local_decision_limit':goal.budgets.max_planner_calls,'remaining_decisions':goal.budgets.max_planner_calls-state['planner_calls'],
        'diagnostic_config':str(directory),'recovered_decision_edited':False,
        'recovered_decision_trace_paths':[old_recovery['trace_path'],str(tracepath)],
        'failure_classification':'provider_preinference_schema_rejection','failed_request_id':rid,
        'failed_request_hash':expected,'failed_gateway_wall_seconds':intent['wall_seconds'],
        'failed_native_wall_seconds':failure['wall_seconds'],'old_schema_sha256':payload_sha256(old_schema),
        'new_schema_sha256':payload_sha256(TYPED_SURFACE_SCHEMA),'new_model_calls_for_recovery':0,
        'new_registered_observation_submissions_allowed':1,'image_experiments_consumed':0}
