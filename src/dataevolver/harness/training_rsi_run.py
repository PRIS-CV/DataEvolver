"""Freeze/run/verify a registered training RSI campaign in EvolutionController.

Bootstrap imports committed effects and their full cost ledger. It supplies no
diagnosis, edit region or proposed repair; the runtime Planner owns those choices.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

from dataevolver.paths import REPO_ROOT
from dataevolver.adapters.model_contracts import ModelProfile, ModelPolicy
from dataevolver.adapters.model_gateway import ModelGateway
from dataevolver.adapters.codex_planner import GatewayPlannerBackend
from dataevolver.adapters.cli_planner_service import CLIPlannerEndpointProvider
from dataevolver.dataset.contracts import ArtifactStore, file_sha256, payload_sha256
from .models import GoalSpec
from .operators import OperatorRegistry
from .state_store import atomic_write_json
from .runtime_rsi import RuntimeRSIPlanner
from .adapters.training_rsi import (TRIAL_ID, OBSERVE_ID, TrainingRSIDomain,
                                    TrainingRSITrialOperator, TrainingRSIObservationOperator)


def build_runtime(goal, campaign_root):
    config = goal.planner.get('runtime', {})
    review_only = config.get('domain', {}).get('capability_review_only') is True
    if (goal.planner.get('kind') != 'runtime_rsi' or goal.initial_operator_id != TRIAL_ID or
            goal.interventions or goal.resource_policy.get('allow_gpu') is not (not review_only) or
            goal.resource_policy.get('allow_network') is not True or
            goal.resource_policy.get('network_scope') != 'registered_planner_inference_only' or
            set(goal.resource_policy.get('allowed_operators', [])) != {TRIAL_ID, OBSERVE_ID} or
            goal.budgets.max_retries != 0):
        raise ValueError('runtime training RSI requires its explicit bounded authority')
    workflow=config['domain'].get('schema_version')=='dataevolver.training_workflow.v1'
    if workflow:
        from .training_workflow_run import validate_workflow_contract
        validate_workflow_contract(goal)
    elif review_only:
        if (goal.budgets.max_experiments != 0 or goal.budgets.max_planner_calls != 1
                or goal.budgets.max_operator_calls != 1):
            raise ValueError('capability review permits one cached bootstrap, one decision and zero experiments')
    elif not (1 <= goal.budgets.max_experiments <= 3 and 1 <= goal.budgets.max_planner_calls <= 12 and
              goal.budgets.max_operator_calls <= 12):
        raise ValueError('runtime pilot exceeds its registered intervention/probe limits')
    campaign = Path(campaign_root) / goal.campaign_id
    profile = ModelProfile.from_dict(config['planner_profile'])
    if profile.model != 'gpt-6.1-sol' or profile.backend != 'local_python':
        raise ValueError('runtime Planner model identity changed')
    gateway = ModelGateway(profiles=(profile,),
        policy=ModelPolicy({'agent.runtime_rsi': profile.profile_id}, None, usage_mode='accounting'),
        receipt_root=campaign / 'planner' / 'model_receipts',
        artifact_store=ArtifactStore(config['domain']['roots']))
    gateway.factories[profile.profile_id] = lambda: CLIPlannerEndpointProvider(
        profile, config['planner_worker'], gateway.store, campaign / 'planner' / 'native')
    backend = GatewayPlannerBackend(gateway, model_id=profile.model,
                                     timeout_seconds=config.get('planner_timeout_seconds', 900))
    registry = OperatorRegistry()
    if workflow:
        from .training_workflow import TrainingWorkflowDomain, WorkflowOperator
        domain=TrainingWorkflowDomain(config['domain'],campaign,backend=backend)
        registry.register(WorkflowOperator(domain))
        registry.register(WorkflowOperator(domain,observation=True))
    else:
        domain = TrainingRSIDomain(config['domain'], campaign, backend=backend)
        registry.register(TrainingRSITrialOperator(domain, visual_backend=backend))
        registry.register(TrainingRSIObservationOperator(domain))
    planner = RuntimeRSIPlanner(backend=backend, domain=domain, trace_root=campaign / 'planner' / 'decisions')
    return registry, planner


def bind_resident_deployment(contract, env, deployment):
    """Bind new hardware to the same model assets; never alter old frozen evidence."""
    from dataevolver.adapters import training_media_worker,training_media_resident,training_prompt_enhancement
    media_path=Path(deployment['media_identity'])
    prompt_path=Path(deployment['prompt_identity'])
    media=json.loads(media_path.read_text());prompt=json.loads(prompt_path.read_text())
    if any(w['kind'] in training_prompt_enhancement.PE_MODELS for w in contract['models']['workers'].values()):
        if not prompt.get('status_path'):
            raise ValueError('PE resident identity lacks its actual status_path; repair deployment metadata before freeze')
        pe_status=json.loads(Path(prompt['status_path']).read_text())
        if pe_status['pid']!=prompt['pid'] or pe_status['state']!='READY':
            raise ValueError('PE resident is not idle for a deployment handoff')
    if media['gpu_uuids']!=deployment['gpu_uuids'] or not media['gpu_uuids']:
        raise ValueError('explicit resumed GPU mapping changed')
    if media['pid']!=prompt['pid']:raise ValueError('combined deployment must share actual resident objects')
    env['CUDA_VISIBLE_DEVICES']=','.join(media['gpu_uuids'])
    for worker in contract['models']['workers'].values():
        kind=worker['kind'];is_prompt=kind in training_prompt_enhancement.PE_MODELS
        document=prompt if is_prompt else media
        model=document['models'][kind]
        if model['assets_sha256']!=payload_sha256(worker['file_hashes']):
            raise ValueError('hardware migration cannot replace frozen model assets')
        if is_prompt:
            worker['implementation_sha256']=file_sha256(Path(training_prompt_enhancement.__file__))
            worker['deployment_sha256']=file_sha256(prompt_path)
            env[worker['deployment_env']]=str(prompt_path)
            continue
        devices=set(map(str,model['device_map'].values()))
        if kind=='sam3_instances':
            if model['device_map'].get('unused_video_tracker')!='cpu':
                raise ValueError('unused SAM tracker must remain CPU resident')
            devices={str(model['device_map']['detector'])}
        execution='cpu' if devices=={'cpu'} else 'cuda'
        if execution=='cuda' and not devices<={'0','cuda:0'}:
            raise ValueError('resumed deployment permits only physical GPU0 or CPU tensors')
        worker['execution_device']=execution
        worker['implementation_sha256']=file_sha256(Path(training_media_worker.__file__))
        worker.pop('shard_max_memory_mib',None);worker.pop('shard_min_free_mib',None)
        resident=worker['resident']
        resident.update(identity_sha256=file_sha256(media_path),
            implementation_sha256=file_sha256(Path(training_media_resident.__file__)),
            activation_reserve_mib=[] if execution=='cpu' else deployment['activation_reserve_mib'][kind])
        worker['resource_profile'].update(cpu_offload=False,
            min_free_mib=65536 if execution=='cpu' else max(resident['activation_reserve_mib']),
            estimate_basis='actual resident GPU0/CPU model placement; activation-only admission')
        env[resident['identity_env']]=str(media_path);env[resident['socket_env']]=media['socket_path']
        env[worker['model_path_env']]=model['path']


def inherited_planner_usage(prior_state,prior_goal,slot):
    safe_terminal=(prior_state['state']=='ABSTAIN' and prior_state.get('terminal_reason')=='RSI_AGENT_ABSTAINED'
                   and prior_state['experiments_used']==0)
    if (prior_state['state']!='PAUSED' and not safe_terminal) or prior_goal.objective['source_family']!=slot:
        raise ValueError('a protocol upgrade needs a paused or explicitly abstained zero-experiment same-source predecessor')
    used=prior_state['planner_calls']+prior_goal.planner['runtime']['domain'].get('prior_runtime',{}).get('planner_calls',0)
    if used>=10:raise ValueError('protocol handoff cannot reset the inherited Planner allowance')
    return used


def freeze(seed_directory, out, campaign_id, slot, *, planner_config, max_experiments=2, wall_seconds=86400, prior_runtime=None, deployment=None,recover_preflight=False,continuation_planner_calls=0,continuation_authority='',capability_review_only=False,capability_request_config=None,surface_context_reconstruction=False,typed_surface_preflight=False,recover_observation_config=None,conservative_surface_authority=False,surface_composition=False):
    seed_directory, out = Path(seed_directory).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('refusing to replace an existing runtime freeze')
    if capability_review_only and (not prior_runtime or max_experiments != 0 or continuation_planner_calls != 1):
        raise ValueError('capability review needs one explicit decision, zero experiments and a verified predecessor')
    if continuation_planner_calls and (not prior_runtime or recover_preflight or
            not 1<=continuation_planner_calls<=12 or not continuation_authority.strip() or
            (max_experiments != 0 if capability_review_only else not 1<=max_experiments<=3)):
        raise ValueError('continuation needs explicit finite authority and an immutable predecessor')
    if conservative_surface_authority and not typed_surface_preflight:
        raise ValueError('conservative surface authority requires the typed reconstruction candidate')
    if surface_composition and (not typed_surface_preflight or conservative_surface_authority):
        raise ValueError('surface composition requires fresh coverage evidence and its own authority contract')
    if recover_observation_config and not typed_surface_preflight:
        raise ValueError('protocol observation recovery requires the typed preflight implementation')
    if typed_surface_preflight and not surface_context_reconstruction:
        raise ValueError('typed surface preflight needs the frozen context reconstruction candidate')
    if surface_context_reconstruction and not capability_request_config:
        raise ValueError('context reconstruction must bind an actual capability request')
    if capability_request_config and (not prior_runtime or capability_review_only or not continuation_planner_calls):
        raise ValueError('surface implementation regression needs an explicit finite new-capability admission')
    dirty = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=REPO_ROOT, text=True)
    if dirty.strip():
        raise ValueError('runtime freeze requires committed source')
    seed = json.loads((seed_directory / 'config.json').read_text())
    validation = json.loads((seed_directory / 'validation.json').read_text())
    if not all(validation[k] for k in ('effect_verified', 'frozen_contract_valid', 'replay_matches_snapshot', 'all_tasks_recorded')):
        raise ValueError('seed batch does not have verified committed effects')
    task = next(t for t in seed['production']['tasks'] if t['slot_id'] == slot)
    family = slot + '-a0'
    seed_goal = json.loads((seed_directory / 'goal.json').read_text())
    work = seed_directory / 'campaigns' / seed_goal['campaign_id'] / 'rounds/r000/execute'
    outcome = json.loads((work / 'families' / (family + '.json')).read_text())
    if outcome['status'] not in {'pass', 'fail'}:
        raise ValueError('seed has an unresolved infrastructure effect')
    if capability_review_only and outcome['status'] != 'fail':
        raise ValueError('capability review needs a committed failed baseline, not a new visual audit')
    records = outcome['model_requests']
    source = next(x['receipt']['output']['artifact'] for x in records if x['request']['role'] == 'training.generate')
    roots = json.loads((seed_directory / 'frozen.json').read_text())['roots']
    env = {}
    for line in (seed_directory / 'env.sh').read_text().splitlines():
        if line.startswith('export '):
            key, value = line[7:].split('=', 1); env[key] = shlex.split(value)[0]
    os.environ.update(env)
    for key in ('endpoint_env', 'token_file_env'):
        name = planner_config['planner_worker'][key]
        if not os.environ.get(name):
            raise ValueError('runtime Planner transport environment is missing: ' + name)
        env[name] = os.environ[name]
    base = json.loads(json.dumps(seed))
    if deployment is not None:
        bind_resident_deployment(base,env,deployment)
        os.environ.update(env)
    production = base['production']
    for name in ('generation_reuse', 'model_reuse', 'shadow_region_corrections', 'surface_effect_contracts', 'rsi_candidate', 'agent_evidence'):
        production.pop(name, None)
    production.update(count=1, tasks=[task], split_counts={k: int(task['split'] == k) for k in production['split_counts']},
                      checker_precheck=False, max_consecutive_failures=1, phase='system_owned_runtime_rsi')
    # Metadata permission changes keep the existing model objects and old identity snapshot.
    live = Path(env['DATAEVOLVER_RESIDENT_IDENTITY'])
    identity = json.loads(live.read_text())
    def deployment_ready():
        snapshots = []
        for path in planner_config['deployment_status_paths']:
            status = json.loads(Path(path).read_text())
            if status['state'] != 'READY' or not (Path('/proc') / str(status['pid'])).exists():
                raise ValueError('resident deployment is not idle; reconcile in-flight requests before metadata handoff')
            snapshots.append({'status_path': path, 'pid': status['pid'], 'state': status['state'],
                              'status_sha256': file_sha256(Path(path))})
        if identity['pid'] not in {x['pid'] for x in snapshots} or not 1<=len(snapshots)<=2:
            raise ValueError('both existing deployments must be accounted for')
        return snapshots
    deployment_ready()
    archive = seed_directory / 'resident-identity.json'
    if archive.exists() and file_sha256(archive) != file_sha256(live) and prior_runtime is None:
        raise ValueError('seed resident identity was already archived differently')
    dataset_parent = Path(roots[seed['dataset_store']]).parent / campaign_id
    portable_parent = dataset_parent.with_name(campaign_id + '-portable')
    scopes = {name: {'dataset': str(dataset_parent / name), 'bundle': str(portable_parent / name)}
              for name in ['baseline', *[f'candidate-{i:03d}' for i in range(max_experiments)]]}
    identity.update(deployment_id=campaign_id, authorized_job_roots=[str(out / 'campaigns')],
                    authorized_dataset_roots=[v['dataset'] for v in scopes.values()])
    out.mkdir(parents=True)
    identity_path = out / 'resident-identity.json'
    shutil.copy2(live, out / 'deployment-parent-identity.json')
    atomic_write_json(identity_path, identity); identity_path.chmod(0o600)
    for worker in base['models']['workers'].values():
        if worker.get('resident'):
            worker['resident']['identity_sha256'] = file_sha256(identity_path)
    # Only committed native outputs/journals are copied on a later owned action.
    seed_files = {p.relative_to(work).as_posix(): file_sha256(p)
                  for name in ('model_receipts', 'pe_model_receipts') for p in (work / name).glob('*.json')
                  if p.name.endswith(('.intent.json', '.receipt.json'))}
    seed_files['families/' + family + '.json'] = file_sha256(work / 'families' / (family + '.json'))
    domain = {'schema_version': 'dataevolver.training_rsi_domain.v1', 'production_contract': base,
        'roots': roots, 'scopes': scopes, 'evidence_root': str(dataset_parent.with_name(campaign_id + '-evidence')),
        'seed': {'work_dir': str(work), 'dataset_root': roots[seed['dataset_store']], 'family_id': family,
                 'source_sha256': source['sha256'], 'config_sha256': file_sha256(seed_directory / 'config.json'),
                 'model_records': records, 'file_hashes': seed_files},
        'experience_snapshot': {'compatible': [], 'reason': 'legacy all2 developer-proposed experience is not verified runtime knowledge for this task'}}
    if capability_review_only:
        domain['capability_review_only'] = True
    inherited_planner_calls = 0
    if prior_runtime is not None:
        prior = Path(prior_runtime).resolve()
        prior_goal = GoalSpec.from_json((prior / 'goal.json').read_text())
        prior_campaign = prior / 'campaigns' / prior_goal.campaign_id
        prior_state = json.loads((prior_campaign / 'state.json').read_text())
        continuation=None
        if continuation_planner_calls:
            from .training_continuation import verified_rejected_continuation,bind_rejected_view
            import sys,time
            from datetime import datetime
            process=subprocess.run([sys.executable,'-m','dataevolver.harness.training_rsi_run','validate',
                '--config-dir',str(prior)],cwd=prior/'source',
                env={**os.environ,'PYTHONPATH':str(prior/'source/src')},capture_output=True,text=True)
            verification=json.loads(process.stdout)
            if process.returncode not in (0,1) or not all(verification.get(k) is True for k in
                    ('frozen_contract_valid','replay_matches_snapshot')):
                raise ValueError('predecessor frozen/replay verification failed')
            continuation=verified_rejected_continuation(prior_campaign,prior_state,prior_goal,slot)
            inherited_planner_calls=continuation['planner_calls']
            view,old_scope=bind_rejected_view(continuation,out/'campaigns'/campaign_id,prior_goal.planner['runtime']['domain']['scopes'])
            domain['continuation_rejected_trial']=view;domain['scopes']['prior-rejected']=old_scope
            domain['continuation']={'authority':continuation_authority,'new_planner_allowance':continuation_planner_calls,
                'new_experiment_allowance':max_experiments,'previous_budget':continuation['previous_budget'],
                'historical_planner_calls':inherited_planner_calls,'historical_experiments_used':continuation['experiments_used'],
                'counter_reset':False,'file_hashes':continuation['file_hashes']}
            left=prior_goal.budgets.max_wall_seconds-(time.time()-datetime.fromisoformat(prior_state['started_at']).timestamp())
            if left<1:raise ValueError('predecessor wall allowance exhausted')
            wall_seconds=min(wall_seconds,int(left))
        elif recover_preflight:
            from .training_preflight_recovery import verified_preflight_proposal
            recovery=verified_preflight_proposal(prior_campaign,prior_state)
            view={**prior_state,'experiments_used':0}
            inherited_planner_calls=inherited_planner_usage(view,prior_goal,slot)
            max_experiments-=recovery['prior_experiments_used']
            if max_experiments<1:raise ValueError('no inherited experiment allowance remains')
            domain['preflight_proposal_recovery']=recovery
        else:
            inherited_planner_calls=inherited_planner_usage(prior_state,prior_goal,slot)
        domain['prior_runtime'] = {'config_dir': str(prior), 'goal_sha256': prior_goal.sha256(),
            'state_sha256': file_sha256(prior_campaign / 'state.json'), 'planner_calls': inherited_planner_calls,
            'handoff_reason': 'registered observation protocol upgrade after safe pause/abstention; preserve all decisions, costs and remaining Planner allowance; no source edit',
            'predecessor_terminal_reason':prior_state.get('terminal_reason','')}
        if continuation:
            domain['prior_runtime'].update(experiments_used=continuation['experiments_used'],
                comparisons=continuation['comparisons'],handoff_reason='explicit bounded continuation after verified rollback; prior counters and effects retained')
        prior_observations=[]
        for item in prior_state['data'].get('rsi_observations',[]):
            if item['verification']['status']!='VERIFIED' or item['stage_result']['status']!='SUCCESS':continue
            path=Path(item['stage_result']['evidence_refs'][0])
            prior_observations.append({'path':str(path),'sha256':file_sha256(path),'item':item})
        domain['prior_observations']=prior_goal.planner['runtime']['domain'].get('prior_observations',[])+prior_observations
        additional = dict(prior_goal.planner['runtime']['domain']['seed'].get('additional_journals',{}))
        for ledger in prior_campaign.glob('observations/*/model_receipts'):
            for path in ledger.glob('*.json'):
                if path.name.endswith(('.intent.json', '.receipt.json')):
                    additional[str(path)] = file_sha256(path)
        domain['seed']['additional_journals'] = additional
        if continuation:
            domain['seed']['additional_journals'].update(continuation['journals']['model_receipts'])
            prior_pe=prior_goal.planner['runtime']['domain']['seed'].get('additional_pe_journals',{})
            domain['seed']['additional_pe_journals']={**prior_pe,**continuation['journals']['pe_model_receipts']}
    if capability_request_config:
        from .training_capability_admission import admit_surface_capability
        implemented=admit_surface_capability(capability_request_config,prior_runtime,source,base['planning']['goal'],reconstruction=surface_context_reconstruction,typed_preflight=typed_surface_preflight,authority_separation=conservative_surface_authority,composition=surface_composition)
        domain['surface_capability']=implemented
        inherited={x['item']['call_id']:x for x in domain.get('prior_observations',[])}
        for item in implemented['registered_observations']:
            name=item['item']['call_id']
            if name in inherited and inherited[name]!=item:raise ValueError('diagnostic observation identity collision')
            inherited[name]=item
        domain['prior_observations']=list(inherited.values())
        domain['prior_runtime']['planner_calls']+=implemented['diagnostic_planner_calls']
        domain['continuation']['historical_planner_calls']=domain['prior_runtime']['planner_calls']
        domain['continuation']['file_hashes'].update(implemented['file_hashes'])
        domain['continuation']['new_capability_not_recipe_extension']=True
    if recover_observation_config:
        from .training_protocol_recovery import verified_observation_handoff
        handoff=verified_observation_handoff(recover_observation_config,prior_config=prior_runtime,
            capability_request_sha256=domain['surface_capability']['request_sha256'])
        if (domain['prior_runtime']['planner_calls']!=handoff['prior_runtime_planner_calls'] or
                continuation_planner_calls!=handoff['local_decision_limit']):
            raise ValueError('observation handoff must retain the original consumed slot and remaining allowance')
        domain['protocol_observation_recovery']=handoff
        # Recovered Planner journals stay in the Planner ledger, copied below;
        # the primary media Gateway has its own immutable policy contract.
        domain['continuation']['file_hashes'].update(handoff['file_hashes'])
        domain['continuation']['new_planner_allowance']=handoff['remaining_decisions']
        domain['continuation']['recovered_local_decisions']=handoff['decision_slots_consumed']
        inherited={x['item']['call_id']:x for x in domain.get('prior_observations',[])}
        for item in handoff.get('registered_observations',[]):
            name=item['item']['call_id']
            if name in inherited and inherited[name]!=item:raise ValueError('recovered observation identity collision')
            inherited[name]=item
        domain['prior_observations']=list(inherited.values())
    if recover_preflight and prior_runtime is None:raise ValueError('preflight recovery needs its immutable predecessor')
    # Recovery may reduce the remaining experiment budget after the original
    # scopes were allocated. Extra unregistered scopes are removed.
    domain['scopes']={k:v for k,v in domain['scopes'].items() if k in {'baseline','prior-rejected'} or int(k.rsplit('-',1)[1])<max_experiments}
    evaluator = {'schema_version': 'dataevolver.harness.evaluator.v1', 'evaluator_id': 'training-runtime-rsi-v1', 'revision': '1',
        'hard_constraints': {k: {'required': True} for k in ('artifact_valid', 'source_identity', 'quality_pass')},
        'target_metrics': {'qualified_ratio': {'required': True, 'minimum': 1.0, 'minimum_improvement': 0.0}}}
    atomic_write_json(out / 'evaluator.json', evaluator)
    key = {'input_artifact_hash': source['sha256'], 'parent_artifact_hash': source['sha256'], 'seed': task['seed'],
           'operator_id': TRIAL_ID, 'operator_version': '1', 'model_config_hash': payload_sha256(base['models']),
           'preprocessing_hash': payload_sha256(base['planning']), 'evaluator_hash': file_sha256(out / 'evaluator.json'),
           'environment_contract_hash': payload_sha256({'roots': roots, 'scopes': scopes})}
    remaining_planner=continuation_planner_calls or 10-inherited_planner_calls
    if remaining_planner<1:raise ValueError('no remaining inherited Planner allowance; cannot reset it in a protocol upgrade')
    goal = {'schema_version': 'dataevolver.harness.goal.v1', 'campaign_id': campaign_id,
        'objective': {'real_data_execution': True, 'kind': 'training_capability_review' if capability_review_only else 'training_runtime_rsi', 'source_family': slot,
                      'initial_trial_key': key, 'canary_trial_key': key},
        'initial_operator_id': TRIAL_ID, 'route_candidates': [TRIAL_ID],
        'budgets': {'max_rounds': max_experiments + 2, 'max_operator_calls': 1 if capability_review_only else 12, 'max_experiments': max_experiments,
                    'max_retries': 0, 'max_planner_calls': remaining_planner, 'max_wall_seconds': wall_seconds},
        'resource_policy': {'allow_gpu': not capability_review_only, 'allow_network': True, 'network_scope': 'registered_planner_inference_only',
                            'allowed_operators': [TRIAL_ID, OBSERVE_ID]},
        'frozen_evaluator': {'path': str(out / 'evaluator.json'), 'sha256': file_sha256(out / 'evaluator.json')},
        'planner': {'kind': 'runtime_rsi', 'runtime': {**planner_config, 'domain': domain}},
        'initial_operator_parameters': {'scope': 'baseline', 'patch': {}, 'experience_id': ''},
        'interventions': [], 'lease_seconds': wall_seconds + 600}
    GoalSpec.from_dict(goal)
    atomic_write_json(out / 'goal.json', goal)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
    archive_source = out / 'source.tar'
    subprocess.run(['git', 'archive', '--format=tar', '--output', str(archive_source), commit,
                    'src', 'configs', 'pyproject.toml'], cwd=REPO_ROOT, check=True)
    with tarfile.open(archive_source) as handle:
        handle.extractall(out / 'source', filter='data')
    atomic_write_json(out / 'frozen.json', {'source_commit': commit, 'goal_sha256': file_sha256(out / 'goal.json'),
        'source_archive_sha256': file_sha256(archive_source), 'source_hashes': {p.relative_to(out / 'source').as_posix(): file_sha256(p)
            for p in (out / 'source').rglob('*') if p.is_file()}, 'seed_usage': validation['usage'],
        'runtime_proposal_supplied_by_developer': False})
    # Fail before rotating API metadata if registration or seed readback is invalid.
    env['DATAEVOLVER_RESIDENT_IDENTITY'] = str(identity_path)
    os.environ.update(env)
    build_runtime(GoalSpec.from_dict(goal), out / 'campaigns')
    if prior_runtime is not None:
        target = out / 'campaigns' / campaign_id / 'planner/model_receipts'
        target.mkdir(parents=True, exist_ok=True)
        for path in (prior_campaign / 'planner/model_receipts').glob('*.json'):
            if path.name.endswith(('.intent.json', '.receipt.json')):
                shutil.copy2(path, target / path.name)
                if file_sha256(path) != file_sha256(target / path.name):
                    raise ValueError('inherited Planner ledger changed')
    for diagnostic_config in [p for p in (capability_request_config,recover_observation_config) if p]:
        diagnostic=json.loads((Path(diagnostic_config)/'goal.json').read_text())
        source_ledger=Path(diagnostic_config)/'campaigns'/diagnostic['campaign_id']/'planner/model_receipts'
        target=out/'campaigns'/campaign_id/'planner/model_receipts';target.mkdir(parents=True,exist_ok=True)
        for path in source_ledger.glob('*.json'):
            if not path.name.endswith(('.intent.json','.receipt.json')):continue
            dest=target/path.name
            if dest.exists() and file_sha256(dest)!=file_sha256(path):
                raise ValueError('capability Planner ledger collision')
            shutil.copy2(path,dest)
    ready = deployment_ready()
    atomic_write_json(out / 'deployment-handoff.json', {'snapshots': ready, 'weights_reloaded': False,
                                                     'old_identity_sha256': file_sha256(live),
                                                     'new_identity_sha256': file_sha256(identity_path)})
    if not archive.exists():
        shutil.copy2(live, archive)
    link = live.with_name(live.name + '.runtime-next')
    link.symlink_to(identity_path); os.replace(link, live)
    (out / 'env.sh').write_text('\n'.join('export ' + k + '=' + shlex.quote(v) for k, v in env.items()) + '\n')
    (out / 'env.sh').chmod(0o600)
    return {'status': 'FROZEN', 'config_dir': str(out), 'source_commit': commit, 'source': source,
            'new_model_calls': 0, 'source_specific_repair_supplied': False}


def validate(directory):
    from .cli import _existing_controller
    directory = Path(directory).resolve()
    frozen = json.loads((directory / 'frozen.json').read_text())
    if file_sha256(directory / 'goal.json') != frozen['goal_sha256']:
        raise ValueError('runtime goal changed')
    for path, digest in frozen['source_hashes'].items():
        if file_sha256(directory / 'source' / path) != digest:
            raise ValueError('runtime source changed')
    goal = GoalSpec.from_json((directory / 'goal.json').read_text())
    controller = _existing_controller(directory / 'campaigns', goal.campaign_id)
    state = controller.status()
    stable, drift = controller._frozen()
    replay = controller.replay().to_dict() == state.to_dict()
    traces = []
    paths=sorted((controller.layout.campaign_dir / 'planner/decisions').glob('p*/trace.json'))
    recovered=goal.planner['runtime']['domain'].get('protocol_observation_recovery')
    if recovered and state.data.get('preflight_proposal_recovered'):
        paths=[Path(p) for p in recovered.get('recovered_decision_trace_paths',[recovered['trace_path']])]+paths
    for path in paths:
        item = json.loads(path.read_text())
        traces.append({'path': str(path), 'outcome': item['outcome'], 'proposal_actor': item['proposal_actor'],
                       'real_model_receipt': bool(item.get('model_call', {}).get('receipt'))})
    histories = state.data.get('rsi_history', [])
    adopted = [x for x in histories if x['decision']['decision'] == 'ADOPT_CANDIDATE']
    value = {'harness_state': state.state.value, 'terminal_reason': state.terminal_reason,
        'frozen_contract_valid': stable, 'contract_drift': list(drift), 'replay_matches_snapshot': replay,
        'runtime_planner_decisions': traces, 'registered_observations': len(state.data.get('rsi_observations', [])),
        'comparisons': histories, 'runtime_recipe_adoption_verified': bool(adopted) and bool(traces) and
            all(x['real_model_receipt'] for x in traces) and stable and replay,
        'planner_usage': controller.planner.backend.gateway.audit_usage(),
        'independent_source_validation_verified': False, 'promoted_runtime_experience_verified': False,
        'recursive_method_improvement_verified': False, 'training_benefit_verified': False}
    requested = state.data.get('rsi_capability_request')
    request_verified = bool(requested and file_sha256(Path(requested['path'])) == requested['sha256'])
    value.update(capability_request_recorded=request_verified,
                 capability_review_completed=bool(goal.planner['runtime']['domain'].get('capability_review_only')
                     and state.terminal_reason == 'CAPABILITY_REQUESTED' and request_verified and stable and replay
                     and len(traces) == 1 and all(t['real_model_receipt'] for t in traces)))
    if requested:
        value['capability_request'] = requested
    if goal.planner['runtime']['domain'].get('schema_version')=='dataevolver.training_workflow.v1':
        wf=state.data.get('training_workflow',{})
        archived=[h for entry in wf.get('completed',[]) for h in entry.get('history',[])]
        completed=state.state.value=='COMPLETE' and wf.get('index')==len(goal.planner['runtime']['domain']['stages'])
        value.update(workflow_completed=bool(completed),completed_stages=wf.get('completed',[]),
            runtime_recipe_adoption_verified=bool(traces) and stable and replay and
                all(x['real_model_receipt'] for x in traces) and
                any(h['decision']['decision']=='ADOPT_CANDIDATE' for h in [*histories,*archived]),
            independent_source_validation_verified=bool(completed and wf.get('validation_pairs')),
            promoted_runtime_experience_verified=bool(completed and wf.get('promoted_sha256')),
            actual_compatible_reuse_verified=bool(completed and wf.get('reuse_pairs')),
            workflow_state=wf)
        if completed:
            from .training_workflow_delivery import attach_terminal_proof
            value['terminal_delivery_proof']=attach_terminal_proof(directory,controller)
    atomic_write_json(directory / 'validation.json', value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze')
    for name in ('seed-config', 'out', 'campaign-id', 'slot', 'planner-config'):
        f.add_argument('--' + name, required=True)
    f.add_argument('--max-experiments', type=int, default=2)
    f.add_argument('--prior-runtime')
    f.add_argument('--deployment-config')
    f.add_argument('--recover-preflight',action='store_true')
    f.add_argument('--continuation-planner-calls',type=int,default=0)
    f.add_argument('--continuation-authority',default='')
    f.add_argument('--capability-review-only',action='store_true')
    f.add_argument('--capability-request-config')
    f.add_argument('--surface-context-reconstruction',action='store_true')
    f.add_argument('--typed-surface-preflight',action='store_true')
    f.add_argument('--recover-observation-config')
    f.add_argument('--conservative-surface-authority',action='store_true')
    f.add_argument('--surface-composition',action='store_true')
    for name in ('run', 'validate'):
        sub.add_parser(name).add_argument('--config-dir', required=True)
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        result = freeze(args.seed_config, args.out, args.campaign_id, args.slot,
                        planner_config=json.loads(Path(args.planner_config).read_text()), max_experiments=args.max_experiments,
                        prior_runtime=args.prior_runtime,
                        recover_preflight=args.recover_preflight,
                        continuation_planner_calls=args.continuation_planner_calls,
                        continuation_authority=args.continuation_authority,
                        capability_review_only=args.capability_review_only,
                        capability_request_config=args.capability_request_config,
                        surface_context_reconstruction=args.surface_context_reconstruction,
                        typed_surface_preflight=args.typed_surface_preflight,
                        recover_observation_config=args.recover_observation_config,
                        conservative_surface_authority=args.conservative_surface_authority,
                        surface_composition=args.surface_composition,
                        deployment=json.loads(Path(args.deployment_config).read_text()) if args.deployment_config else None)
    else:
        directory = Path(args.config_dir).resolve()
        if args.command == 'run':
            from .cli import main as harness_main
            goal = GoalSpec.from_json((directory / 'goal.json').read_text())
            harness_main(['evolve', '--goal', str(directory / 'goal.json'), '--campaign-root', str(directory / 'campaigns'),
                          '--idempotency-key', 'runtime:' + goal.sha256()])
        result = validate(directory)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if args.command=='freeze' or result.get('capability_review_completed') or (result['harness_state']=='COMPLETE' and
        result['frozen_contract_valid'] and result['replay_matches_snapshot'] and
        (not result.get('workflow_completed') or result.get('terminal_delivery_proof',{}).get('accepted'))) else 1


if __name__ == '__main__':
    raise SystemExit(main())
