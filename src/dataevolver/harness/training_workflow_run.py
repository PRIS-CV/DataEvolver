"""Freeze, run and audit the registered multi-source Harness training workflow."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from .adapters.training_rsi import TRIAL_ID

from dataevolver.paths import REPO_ROOT
from dataevolver.dataset.contracts import file_sha256
from dataevolver.dataset.supervision import load_dataset
from .models import GoalSpec,CampaignState
from .state_store import atomic_write_json
from .training_workflow import SCHEMA,task_scope,verify_runtime_pair
from .training_workflow_delivery import verify_trial_bundle
from .training_rsi_run import build_runtime


def workflow_budgets(domain,wall_seconds):
    limits=domain['limits'];n=sum(s['kind'] in {'main','validation'} for s in domain['stages']);stages=len(domain['stages'])
    return {'max_rounds':stages+n*limits['experiments_per_task']+1,
        'max_experiments':n*limits['experiments_per_task'],
        'max_planner_calls':stages*limits['planner_per_task'],
        'max_operator_calls':stages+n*(limits['planner_per_task']+2*limits['experiments_per_task']),
        'max_retries':0,'max_wall_seconds':wall_seconds}


def validate_workflow_contract(goal):
    d=goal.planner['runtime']['domain'];stages=d['stages'];limits=d['limits']
    if d.get('allow_fixture') or d['production_contract']['production']['evidence_kind']!='real_execution':
        raise ValueError('a real training workflow cannot accept fixture evidence')
    if (d['schema_version']!=SCHEMA or len(stages)>40 or len(stages)<5 or
            not 1<=limits['experiments_per_task']<=3 or not 3<=limits['planner_per_task']<=12):
        raise ValueError('workflow exceeds its frozen task/probe limits')
    if goal.budgets.to_dict()!=workflow_budgets(d,goal.budgets.max_wall_seconds):
        raise ValueError('workflow budgets do not match the frozen finite queue')
    if limits['max_experiments']!=goal.budgets.max_experiments:raise ValueError('unbounded candidate scopes')
    kinds=[s['kind'] for s in stages]
    if kinds[0]!='distill' or kinds[-1]!='export' or kinds.count('promote')!=1:
        raise ValueError('workflow must distil, independently validate, promote, produce and export')
    p=kinds.index('promote')
    if not all(k=='validation' for k in kinds[1:p]) or not 1<=p-1<=3 or not all(k=='main' for k in kinds[p+1:-1]):
        raise ValueError('candidate validation and frozen main production must be separate')
    media=[s for s in stages if s['kind'] in {'main','validation'}]
    from dataevolver.dataset.contracts import identifier
    for s in stages:identifier(s['id'],'workflow stage')
    ids=[s['task']['slot_id'] for s in media]
    if len(set(ids))!=len(ids):raise ValueError('workflow source families are not isolated')
    if any(s['task']['split']!='train' for s in media if s['kind']=='validation'):
        raise ValueError('knowledge selection may not read delivery validation/test sources')
    if goal.resource_policy.get('physical_gpu_indices')!=[0]:raise ValueError('GPU1 is forbidden for this workflow')
    if not d['gpu0_uuid'] or len(d['gpu_uuids'])!=1 or d['gpu_uuids']!=[d['gpu0_uuid']]:
        raise ValueError('deployment must expose physical GPU0 only')
    expected={'train':18,'validation':6,'test':6}
    if d['expected_counts']!=expected:raise ValueError('original delivery split changed')
    counts=dict(d['accepted_counts'])
    for s in stages:
        if s['kind']=='main':counts[s['task']['split']]+=1
    if counts!=expected:raise ValueError('accepted sources and queue do not cover the original thirty slots')
    if len(d['accepted_proofs'])!=sum(d['accepted_counts'].values()):raise ValueError('accepted proof count differs')
    if any(s['kind']=='validation' and task_scope(s['task'],d['production_contract']['models'])!=d['experience_scope'] for s in stages):
        raise ValueError('independent validation is outside empirical knowledge scope')


def freeze(pilot_directory,original_config,accepted_proofs,out,campaign_id,validation_tasks,*,resident_pointer):
    """No diagnosis or ROI input: only verified runtime pilot effects and task metadata."""
    pilot_directory=Path(pilot_directory).resolve();out=Path(out).resolve()
    if out.exists():raise ValueError('refusing to replace an existing workflow freeze')
    if subprocess.check_output(['git','status','--porcelain=v1'],cwd=REPO_ROOT,text=True).strip():
        raise ValueError('workflow requires committed source')
    pg=GoalSpec.from_json((pilot_directory/'goal.json').read_text())
    env={}
    for line in (pilot_directory/'env.sh').read_text().splitlines():
        if line.startswith('export '):
            k,v=line[7:].split('=',1);env[k]=shlex.split(v)[0]
    os.environ.update(env)
    campaign=pilot_directory/'campaigns'/pg.campaign_id
    state=CampaignState.from_dict(json.loads((campaign/'state.json').read_text()))
    if state.state.value!='COMPLETE':
        raise ValueError('runtime pilot must naturally complete with frozen/replay verification before workflow handoff')
    # Validate the predecessor with ITS immutable implementation. Loading a
    # predecessor registry from new source would correctly report code drift.
    process=subprocess.run([sys.executable,'-m','dataevolver.harness.training_rsi_run','validate',
        '--config-dir',str(pilot_directory)],cwd=pilot_directory/'source',
        env={**os.environ,'PYTHONPATH':str(pilot_directory/'source/src')},capture_output=True,text=True)
    if process.returncode:
        raise ValueError('predecessor validation failed under its frozen source: '+process.stderr[-1000:])
    validation_path=pilot_directory/'validation.json';validation=json.loads(validation_path.read_text())
    if not all(validation.get(k) is True for k in ('frozen_contract_valid','replay_matches_snapshot','runtime_recipe_adoption_verified')):
        raise ValueError('predecessor has no verified autonomous runtime adoption')
    pilot_path=Path(state.data['last_stage_result']['evidence_refs'][0]);pilot=json.loads(pilot_path.read_text())
    baseline_path=campaign/'rounds/r000/execute/trial.json';baseline=json.loads(baseline_path.read_text())
    verify_runtime_pair(baseline,pilot,state.data.get('rsi_history',[]))
    proof_path=pilot_directory/'workflow-handoff/independent-validation.json'
    verify_trial_bundle(pilot,proof_path.parent,receipt_roots=[campaign])
    seed=copy.deepcopy(pg.planner['runtime']['domain']['production_contract'])
    original=json.loads((Path(original_config)/'config.json').read_text())
    tasks=original['production']['tasks']
    if len(tasks)!=30:raise ValueError('workflow must retain the original thirty-slot schedule')
    all_proofs=[Path(x).resolve() for x in accepted_proofs]+[proof_path]
    accepted={};counts={'train':0,'validation':0,'test':0}
    for path in all_proofs:
        p=json.loads(path.read_text())
        if not p['accepted'] or p['new_model_requests']!=0 or not p['journal_unchanged']:raise ValueError('unverified accepted source')
        root=Path(p['independent_root'])
        if file_sha256(root/'manifest.json')!=p['manifest_sha256'] or file_sha256(root/'checksums.json')!=p['checksums_sha256']:
            raise ValueError('accepted source bundle changed')
        loaded=load_dataset(root)
        if len(loaded)!=1:raise ValueError('each accepted proof must bind one source')
        row=loaded[0][0];slot=row['family_id'].removesuffix('-a0')
        task=next((t for t in tasks if t['slot_id']==slot),None)
        if task is None or task['split']!=row['split'] or slot in accepted:raise ValueError('accepted source slot mismatch')
        accepted[slot]=row;counts[row['split']]+=1
    remaining=[copy.deepcopy(t) for t in tasks if t['slot_id'] not in accepted]
    # Preserve original task identities/requirements while retaining the current
    # authorized execution recipe, rather than reverting older per-task R fields.
    current=seed['production']['tasks'][0]
    defaults={k:copy.deepcopy(current[k]) for k in ('background_true_cfg_scale','selected_edge_margin_pixels',
        'scene_refinement_request','background_refinement_request','refinement_request_origin') if k in current}
    for t in remaining:t.update(defaults)
    # Order from immutable task metadata only; no held-out media used to select K.
    scope=task_scope(seed['production']['tasks'][0],seed['models'])
    remaining.sort(key=lambda t:(t['split']!='train',task_scope(t,seed['models'])!=scope,t['slot_id']))
    if not any(t['split']=='train' and task_scope(t,seed['models'])==scope for t in remaining):
        raise ValueError('no unseen compatible training task remains for actual K reuse')
    if not 1<=len(validation_tasks)<=3:raise ValueError('predeclare one to three independent validation tasks')
    forbidden={t['slot_id'] for t in tasks};seeds={t['seed'] for t in tasks}
    for t in validation_tasks:
        if (t['slot_id'] in forbidden or t['seed'] in seeds or t['split']!='train' or
                task_scope(t,seed['models'])!=scope):raise ValueError('knowledge validation source is not independent and compatible')
        forbidden.add(t['slot_id']);seeds.add(t['seed'])
    for key in ('generation_reuse','model_reuse','shadow_region_corrections','surface_effect_contracts','rsi_candidate','agent_evidence'):
        seed['production'].pop(key,None)
    # The initial cost ledger remains attached; later calls are unioned by request ID.
    roots=pg.planner['runtime']['domain']['roots'];dataset_parent=Path(roots[seed['dataset_store']]).parent
    ds=dataset_parent/campaign_id;bundle=dataset_parent/(campaign_id+'-portable')
    stages=[{'id':'distill','kind':'distill'}]+[{'id':'independent-%02d'%i,'kind':'validation','task':t} for i,t in enumerate(validation_tasks)]+[
        {'id':'promote','kind':'promote'}]+[{'id':t['slot_id'],'kind':'main','task':t} for t in remaining]+[{'id':'export','kind':'export'}]
    identity_path=Path(env['DATAEVOLVER_RESIDENT_IDENTITY']);identity=json.loads(identity_path.read_text())
    pointer=Path(resident_pointer).absolute()
    if not pointer.is_symlink() or pointer.resolve()!=identity_path.resolve():
        raise ValueError('explicit host identity pointer must currently target the parent identity')
    if pointer.is_relative_to(pilot_directory):raise ValueError('cannot rewrite parent frozen identity')
    if len(identity['gpu_uuids'])!=1:raise ValueError('workflow cannot expose GPU1')
    gpu0=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if identity['gpu_uuids']!=[gpu0]:raise ValueError('resident visibility does not match physical GPU0')
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True)
    allocations=[line.strip() for line in processes.splitlines() if line.split(',')[1].strip()==str(identity['pid'])]
    if any(not line.startswith(gpu0+',') for line in allocations):raise ValueError('resident still allocates on GPU1')
    for path in pg.planner['runtime']['deployment_status_paths']:
        live=json.loads(Path(path).read_text())
        if live['state']!='READY' or live['pid']!=identity['pid']:raise ValueError('resident must be idle before identity handoff')
    out.mkdir(parents=True)
    shutil.copy2(pilot_directory/'evaluator.json',out/'evaluator.json')
    newidentity={**identity,'deployment_id':campaign_id,'authorized_job_roots':[str(out/'campaigns')],
                 'authorized_dataset_roots':[str(ds)]}
    atomic_write_json(out/'resident-identity.json',newidentity);(out/'resident-identity.json').chmod(0o600)
    for worker in seed['models']['workers'].values():
        if worker.get('resident'):worker['resident']['identity_sha256']=file_sha256(out/'resident-identity.json')
    env['DATAEVOLVER_RESIDENT_IDENTITY']=str(out/'resident-identity.json')
    # Placement changed only in API authority metadata; model bytes and UUID are fixed.
    scope=task_scope(seed['production']['tasks'][0],seed['models'])
    domain={'schema_version':SCHEMA,'production_contract':seed,'roots':roots,'stages':stages,
        'dataset_root':str(ds),'bundle_root':str(bundle),'evidence_root':str(ds.with_name(campaign_id+'-evidence')),
        'delivery_root':str(out/'delivery'),'accepted_proofs':list(map(str,all_proofs)),
        'accepted_counts':counts,'expected_counts':{'train':18,'validation':6,'test':6},
        'pilot_trial':str(pilot_path),'pilot_baseline':str(baseline_path),'pilot_proof':str(proof_path),
        'pilot_history':state.data['rsi_history'],'experience_scope':scope,
        'task_expression_defaults':defaults,
        'gpu0_uuid':identity['gpu_uuids'][0],'gpu_uuids':identity['gpu_uuids'],
        'limits':{'planner_per_task':8,'experiments_per_task':2},
        'evaluator_sha256':file_sha256(out/'evaluator.json'),
        'input_hashes':{str(p):file_sha256(p) for p in [pilot_path,baseline_path,proof_path,validation_path,*all_proofs]},
        'historical_config_roots':[str(Path(original_config).resolve().parent)],
        'historical_audit_roots':[str(pilot_directory),str(Path(original_config).resolve()),
            *[json.loads(p.read_text())['source_config'] for p in all_proofs if 'source_config' in json.loads(p.read_text())]]}
    n=sum(s['kind'] in {'main','validation'} for s in stages)
    if pg.planner['runtime']['domain'].get('surface_capability'):
        domain['surface_capability']=copy.deepcopy(pg.planner['runtime']['domain']['surface_capability'])
    domain['limits']['max_experiments']=n*domain['limits']['experiments_per_task']
    # Wall allowance is inherited from the parent campaign; no resource-wait deadline reset.
    from datetime import datetime
    left=pg.budgets.max_wall_seconds-(time.time()-datetime.fromisoformat(state.started_at).timestamp())
    if left<=0:raise ValueError('parent wall allowance exhausted')
    goal={'schema_version':'dataevolver.harness.goal.v1','campaign_id':campaign_id,
        'objective':{'real_data_execution':True,'kind':'training_runtime_workflow'},
        'initial_operator_id':TRIAL_ID,'route_candidates':[TRIAL_ID],'interventions':[],
        'budgets':workflow_budgets(domain,int(left)),
        'resource_policy':{**pg.resource_policy,'physical_gpu_indices':[0]},
        'frozen_evaluator':{'path':str(out/'evaluator.json'),'sha256':domain['evaluator_sha256']},
        'planner':{'kind':'runtime_rsi','runtime':{**pg.planner['runtime'],'domain':domain}},
        'initial_operator_parameters':{},'lease_seconds':int(left)+600}
    typed=GoalSpec.from_dict(goal);validate_workflow_contract(typed);atomic_write_json(out/'goal.json',goal)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO_ROOT,text=True).strip()
    subprocess.run(['git','archive','--format=tar','--output',str(out/'source.tar'),commit,'src','configs','pyproject.toml'],cwd=REPO_ROOT,check=True)
    with tarfile.open(out/'source.tar') as handle:handle.extractall(out/'source',filter='data')
    atomic_write_json(out/'frozen.json',{'source_commit':commit,'goal_sha256':file_sha256(out/'goal.json'),
        'source_hashes':{p.relative_to(out/'source').as_posix():file_sha256(p) for p in (out/'source').rglob('*') if p.is_file()},
        'source_archive_sha256':file_sha256(out/'source.tar'),'source_specific_repair_supplied':False,
        'parent_resource_deadline_epoch':seed['production']['resource_deadline_epoch']})
    atomic_write_json(out/'deployment-handoff.json',{'physical_gpu_indices':[0],'gpu0_uuid':gpu0,
        'resident_pid':identity['pid'],'allocations':allocations,'weights_reloaded':False,
        'old_identity_sha256':file_sha256(identity_path),'new_identity_sha256':file_sha256(out/'resident-identity.json')})
    os.environ.update(env);build_runtime(typed,out/'campaigns')
    # The host follows the existing identity pointer. Do not rotate during a model request.
    for path in pg.planner['runtime']['deployment_status_paths']:
        if json.loads(Path(path).read_text())['state']!='READY':raise ValueError('resident became busy; handoff aborted')
    link=pointer.with_name(pointer.name+'.workflow-next');link.symlink_to(out/'resident-identity.json');os.replace(link,pointer)
    (out/'env.sh').write_text('\n'.join('export '+k+'='+shlex.quote(v) for k,v in env.items())+'\n');(out/'env.sh').chmod(0o600)
    return {'status':'FROZEN','stages':len(stages),'remaining_main_records':len(remaining),
        'physical_gpu_indices':[0],'source_commit':commit,'model_calls':0,'source_specific_repair_supplied':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze')
    for name in ('pilot-config','original-config','out','campaign-id','validation-tasks','resident-pointer'):f.add_argument('--'+name,required=True)
    f.add_argument('--accepted-proof',action='append',required=True)
    for name in ('run','validate'):sub.add_parser(name).add_argument('--config-dir',required=True)
    a=p.parse_args(argv)
    if a.command=='freeze':value=freeze(a.pilot_config,a.original_config,a.accepted_proof,a.out,a.campaign_id,json.loads(Path(a.validation_tasks).read_text()),resident_pointer=a.resident_pointer)
    else:
        from .training_rsi_run import main as run
        return run([a.command,'--config-dir',a.config_dir])
    print(json.dumps(value));return 0


if __name__=='__main__':raise SystemExit(main())
